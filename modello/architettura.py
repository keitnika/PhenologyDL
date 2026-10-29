"""DINOv2 backbone with classification and continuous BBCH regression heads."""
from pathlib import Path
import torch
from torch import nn
from modello.supporto import read_json


def validate_config(config):
    classes = config['classes']
    assert [c['id'] for c in classes] == list(range(12)), 'Sono richieste le 12 classi ordinate'
    assert config['backbone']['torch_hub_model'] == 'dinov2_vitb14'
    assert config['architecture']['representation_dimension'] == 1536
    phases = {p['id']: p for p in config['level1_phases']}
    for item in classes:
        phase = phases[item['level1']]
        assert phase['bbch_min'] <= item['bbch_min'] <= item['bbch_max'] <= phase['bbch_max']


def load_backbone(config, repository, pretrained):
    repository = Path(repository).expanduser().resolve()
    if not (repository / 'hubconf.py').is_file():
        raise FileNotFoundError(f'Repository ufficiale DINOv2 locale non trovata: {repository}')
    return torch.hub.load(str(repository), config['backbone']['torch_hub_model'], source='local', pretrained=pretrained)


class BBCHClassifier(nn.Module):
    """Shared DINOv2 representation, 12-class head and scalar BBCH head."""
    def __init__(self, backbone, class_count=12):
        super().__init__()
        self.backbone = backbone
        self.feature_norm = nn.LayerNorm(1536)
        self.head = nn.Linear(1536, class_count)
        self.bbch_linear = nn.Linear(1536, 1)

    def _heads(self, representation):
        representation = self.feature_norm(representation)
        logits = self.head(representation)
        # BBCH is defined on [0, 99]; sigmoid keeps the regression estimate in this range.
        bbch_estimate = 99.0 * torch.sigmoid(self.bbch_linear(representation).squeeze(1))
        return logits, bbch_estimate

    def forward(self, images):
        features = self.backbone.forward_features(images)
        representation = torch.cat([features['x_norm_clstoken'], features['x_norm_patchtokens'].mean(dim=1)], dim=1)
        return self._heads(representation)


class PartialClassifier(BBCHClassifier):
    def __init__(self, backbone, class_count, trainable_blocks):
        super().__init__(backbone, class_count)
        self.trainable_blocks = trainable_blocks
        if not 1 <= trainable_blocks <= len(backbone.blocks):
            raise ValueError('Numero di blocchi allenabili non valido')
        self.backbone.requires_grad_(False)
        for block in self.backbone.blocks[-trainable_blocks:]:
            block.requires_grad_(True)
        self.backbone.norm.requires_grad_(True)

    def train(self, mode=True):
        super().train(mode)
        for block in self.backbone.blocks[:-self.trainable_blocks]:
            block.eval()
        return self

    def forward(self, images):
        with torch.no_grad():
            tokens = self.backbone.prepare_tokens_with_masks(images, None)
            for block in self.backbone.blocks[:-self.trainable_blocks]:
                tokens = block(tokens)
        tokens = tokens.detach()
        for block in self.backbone.blocks[-self.trainable_blocks:]:
            tokens = block(tokens)
        normalized = self.backbone.norm(tokens)
        representation = torch.cat([normalized[:, 0], normalized[:, self.backbone.num_register_tokens + 1:].mean(dim=1)], dim=1)
        return self._heads(representation)


class CachedHead(nn.Module):
    def __init__(self, class_count):
        super().__init__()
        self.norm = nn.LayerNorm(1536)
        self.linear = nn.Linear(1536, class_count)
        self.bbch_linear = nn.Linear(1536, 1)

    def forward(self, features):
        representation = self.norm(features)
        logits = self.linear(representation)
        bbch_estimate = 99.0 * torch.sigmoid(self.bbch_linear(representation).squeeze(1))
        return logits, bbch_estimate


def load_checkpoint(path, repository, device, config_path=None):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if 'state_dict' in checkpoint:
        config, state = (checkpoint['config'], checkpoint['state_dict'])
    else:
        if config_path is None:
            raise ValueError('Per un modello storico con soli pesi serve anche la sua configurazione di architettura.')
        config, state = (read_json(config_path), checkpoint)
    validate_config(config)
    model = BBCHClassifier(load_backbone(config, repository, False), len(config['classes']))
    model.load_state_dict(state, strict=True)
    return (model.to(device).eval(), config)


def save_checkpoint(path, model, config):
    state = {key: value.detach().cpu().float() for key, value in model.state_dict().items()}
    torch.save({'state_dict': state, 'config': config}, path)
