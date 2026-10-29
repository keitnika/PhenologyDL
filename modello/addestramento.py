"""Due stadi invariati: apprendimento della testa e degli ultimi quattro blocchi."""
import math
import numpy as np
import torch
import hashlib
import json
from modello.supporto import clear_device
from modello.dati import CacheDataset, make_loader
from modello.architettura import load_backbone
import copy
import time
import pandas as pd
from modello.supporto import choose_device, clear_device, read_json, set_seed, write_json, project_path
from modello.dati import ImageDataset, make_loader, make_transforms
from modello.dati import load_splits
from modello.architettura import CachedHead, PartialClassifier, load_backbone, save_checkpoint, validate_config
from modello.valutazione import predict_frame
from modello.valutazione import metrics

def class_weights(labels, class_count):
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=class_count)
    if np.any(counts == 0):
        raise ValueError(f'Training senza le classi {np.flatnonzero(counts == 0).tolist()}')
    weights = 1 / np.sqrt(counts)
    return torch.tensor(weights / weights.mean(), dtype=torch.float32)

def ordinal_loss(logits, labels, weights, sigma):
    if sigma <= 0:
        raise ValueError('Il modello selezionato richiede sigma > 0')
    classes = torch.arange(logits.shape[1], device=labels.device, dtype=torch.float32)
    distances = classes.unsqueeze(0) - labels.float().unsqueeze(1)
    targets = torch.exp(-distances ** 2 / (2 * sigma ** 2))
    targets /= targets.sum(dim=1, keepdim=True)
    losses = -(targets * torch.log_softmax(logits, dim=1)).sum(dim=1)
    return (losses * weights[labels]).mean()

def joint_loss(logits, bbch_prediction, class_labels, bbch_targets, weights, sigma, regression_weight=1.0):
    class_loss = ordinal_loss(logits, class_labels, weights, sigma)
    # Smooth-L1 on BBCH normalized to [0,1], rescaled to keep the regression
    # contribution interpretable in approximately BBCH-unit loss scale.
    regression_loss = torch.nn.functional.smooth_l1_loss(
        bbch_prediction / 99.0, bbch_targets.float() / 99.0, beta=0.05
    ) * 99.0
    return class_loss + regression_weight * regression_loss, class_loss.detach(), regression_loss.detach()

def schedule(step, warmup_steps, total_steps):
    if step < warmup_steps:
        return (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return 0.5 * (1 + math.cos(math.pi * min(progress, 1)))

def partial_optimizer(model, config):
    s = config['stage2']
    groups = [{'params': list(model.feature_norm.parameters()) + list(model.head.parameters()) + list(model.bbch_linear.parameters()), 'lr': s['head_learning_rate'], 'name': 'heads'}, {'params': model.backbone.norm.parameters(), 'lr': s['last_block_learning_rate'], 'name': 'final_norm'}]
    for depth, block in enumerate(reversed(model.backbone.blocks[-s['trainable_blocks']:])):
        groups.append({'params': block.parameters(), 'lr': s['last_block_learning_rate'] * s['layerwise_decay'] ** depth, 'name': f'block_{len(model.backbone.blocks) - 1 - depth}'})
    return torch.optim.AdamW(groups, weight_decay=config['weight_decay'])

def build_cache(parts, image_root, config, training, destination, device):
    allowed = set(parts['train'].image_name) | set(parts['dev'].image_name)
    assert not allowed & set(parts['test'].image_name)
    files = []
    for name in sorted(allowed):
        stat = (image_root / name).stat()
        files.append((name, stat.st_size, stat.st_mtime_ns))
    fingerprint = hashlib.sha256(json.dumps({'files': files, 'model': config, 'seed': training['seed'], 'views': training['augmented_views'], 'repository': str(training['dinov2_repo']), 'train_manifest': parts['train'].to_dict('records'), 'dev_manifest': parts['dev'].to_dict('records')}, sort_keys=True).encode()).hexdigest()
    path = destination / 'embeddings.pt'
    backbone = load_backbone(config, training['dinov2_repo'], True).to(device).eval()
    backbone.requires_grad_(False)
    cache = {'fingerprint': fingerprint}
    for name in ('train', 'dev'):
        augmented = training['augmented_views'] if name == 'train' else 0
        dataset = CacheDataset(parts[name], image_root, config, training['seed'], augmented)
        tensor = torch.full((augmented + 1, len(parts[name]), 1536), float('nan'))
        loader = make_loader(dataset, training['cache_batch_size'], training['workers'], False, training['seed'])
        with torch.inference_mode():
            for batch, (images, views, indices) in enumerate(loader):
                features = backbone.forward_features(images.to(device))
                values = torch.cat([features['x_norm_clstoken'], features['x_norm_patchtokens'].mean(dim=1)], dim=1)
                tensor[views, indices] = values.float().cpu()
                if batch % 10 == 0:
                    insieme = 'addestramento' if name == 'train' else 'validazione'
                    print(f'Preparazione delle immagini di {insieme}: gruppo {batch + 1} di {len(loader)}.', flush=True)
        if not torch.isfinite(tensor).all():
            raise ValueError('Cache incompleta o valori non finiti')
        cache[name] = tensor
    destination.mkdir(parents=True, exist_ok=True)
    torch.save(cache, path)
    del backbone
    clear_device(device)
    return cache

def run_training(training):
    config = read_json(project_path(training['model_config']))
    size = training['input_size']
    if size < 14 or size % 14:
        raise ValueError("'lato_immagini' deve essere un multiplo di 14, per esempio 336.")
    config['preprocessing'].update(input_size=size, center_crop=size, resize_short_side=int(round(size / 0.875)))
    validate_config(config)
    if not training['image_root'] or not training['dinov2_repo']:
        raise ValueError('Configurare image_root e dinov2_repo oppure passarli dalla riga di comando')
    image_root = project_path(training['image_root'])
    training['dinov2_repo'] = str(project_path(training['dinov2_repo']))
    parts = load_splits(project_path(training['splits_dir']))
    output = project_path(training['output_dir'])
    if output.exists():
        raise FileExistsError('Usare una nuova cartella di training: non è prevista la ripresa')
    s1, s2 = (training['stage1'], training['stage2'])
    if min(s1['epochs'], s2['epochs']) < 1:
        raise ValueError("Entrambi gli stadi richiedono almeno un'epoca")
    if training['augmented_views'] < 1:
        raise ValueError('È richiesta almeno una vista aumentata')
    if s2['effective_batch_size'] % s2['physical_batch_size']:
        raise ValueError('Il batch effettivo deve essere multiplo di quello fisico')
    weights = class_weights(parts['train'].class_id, len(config['classes']))
    output.mkdir(parents=True)
    runtime = copy.deepcopy(config)
    runtime['training'] = copy.deepcopy(training)
    write_json(output / 'config.json', runtime)
    weights_frame = pd.DataFrame({'class_id': range(len(weights)), 'weight': weights.numpy()})
    weights_frame.to_csv(output / 'class_weights.csv', index=False)
    device = choose_device(training['device'])
    print(f'Dispositivo: {device}', flush=True)
    set_seed(training['seed'])
    cache = build_cache(parts, image_root, config, training, output / 'cache', device)
    set_seed(training['seed'])
    head = CachedHead(len(config['classes'])).to(device)
    labels = torch.tensor(parts['train'].class_id.to_numpy(), dtype=torch.long)
    bbch_labels = torch.tensor(parts['train'].BBCH.to_numpy(dtype=np.float32), dtype=torch.float32)
    dev_labels = parts['dev'].class_id.to_numpy()
    dev_bbch = parts['dev'].BBCH.to_numpy(dtype=np.float32)
    regression_weight = float(training.get('regression_loss_weight', 1.0))
    weights = weights.to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=s1['learning_rate'], weight_decay=training['weight_decay'])
    updates = math.ceil(len(labels) / s1['batch_size'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: schedule(step, updates * s1['warmup_epochs'], updates * s1['epochs']))
    history = []
    best = -1.0
    for epoch in range(s1['epochs']):
        started = time.perf_counter()
        generator = torch.Generator().manual_seed(training['seed'] + epoch * 101)
        views = torch.randint(1, training['augmented_views'] + 1, (len(labels),), generator=generator)
        order = torch.randperm(len(labels), generator=generator)
        head.train()
        total = 0.0
        for start in range(0, len(order), s1['batch_size']):
            index = order[start:start + s1['batch_size']]
            features = cache['train'][views[index], index].to(device)
            target = labels[index].to(device)
            bbch_target = bbch_labels[index].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, bbch_prediction = head(features)
            loss, class_loss, regression_loss = joint_loss(logits, bbch_prediction, target, bbch_target, weights, training['sigma'], regression_weight)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total += float(loss.detach().cpu()) * len(index)
        head.eval()
        with torch.inference_mode():
            dev_outputs = [head(batch.to(device)) for batch in cache['dev'][0].split(s1['batch_size'])]
            predictions = torch.cat([item[0].cpu() for item in dev_outputs])
            dev_bbch_predictions = torch.cat([item[1].cpu() for item in dev_outputs]).numpy()
        values, _ = metrics(dev_labels, predictions.argmax(1).numpy())
        values['bbch_mae'] = float(np.mean(np.abs(dev_bbch - dev_bbch_predictions)))
        row = {'stage': 1, 'epoch': epoch + 1, 'train_loss': total / len(labels), 'seconds': time.perf_counter() - started, **{f'dev_{k}': v for k, v in values.items()}}
        history.append(row)
        pd.DataFrame(history).to_csv(output / 'training_log.csv', index=False)
        if values['macro_f1'] > best:
            best = values['macro_f1']
            torch.save(head.state_dict(), output / 'best_head_validation.pt')
        print(f"Prima fase: epoca {epoch + 1} di {s1['epochs']} completata. Immagini corrette in validazione: {values['accuracy']:.1%}; indice F1 medio tra le classi: {values['macro_f1']:.3f}; errore di apprendimento: {row['train_loss']:.4f}.", flush=True)
    final_head = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
    torch.save(final_head, output / 'final_head.pt')
    del head, optimizer, scheduler, cache
    clear_device(device)
    set_seed(training['seed'])
    model = PartialClassifier(load_backbone(config, training['dinov2_repo'], True), len(config['classes']), s2['trainable_blocks']).to(device)
    model.feature_norm.load_state_dict({'weight': final_head['norm.weight'], 'bias': final_head['norm.bias']})
    model.head.load_state_dict({'weight': final_head['linear.weight'], 'bias': final_head['linear.bias']})
    model.bbch_linear.load_state_dict({'weight': final_head['bbch_linear.weight'], 'bias': final_head['bbch_linear.bias']})
    optimizer = partial_optimizer(model, training)
    accumulation = s2['effective_batch_size'] // s2['physical_batch_size']
    batches = math.ceil(len(labels) / s2['physical_batch_size'])
    updates = math.ceil(batches / accumulation)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: schedule(step, updates * s2['warmup_epochs'], updates * s2['epochs']))
    train_transform, _ = make_transforms(config)
    best_bbch_mae = float('inf')
    best_macro_f1 = -1.0
    for epoch in range(s2['epochs']):
        started = time.perf_counter()
        loader = make_loader(ImageDataset(parts['train'], image_root, train_transform), s2['physical_batch_size'], training['workers'], True, training['seed'] + epoch * 101)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for batch_index, (images, target, bbch_target) in enumerate(loader):
            logits, bbch_prediction = model(images.to(device))
            loss, class_loss, regression_loss = joint_loss(logits, bbch_prediction, target.to(device), bbch_target.to(device), weights, training['sigma'], regression_weight)
            (loss / accumulation).backward()
            if (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(loader):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            total += float(loss.detach().cpu()) * len(target)
        probabilities, bbch_predictions = predict_frame(model, parts['dev'], image_root, config, device, s2['physical_batch_size'], training['workers'], training['seed'])
        values, _ = metrics(dev_labels, probabilities.argmax(1))
        values['bbch_mae'] = float(np.mean(np.abs(dev_bbch - bbch_predictions)))
        row = {'stage': 2, 'epoch': epoch + 1, 'train_loss': total / len(labels), 'seconds': time.perf_counter() - started, **{f'dev_{k}': v for k, v in values.items()}}
        history.append(row)
        pd.DataFrame(history).to_csv(output / 'training_log.csv', index=False)
        if values['bbch_mae'] < best_bbch_mae:
            best_bbch_mae = values['bbch_mae']
            save_checkpoint(output / 'best_validation.pt', model, runtime)
        if values['macro_f1'] > best_macro_f1:
            best_macro_f1 = values['macro_f1']
            save_checkpoint(output / 'best_classification_validation.pt', model, runtime)
        print(f"Seconda fase: epoca {epoch + 1} di {s2['epochs']} completata. Immagini corrette in validazione: {values['accuracy']:.1%}; indice F1 medio tra le classi: {values['macro_f1']:.3f}; errore di apprendimento: {row['train_loss']:.4f}.", flush=True)
        del loader
        clear_device(device)
    save_checkpoint(output / 'model_final.pt', model, runtime)
    write_json(output / 'complete.json', {'completed': True, 'test_images_loaded': 0, 'final_checkpoint_policy': 'model_final.pt is last epoch; best_validation.pt minimizes raw validation BBCH MAE; best_classification_validation.pt maximizes validation macro F1', 'outputs': ['12-class BBCH prediction', 'continuous BBCH estimate', 'top-class softmax confidence'], 'training_images': len(parts['train']), 'dev_images': len(parts['dev'])})
    return output / 'model_final.pt'
