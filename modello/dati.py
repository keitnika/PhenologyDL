"""Immagini, gruppi descrittivi e suddivisione dei dati."""
import random
from functools import partial
import torch
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from modello.supporto import seed_worker
from datetime import datetime
from pathlib import Path
from typing import Iterable
import re
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from modello.supporto import choose_device, seed_worker
from sklearn.model_selection import StratifiedGroupKFold
from modello.supporto import read_json, write_json, project_path, set_seed

def make_transforms(config):
    p = config['preprocessing']
    a = config['augmentation']
    tail = [transforms.ToTensor(), transforms.Normalize(p['normalization_mean'], p['normalization_std'])]
    train = transforms.Compose([transforms.RandomResizedCrop(p['input_size'], scale=tuple(a['crop_scale']), ratio=tuple(a['crop_ratio']), interpolation=InterpolationMode.BICUBIC), transforms.RandomHorizontalFlip(a['horizontal_flip_probability']), transforms.ColorJitter(brightness=a['brightness'], contrast=a['contrast']), *tail])
    return (train, make_evaluation_transform(config))

def make_evaluation_transform(config):
    p = config['preprocessing']
    return transforms.Compose([transforms.Resize(p['resize_short_side'], interpolation=InterpolationMode.BICUBIC), transforms.CenterCrop(p['center_crop']), transforms.ToTensor(), transforms.Normalize(p['normalization_mean'], p['normalization_std'])])

def image_tensor(path, transform):
    try:
        with Image.open(path) as image:
            return transform(ImageOps.exif_transpose(image).convert('RGB'))
    except Exception as error:
        raise RuntimeError(f'Impossibile leggere {path}: {error}') from error

class ImageDataset(Dataset):

    def __init__(self, frame, image_root, transform):
        self.frame = frame.reset_index(drop=True)
        self.image_root, self.transform = (image_root, transform)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        return (image_tensor(self.image_root / row.image_name, self.transform), int(row.class_id), float(row.BBCH))

class CacheDataset(ImageDataset):

    def __init__(self, frame, image_root, config, seed, augmented_views):
        train, evaluation = make_transforms(config)
        super().__init__(frame, image_root, evaluation)
        self.train_transform, self.seed, self.views = (train, seed, augmented_views + 1)

    def __len__(self):
        return len(self.frame) * self.views

    def __getitem__(self, item):
        view, index = divmod(item, len(self.frame))
        state, python_state = (torch.random.get_rng_state(), random.getstate())
        try:
            if view:
                seed = self.seed + view * 1000003 + index
                torch.manual_seed(seed)
                random.seed(seed)
            row = self.frame.iloc[index]
            tensor = image_tensor(self.image_root / row.image_name, self.train_transform if view else self.transform)
        finally:
            torch.random.set_rng_state(state)
            random.setstate(python_state)
        return (tensor, view, index)

def make_loader(dataset, batch_size, workers, shuffle, seed):
    return DataLoader(dataset, batch_size=batch_size, num_workers=workers, shuffle=shuffle, pin_memory=False, worker_init_fn=partial(seed_worker, base_seed=seed), generator=torch.Generator().manual_seed(seed))
register_heif_opener()
REQUIRED_DUPLICATE_COLUMNS = {'group_id', 'tipo_duplicato', 'image_name'}
FILENAME_TIMESTAMP_RE = re.compile('(?<!\\d)((?:19|20)\\d{6})[_-](\\d{6})(?!\\d)', re.IGNORECASE)
EXIF_DATETIME_RE = re.compile('((?:19|20)\\d{2})[:\\-](\\d{2})[:\\-](\\d{2})[ T](\\d{2}):(\\d{2}):(\\d{2})')

class UnionFind:

    def __init__(self, items: Iterable[str]):
        self.parent = {item: item for item in items}
        self.rank = {item: 0 for item in items}

    def find(self, item: str) -> str:
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            parent = self.parent[item]
            self.parent[item] = root
            item = parent
        return root

    def union(self, first: str, second: str) -> None:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root == second_root:
            return
        if self.rank[first_root] < self.rank[second_root]:
            first_root, second_root = (second_root, first_root)
        self.parent[second_root] = first_root
        if self.rank[first_root] == self.rank[second_root]:
            self.rank[first_root] += 1

def parse_exif_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode('utf-8', errors='ignore')
    match = EXIF_DATETIME_RE.search(str(value))
    if not match:
        return None
    try:
        return datetime(*map(int, match.groups()))
    except ValueError:
        return None

def extract_exif_timestamp(path: Path) -> tuple[datetime | None, str | None]:
    try:
        with Image.open(path) as image:
            exif = image.getexif()
            original = None
            try:
                original = exif.get_ifd(34665).get(36867)
            except Exception:
                original = None
            if original is None:
                original = exif.get(36867)
            timestamp = parse_exif_datetime(original)
            if timestamp is not None:
                return (timestamp, None)
            timestamp = parse_exif_datetime(exif.get(306))
            return (timestamp, None)
    except Exception as exc:
        return (None, f'{type(exc).__name__}: {exc}')

def extract_filename_timestamp(value: object) -> datetime | None:
    match = FILENAME_TIMESTAMP_RE.search(str(value))
    if not match:
        return None
    try:
        return datetime.strptime('_'.join(match.groups()), '%Y%m%d_%H%M%S')
    except ValueError:
        return None

def scan_timestamps(frame: pd.DataFrame):
    timestamps: list[pd.Timestamp | pd.NaT] = []
    sources: list[str] = []
    errors: list[dict] = []
    total = len(frame)
    for position, row in frame.iterrows():
        timestamp, error = extract_exif_timestamp(Path(row['image_path']))
        if error:
            errors.append({'image_name': row['image_name'], 'error': error})
        if timestamp is not None:
            timestamps.append(pd.Timestamp(timestamp))
            sources.append('exif')
        else:
            timestamp = extract_filename_timestamp(row['original_name'])
            if timestamp is not None:
                timestamps.append(pd.Timestamp(timestamp))
                sources.append('nome_file')
            else:
                timestamps.append(pd.NaT)
                sources.append('nessuna')
        if (position + 1) % 100 == 0 or position + 1 == total:
            print(f'Controllo data e ora degli scatti: {position + 1} di {total}', flush=True)
    result = frame.copy()
    result['timestamp'] = pd.to_datetime(timestamps)
    result['fonte_timestamp'] = sources
    return (result, pd.DataFrame(errors, columns=['image_name', 'error']))

def union_timestamp_sessions(frame: pd.DataFrame, union_find: UnionFind, gap_minutes: float):
    timestamped = frame.loc[frame['timestamp'].notna()].sort_values(['vineyard_key', 'variety_key', 'timestamp', 'image_name'])
    max_gap = pd.Timedelta(minutes=gap_minutes)
    for _, subgroup in timestamped.groupby(['vineyard_key', 'variety_key'], sort=True, observed=True):
        previous_name = None
        previous_timestamp = None
        for row in subgroup.itertuples(index=False):
            if previous_name is not None and row.timestamp - previous_timestamp <= max_gap:
                union_find.union(previous_name, row.image_name)
            previous_name = row.image_name
            previous_timestamp = row.timestamp

class EmbeddingDataset(Dataset):

    def __init__(self, frame: pd.DataFrame):
        self.frame = frame.reset_index(drop=True)
        self.transform = transforms.Compose([transforms.Resize(256, interpolation=InterpolationMode.BICUBIC), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        path = Path(self.frame.iloc[index]['image_path'])
        try:
            with Image.open(path) as image:
                image = ImageOps.exif_transpose(image).convert('RGB')
                tensor = self.transform(image)
        except Exception as exc:
            raise RuntimeError(f'Impossibile leggere immagine {path}') from exc
        return (tensor, index)

def load_dinov2(repo: Path | None, device: torch.device):
    if repo is None or not repo.is_dir():
        raise FileNotFoundError(f'Repository ufficiale DINOv2 locale non trovata: {repo}')
    model = torch.hub.load(str(repo), 'dinov2_vitb14', source='local')
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model.to(device)

def compute_embeddings(frame: pd.DataFrame, repo: Path | None, batch_size: int, workers: int, seed: int):
    if frame.empty:
        return np.empty((0, 768), dtype=np.float32)
    device = choose_device()
    print(f'Calcolo la somiglianza tra immagini senza data e ora usando: {device}', flush=True)
    model = load_dinov2(repo, device)
    dataset = EmbeddingDataset(frame)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=device.type == 'cuda', worker_init_fn=partial(seed_worker, base_seed=seed), generator=generator)
    result = np.empty((len(dataset), 768), dtype=np.float32)
    completed = 0
    with torch.inference_mode():
        for images, indices in loader:
            features = model(images.to(device))
            if isinstance(features, dict):
                features = features['x_norm_clstoken']
            if features.ndim != 2 or features.shape[1] != 768:
                raise RuntimeError(f'Risposta inattesa di DINOv2: {tuple(features.shape)}')
            features = torch.nn.functional.normalize(features.float(), dim=1)
            result[indices.numpy()] = features.cpu().numpy()
            completed += len(indices)
            print(f'Somiglianza delle immagini calcolata: {completed} di {len(dataset)}', flush=True)
    del model
    if device.type == 'mps':
        torch.mps.empty_cache()
    return result

def union_embedding_clusters(frame: pd.DataFrame, embeddings: np.ndarray, union_find: UnionFind, threshold: float):
    if frame.empty:
        return
    working = frame.reset_index(drop=True).copy()
    working['embedding_index'] = np.arange(len(working))
    for _, subgroup in working.groupby(['vineyard_key', 'variety_key'], sort=True, observed=True):
        if len(subgroup) == 1:
            continue
        indices = subgroup['embedding_index'].to_numpy()
        clusterer = AgglomerativeClustering(n_clusters=None, metric='cosine', linkage='average', distance_threshold=threshold)
        labels = clusterer.fit_predict(embeddings[indices])
        names = subgroup['image_name'].to_numpy()
        for label in np.unique(labels):
            members = names[labels == label]
            for name in members[1:]:
                union_find.union(str(members[0]), str(name))

def confirmed_duplicate_group_ids(evaluation_path: Path) -> set[str]:
    evaluations = pd.read_csv(evaluation_path)
    required = {'group_id', 'valutazione'}
    missing = sorted(required - set(evaluations.columns))
    if missing:
        raise ValueError(f'Nel CSV delle valutazioni dei duplicati mancano le colonne {missing}: {evaluation_path}')
    normalized = evaluations['valutazione'].fillna('').astype(str).str.strip().str.casefold()
    confirmed = normalized.isin({'duplicato confermato', 'confirmed'})
    return set(evaluations.loc[confirmed, 'group_id'].astype(str))

def union_duplicate_groups(duplicate_csv: Path, evaluation_path: Path | None, valid_names: set[str], union_find: UnionFind):
    duplicates = pd.read_csv(duplicate_csv)
    missing = sorted(REQUIRED_DUPLICATE_COLUMNS - set(duplicates.columns))
    if missing:
        raise ValueError(f'Nel report dei duplicati mancano le colonne: {missing}')
    if evaluation_path is None:
        selected = duplicates
        policy = 'all exact and near-duplicate groups (no evaluation CSV found)'
    else:
        confirmed_ids = confirmed_duplicate_group_ids(evaluation_path)
        selected = duplicates[duplicates['group_id'].astype(str).isin(confirmed_ids)]
        policy = f'confirmed groups from {evaluation_path}'
    used_groups = 0
    ignored_names: set[str] = set()
    for _, subgroup in selected.groupby('group_id', sort=True, observed=True):
        names = [str(name) for name in subgroup['image_name']]
        eligible = [name for name in names if name in valid_names]
        ignored_names.update((name for name in names if name not in valid_names))
        if len(eligible) < 2:
            continue
        used_groups += 1
        for name in eligible[1:]:
            union_find.union(eligible[0], name)
    return (policy, used_groups, sorted(ignored_names))

def read_metadata(path, sheet, image_root, model_config):
    frame = pd.read_excel(path, sheet_name=sheet)
    required = {'image_name', 'original_name', 'vineyard', 'variety', 'data', 'BBCH'}
    if required - set(frame.columns):
        raise ValueError(f'Colonne mancanti: {sorted(required - set(frame.columns))}')
    if frame.image_name.isna().any() or frame.image_name.duplicated().any():
        raise ValueError('image_name contiene valori mancanti o duplicati')

    def class_id(value):
        try:
            number = float(value)
        except (ValueError, TypeError):
            return None
        if not number.is_integer():
            return None
        for item in model_config['classes']:
            if item['bbch_min'] <= number <= item['bbch_max']:
                return item['id']
        return None
    frame['class_id'] = frame.BBCH.map(class_id)
    invalid = frame[frame.class_id.isna()]
    if not invalid.empty:
        raise ValueError(f'BBCH fuori mappatura; correggere prima di procedere: {invalid.BBCH.value_counts(dropna=False).to_dict()}')
    frame['class_id'] = frame.class_id.astype(int)
    root = Path(image_root).resolve()

    def safe_path(name):
        if Path(str(name)).is_absolute() or '..' in Path(str(name)).parts:
            raise ValueError(f"image_name deve essere relativo e senza '..': {name}")
        path = (root / str(name)).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f'image_name esterno alla radice immagini: {name}')
        if not path.is_file():
            raise FileNotFoundError(path)
        return str(path)
    frame['image_path'] = frame.image_name.map(safe_path)
    for field in ('vineyard', 'variety'):
        frame[f'{field}_key'] = frame[field].astype(str).str.strip().str.casefold()
    frame['class_name'] = frame.class_id.map({c['id']: c['name'] for c in model_config['classes']})
    return frame

def save_splits(frame, destination, seed, ratios):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f'Usare una nuova cartella per non cambiare uno split esistente: {destination}')
    train_fraction, dev_fraction, test_fraction = ratios
    if min(ratios) <= 0 or abs(sum(ratios) - 1) > 1e-10:
        raise ValueError('Le frazioni devono essere positive e sommare a 1')
    # Group-aware 10-fold split: every correlated group stays entirely within
    # one partition. Two folds are reserved for dev/test; the remaining eight
    # form training (approximately 80/10/10). This avoids image-level leakage.
    if not np.isclose([train_fraction, dev_fraction, test_fraction], [0.8, 0.1, 0.1]).all():
        raise ValueError('La suddivisione group-aware attuale richiede quote 0.8/0.1/0.1.')
    if frame.group_id.nunique() < 10:
        raise ValueError('Servono almeno 10 gruppi distinti per la suddivisione group-aware 80/10/10.')
    splitter = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
    folds = list(splitter.split(frame, frame.class_id, groups=frame.group_id.astype(str)))
    dev_indices = folds[0][1]
    test_indices = folds[1][1]
    held_out = set(dev_indices) | set(test_indices)
    train_indices = np.array([i for i in range(len(frame)) if i not in held_out], dtype=int)
    train, dev, test = (frame.iloc[train_indices].copy(), frame.iloc[dev_indices].copy(), frame.iloc[test_indices].copy())
    group_sets = [set(part.group_id.astype(str)) for part in (train, dev, test)]
    if (group_sets[0] & group_sets[1]) or (group_sets[0] & group_sets[2]) or (group_sets[1] & group_sets[2]):
        raise RuntimeError('Errore interno: un gruppo correlato compare in più split.')
    missing_train = sorted(set(range(12)) - set(train.class_id.astype(int)))
    if missing_train:
        raise ValueError(f'La suddivisione per gruppi lascia classi senza immagini di training: {missing_train}. Occorre rivedere i gruppi o raccogliere altri dati.')
    destination.mkdir(parents=True)
    columns = ['image_name', 'group_id', 'class_id', 'class_name', 'BBCH']
    for name, part in (('train', train), ('dev', dev), ('test', test)):
        part[columns].sort_values('image_name').to_csv(destination / f'{name}.csv', index=False)
    parts = load_splits(destination)
    summary = {name: len(part) for name, part in parts.items()}
    summary['seed'] = seed
    summary['split_method'] = 'StratifiedGroupKFold 10 folds; groups kept intact'
    summary['fractions'] = {'train': train_fraction, 'dev': dev_fraction, 'test': test_fraction}
    summary['groups_are_not_split_constraints'] = False
    summary['correlated_groups_across_splits'] = {f'{a}_{b}': len(set(parts[a].group_id) & set(parts[b].group_id)) for a, b in (('train', 'dev'), ('train', 'test'), ('dev', 'test'))}
    write_json(destination / 'config.json', summary)
    counts = pd.DataFrame({name: part.class_id.value_counts() for name, part in parts.items()}).fillna(0).astype(int).sort_index()
    counts.to_csv(destination / 'class_counts.csv', index_label='class_id')
    print(f"Dati suddivisi per gruppi: {len(parts['train'])} immagini per imparare, {len(parts['dev'])} per la validazione, {len(parts['test'])} per il test finale, che resta chiuso.", flush=True)
    return parts

def load_splits(directory):
    directory = Path(directory)
    parts = {name: pd.read_csv(directory / f'{name}.csv') for name in ('train', 'dev', 'test')}
    for name, frame in parts.items():
        if frame.empty or frame.image_name.isna().any() or frame.image_name.duplicated().any():
            raise ValueError(f'Split {name} vuoto o nomi immagine non validi')
        if not {'image_name', 'class_id', 'group_id', 'BBCH'} <= set(frame):
            raise ValueError(f'Colonne mancanti nello split {name}')
        if not frame.class_id.isin(range(12)).all():
            raise ValueError(f'Classi non valide in {name}')
        if frame.BBCH.isna().any() or not frame.BBCH.between(0, 99).all():
            raise ValueError(f'Valori BBCH non validi in {name}')
        if frame.group_id.isna().any():
            raise ValueError(f'Gruppi mancanti in {name}')
        for filename in frame.image_name:
            if Path(filename).is_absolute() or '..' in Path(filename).parts:
                raise ValueError(f'Nome immagine non relativo: {filename}')
    for a, b in (('train', 'dev'), ('train', 'test'), ('dev', 'test')):
        assert not set(parts[a].image_name) & set(parts[b].image_name), 'Immagini presenti in più split'
    config = directory / 'config.json'
    if config.is_file():
        recorded = read_json(config)
        for name in parts:
            assert len(parts[name]) == recorded[name], 'Dimensione dello split modificata'
    return parts


def prepara_dati(settings, run):
    """Prepara i dati senza sovrascrivere gli split o aprire il test esistente."""
    data = settings['dati']
    ratios = settings['suddivisione']
    metadata = project_path(data['metadati'])
    image_root = project_path(data['immagini'])
    if not metadata.is_file():
        raise ValueError(f"Non trovo il file delle etichette '{metadata}'. Controlla dati.metadati in config.json.")
    if not image_root.is_dir():
        raise ValueError(f"Non trovo la cartella immagini '{image_root}'. Controlla dati.immagini in config.json.")
    set_seed(ratios['seme'])
    config = read_json(project_path('configs/model.json'))
    frame = read_metadata(metadata, data['foglio'], image_root, config)
    # Riutilizzo solo se immagini, etichette, seed e percentuali coincidono.
    existing = project_path('outputs/data/splits')
    requested = [ratios[key] for key in ('training', 'dev', 'test')]
    if min(requested) <= 0 or abs(sum(requested) - 1) > 1e-10:
        raise ValueError("Le quote training, dev e test devono essere positive e sommare a 1.")
    if all((existing / f'{name}.csv').is_file() for name in ('train', 'dev', 'test')):
        # Older classification-only split manifests do not contain the numeric BBCH target.
        headers = [set(pd.read_csv(existing / f'{name}.csv', nrows=0).columns) for name in ('train', 'dev', 'test')]
        if all('BBCH' in columns for columns in headers):
            previous = load_splits(existing)
            previous_config = read_json(existing / 'config.json')
            old_frame = pd.concat(previous.values())
            old_labels = dict(zip(old_frame.image_name, zip(old_frame.class_id, old_frame.BBCH)))
            new_labels = dict(zip(frame.image_name, zip(frame.class_id, frame.BBCH)))
            same_ratios = previous_config.get('fractions') == dict(zip(('train','dev','test'), requested))
            if (old_labels == new_labels and same_ratios and previous_config.get('seed') == ratios['seme']
                    and previous_config.get('groups_are_not_split_constraints') is False):
                print('Immagini ed etichette invariate: uso la suddivisione già presente, senza modificarla.', flush=True)
                return existing
        else:
            print('La suddivisione precedente non contiene i valori BBCH numerici: ne creo una nuova per il modello multi-output.', flush=True)
    print('Preparo una nuova suddivisione in una cartella separata. Le suddivisioni precedenti restano intatte.', flush=True)
    group_path = project_path(data['gruppi']) if data.get('gruppi') else None
    groups = None
    if group_path and group_path.is_file():
        groups = pd.read_csv(group_path, usecols=['image_name', 'group_id'])
        if groups.image_name.duplicated().any():
            raise ValueError('Il file dei gruppi contiene nomi ripetuti. Correggilo o imposta dati.gruppi a null per ricalcolarlo.')
        if not set(frame.image_name) <= set(groups.image_name):
            print('Il file dei gruppi non copre le nuove immagini: ricalcolo i gruppi prima della divisione.', flush=True)
            groups = None
    if groups is not None:
        frame = frame.merge(groups, on='image_name', how='left', validate='one_to_one')
        if frame.group_id.isna().any():
            raise ValueError('Nel file dei gruppi mancano alcuni identificativi.')
    else:
        frame, errors = scan_timestamps(frame)
        union = UnionFind(frame.image_name)
        union_timestamp_sessions(frame, union, config['grouping_defaults']['session_gap_minutes'])
        missing = frame[frame.timestamp.isna()].reset_index(drop=True)
        if not missing.empty:
            features = compute_embeddings(missing, project_path(settings['ambiente']['dinov2']),
                settings['addestramento']['immagini_insieme'], settings['addestramento']['processi_lettura'], ratios['seme'])
            union_embedding_clusters(missing, features, union, config['grouping_defaults']['cluster_distance'])
        if data.get('duplicati'):
            duplicates = project_path(data['duplicati'])
            if not duplicates.is_file():
                raise ValueError(f"Non trovo il report duplicati '{duplicates}'. Correggi dati.duplicati o imposta null se non disponibile.")
            reviews = project_path(data['valutazioni_duplicati']) if data.get('valutazioni_duplicati') else None
            union_duplicate_groups(duplicates, reviews, set(frame.image_name), union)
        else:
            print('Avviso: nessun report dei duplicati fornito; i gruppi si basano su sessioni e similarità.', flush=True)
        components = {}
        for name in sorted(frame.image_name):
            components.setdefault(union.find(name), []).append(name)
        mapping = {name: f'G{index:04d}' for index, names in enumerate(components.values(), 1) for name in names}
        frame['group_id'] = frame.image_name.map(mapping)
        if not errors.empty:
            print(f'Avviso: {len(errors)} immagini con informazioni EXIF non leggibili; usati i criteri alternativi previsti.', flush=True)
    destination = run / 'dati'
    destination.mkdir(parents=True)
    frame[['image_name','group_id','class_id','class_name']].to_csv(destination / 'groups.csv', index=False)
    splits = destination / 'splits'
    save_splits(frame, splits, ratios['seme'], requested)
    return splits
