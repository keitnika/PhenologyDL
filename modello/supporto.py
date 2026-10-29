"""Configurazioni e strumenti condivisi; nessun dato viene modificato."""
import gc
import hashlib
import json
import random
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')

def project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path

def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def seed_worker(worker_id, *, base_seed):
    worker_seed = (base_seed + worker_id) % 2 ** 32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def choose_device(requested='auto'):
    if requested != 'auto':
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

def clear_device(device):
    gc.collect()
    if device.type == 'mps':
        torch.mps.empty_cache()
    elif device.type == 'cuda':
        torch.cuda.empty_cache()


def impostazioni(path="config.json"):
    """Legge l'unico file pubblico, senza richiedere modifiche ai sorgenti."""
    path = project_path(path)
    if not path.is_file():
        raise ValueError(f"Manca '{path.name}'. Ripristina il file di configurazione del progetto.")
    try:
        settings = read_json(path)
    except json.JSONDecodeError as error:
        raise ValueError(f"Il file '{path.name}' non è valido: controlla virgole e virgolette, riga {error.lineno}.") from error
    required = {"dati", "suddivisione", "addestramento", "previsione", "ambiente", "risultati"}
    if required - settings.keys():
        raise ValueError(f"Sezioni mancanti nella configurazione: {', '.join(sorted(required - settings.keys()))}")
    return settings


def messaggio_errore(error):
    text = str(error)
    if isinstance(error, KeyError):
        return f"Manca l'impostazione {text} in config.json. Confronta il file con le istruzioni del README."
    if 'least populated class' in text or ('number of classes' in text and ('test_size' in text or 'train_size' in text)):
        return "Le immagini per classe non bastano per la suddivisione richiesta. Aggiungi esempi delle classi rare o chiedi al responsabile di controllare le quote."
    if 'Training senza le classi' in text:
        return "Nel training mancano alcune delle 12 classi. Controlla etichette e numerosità prima di riavviare. Dettagli: " + text
    if 'Error(s) in loading state_dict' in text:
        return "Il file del modello non è compatibile con questa architettura. Usa un modello prodotto da questo progetto o chiedi al responsabile tecnico."
    if "out of memory" in text.lower():
        return "Memoria insufficiente. Riduci 'immagini_insieme' in config.json e mantieni 'immagini_per_aggiornamento' un suo multiplo. Poi rilancia il comando."
    return text or "Operazione non completata. Controlla i percorsi e la configurazione."
