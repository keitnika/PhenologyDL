"""Codice interno: usare train.py e predict.py per le operazioni normali."""
import warnings

# xFormers è facoltativo; la sua assenza è normale sui Mac e non richiede installazioni.
warnings.filterwarnings('ignore', message='xFormers is not available.*', category=UserWarning)
