# Requisiti confermati

Questo documento descrive cosa deve fare la versione pulita. I valori derivano dal codice originale e dalle decisioni approvate dall'utente. Non descrive risultati storici.

## Funzione del modello

- Ricevere una o più immagini di vite e assegnare una delle 12 classi BBCH ordinate.
- Restituire classe, intervallo BBCH, probabilità di ciascuna classe, fase generale A–G, stima BBCH continua grezza (`BBCH_stimato_float`, non limitata all'intervallo della classe) e `confidence_score` (probabilità softmax massima, non calibrata).
- I checkpoint precedenti, con sola testa di classificazione, NON sono compatibili: serve un nuovo addestramento. Il checkpoint è il file con i pesi, cioè i numeri appresi durante l'addestramento.
- Non introdurre classificazione del singolo codice BBCH: per esempio 51 e 53 appartengono ancora alla stessa classe.

## Architettura (backbone invariato, due teste)

- DINOv2 ufficiale `dinov2_vitb14`: il backbone, cioè la parte che estrae informazioni visive.
- Concatenazione del token CLS, una descrizione generale dell'immagine, e della media dei patch token, le descrizioni delle sue porzioni.
- 1536 valori, LayerNorm, uno strato lineare con 12 uscite (`head`) e uno strato lineare con 1 uscita (`bbch_linear`), con sigmoide scalata a 0–99, per il BBCH continuo.
- Ingresso 336 × 336, normalizzazione ImageNet e interpolazione bicubica.
- Ordine delle classi e fasi definito soltanto in `configs/model.json`.

| Indice | Classe | BBCH inclusi | Fase |
|---|---|---|---|
| 0 | gem_dev | 00–09 | A |
| 1 | leaf_dev | 10–19 | B |
| 2 | 53 | 51–53 | C |
| 3 | 55 | 55–56 | C |
| 4 | 57 | 57–59 | C |
| 5 | flowering | 61–69 | D |
| 6 | 73 | 71–73 | E |
| 7 | 75 | 75 | E |
| 8 | 77 | 77–79 | E |
| 9 | 83 | 81–83 | F |
| 10 | 85 | 85 | F |
| 11 | 87_senescence | 87–99 | G |

## Dati e nuova divisione approvata

- Dati esterni alla nuova cartella, non copiati né modificati.
- Metadati Excel: `image_name`, `original_name`, `vineyard`, `variety`, `data`, `BBCH`. Nome del foglio configurabile.
- Immagini JPEG/JPG, HEIC/HEIF e PNG per la previsione; file singolo o cartella con ricerca nelle sottocartelle. Apertura HEIC registrata a livello di modulo, orientamento EXIF corretto prima delle trasformazioni. I formati aggiunti non cambiano l'architettura né le trasformazioni.
- Nomi univoci e relativi alla radice immagini. Nomi duplicati, file mancanti e BBCH non mappati interrompono la preparazione con un messaggio esplicito.
- Nuovo split **per gruppi** (`StratifiedGroupKFold`, 10 fold, seed 42): 8 fold training, 1 dev, 1 test, circa 80/10/10. Servono almeno 10 gruppi e quote 0.8/0.1/0.1. Ogni classe deve avere immagini di training.
- Il training serve ad apprendere; il dev serve a controllare l'andamento; il test serve alla valutazione finale.
- Un gruppo correlato resta interamente in un solo insieme (nessuna contaminazione tra training, dev e test). Gli split salvati contengono anche il BBCH numerico; split senza questa colonna non vengono riusati.
- Conservare separatamente il calcolo dei gruppi per descrizione e bootstrap, cioè ricampionamento di gruppi per stimare l'incertezza: sessioni con pausa massima 10 minuti; immagini senza timestamp raggruppate per similarità DINOv2 con distanza coseno 0,15; unione dei duplicati da un CSV esterno, facoltativamente filtrato dalle valutazioni confermate.
- La preparazione può riusare una tabella esterna di gruppi che copra tutte le immagini. Non riusa i vecchi split.

## Training e validation

- Una sola configurazione: quella derivata da B_lpft_4blocks. Nessun confronto fra varianti e nessuna cross-validation, cioè nessun addestramento ripetuto su cinque divisioni.
- Stadio 1: backbone congelato, testa addestrata per 5 epoche. Un'epoca è un passaggio su tutte le immagini di training.
- Cache, cioè risultati intermedi salvati: 1 vista normale e 5 viste aumentate per immagine di training. Solo la vista normale per il dev. Nessuna immagine del test nella cache.
- Una vista aumentata scelta con seed fissato per ogni immagine e ogni epoca dello stadio 1.
- Stadio 2: 9 epoche, ultimi 4 blocchi e normalizzazione finale allenabili. I primi 8 blocchi sono eseguiti senza conservare i calcoli per il gradiente.
- AdamW; weight decay 0,05; learning rate testa 0,001 nello stadio 1 e 0,0001 nello stadio 2; ultimo blocco 0,00002, decadimento 0,75 per blocco precedente. Il learning rate regola quanto cambiano i pesi a ogni aggiornamento.
- Programma coseno con avvio graduale di un'epoca.
- Loss congiunta: loss ordinale morbida per la classe più Smooth-L1 (beta 0,05, su BBCH/99, riscalata ×99) per il BBCH continuo, peso `regression_loss_weight` = 1,0. Loss ordinale morbida, cioè penalità che considera la vicinanza tra classi, sigma 0,75. Pesi di classe inversamente proporzionali alla radice delle frequenze del solo training, normalizzati a media 1.
- Batch fisico 8, effettivo 16 con accumulo dei gradienti; 2 worker. Un batch è un insieme di immagini elaborate insieme; i worker caricano le immagini in parallelo. Testa in cache con batch 256.
- Nessun activation checkpointing, nessuna scadenza oraria, nessun limite di durata, nessuna prova temporale obbligatoria.
- Validation a ogni epoca, senza arresto anticipato. `model_final.pt` è l'ultima epoca (copiato in `modello_corrente.pt`); `best_validation.pt` minimizza il MAE BBCH di validazione; `best_classification_validation.pt` massimizza il macro-F1 sul dev. La valutazione finale usa `best_validation.pt`.
- Non riprendere training interrotti: usare una cartella nuova e ricominciare.
- Conservare la gestione originale dell'accumulo anche per l'ultimo insieme incompleto di mini-batch. Correggerla richiederebbe una decisione distinta perché cambierebbe gli aggiornamenti.
- Parametri effettivi e log per epoca salvati insieme al modello.

## Valutazione e inferenza

- Test separato dal training. Non avviarlo automaticamente. La valutazione finale protetta resta una funzione interna per il data scientist, non un terzo comando pubblico.
- Impedire il test se il training non è terminato e registrare l'apertura del test prima di leggere le immagini. Non ripetere la valutazione nello stesso addestramento sullo stesso elenco di test.
- MAE e RMSE del BBCH continuo grezzo (senza clipping). Accuracy; precision, recall e F1 per classe; medie macro e pesate; accuratezza entro una classe; errore medio assoluto ordinale; matrici di confusione numeriche e in figura.
- Accuracy: frazione di immagini corrette. Precision: quanto sono attendibili le assegnazioni a una classe. Recall: quante immagini vere della classe sono riconosciute. F1 combina precision e recall. La media macro attribuisce lo stesso peso a ogni classe; quella pesata considera la numerosità delle classi.
- Bootstrap per gruppo opzionale, normalmente 2000 ripetizioni, seed 42, intervalli al 95%.
- Classe non interpretabile con meno di 10 immagini oppure meno di 3 gruppi nel test; spiegare quale condizione manca. Riportare anche macro-F1 sulle sole classi interpretabili.
- Non mantenere l'analisi secondaria a 11 classi.
- Nessuna soglia aziendale di accettazione inventata: va definita dal team.

## Ambiente e verifica

- Nuova cartella separata: nessun file dell'originale modificato, nessun Git inizializzato, nessun commit o push.
- Ambiente `.venv` dedicato;
- Dipendenze dirette effettivamente usate, con le stesse versioni fissate dell'originale. Nessuna installazione a livello di sistema.
- DINOv2 dalla copia ufficiale locale disponibile, senza download superflui.
- Verificare sintassi e import, un breve training su dati reali, validation e test funzionali separati e ricaricamento del modello. Il test reale non serve al debug.
- Il checkpoint originale (sola classificazione) non si carica in questa versione. Dopo un nuovo training le metriche non sono confrontabili con quelle vecchie: lo split è cambiato.
- Documentazione italiana, breve README e questo elenco di requisiti. Escludere vecchi report, fotografie, documentazione LaTeX, esperimenti e test automatici distribuiti con il progetto.

## Interfaccia

- Due soli comandi pubblici: `python train.py` e `python predict.py`. Gli argomenti sono facoltativi.
- Un solo file modificabile nelle operazioni normali: `config.json`, con spiegazioni nei campi `_guida`. Architettura, mappatura e dettagli tecnici interni restano in `configs/model.json`.
- Il training prepara i dati e la suddivisione senza un comando preliminare. Riusa lo split principale se immagini, etichette, seed e percentuali coincidono; con dati nuovi conserva una nuova suddivisione nella nuova esecuzione, senza sovrascrivere quella principale.
- Ogni training riparte dai pesi ufficiali DINOv2, mai dal precedente modello BBCH. Le esecuzioni sono versionate e quelle interrotte non sono riprese.
- Il modello utilizzabile da predict.py è sempre `outputs/modello_corrente.pt` con la configurazione predefinita. Si aggiorna soltanto dopo il completamento del nuovo training; le versioni precedenti restano nelle rispettive cartelle di esecuzione.
- A schermo: nome dell'immagine, classe, BBCH stimato (3 decimali) e confidence. Nel CSV: anche intervallo BBCH, fase, `confidence_percent` e le probabilità di tutte le classi.
- I dati di training restano esterni nelle posizioni attuali. La cartella predefinita per le immagini nuove è `../PhenologyDeepLearning-data/da_prevedere`, modificabile in config.json. Non si copiano o spostano le immagini originali.
