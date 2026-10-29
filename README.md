# Riconoscere lo stadio BBCH della vite

Il progetto elabora fotografie di vite e assegna una delle 12 classi BBCH previste e stima anche un valore BBCH continuo (numero decimale 0–99). Si puo' addestrare nuovamente il modello con nuovi dati oppure usare il modello salvato per classificare foto nuove.

## 1. Preparazione iniziale

Apri il terminale nella cartella del progetto e attiva l'ambiente:

```bash
source .venv/bin/activate
```

Le dipendenze sono i pacchetti Python necessari. Su un nuovo computer, dopo aver predisposto un ambiente Python 3.11, installali con un solo comando:

```bash
python -m pip install -r requirements.txt
```

Se .venv non esiste, bisogna crearlo una sola volta con Python 3.11 prima di attivarlo:

```bash
python3.11 -m venv .venv
```

## 2. Un solo file da configurare

Bisogna aprire **config.json** con un editor di testo. Non modificare i file Python né configs/model.json. I campi _guida spiegano cosa puoi cambiare: JSON non accetta commenti con # o //.

| Sezione | Cosa contiene |
|---|---|
| dati | Cartella delle immagini, file Excel, nome del foglio e file dei gruppi/duplicati |
| suddivisione | Quote training 0,8, dev 0,1, test 0,1; il seme rende ripetibile la divisione |
| addestramento | Epoche delle due fasi, dimensioni delle immagini e carico di lavoro |
| previsione | File o cartella da classificare quando non specifichi un percorso nel comando |
| ambiente | Percorso della copia DINOv2 e dispositivo; lascia auto per la scelta automatica |
| risultati | Cartella di destinazione, normalmente outputs |

Un'**epoca** è un passaggio su tutte le immagini usate per imparare. I valori normali sono 5 epoche nella prima fase e 9 nella seconda. **immagini_insieme** indica quante immagini sono elaborate contemporaneamente: da abbassare se manca memoria. **immagini_per_aggiornamento** mantiene il carico complessivo di ogni aggiornamento e deve essere un multiplo del primo valore.

I percorsi relativi partono dalla radice del progetto, non dalla cartella del file Excel. Il simbolo ~ indica la cartella personale dell'utente.

## 3. Dove mettere i dati

Le fotografie e le etichette restano fuori dal progetto. I percorsi iniziali di training puntano al dataset esterno già disponibile, nella cartella vicina grapevine_dataset. Non vengono spostate, rinominate o modificate fotografie.

L'Excel richiede queste colonne:

| Colonna | Contenuto |
|---|---|
| image_name | Nome o percorso relativo della fotografia dentro dati.immagini; deve essere univoco |
| original_name | Nome originale, utile per ricavare l'ora dello scatto |
| vineyard | Vigneto |
| variety | Varietà |
| data | Data |
| BBCH | Codice numerico annotato |

Il foglio predefinito è **Catalogo foto**. Per esempio, se image_name contiene 0005.jpg, il file deve trovarsi nella cartella immagini configurata. I codici devono appartenere alle classi definite in configs/model.json: ad esempio 51 e 53 appartengono alla stessa classe, chiamata 53.

I file dei gruppi e dei duplicati sono facoltativi. Un gruppo descrive fotografie correlate. Se il file dei gruppi non copre nuovi dati, il programma li ricalcola dalle sessioni di scatto e dalla similarità delle immagini, prima di creare la divisione dei nuovi dati. Se i gruppi non sono già forniti e il report duplicati non è disponibile, imposta dati.duplicati a null: riceverai un avviso. Non scrivere la parola null tra virgolette.

## 4. Addestrare: un comando

Dopo aver aggiornato config.json, esegui:

```bash
python train.py
```

Il programma controlla i dati, prepara o riusa la suddivisione e avvia entrambe le fasi di apprendimento. Non devi avviare altri script.

- **Training**: immagini da cui il modello impara.
- **Dev o validazione**: immagini separate con cui si controlla l'andamento.
- **Test**: immagini riservate alla valutazione finale, che questo comando non esegue.

La divisione è 80% / 10% / 10% **per gruppi** di fotografie correlate (StratifiedGroupKFold): un gruppo non viene mai diviso tra training, dev e test. Servono almeno 10 gruppi e ogni classe deve avere immagini di training; altrimenti il programma si ferma con un messaggio esplicito.

Se immagini, etichette (classe e BBCH numerico), quote e seme coincidono con lo split principale già disponibile, il programma lo conserva. Gli split vecchi senza la colonna BBCH non vengono riusati. Con dati diversi salva una nuova divisione nella nuova esecuzione, senza sovrascrivere quella principale.

A schermo compaiono la preparazione delle immagini e le epoche completate. La percentuale indica quante immagini di validazione sono riconosciute correttamente; bbch_mae è l'errore medio assoluto del valore BBCH continuo. F1 è un indice che combina la capacità di trovare una classe e l'affidabilità delle assegnazioni; la sua media dà qui lo stesso peso alle 12 classi.

Ogni nuovo addestramento parte dai pesi ufficiali DINOv2, non dal precedente modello BBCH. Non riprende esecuzioni interrotte. Se si interrompe, il modello corrente non viene sostituito e puoi rilanciare lo stesso comando.

A fine esecuzione trovi:

- **outputs/modello_corrente.pt**: modello caricato automaticamente per le previsioni;
- **outputs/modello_corrente.json**: riferimento alla versione e impronta SHA-256, un identificatore del contenuto;
- **outputs/addestramenti/**: cartelle delle singole esecuzioni, con configurazione effettiva, dati di suddivisione, log, risultati intermedi e modelli;
- nella cartella `modello/` di ogni esecuzione: `model_final.pt` (ultima epoca, copiato in modello_corrente.pt), `best_validation.pt` (minimo MAE BBCH di validazione) e `best_classification_validation.pt` (massimo Macro-F1).

## 5. Prevedere: un comando

Metti le fotografie nuove nella cartella esterna **../PhenologyDeepLearning-data/da_prevedere**, oppure cambia previsione.immagini in config.json. Sono accettati JPEG/JPG, HEIC/HEIF e PNG, anche nelle sottocartelle.

Poi esegui:

```bash
python predict.py
```

Il programma carica automaticamente outputs/modello_corrente.pt. Per ogni immagine stampa nome, classe, BBCH stimato (decimale) e confidence. La confidence è la probabilità softmax massima: non è calibrata e non è una garanzia.

Il risultato completo è salvato in **outputs/previsioni.csv**, leggibile anche con Excel: include classe, BBCH_min/BBCH_max, fase, BBCH_stimato_float, BBCH_stimato_display, confidence_score, confidence_percent e le probabilità delle 12 classi. Un nuovo comando riuscito aggiorna quel CSV; conserva una copia prima di una nuova analisi se vuoi mantenere i risultati precedenti.

Se preferisci indicare direttamente un file o una cartella, puoi usare lo stesso comando con un percorso facoltativo:

```bash
python predict.py "cartella_con_fotografie"
```

L'orientamento viene corretto per la lettura, ma gli originali non vengono modificati.

## 6. Errori comuni

| Messaggio o problema | Cosa fare |
|---|---|
| Nessuna immagine trovata / percorso inesistente | Controlla previsione.immagini o il percorso passato al comando; usa i formati elencati |
| Nessun modello addestrato disponibile | Esegui prima python train.py |
| Memoria insufficiente | Riduci immagini_insieme e mantieni immagini_per_aggiornamento multiplo del nuovo valore |
| File delle etichette o fotografia mancante | Correggi i percorsi e image_name nell'Excel, senza rinominare automaticamente i dati originali |
| BBCH fuori mappatura | Controlla l'annotazione: il programma non scarta in silenzio i codici non previsti |
| Training senza alcune classi / campione troppo piccolo | Aggiungi esempi delle classi mancanti o chiedi al responsabile; non inventare etichette |
| Configurazione non valida | Controlla virgole, virgolette e i valori spiegati nei campi _guida |
| DINOv2 o pesi ufficiali mancanti | Chiedi al responsabile di predisporre la copia e la cache indicate dalla configurazione |

## File da conoscere

| Elemento | Ruolo |
|---|---|
| train.py | Unico comando di addestramento e controllo dev |
| predict.py | Unico comando di previsione |
| config.json | Impostazioni per l'utilizzatore |
| configs/model.json | Dettagli interni, classi e valori tecnici (incluso regression_loss_weight) |
| modello/ | Codice interno, separato per architettura, dati, apprendimento, metriche e supporto |
| REQUIREMENTS.md | Requisiti e decisioni concordate |
| requirements.txt | Dipendenze con versioni fissate |
| outputs/ | Risultati, split e registri da non azzerare |
| .venv/ | Ambiente Python locale |

La valutazione finale protetta resta disponibile internamente al data scientist, separata dai due comandi. Non viene avviata durante addestramento o previsione. I vecchi registri di utilizzo del test sono conservati: non cancellarli per aggirare i blocchi.

Non sono distribuiti test automatici, vecchi esperimenti o configurazioni alternative.


## Dual output: BBCH class + continuous estimate

This version trains two heads on the shared DINOv2 representation:

- a 12-class BBCH classifier (the existing grouped classes);
- a regression head trained against the original integer `BBCH` metadata value, producing a continuous estimate in the 0-99 range.

The split CSVs preserve the original `BBCH` value for regression. `predict.py` writes `BBCH_stimato_float` and `confidence_score` (the maximum softmax class probability) in addition to the predicted class. The confidence score is not calibrated and should not be interpreted as a guaranteed probability of correctness. The continuous estimate is the raw regression output (sigmoid-bounded to 0-99) and is **not** clipped to the predicted class interval, so class and estimate can disagree.

Because older checkpoints have only a classification head, this version requires a fresh training run; old model checkpoints are not compatible.


## Note sulla versione corretta

- Le predizioni BBCH continue sono conservate come valori grezzi della testa di regressione: non vengono più forzate nell'intervallo della classe classificata. Il CSV include `BBCH_stimato_float` e `BBCH_stimato_display` (tre decimali). La classe e la stima numerica sono due output distinti e possono quindi discordare; questa discordanza è visibile invece di essere nascosta dal clipping.
- La valutazione salva MAE/RMSE della regressione grezza e le predizioni grezze.
- Il checkpoint `best_validation.pt` è selezionato sul MAE BBCH di validazione; `best_classification_validation.pt` è selezionato sul Macro-F1. La valutazione finale usa il primo.
- I nuovi split usano `StratifiedGroupKFold` con gruppi interi, per ridurre la contaminazione tra immagini correlate. La configurazione supporta quote 80/10/10. Le vecchie suddivisioni a livello di singola immagine non vengono riutilizzate.
- I vecchi checkpoint (sola classificazione) non sono compatibili: mancano i pesi `bbch_linear` e `predict.py` non li carica. Serve un nuovo addestramento con questa versione.


## Uso su Google Colab

Il notebook `Run_Phenology_BBCH_Fixed_Project.ipynb` esegue tutto su Colab: prepara il progetto su Drive, imposta i percorsi Colab in config.json senza toccare il repository, addestra e, a fine addestramento, crea la cartella `outputs/consegna/` con tutto ciò che serve per l'integrazione web:

- `modello_corrente.pt` e `best_validation.pt` (checkpoint con testa di classificazione e di regressione);
- `model.json` (configurazione di architettura e classi) e `config_usata.json`;
- `previsioni_esempio.csv` (se c'erano immagini da prevedere) e `LEGGIMI_consegna.txt`.

I file `.pt` sono pesanti: non vanno committati su GitHub normale (usare Git LFS, una Release o Drive).
