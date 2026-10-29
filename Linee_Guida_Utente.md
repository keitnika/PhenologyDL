# Linee guida per l'acquisizione delle foto

- Una gemma, un organo o un grappolo d'uva per foto
- Il soggetto deve occupare più della metà dell'immagine
- Mantenere il soggetto al centro dell'inquadratura
- Il soggetto deve essere a fuoco
- Usare la luce naturale quando possibile ed evitare ombre forti
- Evitare oggetti che ostruiscono la pianta (mani, attrezzi, fili o altra vegetazione)
- Mantenere visibili le strutture rilevanti della pianta (gemme, foglie, infiorescenze, fiori o acini)
- Evitare angolazioni insolite della fotocamera
- Registrare i metadati quando disponibili: data, varietà d'uva, vigneto e ID della pianta

## Come scattare le foto per stimare l'intero vigneto

Il modello di deep learning analizza ogni fotografia e assegna a quella singola immagine uno stadio BBCH. Tuttavia, l'obiettivo del progetto non è conoscere lo stato di una sola pianta, ma stimare la crescita dell'intero vigneto. Per questo motivo, il valore finale dipende da come vengono scattate le foto, non solo dalla loro qualità.

**Scatta più foto possibili.** L'algoritmo lavora meglio quando ha a disposizione un campione numeroso. Ogni pianta, anche nello stesso stadio, presenta differenze naturali di sviluppo. Con poche foto, queste differenze possono pesare troppo sul risultato. Con molte foto, le variazioni individuali si compensano e la stima riflette lo stato medio reale del vigneto.

**Distribuisci gli scatti su tutta la superficie.** Non fotografare tutte le piante dello stesso filare o della stessa zona. Il vigneto non è uniforme: esposizione al sole, tipo di suolo, irrigazione, pendenza e vicinanza al bordo possono far sviluppare le piante in modo diverso. Fotografare solo una porzione significa ottenere un quadro parziale, che rischia di sovrastimare o sottostimare la crescita complessiva.

**Scegli punti di scatto distanti e indipendenti.** Scattare foto molto vicine tra loro produce immagini simili, quindi porta poche informazioni nuove. Più i punti sono lontani e non collegati tra loro, più ogni foto aggiunge un dato diverso al campione. Un buon metodo è attraversare il vigneto in diagonale o secondo percorsi a zig-zag, fermandosi a intervalli regolari e scattando una foto in ogni punto, senza seguire sempre lo stesso filare.

**Esempio pratico.** Dieci foto scattate in dieci filari diversi, distribuiti da un lato all'altro dell'appezzamento, forniscono un'informazione molto più affidabile di dieci foto scattate di seguito sulla stessa fila. Nel primo caso la stima tiene conto della variabilità reale del campo; nel secondo rischia di descrivere solo la parte del vigneto che si trova in quel punto.
