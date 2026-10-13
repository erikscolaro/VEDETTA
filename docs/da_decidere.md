# Da decidere, rischi, limiti

## Da decidere

| punto | quando |
|---|---|
| **minimo di istanze e di video per cella** della matrice classe × fascia | passo 1, prima di toccare i dati |
| **fonti per i buchi** (uccelli grandi, droni grandi, aerei grandi), da portare a 640×512 | passo 5, dopo il conteggio |
| **detector a vocabolario aperto** per proporre i box delle aggiunte | passo 5 |
| **soglie dell'hash semantico** (oggi `--hash-thr 8`, `--geo-thr 0.05`) | passo 6 |

## Rischi

| rischio | quando lo si scopre | piano B |
|---|---|---|
| SAM 2 non rende aderenti i box degli oggetti minuscoli (sagome vuote o sbagliate) | passo 2 | soglia di contrasto locale su cielo uniforme; revisione a mano dei casi rimasti |
| aggiunte (uccelli e aerei grandi) non trovate o troppo lente da etichettare | passo 5 | si usa quello che c'è; le celle scoperte si dichiarano come limite |
| dopo la correzione dei box molti oggetti scendono sotto i 16 px, e val e test si svuotano | passo 3 | si rivede il minimo per cella; le immagini sotto i 16 px restano comunque nel train |

## Limiti da dichiarare nella tesi

- **Val e test non contengono oggetti sotto i 16 px**: in deploy ci saranno droni più lontani di così.
- **Mosaici**: le giunture fra un frame e l'altro sono bordi artificiali (attenuati dallo sfondo dello stesso video).
  Senza fotogrammi veri ad alta risoluzione non si possono confrontare con un fotogramma vero.
- **Celle scoperte** della matrice classe × fascia (per esempio uccelli grandi ripresi da terra, rari in natura).
- **Sessioni di volo** di Svanström non note: video della stessa sessione possono finire in split diversi.
