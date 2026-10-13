# Regole fisse

Valgono per tutto il dataset. Vengono da `NEW_MASTERPLAN.md` di yoloeval (§2 Regole fisse e §5 Fase 0).

## Formato

- **Classi**: `0 airplane`, `1 bird`, `2 drone`, `3 helicopter` (stesso ordine di AOD4 e Svanström in yoloeval).
- **Immagini 640×512**, come Svanström, per tutte le fonti. Nessun ingrandimento.
- **Label** in formato Ultralytics (YOLO txt), una riga per box.
- **Training** in yoloeval sempre a 640×640: il frame 640×512 non viene scalato, quindi le grandezze in pixel delle
  label sono quelle che vede il modello.

## Grandezza degli oggetti

- La grandezza di un oggetto è **√(w·h)** del suo box, in pixel, come in COCO.
- **Fasce**: 16–32, 32–64, 64–96, sopra 96 px. 16–32 e 32–64 come "small" e "medium" di AI-TOD, 64–96 e sopra 96
  come "medium" e "large" di COCO.
- **Campo della tesi: oggetti ≥ 16 px.** Le immagini con almeno un oggetto sotto i 16 px **non entrano in val e
  test**; nel train restano, con la loro label (togliere solo la label insegnerebbe al modello che lì non c'è
  niente). Così le metriche standard di Ultralytics valgono già solo per oggetti ≥ 16 px.
- Perché 16 px: la testa più fine di YOLO26 lavora a passo 8 (sotto i 16 px un oggetto copre 1–2 celle per lato);
  sui benchmark di oggetti minuscoli (AI-TOD) i detector diventano usabili sopra i 16 px; un box di pochi pixel non
  si disegna in modo affidabile; con pochi oggetti minuscoli la mAP oscillerebbe per quella fascia.

## Annotazione

- **Box aderente all'oggetto**, con la stessa regola per tutte le fonti.
- **Tutti gli oggetti visibili etichettati** (stormi, oggetti diversi dal bersaglio). Si riporta quanti mancavano per
  fonte.

## Split e bilanciamento

- **Split per video** (70/20/10), mai per frame: frame vicini dello stesso video sono quasi identici (era il leakage
  di AOD4).
- **Quasi-duplicati** tolti con l'hash semantico (dHash del ritaglio attorno al bersaglio + posizione e grandezza
  del box): dentro ogni video, e fra split dopo lo split.
- **Bilanciamento** su classe × fascia (4 × 4 = 16 celle): un **minimo di istanze e di video distinti per cella**,
  non celle uguali. Anche val e test hanno un minimo per cella.
- **Background**: circa il 10% di frame senza oggetti, dagli stessi video.
- **Requisiti delle fonti**: a colori, da terra verso il cielo, ID dei video, licenza adatta a una tesi.

## Cosa consegna questa repo a yoloeval

| cosa | dettaglio |
|---|---|
| `all/` | train, val, test, `calib` (512 immagini di train a seed fisso), con `dataset.yaml` |
| sottoinsiemi per fascia | val e test per fascia (16–32, 32–64, 64–96, sopra 96), per la mAP per fascia con il `val` standard |
| `val_480p`, `val_720p`, `val_1080p` | set di val alle risoluzioni di deploy (ingressi 864×480, 1280×736, 1920×1088), a mosaico |
| `test_480p`, `test_720p`, `test_1080p` | come sopra per il test, **chiusi fino alla fase 5** |
| report | matrice classe × fascia per split (istanze e video), quota di ogni fonte per classe, immagini senza oggetti, oggetti sotto i 16 px, quanto erano larghi i box originali di Svanström |
| `manifest.json` | fonti, parametri, frame scelti, hash; dopo il congelamento il dataset non cambia più |

## Set di deploy (mosaici)

- Ogni fotogramma di deploy ha **un solo frame con l'oggetto**, in posizione casuale a seed fisso, e intorno frame di
  **solo sfondo dello stesso video** (oggetto cancellato per inpainting): il cielo è lo stesso, niente giunture fra
  cieli diversi.
- 1080p: 3×2 frame da 640×512 (1920×1024) più una striscia di sfondo fino a 1088. 720p e 480p si ritagliano di
  conseguenza; a 480p il frame con l'oggetto (alto 512) si taglia in modo che l'oggetto resti intero.
- **Nessun ridimensionamento**: gli oggetti hanno la stessa grandezza in pixel a ogni risoluzione. È voluto: con la
  stessa camera, a risoluzione più alta lo stesso oggetto occupa più pixel e il modello va meglio per forza; a
  grandezza invariata la mAP confronta solo i modelli, e il vantaggio della risoluzione si esprime come raggio di
  rilevamento utile (calcolato a parte). E dati veri a 1080p per tutte le classi non ci sono.
- La validazione di Ultralytics non fa mosaic (è solo augmentation di training), per questo i set si costruiscono a
  parte, una volta su disco, uguali per tutti i modelli.
