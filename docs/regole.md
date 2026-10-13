# Regole

Da `NEW_MASTERPLAN.md` (§2, §5).

## Formato

- Classi `0 airplane`, `1 bird`, `2 drone`, `3 helicopter`; label YOLO txt.
- Tutte le immagini **640×512**, mai ingrandite. Il training a 640 non le scala: i pixel delle label sono quelli che
  vede il modello.

## Grandezza

- Grandezza = **√(w·h)** del box, in pixel.
- Fasce: **16–32, 32–64** (AI-TOD), **64–96, > 96** (COCO).
- Immagini con un oggetto **sotto i 16 px**: solo nel train, con la label. Perché 16: passo 8 della testa (1–2 celle
  per lato), AI-TOD (sotto i 16 px i detector falliscono), label poco affidabili, mAP instabile.

## Annotazione

- Box **aderente** all'oggetto, stessa regola per tutte le fonti.
- Casi dubbi (decisi il 13/10/2026):
  - pale ed eliche nel box **se visibili** nel frame; sfocate e invisibili, il box resta sul corpo;
  - oggetto tagliato dal bordo: box sulla parte visibile;
  - stormi: un box per uccello visibile;
  - oggetto coperto (alberi, pali): box sulla parte visibile.
  - scie degli aerei escluse: box solo sull'aereo; le clip con la scia si tolgono dal dataset.
- **Tutti** gli oggetti visibili delle 4 classi etichettati (anche stormi e oggetti non bersaglio).
- Oggetti fuori dalle 4 classi (mongolfiere, alianti, parapendii, aquiloni): **senza label**, immagine tenuta.

## Split e bilanciamento

- Split **per sessione** 70/20/10: clip (scene dei video) della stessa sessione di ripresa insieme, nel dubbio
  unite.
- Quasi-duplicati via con hash semantico (dHash del ritaglio sul bersaglio + posizione e grandezza del box): dentro
  la sequenza e fra split.
- Background circa 10%, dagli stessi video.
- Fonti: RGB, da terra verso il cielo, ID dei video, licenza adatta a una tesi.

| minimo | valore | fonte |
|---|---|---|
| train, per classe | ≥ 1500 immagini, ≥ 10.000 istanze | Ultralytics, "Tips for best training results" |
| train, per cella | > 100 immagini, ≥ 10 video | LVIS, categoria "frequente" |
| val e test, per cella | ≥ 385 istanze | AP entro ±5 punti al 95%: (1,96 · 0,5 / 0,05)² |
| val, complessivo | rumore della mAP < 1 punto | soglia di scelta del piano |

- I minimi di Ultralytics sono generici: la guida consiglia il training da zero solo per dataset grandi, senza
  soglia. Per il training da zero: He et al., "Rethinking ImageNet Pre-training" (ICCV 2019), alla pari con il
  preaddestrato fino a circa 10k immagini COCO (da verificare sul paper). Controllo vero: curve di training
  (yoloeval T2-03).
- Val e test si verificano con **bootstrap per sessione** prima del congelamento.
- Cella sotto il minimo: si riempie con aggiunte, altrimenti è un limite dichiarato.

## Consegna a yoloeval

| cosa | dettaglio |
|---|---|
| `all/` | train, val, test, `calib` (512 immagini di train), `dataset.yaml` |
| sottoinsiemi per fascia | val e test, per la mAP per fascia con il `val` standard |
| `val_480p`, `val_720p`, `val_1080p` | mosaici, ingressi 864×480, 1280×736, 1920×1088 |
| `test_480p`, `test_720p`, `test_1080p` | come sopra, chiusi fino alla fase 5 |
| report | matrice classe × fascia per split (istanze e video), quota per fonte, background, oggetti < 16 px, larghezza dei box originali di Svanström |
| `manifest.json` | fonti, parametri, frame, hash; poi il dataset si congela |

## Mosaici di deploy

- Un solo frame con l'oggetto, in posizione casuale a seed fisso; intorno frame di solo sfondo dello **stesso video**
  (bersaglio cancellato per inpainting).
- 1080p: 3×2 frame (1920×1024) + striscia di sfondo fino a 1088. 720p e 480p ritagliati; a 480p il frame con
  l'oggetto si taglia tenendo l'oggetto intero.
- Nessun ridimensionamento: a grandezza invariata la mAP confronta solo i modelli; la risoluzione si traduce in
  raggio di rilevamento, calcolato a parte.
