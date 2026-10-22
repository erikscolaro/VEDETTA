# Regole

Da `NEW_MASTERPLAN.md` (§2, §5).

## Formato

- Classi `0 airplane`, `1 bird`, `2 drone`, `3 helicopter`; label YOLO txt.
- Tutte le immagini **640×512**, mai ingrandite. Fonti più grandi: **ritaglio** a 640×512 attorno agli oggetti, senza
  ridimensionare (decisione del 16/10/2026). Il training a 640 non le scala: i pixel delle label sono quelli che
  vede il modello.

## Grandezza

- Grandezza = **√(w·h)** del box, in pixel.
- Fasce: **8–16, 16–32, 32–64** (AI-TOD), **64–96, > 96** (COCO). La 8–16 è in tutti gli split (decisione
  dell'utente, 19/10/2026; prima gli oggetti sotto i 16 px stavano solo nel train).
- Oggetti con area **sotto i 4 px²** (√(w·h) < 2 px: 4×1, 2×2…): via (label tolta; utente, 19/10/2026, prima
  era "sotto i 4 px" di lato). Fino a 8 px: solo nel train, con la label, senza obiettivo.
- **Niente dominio notturno** (utente, 19/10/2026): via i ritagli bui (luminosità media < 35) di Anti-UAV e DVIDS
  (luci nel buio, visori notturni); di notte c'erano quasi solo droni. Lo scenario è diurno.
- **Fumo degli airshow** = scia (utente, 19/10/2026): via i tratti DVIDS con scie di fumo o condensa.
- FBD-SV-2024 (uccelli su campi, alberi, fili) tenuto come **estensione del dominio all'orizzonte** (utente,
  19/10/2026).
- Celle accettate sotto il margine: elicotteri 16–32 (utente, 19/10/2026).
- **Obiettivo: distribuzione continua per classe** (decisione dell'utente del 22/10/2026, sostituisce i 2.857
  per cella del 19/10/2026): **14.286 box per classe** in tutto (10.000 nel train, Ultralytics, con lo split
  70/20/10), con densità in log s proporzionale a **s^−α tra 8 e 320 px** (s = √(w·h)): **α = 0,3 per i droni,
  0,45 per uccelli, aerei ed elicotteri**. Più esempi dove gli oggetti sono piccoli: è lì che la torretta deve
  vederli e che le classi si confondono. Ragionamento e grafici in [`distribuzioni.md`](distribuzioni.md).
- **Priorità**: drone e uccello sono le classi principali (distinguere un drone da un uccello); aereo ed
  elicottero sono secondarie (un aereo o un elicottero che arriva alla base si sa). Il deficit sotto la curva di
  aerei ed elicotteri piccoli è un **compromesso accettato** (utente, 22/10/2026); per gli uccelli resta da
  raffinare la distribuzione.
- Le fasce restano per stratificare val e test e per la mAP per fascia; `tools/buchi.py` conta ancora per cella.
- Storico (19/10/2026): obiettivo per cella di 2.000 istanze nel train (2.857 in tutto), +10% prima di CVAT.
- Bilanciamento (utente, 19/10/2026): **val e test piatti** (~571 e ~286 per cella; nelle celle sotto il minimo il
  30% delle istanze); **train con tetto di 4.000 istanze per cella** (sopra si scarta, sotto si tiene tutto). Niente
  copie ridotte per la fascia 8–16: lo scale lo fa Ultralytics in training (`scale: 0.25`, zoom 0,75–1,25). Stima
  sui box proposti: train ~60.900, val + test ~14.900.

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
  unite. **Sessione = stesso posto e stesse condizioni**: posto diverso = sessione diversa, e anche lo stesso posto
  in condizioni diverse (altro giorno, altra luce) = sessione diversa (regola dell'utente, 16/10/2026).
  Lo stesso sfondo può quindi stare in train e in test: **leakage dello sfondo accettato** (decisione dell'utente,
  16/10/2026), perché in produzione la torretta è fissa e lo sfondo resta quello; quello che cambia sono i droni.
- Quasi-duplicati via con hash semantico (dHash del ritaglio sul bersaglio + posizione e grandezza del box): dentro
  la sequenza e fra split.
- Background circa 10%, dagli stessi video.
- Fonti: RGB, da terra verso il cielo, ID dei video, licenza **aperta** (pubblico dominio, CC0, CC BY, Apache/MIT),
  verificata prima del download e scritta in [`licenze.md`](licenze.md) (utente, 19/10/2026).

| minimo | valore | fonte |
|---|---|---|
| train, per classe | ≥ 1500 immagini, ≥ 10.000 istanze | Ultralytics, "Tips for best training results" |
| train, per classe e grandezza | curva s^−α con 10.000 istanze per classe (14.286 in tutto); per fascia > 100 immagini, ≥ 10 video | Ultralytics; scenario della torretta (`distribuzioni.md`); LVIS, categoria "frequente" |
| val e test, per cella | le quote del 70/20/10 (~571 e ~286 con 2.000 nel train); il vecchio 385 (AP entro ±5 punti al 95%: (1,96 · 0,5 / 0,05)²) resta come riferimento | Compromessi: incertezza dal bootstrap per sessione |
| val, complessivo | rumore della mAP < 1 punto | soglia di scelta del piano |

- I minimi di Ultralytics sono generici: la guida consiglia il training da zero solo per dataset grandi, senza
  soglia. Per il training da zero: He et al., "Rethinking ImageNet Pre-training" (ICCV 2019), alla pari con il
  preaddestrato fino a circa 10k immagini COCO (da verificare sul paper). Controllo vero: curve di training
  (yoloeval T2-03).
- Val e test si verificano con **bootstrap per sessione** prima del congelamento.
- Sotto la curva: si riempie con aggiunte, altrimenti è un limite dichiarato (aerei ed elicotteri piccoli: accettato).

## Flusso per ogni fonte (deciso dall'utente il 16/10/2026)

1. download;
2. label con un metodo automatico (YOLO, sfondo fermo, contrasto o SAM, secondo la fonte);
3. deduplica dentro la clip (hash a 16 bit);
4. revisione in CVAT dei frame tenuti;
5. impronta semantica di ogni clip e gruppi per somiglianza, su tutte le fonti insieme;
6. split per gruppi.

## Compromessi (decisi il 16/10/2026)

Il dataset non può costare un mese. Dominio: torretta fissa, lo sfondo resta quello, cambiano i droni; l'unico asse
da tenere pulito è il **drone e la sua traiettoria** (lo stesso volo mai in due split).

| asse | compromesso | motivo |
|---|---|---|
| sfondo | leakage accettato | in produzione lo sfondo è quello |
| sessioni | **gruppi automatici** dall'impronta semantica delle clip (DINOv2, soglia **0,85** per tutte le fonti), **uniti** alle sessioni note (manuali, registrazioni, video); dopo lo split si misura il leakage rimasto (vicino più simile nel train) | stessa procedura per tutte le fonti; a 0,95 il 43% delle coppie di sessioni manuali finiva in gruppi diversi |
| meteo e luce | non decidono sessioni né split; si riportano come distribuzione nel report di val e test | descrivono il dataset, non evitano leakage |
| drone | ogni modello di drone in train e in test (val se possibile); split stratificato su classe × fascia | è l'asse che misura la generalizzazione |
| box | proposte automatiche (YOLO sul cielo, sfondo fermo sui cortili), poi revisione in CVAT di **tutti i frame tenuti dalla deduplica** (non di tutti i frame), prima dello split (decisione dell'utente, 16/10/2026; sostituisce "solo val e test") | la deduplica toglie già la maggior parte del lavoro |
| celle vuote (droni e uccelli > 64 px) | limite dichiarato; al più una fonte in più se veloce | rare anche nel dominio vero |
| minimo per cella | **da rivedere** (proposta: 150–200 istanze in val e test invece di 385, con l'incertezza dal bootstrap per sessione) | meglio una barra d'errore che una cella vuota |
| FRED | si ferma alle sequenze già scaricate | bastano per varietà di drone e numero di istanze |

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
