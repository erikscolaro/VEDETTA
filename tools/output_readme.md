# air-detection: dataset per la rilevazione di velivoli e uccelli

Generato da `tools/split.py --step export` il {data} (repo air-detection; procedura completa nel README della repo).
Formato Ultralytics YOLO: `images/<split>/*.jpg`, `labels/<split>/*.txt` (una riga per box: classe, centro x,
centro y, larghezza, altezza, normalizzati), `dataset.yaml`. Tutte le immagini sono 640×512, mai ingrandite. Le immagini sono hardlink ai file del repo, le label
sono riscritte qui con gli id sotto (nel repo restano nell'ordine airplane, bird, drone, helicopter).

| classe | id |
|---|---|
| drone | 0 |
| airplane | 1 |
| helicopter | 2 |
| bird | 3 |

## Contenuto

{conteggi}

- **train**: immagini reali (con un tetto per classe e grandezza), mosaici di compensazione per gli oggetti
  piccoli, background.
- **val**: immagini reali con la distribuzione obiettivo, più background. Serve a scegliere il modello migliore.
- **test**: immagini reali con lo stesso numero di box per fascia di grandezza, più background. Si usa una volta.
- **calib**: 512 immagini del train (reali e background, niente mosaici) bilanciate per classe e grandezza, per la
  calibrazione int8 (`calib.yaml`).
- **raw/**: `images/` e `labels/` di train, val e test insieme, in una cartella sola (senza calib), per chi vuole
  rifare gli split a modo suo; lo split di ogni immagine è nel manifest.
- Almeno 10.000 istanze per classe in train, 2.000 in val e 1.000 in test. Gli uccelli sopra i 96 px stanno solo nel
  train (troppo pochi per misurarli).
- Nessuna immagine ha box sotto gli 8 px di lato (√(w·h)). Lo split è per **gruppo**: tutte le immagini di un gruppo
  stanno nello stesso split, quindi la stessa ripresa non è mai in due split.

![distribuzione](distribuzione.png)

`distribuzione.png` (e `.pdf`): istanze per grandezza √(w·h) e classe. Blu: addestramento, immagini reali; giallo:
mosaici sintetici; linee: validazione e test; ombra grigia: i dati originali prima di ogni selezione. Obiettivi in nero:
densità in scala logaritmica ∝ s^−0,3 tra 8 e 320 px (uccelli fino a 96), alzata ×1,2, al 70% per l'addestramento
(linea continua) e al 20% per la validazione (tratteggiata); test uniforme in scala logaritmica (punteggiata). In alto
a destra l'errore di ogni split, Σ|osservato − obiettivo| / Σ obiettivo per mezza ottava.

![leakage](leakage.png)

`leakage.png` (e `.pdf`, tabella in `leakage.md`): per ogni immagine reale di validazione e test, la somiglianza
DINOv2 con l'immagine più vicina dell'addestramento, confrontata con due riferimenti dentro l'addestramento: stesso
gruppo e altra clip (com'è fatto un leakage vero) e gruppi diversi (il fondo). Validazione e test coincidono con il
fondo: non somigliano all'addestramento più di quanto due gruppi diversi si somiglino fra loro.
Tabelle di `distribuzione.png` in `distribuzione.md`.

## Legenda dei nomi

| prefisso | tipo | formato | esempio |
|---|---|---|---|
| `R` | immagine **reale** di una ripresa | `R_<gruppo>_<clip>_<n>` | `R_G0000_a7bdc852_000001.jpg` |
| `M` | **mosaico** di compensazione (solo train) | `M_x<griglia>_<n>` | `M_x4_000012.jpg` |
| `B` | **background**, solo cielo e orizzonte, senza label | `B_<camera>_<n>` | `B_f88e9c93_000033.jpg` |

- `<gruppo>` (`G0000`…): unità dello split. Un gruppo riunisce le sessioni di ripresa che si somigliano (stesso
  posto e stesse condizioni, oppure lo stesso girato ripubblicato).
- `<clip>` / `<camera>`: primi 8 caratteri dello SHA-1 di `fonte:clip` (o `fonte:camera` per il background); stabili
  fra un export e l'altro.
- `<n>`: numero progressivo a 6 cifre, in ordine di frame dentro la clip o la camera.
- `M_x2`, `M_x4`, `M_x8`: mosaici 640×512 di 2×2, 4×4 o 8×8 ritagli del train ridotti di ×0,5, ×0,25, ×0,125
  (sfocatura e rumore leggeri), per gli oggetti piccoli che mancavano. Sono sintetici e stanno solo nel train.

## Risalire all'origine

`manifest.csv` ha una riga per immagine: `immagine` (percorso qui), `split`, `tipo` (reale, mosaico, background),
`gruppo`, `fonte`, `sessione`, `clip` (o camera), `frame` (indice nel video originale, o nome del fotogramma per il
background), `sorgente` (file nel repo air-detection, sotto `data/`; per gli export fatti prima del 23/10/2026 i percorsi vanno letti sotto `data/input/` o `data/intermediate/`). Per esempio:

    images/train/R_G0000_a7bdc852_000001.jpg → fonte svanstrom, clip V_AIRPLANE_001/s0, frame 24
                                             → video V_AIRPLANE_001 del dataset di Svanström, fotogramma 24

Per le clip DVIDS l'id del video è nel nome della clip (`DVIDS_<id>/s0`): pagina, autore e licenza di ogni video in
`fonti_dvids.csv`. Le sorgenti dei mosaici (ritaglio per ritaglio) sono in `intermediate/13_train_compensation/mosaici.csv`
del repo.

## Fonti, licenze e attribuzione

| fonte | cosa dà | licenza | download | attribuzione |
|---|---|---|---|---|
| Svanström et al., "A dataset for multi-sensor drone detection", Data in Brief 2021 | video RGB 640×512 di droni, aerei, elicotteri, uccelli | **CC0 1.0** | [github.com/DroneDetectionThesis/Drone-detection-dataset](https://github.com/DroneDetectionThesis/Drone-detection-dataset) | citare l'articolo |
| FRED, Magrini et al., ACM MM 2025 | droni (sequenze RGB) | **Apache 2.0** | [huggingface.co/datasets/GabrieleMagrini/FRED](https://huggingface.co/datasets/GabrieleMagrini/FRED) | citare l'articolo, riportare la licenza |
| Anti-UAV300 (parte RGB), Jiang et al., IEEE TMM 2021 | droni | **MIT** | [github.com/ZhaoJ9014/Anti-UAV](https://github.com/ZhaoJ9014/Anti-UAV), mirror [huggingface.co/datasets/DestinyLove/AntiUAV300](https://huggingface.co/datasets/DestinyLove/AntiUAV300) | copyright e licenza MIT, citare l'articolo |
| DVIDS (Defense Visual Information Distribution Service) | aerei, elicotteri, droni | **pubblico dominio** (opere di dipendenti federali USA), video per video | [dvidshub.net](https://www.dvidshub.net) (elenco in `fonti_dvids.csv`) | "The appearance of U.S. Department of War (DoW) visual information does not imply or constitute DoW endorsement."; niente insegne in usi commerciali |
| FBD-SV-2024, Sun et al., Scientific Data 2025 | uccelli | **solo ricerca, non commerciale** ⚠ | [figshare 27377595](https://figshare.com/articles/dataset/FBD-SV-2024/27377595), Kaggle `swjtuziwei/fbd-sv-2024` | citare l'articolo |
| Clouds-1500 (Mendeley Data, v2, 2023) | background: cielo verso l'orizzonte, meteo vario | **CC BY 4.0** | [data.mendeley.com/datasets/2khchjbgzr/2](https://data.mendeley.com/datasets/2khchjbgzr/2) | citare il dataset e gli autori, indicare la licenza |
| Skyfinder, Mihail et al., WACV 2016 (copia nel record Zenodo 2562396, Nice & Wijnands 2019) | background: 18 webcam AMOS | **incerta** ⚠: il record dichiara CC BY 4.0, ma le immagini sono di webcam di terzi | [zenodo.org/records/2562396](https://zenodo.org/records/2562396) (`OriginalImages.zip`, senza la cartella `GSV`); progetto [cs.valdosta.edu/~rpmihail/skyfinder](https://cs.valdosta.edu/~rpmihail/skyfinder/) | citare Mihail et al. 2016 e Nice & Wijnands 2019 |

⚠ **Eccezioni dichiarate.** FBD-SV-2024 (solo ricerca) e Skyfinder (diritti incerti) sono usati perché il dataset
serve a una tesi di ricerca e non a un prodotto. Sono separabili (colonna `fonte` del manifest): vanno tolti prima
di pubblicare il dataset o di usarlo a fini commerciali. Le label di tutte le fonti sono state riviste a mano (CVAT);
i box di DVIDS, Anti-UAV e FRED sono proposte automatiche corrette a mano, non le annotazioni originali.
