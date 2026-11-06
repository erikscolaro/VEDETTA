# VEDETTA: dataset per la rilevazione di velivoli e uccelli

Generato da `tools/dataset.py --step export` il 28/10/2026 (repo vedetta; procedura completa nel README della
repo). Formato Ultralytics YOLO: `images/<split>/*.jpg`, `labels/<split>/*.txt` (una riga per box: classe, centro x,
centro y, larghezza, altezza, normalizzati), `dataset.yaml`. Tutte le immagini sono 640×512, mai ingrandite. Le
immagini sono hardlink ai file del repo; le label sono riscritte qui con gli id sotto (nel repo restano nell'ordine
airplane, bird, drone, helicopter).

| classe | id |
|---|---|
| drone | 0 |
| airplane | 1 |
| helicopter | 2 |
| bird | 3 |

## Dominio

Il dataset serve a una torretta anti-drone: telecamera RGB a terra che guarda il cielo e l'orizzonte. Ogni clip ha
uno **scenario** (colonna `scenario` del manifest), cioè lo sfondo attorno ai bersagli:
- `cielo`;
- `orizzonte`: in volo basso, con alberi, colline o edifici dietro;
- `terra`: fermo o posato, in rullaggio, nell'erba;
- `muro`: davanti a facciate o cortili.

Val e test contengono **solo cielo e orizzonte**. Il train ammette al massimo il 25% per classe di terra e muro,
perché aiuta a imparare la forma degli oggetti.

## Contenuto

| split | immagini | reali | mosaici | background |
|---|---|---|---|---|
| train | 21.958 | 18.397 | 511 | 3.050 |
| val | 6.659 | 5.759 | 0 | 900 |
| test | 3.441 | 2.925 | 0 | 516 |
| calib | 511 | 460 | 0 | 51 |

- **train**: immagini reali, con queste regole:
  - tetto per classe e quarto d'ottava di grandezza (70% della curva obiettivo);
  - nessuna sessione oltre il 6% di una classe;
  - nessuna fonte oltre 1,5 volte la sua quota in val e test.

  Poi i mosaici di compensazione per gli oggetti piccoli, e il background.
- **val**: immagini reali, al 20% della curva obiettivo e mai oltre il 20% del materiale disponibile, più background.
  Serve a scegliere il modello migliore.
- **test**: immagini reali con lo stesso numero di box per fascia di grandezza (uniforme in scala logaritmica), più
  background. Si usa una volta sola.
- **calib**: circa 500 immagini del train (reali e background, niente mosaici), bilanciate per classe e grandezza,
  per la calibrazione int8 (`calib.yaml`).
- **raw/**: `images/` e `labels/` di train, val e test in una cartella sola (senza calib), per chi vuole rifare gli
  split; lo split di ogni immagine è nel manifest.
- Nessuna immagine ha box sotto gli 8 px di lato (√(w·h)).
- Lo split è per **gruppo**: tutte le immagini di un gruppo stanno nello stesso split, quindi la stessa ripresa non
  è mai in due split.
- **Uccelli sotto-rappresentati** (scelta dichiarata): in cielo o all'orizzonte ce ne sono pochi. FBD-SV-2024 è
  quasi tutto vegetazione, quindi resta solo nel train e dentro la quota fuori dominio.
- **Droni ad ala fissa esclusi**: troppo pochi per stare in tutti gli split.

![distribuzione](distribuzione.png)

`distribuzione.png` (e `.pdf`): istanze per grandezza √(w·h) e classe.
- Blu: addestramento, immagini reali; giallo: mosaici sintetici; linee colorate: validazione e test.
- Obiettivi in nero: densità in scala logaritmica ∝ s^−0,3 tra 8 e 320 px (uccelli fino a 96), alzata ×1,2.
  - Addestramento al 70% (linea continua).
  - Validazione al 20% (tratteggiata).
  - Test uniforme in scala logaritmica (punteggiata).

![fonti](fonti.png)

`fonti.png`: quota delle box di ogni fonte per split e classe (i mosaici contano la fonte delle loro caselle).

![contesto](contesto.png)

`contesto.png`: quota delle box di ogni scenario per split e classe. Il meteo comparirà come seconda riga quando la
colonna `meteo` del manifest sarà riempita.

![leakage](leakage.png)

`leakage.png` (tabella in `leakage.md`): per ogni immagine reale di validazione e test, la somiglianza DINOv2 con
l'immagine più vicina dell'addestramento. Si confronta con due riferimenti presi dentro l'addestramento:
- stesso gruppo e altra clip, cioè com'è fatto un leakage vero;
- gruppi diversi, cioè il fondo.

## Legenda dei nomi

| prefisso | tipo | formato | esempio |
|---|---|---|---|
| `R` | immagine **reale** di una ripresa | `R_<gruppo>_<clip>_<n>` | `R_G0000_a7bdc852_000001.jpg` |
| `M` | **mosaico** di compensazione (solo train) | `M_x<griglia>_<n>` | `M_x4_000012.jpg` |
| `B` | **background**, senza label | `B_<camera>_<n>` | `B_f88e9c93_000033.jpg` |

- `<gruppo>`: unità dello split.
  - `G0000`…: un gruppo riunisce le sessioni di ripresa che si somigliano.
  - `GO…`: foto di Open Images.
  - `GN…`: sessioni Anti-UAV aggiunte il 26/10/2026.
- `<clip>` / `<camera>`: primi 8 caratteri dello SHA-1 di `fonte:clip` (o `fonte:camera` per il background).
- `<n>`: numero progressivo a 6 cifre, in ordine di frame dentro la clip o la camera.
- `M_x2`, `M_x4`, `M_x8`: mosaici 640×512 di 2×2, 4×4 o 8×8 ritagli del train ridotti di ×0,5, ×0,25 o ×0,125, con
  sfocatura, rumore e JPEG leggeri.
  - Le caselle di un mosaico sono prese **a caso dalla stessa mezza ottava di grandezza**, solo da immagini in
    dominio, a turno tra le sessioni.
  - Sono sintetici e stanno solo nel train.

## Risalire all'origine

`manifest.csv` ha una riga per immagine, con queste colonne:
- `immagine` (percorso qui), `split`, `tipo` (reale, mosaico, background);
- `gruppo`, `fonte`, `sessione`, `clip` (o camera);
- `frame`: indice nel video originale;
- `scenario`, `meteo` (vuoto per ora);
- `sorgente`: file nel repo vedetta, sotto `data/`.

Attribuzioni e origini:
- **DVIDS**: pagina, autore e licenza di ogni video in `fonti_dvids.csv`.
- **Open Images**: autore, licenza e pagina di ogni foto in `fonti_openimages.csv`.
- **Mosaici**: le sorgenti, ritaglio per ritaglio, in `intermediate/17_dataset/mosaici.csv` del repo.

## Fonti, licenze e attribuzione

Il dataset nel suo insieme è rilasciato con licenza **CC BY 4.0**. Ogni immagine mantiene la licenza della sua fonte
(colonna `fonte` del manifest), che va rispettata con l'attribuzione indicata.

| fonte | cosa dà | licenza | download | attribuzione |
|---|---|---|---|---|
| Svanström et al., "A dataset for multi-sensor drone detection", Data in Brief 2021 | video RGB 640×512 di droni, aerei, elicotteri, uccelli | **CC0 1.0** | [github.com/DroneDetectionThesis/Drone-detection-dataset](https://github.com/DroneDetectionThesis/Drone-detection-dataset) | citare l'articolo |
| FRED, Magrini et al., ACM MM 2025 | droni (sequenze RGB) | **Apache 2.0** | [huggingface.co/datasets/GabrieleMagrini/FRED](https://huggingface.co/datasets/GabrieleMagrini/FRED) | citare l'articolo; testo della licenza in `LICENSES/Apache-2.0-FRED.txt` |
| Anti-UAV300 (parte RGB), Jiang et al., IEEE TMM 2021 | droni | **MIT** | [github.com/ZhaoJ9014/Anti-UAV](https://github.com/ZhaoJ9014/Anti-UAV), mirror [huggingface.co/datasets/DestinyLove/AntiUAV300](https://huggingface.co/datasets/DestinyLove/AntiUAV300) | citare l'articolo; copyright e licenza in `LICENSES/MIT-Anti-UAV.txt` |
| DVIDS (Defense Visual Information Distribution Service) | aerei, elicotteri, droni, uccelli | **pubblico dominio** (opere di dipendenti federali USA), video per video | [dvidshub.net](https://www.dvidshub.net) (elenco in `fonti_dvids.csv`) | "The appearance of U.S. Department of War (DoW) visual information does not imply or constitute DoW endorsement."; niente insegne in usi commerciali |
| Open Images V7 (Google) | uccelli in cielo (box individuali) | annotazioni **CC BY 4.0**; immagini **CC BY 2.0** | [storage.googleapis.com/openimages/web](https://storage.googleapis.com/openimages/web/index.html) | autore e link di ogni foto in `fonti_openimages.csv`; citare Kuznetsova et al., IJCV 2020 |
| FBD-SV-2024, Sun et al., Scientific Data 2025 | uccelli | **CC BY 4.0** | [figshare 27377595](https://figshare.com/articles/dataset/FBD-SV-2024/27377595), Kaggle `swjtuziwei/fbd-sv-2024` | citare l'articolo |
| Clouds-1500 (Mendeley Data, v2, 2023) | background: cielo verso l'orizzonte, meteo vario | **CC BY 4.0** | [data.mendeley.com/datasets/2khchjbgzr/2](https://data.mendeley.com/datasets/2khchjbgzr/2) | citare il dataset e gli autori, indicare la licenza |
| SkyFinder, Mihail et al., WACV 2016 (copia nel record Zenodo 2562396, Nice & Wijnands 2019) | background: 18 webcam AMOS | **CC BY 4.0** | [zenodo.org/records/2562396](https://zenodo.org/records/2562396) | citare Mihail et al. 2016 e Nice & Wijnands 2019 |

**Revisione delle label:**
- Le label sono state riviste a mano in CVAT.
- Il 25–26/10/2026 c'è stata una seconda revisione per candidati. Due detector indipendenti hanno segnalato oggetti
  senza box, box vuote e classi in disaccordo; le decisioni sono state prese dall'operatore su fogli di controllo
  e sono tracciate in `intermediate/13_revisione/` del repo.
- I droni Anti-UAV aggiunti il 26/10/2026 (`input/anti_uav/8_extra`) hanno le label originali del dataset, non
  riviste.
