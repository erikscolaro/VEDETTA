# Status — 19/10/2026

Stato del lavoro e passaggio di consegne. Procedura completa: [`docs/procedura.md`](docs/procedura.md); regole:
[`docs/regole.md`](docs/regole.md). Rispondere in italiano, poche parole, spiegazioni semplici; l'utente rivede a mano
su fogli numerati e risponde con i numeri.

## Cosa stiamo facendo

Costruzione del dataset da Svanström (284 video RGB 640×512). Passo 5 (box) **finito** il 16/10/2026; prossimo il
passo 10, aggiunte: FRED per i droni (`tools/fred.py`, `data/fred/`), FBD-SV-2024 per gli uccelli e DVIDS per aerei ed
elicotteri (`tools/aggiunte.py`, `data/fbd_sv/`, `data/dvids/`), tutte con i task CVAT pronti da rivedere. Una cartella per passo in `data/svanstrom/` (fuori da git). Ogni passaggio fatto va
scritto in `docs/procedura.md` (richiesta dell'utente).

## A che punto siamo

| passo | stato | uscita |
|---|---|---|
| 2 frame | fatto | `1_frame/`: 284 video, 88.932 frame (V_AIRPLANE_002 scartato) |
| 3 clip (tagli di scena) | fatto, rivisto a mano | `2_sequenze/sequenze.csv`: 57 tagli, 341 clip |
| 4 sessioni | fatto, rivisto a mano | `3_sessioni/sessioni.csv`: 341 clip in 112 sessioni |
| 5 box | fatto, tutto rivisto in CVAT | `4_box/manuale/`: 82.315 frame con box, 323 clip |
| 6 meteo | fatto, rivisto in CVAT | `5_meteo/meteo.csv`: sole 219, nuvole 116, nebbia 6 |
| 7 quasi-duplicati | fatto, 16 bit scelti sui fogli | `6_dedup/tenuti.csv`: 14.505 frame su 83.548 (17%) |
| 8 conteggio | fatto | `7_conteggio/conteggio.csv` |
| 9 buchi (tutte le fonti, `tools/buchi.py`) | fatto | `data/aggregati/8_buchi/buchi.csv`: 15 buchi su 16 celle |
| 10 aggiunte | raccolta finita, preselezione fatta, **task CVAT pronti** | FRED 541–551 (8.996 ritagli); DVIDS 599–628 (28.520); Anti-UAV 629–633 (4.616); FBD 634–644 (10.544). Preselezione in `data/<fonte>/5_box/preselezione/` (`tools/preselezione.py`) |
| 9 buchi dopo la revisione CVAT (22/10/2026, 2.857 per cella) | **8 buchi su 20** (prima, sui box proposti, 4) | aerei 8–16 (865), 16–32 (2.387); uccelli 8–16 (2.571), 64–96 (562), > 96 (209); elicotteri 8–16 (762), 16–32 (1.864), 64–96 (2.596); droni chiusi |
| 11 split (23/10/2026) | fatto: doppioni rivisti (146 coppie unite), gruppi (`11_gruppi/`, 1.010), split per gruppi con val e tetto su mezze ottave (`12_split/`: via 7.198 immagini con box < 8 px); tetto e mosaici (`13_train_compensation/`: 26.841 reali + 929 mosaici); background (`14_background/`, Clouds-1500 e Skyfinder); export in `data/output/` (train 30.579, val 9.013, test 5.165; `distribuzione.{png,pdf,md}`, `leakage.{png,pdf,md}`; minimi 10k/2k/1k per classe rispettati; verificato con Ultralytics; classi 0 drone, 1 airplane, 2 helicopter, 3 bird; README con licenze e legenda, `raw/`, `distribuzione.png`); calib (512, `15_calib/`, `data/output/calib.yaml`); **prossimo**: consegna (sottoinsiemi per fascia, mosaici di deploy, report, manifest) | `aggregati/9_split/frame.csv` (58.960 frame annotati), `embedding_s.npy` (DINOv2 ViT-S; B e L provati e scartati), `somiglianze_s.png`, `somiglianze_tutte.png` |

Passo 5: tolte 12 clip con la scia (`scie.txt`) e 6 scartate in CVAT (`scartate.txt`); 1.233 frame senza box in 18
clip tenuti apposta. Il rapporto √area originale/corretta non si calcola (decisione dell'utente).

## Aperti

1. Obiettivo per cella deciso il 19/10/2026 (`regole.md`): 2.000 nel train, 2.857 in tutto, +10% prima di CVAT.
2. **Raccolta finita (19/10/2026)**: 4 celle sotto il minimo (aerei e elicotteri 8–16, uccelli > 64) come limite dichiarato. Prossimo: revisione CVAT. Scartati: zoom ×2 e YOLO a risoluzione piena sui 4K; DVIDS non serve per gli uccelli grandi (eventuale Wikimedia dopo).
3. Revisione in CVAT dei task 541–551 e 599–644, poi `cvat_import` e conteggio.
4. Fumo degli airshow DVIDS = scia (deciso nel giro 10): 86 tratti in `data/dvids/0_raw/fumo.txt`, 7.233 ritagli tolti.
5. Licenze in `docs/licenze.md`: solo aperte (pubblico dominio, CC0, CC BY). FBD-SV-2024 ("solo ricerca") tenuto
   come eccezione dichiarata. DVIDS: frase di non-endorsement nella tesi.
6. Oggetti sotto i 4 px via; immagini con box sotto gli 8 px via da tutti gli split (23/10/2026, prima solo nel train).

7. **A fine progetto** (utente, 19/10/2026): una sola procedura per rifare **lo stesso identico dataset**. Cartella
   `manifest/` in git con i soli video usati (DVIDS: id, url, mp4, licenza, autore, tratti, sha256; Anti-UAV,
   FBD, FRED: sequenze e frame), le scelte dell'operatore (`esclusi.txt`, `fumo.txt`, `scarti*.txt`, sessioni,
   esclusi della preselezione) e le label finali di CVAT; un comando (`tools/rifai.py`) che scarica, rifà i
   passi e verifica gli hash di immagini e label; versioni di ffmpeg/OpenCV fissate. `procedura.md` riscritta
   con l'essenziale; la cronaca dei giri in un diario a parte (query in un file di dati, tabella dei giri
   generata da script).

8. **Obiettivo nuovo** (22/10/2026, `regole.md`): curva continua s^−α, 14.286 box per classe; drone e uccello principali, deficit di aerei ed elicotteri piccoli accettato. α = 0,3 per tutte le classi (23/10/2026, stima: dominio sconosciuto). Uccelli grandi sotto la curva accettati; piccoli del train con mosaici ridotti dopo lo split (`regole.md`). **Da fare**: allineare `tools/buchi.py` alla curva.

## Note pratiche

- Task CVAT DVIDS da rivedere (20/10/2026): 645–667 (655 → 668 passato da Sonnet, 645 → 669 passato da Haiku, 646 → 671 passato da Sonnet; 647–654 e 656–664 fatti sugli originali e in stage validation, 665 chiuso dall'utente; controllo dei task chiusi dall'utente fatto (task 672, correzioni riportate sugli originali il 22/10); secondo controllo (task 673) riportato sugli originali e task di controllo cancellati; 669 svuotato per errore e ricostruito il 22/10, passato da Sonnet, in validation per la revisione dell'utente; per gli altri un solo agente Sonnet a campione, 1 frame ogni 10 ±5 intorno agli errori, 10 immagini per turno), rifatti senza stella, visore e box doppie o interne.
- Python: `air-detection/.venv/bin/python` (torch, ultralytics, cvat-sdk). Pesi SAM in `data/modelli/sam2.1_l.pt`.
- CVAT dagli script: `NO_PROXY=localhost,127.0.0.1` (c'è un proxy di sistema); credenziali in `~/.config/cvat.env`
  (`CVAT_USER`, `CVAT_PASS`, utente `erikscolaro`). Docker vuole `sudo` (password): se CVAT è spento, l'utente lancia
  `cd ~/Desktop/cvat && sudo docker compose up -d`.
- Modifiche non committate da `30a5a44`: `tools/svanstrom.py` (job da 10k frame, meteo), `tools/fred.py` e `tools/aggiunte.py` (non tracciati), `docs/procedura.md`, `README.md`, `requirements.txt`, questo file. Commit a una riga, in italiano, senza
  co-author. Le revisioni umane in `data/` si versionano a fine progetto (decisione dell'utente).
