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
| 9 buchi (obiettivo nuovo: 2.857 per cella, 5 fasce da 8–16, +10% prima di CVAT) | 4 buchi su 20 dopo giro 10 e uccelli DVIDS (box proposti) | aereo 8–16 (1.392); elicottero 8–16 (1.665); uccello 64–96 (578), > 96 (215); droni ed elicotteri 16–96 chiusi col margine |

Passo 5: tolte 12 clip con la scia (`scie.txt`) e 6 scartate in CVAT (`scartate.txt`); 1.233 frame senza box in 18
clip tenuti apposta. Il rapporto √area originale/corretta non si calcola (decisione dell'utente).

## Aperti

1. Obiettivo per cella deciso il 19/10/2026 (`regole.md`): 2.000 nel train, 2.857 in tutto, +10% prima di CVAT.
2. **Raccolta finita (19/10/2026)**: 4 celle sotto il minimo (aerei e elicotteri 8–16, uccelli > 64) come limite dichiarato. Prossimo: revisione CVAT. Scartati: zoom ×2 e YOLO a risoluzione piena sui 4K; DVIDS non serve per gli uccelli grandi (eventuale Wikimedia dopo).
3. Revisione in CVAT dei task 541–551 e 599–644, poi `cvat_import` e conteggio.
4. Fumo degli airshow DVIDS = scia (deciso nel giro 10): 86 tratti in `data/dvids/0_raw/fumo.txt`, 7.233 ritagli tolti.
5. Licenze in `docs/licenze.md`: solo aperte (pubblico dominio, CC0, CC BY). FBD-SV-2024 ("solo ricerca") tenuto
   come eccezione dichiarata. DVIDS: frase di non-endorsement nella tesi.
6. Oggetti sotto i 4 px via; 4–8 px solo nel train (deciso il 19/10/2026).

7. **A fine progetto** (utente, 19/10/2026): una sola procedura per rifare **lo stesso identico dataset**. Cartella
   `manifest/` in git con i soli video usati (DVIDS: id, url, mp4, licenza, autore, tratti, sha256; Anti-UAV,
   FBD, FRED: sequenze e frame), le scelte dell'operatore (`esclusi.txt`, `fumo.txt`, `scarti*.txt`, sessioni,
   esclusi della preselezione) e le label finali di CVAT; un comando (`tools/rifai.py`) che scarica, rifà i
   passi e verifica gli hash di immagini e label; versioni di ffmpeg/OpenCV fissate. `procedura.md` riscritta
   con l'essenziale; la cronaca dei giri in un diario a parte (query in un file di dati, tabella dei giri
   generata da script).

## Note pratiche

- Task CVAT DVIDS da rivedere (20/10/2026): 645–667 (655 → 668 passato da Sonnet, 645 → 669 passato da Haiku, 646 → 671 passato da Sonnet; per gli altri agenti Sonnet a 10 immagini per turno, direttamente sugli originali), rifatti senza stella, visore e box doppie o interne.
- Python: `air-detection/.venv/bin/python` (torch, ultralytics, cvat-sdk). Pesi SAM in `data/modelli/sam2.1_l.pt`.
- CVAT dagli script: `NO_PROXY=localhost,127.0.0.1` (c'è un proxy di sistema); credenziali in `~/.config/cvat.env`
  (`CVAT_USER`, `CVAT_PASS`, utente `erikscolaro`). Docker vuole `sudo` (password): se CVAT è spento, l'utente lancia
  `cd ~/Desktop/cvat && sudo docker compose up -d`.
- Modifiche non committate da `30a5a44`: `tools/svanstrom.py` (job da 10k frame, meteo), `tools/fred.py` e `tools/aggiunte.py` (non tracciati), `docs/procedura.md`, `README.md`, `requirements.txt`, questo file. Commit a una riga, in italiano, senza
  co-author. Le revisioni umane in `data/` si versionano a fine progetto (decisione dell'utente).
