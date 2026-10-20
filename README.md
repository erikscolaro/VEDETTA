# air-detection

Dataset per la tesi in `~/Desktop/yoloeval` (piano: `NEW_MASTERPLAN.md`): torretta anti-drone a terra, camera RGB
verso il cielo. Classi: `0 airplane`, `1 bird`, `2 drone`, `3 helicopter`.

Base Svanström (640×512), più aggiunte per le celle classe × fascia scoperte, tutte a 640×512. In costruzione: stato
in [`STATUS.md`](STATUS.md).

| file | contenuto |
|---|---|
| [`STATUS.md`](STATUS.md) | a che punto siamo, cosa resta da fare |
| [`docs/regole.md`](docs/regole.md) | regole e consegna a yoloeval |
| [`docs/procedura.md`](docs/procedura.md) | passi di costruzione, in ordine |
| [`docs/svanstrom.md`](docs/svanstrom.md) | Svanström: label, numeri, box larghi |
| [`docs/ricerca_fonti.md`](docs/ricerca_fonti.md) | fonti candidate e scartate |
| [`docs/da_decidere.md`](docs/da_decidere.md) | punti aperti, rischi, limiti |
| `tools/svanstrom.py` | un `--step` per passo della procedura (sotto); senza `--step`, la pipeline vecchia da adattare |
| `tools/buchi.py` | passo 9, buchi su tutte le fonti: `data/*/*_conteggio` -> `data/aggregati/8_buchi/buchi.csv` |
| `tools/fred.py` | FRED, aggiunta per i droni: `--step label` (passo 0, solo le label via HTTP a intervalli), `--step anteprima` (un frame per sequenza, fogli per togliere le indoor) |
| `tools/aod4_fonti.py` | fonti di AOD4 dai nomi dei file (solo riferimento) |

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # poi .venv/bin/python al posto di python
python tools/svanstrom.py --step frames   # passo 2: data/svanstrom/0_raw -> 1_frame
python tools/svanstrom.py --step tagli    # passo 3: candidati taglio da rivedere in 2_sequenze/candidati
python tools/svanstrom.py --step sequenze # passo 3: dai tagli veri in 2_sequenze/tagli.txt a sequenze.csv
python tools/svanstrom.py --step sessioni # passo 4: proposta di sessioni da rivedere in 3_sessioni/fogli
python tools/svanstrom.py --step unione   # passo 4: proposta + 3_sessioni/correzioni.txt -> sessioni.csv
python tools/svanstrom.py --step box      # passo 5: box a contrasto, clip pulite/difficili, fogli
python tools/svanstrom.py --step sam      # passo 5: SAM 2 con punto sulle difficili e sulle pulite sbagliate
python tools/svanstrom.py --step fogli_sam  # passo 5: fogli di revisione dei box SAM
python tools/svanstrom.py --step cvat     # passo 5: task CVAT 'box', job da ~10k frame di clip intere (tutte tranne le scie)
python tools/svanstrom.py --step cvat_import    # passo 5: zip YOLO di CVAT in 4_box/cvat_zip -> 4_box/manuale
python tools/svanstrom.py --step meteo          # passo 6: meteo per clip (SigLIP zero-shot) -> 5_meteo + task CVAT 'meteo'
python tools/svanstrom.py --step meteo_import   # passo 6: tag corretti in CVAT -> 5_meteo/meteo.csv
python tools/svanstrom.py --step dedup          # passo 7: quasi-duplicati dentro la clip -> 6_dedup/tenuti.csv
python tools/svanstrom.py --step fogli_dedup    # passo 7: fogli di controllo, tenuti/tolti -> 6_dedup/fogli
python tools/svanstrom.py --step conta          # passo 8: istanze, frame, video, sessioni per cella -> 7_conteggio/conteggio.csv
```

![box originali di Svanström: 16 px attorno a oggetti di 4–6 px](immagini/svanstrom_16px.png)
