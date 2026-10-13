# Status — 13/10/2026, 19:15

Passaggio di consegne per il prossimo agente. Procedura completa: [`docs/procedura.md`](docs/procedura.md); regole:
[`docs/regole.md`](docs/regole.md). Rispondere in italiano, poche parole, spiegazioni semplici; l'utente rivede a mano
su fogli numerati e risponde con i numeri.

## Cosa stiamo facendo

Costruzione del dataset da Svanström (284 video RGB 640×512), passo 5 della procedura: **correggere i box**, che in
Svanström sono larghi circa 3 volte l'oggetto. Una cartella per passo in `data/svanstrom/` (fuori da git).

## A che punto siamo

| passo | stato | uscita |
|---|---|---|
| 2 frame | fatto | `1_frame/`: 284 video, 88.932 frame (V_AIRPLANE_002 scartato) |
| 3 clip (tagli di scena) | fatto, rivisto a mano | `2_sequenze/sequenze.csv`: 57 tagli, 341 clip |
| 4 sessioni | fatto, rivisto a mano | `3_sessioni/sessioni.csv`: 341 clip in 112 sessioni (S076: 49 clip di droni, 16,6% dei frame) |
| 5 box | **in corso** | `4_box/` |

Passo 5, dettaglio (`data/svanstrom/4_box/`):

- `box.csv`: box a contrasto locale per tutti i 95.944 box; clip **pulite** (226) e **difficili** (115).
- `pulite_sbagliate.txt`: 50 pulite bocciate in revisione → passate a SAM. Le altre 176 pulite sono **buone** (box a
  contrasto, con interpolazione dei box segnalati).
- `scie.txt`: 12 clip di aerei con la scia, **da togliere dal dataset** (decisione dell'utente).
- **SAM 2.1 Large con prompt a punto**, in esecuzione in background (`--step sam`, log in `sam.log`) sulle 165
  clip difficili + pulite sbagliate: alle 18:00 era a 94/165, fine prevista verso le 18:40. Un json per clip in
  `sam/`; se il processo è morto, rilanciare `--step sam`: riparte dalle clip mancanti.
- `fogli_sam/` + `fogli_sam/clip.txt`: fogli di revisione dei box SAM (rigenerare con `--step fogli_sam`; i numeri
  già dati restano stabili). L'utente ha rivisto le clip **1–34**: 9 sbagliate in `sam_sbagliate.txt`.
- **CVAT** per le correzioni a mano: `~/Desktop/cvat` (v2.78.0, Docker), `http://localhost:8080`, cartella
  `data/svanstrom` montata in sola lettura. Task 4 = V_AIRPLANE_017/s0, già corretto dall'utente e importato in
  `manuale/V_AIRPLANE_017_s0/`.

## Decisione delle 18:05: tutto in CVAT

L'utente vuole rivedere **tutto** il dataset in CVAT, non solo le clip bocciate: un task per ogni clip tranne le
scie (329 clip), con i box migliori disponibili (SAM se la clip è passata da SAM, altrimenti contrasto). Le label
finali vengono tutte dagli export CVAT (`4_box/cvat_zip/` → `--step cvat_import` → `4_box/manuale/`).

Aggiornamento 19:15: **SAM finito** (165/165, json in `4_box/sam/`) e **tutte le 329 clip caricate in CVAT**
(task 4–332; il 4 è V_AIRPLANE_017/s0, già corretto). Log in `4_box/cvat.log`.

## Cosa deve fare il prossimo agente

1. SAM è finito (165/165).
2. I task ci sono tutti (329, chiavi ogni 50 frame). La revisione con i
   fogli di SAM (`fogli_sam`, `sam_sbagliate.txt`) non serve più: l'utente rivede tutto in CVAT; `sam_sbagliate.txt`
   resta come elenco delle clip da guardare per prime.
3. L'utente corregge in CVAT, esporta **Ultralytics YOLO Detection 1.0** e lascia gli zip in `4_box/cvat_zip/`; poi
   `--step cvat_import`. Le clip irrecuperabili si scartano (chiedere).
4. Label finali di `4_box/` = quelle di `manuale/` (tutte le clip tranne le scie). Poi il rapporto fra √area originale
   e corretta per classe e fascia (in `procedura.md`, passo 5).
5. Ancora aperti nel passo 5: oggetti non etichettati (stormi).
6. Passo 6, conteggio per cella. Nota: con i box a contrasto il **64% dei box è sotto i 16 px** (uccelli 93%, droni
   74%). L'utente ha proposto di scartare i frame sotto i 16 px; si è rimandato al conteggio, perché toglierebbe quasi
   tutti gli uccelli. Le regole dicono: sotto i 16 px solo nel train.

## Note pratiche

- Python: `air-detection/.venv/bin/python` (torch, ultralytics, cvat-sdk). Pesi SAM in `data/modelli/sam2.1_l.pt`.
- CVAT dagli script: `NO_PROXY=localhost,127.0.0.1` (c'è un proxy di sistema); credenziali in `~/.config/cvat.env`
  (`CVAT_USER`, `CVAT_PASS`, utente `erikscolaro`). Docker vuole `sudo` (password): se CVAT è spento, l'utente lancia
  `cd ~/Desktop/cvat && sudo docker compose up -d`.
- Modifiche non committate da `1fdd84d`: `tools/svanstrom.py` (passi `fogli_sam`, `cvat`, `cvat_import`),
  `docs/procedura.md`, `docs/regole.md` (scie), `README.md`, questo file. Commit a una riga, in italiano, senza
  co-author. Le revisioni umane in `data/` si versionano a fine progetto (decisione dell'utente).
- La sessione opus-2 sta allineando yoloeval al `NEW_MASTERPLAN.md` (senza testare sul dataset, che non esiste ancora).
