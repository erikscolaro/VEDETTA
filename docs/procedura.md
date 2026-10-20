# Procedura

Ogni passo dipende dal precedente, tranne il meteo (6), che non usa i box. Regole in [`regole.md`](regole.md).

Ogni passo di una fonte ha la sua cartella in `data/<fonte>/`; i passi su tutte le fonti (9 e 11–14) hanno la loro
cartella numerata in `data/aggregati/` (`8_buchi/`, `9_split/`). Ogni cartella legge solo da quelle dei passi prima, così si rifà un passo senza ripartire da
zero. Nelle cartelle di soli indici (2, 3, 5, 6, 7) vanno CSV, non copie delle immagini.

| cartella | contenuto | passo |
|---|---|---|
| `0_raw/` | video `.mp4` + label `.mat` | download |
| `1_frame/` | frame `.png` + label originali YOLO, per frame | 2 |
| `2_sequenze/` | `candidati/` (fogli + `candidati.csv`) e `sequenze.csv` (video, frame, sequenza) | 3 |
| `3_sessioni/` | `embedding.npz`, `fogli/`, `proposta.csv`, `correzioni.txt`, `sessioni.csv` (video, sessione) | 4 |
| `4_box/` | `box.csv`, `sam/`, fogli di revisione, `cvat_zip/`, `manuale/` (label finali), `scie.txt`, `scartate.txt` | 5 |
| `5_meteo/` | `proposta.csv`, `meteo.csv` (clip, sessione, meteo), `meteo.log` | 6 |
| `6_dedup/` | `tenuti.csv` (frame che restano) | 7 |
| `7_conteggio/` | `conteggio.csv` (classe, fascia, istanze, frame, clip, video, sessioni) | 8 |
| `data/aggregati/9_split/` (tutte le fonti) | `images/` e `labels/{train,val,test}` in hardlink + `manifest.json` + report | 11–14 |

## Come rifare le aggiunte (passo 10)

Comandi in ordine; **[operatore]** = una persona guarda dei fogli e scrive un file di testo, che il passo dopo legge.
Tutti i file scritti dall'operatore restano in `data/` (sono la traccia delle scelte). Python: `.venv/bin/python`.

Fonti a dataset (label già fatte): FRED (`tools/fred.py`, passi `frame … proposte`, vedi sotto), FBD
(`aggiunte.py --fonte fbd --step frame`), Anti-UAV (`--fonte antiuav --step frame`). Poi per ognuna `--step catena`,
`--step sessioni`.

DVIDS (video di pubblico dominio), a giri, per classe (`--classe airplane|helicopter|drone|bird`):

1. `aggiunte.py --fonte dvids --step cerca --classe C`: query di `QUERY` → `0_raw/candidati.csv`.
2. `--step info --classe C`: licenza (solo "PUBLIC DOMAIN"), mp4 originale, durata, risoluzione → `0_raw/video.csv`.
3. `--step anteprima --classe C`, `--step doppioni`, `--step scegli`: anteprime ogni 10 s, video doppi, tratti di
   ±5 s dove la YOLO vede un velivolo in cielo → `0_raw/tratti.csv`.
4. `--step fogli --classe C [--grandezza 96 | --minimo 48]` → `0_raw/fogli_trattiN/` (un'anteprima per tratto).
5. **[operatore]** guarda i fogli e scrive `0_raw/scartiN.txt`, una riga `numero codice` per ogni tratto da
   scartare: C ripresa da dentro un velivolo; A aria-aria o dall'alto; P persone in primo piano senza velivoli in volo;
   I interni; T solo velivoli a terra in primo piano; N nessun velivolo; F scia di fumo o condensa. Nel dubbio tiene.
6. `--step escludi --fogli fogli_trattiN --scarti data/dvids/0_raw/scartiN.txt` → `0_raw/esclusi.txt`.
7. `--step frame --classe C [--grandezza 96]`: 5 fps dei tratti tenuti, label YOLO; poi `--step catena`,
   `--step sessioni`.
8. Fumo nei tratti già scaricati: `--step fogli_fumo` → `0_raw/fogli_fumoN/` (due ritagli per tratto di aerei);
   **[operatore]** scrive `0_raw/scarti_fumoN.txt` (righe `numero F` dei tratti con una scia di fumo o condensa
   dietro un velivolo; non nuvole, polvere, foschia); `--step fumo --fogli fogli_fumoN --scarti …` → `0_raw/fumo.txt`.

Su tutte le fonti:

9. `tools/preselezione.py` → `data/<fonte>/5_box/preselezione/` (label corrette ed `esclusi.txt`: classe del video,
   primi piani, notte, fumo, box sotto 4 px², tetto). Si rilancia dopo ogni `catena`.
10. `--step conta` per fonte (`fred.py --step conta` per FRED), poi `tools/buchi.py` → `aggregati/8_buchi/buchi.csv`.
11. Facoltativo, stima del lavoro in CVAT: `tools/campione.py --step fogli --nome N --fonti dvids:70 …`;
    **[operatore]** scrive `aggregati/campioni/N/giudizi.csv` (righe `n,fix|dubbio,tipo` dei ritagli da correggere);
    `--step stima --nome N`.
12. `--step cvat` per fonte (FRED: `fred.py --step cvat`), con `NO_PROXY=localhost,127.0.0.1`: un task per gruppo di
    sequenze, le sequenze già in un task si saltano. **[operatore]** corregge i box in CVAT (classe, box mancanti,
    falsi positivi, box che uniscono più velivoli), esporta "Ultralytics YOLO Detection 1.0" senza immagini in
    `5_box/cvat_zip/`; `--step cvat_import` → `5_box/manuale/`.

1. **Regole** fissate prima di toccare i dati (fatto, salvo i casi dubbi del box: [`da_decidere.md`](da_decidere.md)).
2. **Frame** (`0_raw → 1_frame`): frame estratti, label `.mat` in YOLO. Clip scartata se label ≠ frame.
3. **Sequenze** (`1_frame → 2_sequenze`): un video ne contiene più di una.
   - Candidati a soglia bassa, tanti falsi positivi: per ogni coppia di frame consecutivi, picco rispetto ai 5 frame
     prima e dopo di due misure, distanza degli istogrammi HSV e 1 − correlazione in scala di grigi (80×64); punteggio
     = somma, candidato sopra 0,07 (398 coppie in 182 video, 34 fogli da 12 coppie).
   - Revisione umana di tutti i candidati: si elencano i numeri dei tagli veri, il resto è scartato.
   - Nessuna soglia automatica regge: panoramiche, cambi di esposizione e tagli si sovrappongono anche dopo aver
     compensato il movimento della camera (taglio vero più debole: 0,13, V_DRONE_101 frame 228).
   - Numeri dei tagli veri in `tagli.txt`. Esito: 57 tagli in 33 video, 341 sequenze su 284 video.
   - Uscita: `sequenze.csv` con l'ID `video/sequenza` per ogni frame (`--step sequenze`).
4. **Sessioni** (`1_frame + 2_sequenze → 3_sessioni`): l'unità è la **clip** (`video/sN`), non il video. Clip di
   video diversi possono essere la stessa ripresa (il tramonto in fondo a V_AIRPLANE_005 è V_AIRPLANE_014); clip dello
   stesso video possono essere di giorni diversi.
   - Striscia per clip: il primo frame e poi uno al secondo (341 clip, 3.164 frame).
   - Embedding DINOv2 ViT-S/14 (token CLS) di ogni frame della striscia; somiglianza fra due clip = massimo fra le
     coppie dei loro frame.
   - Proposta automatica larga: gruppi per legame medio sopra 0,95 (`--session-thr`). Il legame singolo incatena
     tutto, su cielo uniforme quasi ogni coppia supera 0,9. Esito: 47 gruppi e 131 clip sole; dopo la revisione 112
     sessioni (62 da una clip, la più grande 49 clip).
   - Fogli in `fogli/`: uno per gruppo (`gNN.jpg`, una riga per clip, in fondo in grigio la clip più simile fuori dal
     gruppo) e le clip sole a 6 per foglio (`soli_NN.jpg`), ognuna con la sua più simile.
   - Revisione umana: si confrontano le righe fra loro (stesso posto, stesso giorno?), non i frame di una riga. Si
     scrivono solo le eccezioni in `correzioni.txt`: `gNN: via <clip>`, `<clip> con gNN`, `gNN + gMM`. Nel dubbio,
     stessa sessione; le clip di solo cielo, senza indizi sul luogo, restano come proposte.
   - Le risposte (anche "tutto stessa sessione", "v/f" per coppia sui fogli dei soli) si traducono in righe
     esplicite per clip in `correzioni.txt`.
   - Uscita: `sessioni.csv` (clip, sessione) con `--step unione`: gruppi meno i `via`, più le unioni, per
     transitività.
   - Verifica di S076 dopo la regola nuova (16/10/2026): 49 clip di droni, 16,6% dei frame. Sottogruppi DINOv2 a 0,96
     (fogli in `3_sessioni/S076/fogli/`, `S076/proposta.csv`): 26, 11, 7, 4 e 1 clip, ma stesso campo, stesso cielo,
     stesse condizioni, somiglianza fra gruppi fino a 0,99 e meteo tutto sole: è **una sessione sola** e resta tale
     (dividerla metterebbe lo stesso giorno in train e test). Il suo peso si limita al bilanciamento (passo 12, tetto
     per sessione). Le altre sessioni di Svanström restano con la regola del 13/10 (nel dubbio unite, più prudente).
5. **Correggere i box di Svanström** (`1_frame → 4_box`). Le label sono per frame, il tracking non serve.
   - SAM 2.1 Large con il box originale come prompt non serve: ridà il box stesso (prova su 44 box).
   - **Contrasto locale** su tutti i box (`--step box`, 20 s): pixel che si staccano dallo sfondo (mediana e MAD
     dell'anello attorno), dentro il box allargato del 20%. I box corretti sono circa un terzo degli originali come
     lato (mediana 0,35): il 64% sotto i 16 px.
   - Box segnalati: vuoto, sotto i 3 px, allungato oltre 3:1 (scie), sfondo non uniforme (MAD > 15). Clip **pulite**
     con al più il 10% di segnalati (226), **difficili** le altre (115). Nelle pulite i segnalati si interpolano dai
     box buoni vicini della stessa clip. Tutto in `box.csv`.
   - Revisione umana per clip delle pulite (`fogli_pulite/`, 6 frame ingranditi per clip): 50 sbagliate (nuvole,
     rami, pale), in `pulite_sbagliate.txt`.
   - **SAM 2 con un punto** come prompt (centro del box a contrasto) sulle difficili e sulle pulite sbagliate (`--step
     sam`, 47.920 box, un json per clip in `sam/`): meglio con nuvole, alberi e pale; sbaglia ancora su scie e rami.
   - Clip con scia tolte dal dataset (revisione umana di tutte le clip di aerei, `fogli_scie/`): 12 clip in
     `scie.txt`.
   - Poi revisione per clip dei box SAM (`--step fogli_sam`); basta un frame sbagliato e la clip va a mano
     (`sam_sbagliate.txt`).
   - **A mano in CVAT** (locale, `~/Desktop/cvat`, `localhost:8080`, cartella `data/svanstrom` montata in sola
     lettura): `--step cvat` crea un task `box` con **job di clip intere da al più 10.000 frame** (tutte tranne le scie, in ordine di
     sessione), box attuali come traccia (a mano se la clip è in `manuale/`, altrimenti SAM, altrimenti contrasto);
     ogni frame è un frame chiave (l'interpolazione di CVAT sbagliava sugli elicotteri): si rivede frame per frame. Credenziali in `~/.config/cvat.env`. In alternativa si scarta la clip.
   - Rientro: da CVAT si esporta il task in **Ultralytics YOLO Detection 1.0**, senza immagini (Actions → Export
     task dataset, poi Requests → Download) e lo zip si lascia in `4_box/cvat_zip/`; `--step cvat_import` scrive le
     label per frame in `4_box/manuale/<video>_sN/`. I frame senza box non hanno il txt.
   - Esito della revisione di tutte le 329 clip (task `box` 449, 85.437 frame, export del 16/10/2026): 82.315 frame
     con box in 323 clip. Le label importate prima (14/10) sono in `manuale_14-10/`, non usate.
   - **6 clip scartate** (svuotate in CVAT, decisione dell'utente), in `scartate.txt`: V_AIRPLANE_054/s0,
     V_AIRPLANE_055/s0, V_HELICOPTER_057/s0, V_HELICOPTER_058/s0, V_HELICOPTER_059/s0, V_HELICOPTER_060/s0.
   - **Frame senza box tenuti** (voluto: oggetto non visibile), 1.233 frame in 18 clip, i più in V_DRONE_050/s0 (219),
     V_DRONE_039/s0 (185), V_DRONE_105/s0 (132), V_DRONE_052/s0 (123), V_BIRD_009/s0 (117), V_BIRD_029/s0 (114).
   - Rapporto fra √area originale e corretta: non si calcola (decisione dell'utente, 16/10/2026).
   - Label finali di Svanström = `manuale/`, meno scie e scartate.
   - Oggetti non etichettati (stormi): in CVAT l'utente ha aggiunto i più importanti, non i minuscoli (16/10/2026).
6. **Meteo per clip** (`1_frame + 2_sequenze + 3_sessioni → 5_meteo`; non usa i box, si può rifare da solo).
   - `--step meteo`: SigLIP zero-shot (ViT-SO400M-14-SigLIP-384) sui frame della striscia, 5 classi (sole, nuvole, pioggia, neve,
     nebbia), proposta in `proposta.csv` e task CVAT `meteo` (333) con un frame per clip e il tag proposto.
   - Revisione umana in CVAT, un tag per frame; `--step meteo_import` legge i tag da CVAT (niente export) e scrive
     `meteo.csv` (clip, sessione, meteo, modello).
   - Definizione (dell'utente, 16/10/2026): *nuvole* se nel cielo c'è almeno una nuvola, ovunque nell'immagine;
     *sole* se il cielo è sereno.
   - Prima revisione rifatta dall'utente con questa definizione (la prima importazione è in `5_meteo/meteo_16-10_prima.csv`).
     Esito finale (16/10/2026): 341 clip, 74 cambiate a mano rispetto al modello; sole 219, nuvole 116, nebbia 6,
     pioggia e neve 0. 8 sessioni su 112 hanno clip con meteo diversi.
7. **Quasi-duplicati** (`4_box + 2_sequenze → 6_dedup`): hash semantico dentro ogni clip, prima di contare (i frame
   vicini della stessa clip sono quasi uguali e gonfiano i conteggi; ordine cambiato il 16/10/2026, prima veniva
   dopo le aggiunte).
   - `--step dedup` (`--hash-thr 16`, `--geo-thr 0.05`): per ogni clip tranne scie e
     scartate, frame in ordine; un frame resta se nessun frame già tenuto della clip ha lo stesso numero di box, dHash
     del ritaglio sui box (con contesto di un lato, almeno 8 px) entro 16 bit su 64, centro entro 0,05 della diagonale
     normalizzata e lati entro il 35% (0,3 in log). Frame senza box: dHash del frame intero. Label da `4_box/manuale/`.
   - Prima prova con 8 bit (il valore della pipeline vecchia): 39.858 frame tenuti su 83.548 (48%).
   - Controllo a occhio: `--step fogli_dedup` scrive in `6_dedup/fogli/` 12 fogli (3 clip a caso per classe, 24
     frame consecutivi a metà clip, ritaglio 64×64 sull'oggetto; bordo verde tenuto, rosso tolto; `clip.txt`).
     Con 8 bit (16/10/2026, foglio 08, V_DRONE_100/s0) i frame consecutivi quasi uguali restavano quasi tutti: su
     oggetti di pochi pixel rumore e compressione cambiano i bit dell'hash. Decisione dell'utente: alzare la soglia.
   - Prove: 12 bit 28% dei frame, 16 bit 17%, 20 bit 12%. Scelto **16 bit** (nuovo default) dopo i fogli: tra i
     frame tenuti cambia la posa (droni ogni 5–10 frame), i frame identici vanno via (elicottero fermo: 1 su 24).
   - Esito (16/10/2026, 16 bit): **14.505 frame tenuti su 83.548 (17%)**: airplane 14%, bird 23%, drone 17%,
     helicopter 15%. Dei 1.233 frame senza box ne restano 25.
8. **Contare** istanze, frame, clip, video e sessioni per cella classe × fascia, sui box corretti (`--step conta`,
   `4_box + 6_dedup + 3_sessioni → 7_conteggio`: legge i frame di `6_dedup/tenuti.csv` con le label di
   `4_box/manuale/`, scrive `conteggio.csv`). Fasce sulla √(w·h) in px a 640×512:
   < 8, 8–16, 16–32, 32–64, 64–96, > 96 (le due sotto i 16 px solo per sapere quanti sono). Un frame conta in ogni
   cella di cui ha un box.
   - Esito dopo la deduplica a 16 bit (16/10/2026): 323 clip, 14.480 frame con box, 25 senza. Istanze:

     | classe | < 8 | 8–16 | 16–32 | 32–64 | 64–96 | > 96 |
     |---|---|---|---|---|---|---|
     | airplane | 944 | 489 | 315 | 125 | 88 | 136 |
     | bird | 3.027 | 970 | 376 | 16 | 1 | 41 |
     | drone | 2.666 | 1.937 | 1.225 | 304 | 20 | 32 |
     | helicopter | 283 | 365 | 424 | 701 | 421 | 502 |

     Sessioni (≥ 16 px): airplane 13/14/10/10, bird 9/6/1/8, drone 6/3/2/2, helicopter 21/27/22/21. Video: airplane
     17/11/9/8, bird 17/8/1/10, drone 67/23/6/7, helicopter 30/36/32/26.
   - Conteggio preliminare (16/10/2026), **prima della deduplica** (`conteggio_prima_dedup.csv`): 323 clip, 82.315
     frame con box, 1.233 senza. Istanze:

     | classe | < 8 | 8–16 | 16–32 | 32–64 | 64–96 | > 96 |
     |---|---|---|---|---|---|---|
     | airplane | 5.656 | 3.633 | 1.772 | 1.411 | 761 | 1.273 |
     | bird | 16.014 | 3.317 | 956 | 49 | 9 | 172 |
     | drone | 13.639 | 13.417 | 7.004 | 1.312 | 40 | 120 |
     | helicopter | 2.043 | 3.027 | 2.578 | 4.201 | 2.415 | 4.030 |

     Sessioni per cella (≥ 16 px): airplane 15/14/10/10, bird 9/6/2/8, drone 6/3/2/2, helicopter 21/27/23/22.
     Video: airplane 19/11/9/8, bird 18/8/2/10, drone 70/24/7/7, helicopter 32/36/33/28.
   - Il 68% delle istanze è sotto i 16 px (uccelli 94%, droni 76%); sotto gli 8 px il 42%.
9. **Buchi**, su **tutte le fonti insieme** (`data/*/*_conteggio → data/aggregati/8_buchi/`, `python tools/buchi.py`, fuori da
   `svanstrom.py` dal 16/10/2026): somma i conteggi delle fonti (sessioni e video di fonti diverse sono distinti) e
   scrive le celle classe × fascia ≥ 16 px sotto i minimi, con quanto manca e le istanze per fonte, in `buchi.csv`.
   Minimi per cella ricavati da `regole.md`: istanze ≥ 870 (385 in val + 385 in test + 100 in train), sessioni ≥ 3 (lo split è per sessione: almeno una per split), video ≥ 12 (10 in train + 1 in val + 1 in
   test). Per classe, nel train: ≥ 1.500 frame e ≥ 10.000 istanze su tutte le fasce.
   - Esito dopo la deduplica a 16 bit (16/10/2026): **15 buchi su 16 celle**, regge solo drone 16–32 (1.225
     istanze). Sessioni e video bastano quasi ovunque: mancano soprattutto istanze. Per classe tutte sotto le 10.000
     istanze (airplane 2.097, bird 4.431, drone 6.184, helicopter 2.696).

     | classe | 16–32 | 32–64 | 64–96 | > 96 |
     |---|---|---|---|---|
     | airplane | 315 | 125 (1 video) | 88 (3 video) | 136 (4 video) |
     | bird | 376 | 16 (4 video) | 1 (2 sessioni, 11 video) | 41 (2 video) |
     | drone | ok | 304 | 20 (1 sessione, 6 video) | 32 (1 sessione, 5 video) |
     | helicopter | 424 | 701 | 421 | 502 |

     Istanze presenti su 870; tra parentesi gli altri minimi mancanti.
   - Con la deduplica a 8 bit: 11 buchi su 16.
   - Esito preliminare, **prima della deduplica** (`data/aggregati/8_buchi/buchi_prima_dedup.csv`): 8 buchi su 16 celle.
10. **Aggiunte** per i buchi, a 640×512: box proposti da un detector a vocabolario aperto, resi aderenti con SAM 2,
   rivisti a mano. Ogni fonte ha le sue cartelle `0_raw`…`4_box` e la sua deduplica; poi si
    rifanno conteggio e buchi su tutte le fonti.
    - Ricerca delle fonti per i buchi (16/10/2026): candidate in [`ricerca_fonti.md`](ricerca_fonti.md), da scegliere
      dopo aver verificato licenza, punto di vista e ID dei video.
    - **FRED** scelto dall'utente (16/10/2026). Hugging Face `GabrieleMagrini/FRED`: 231 zip, uno per sequenza (184
      train, 47 test), 205 GB; sul disco 72 GB liberi. Prova sulla sequenza 36 (`data/fred/0_raw/36.zip`, 314 MB):
      frame RGB jpg 1280×720 con bordo nero di padding, 3.409 frame, label `coordinates_rgb.txt` (tempo: x1, y1, x2, y2,
      id, modello di drone); nello zip prima i dati a eventi (46%), poi label e RGB in un blocco unico in fondo (54%),
      quindi si scaricano solo questi con richieste HTTP a intervalli. Nomi dei frame con l'ora (20:04–20:06), data
      nello zip 22/05/2025 (da verificare se è la data di ripresa). La 36 è un cortile al crepuscolo, a terra e non
      verso il cielo: le sequenze vanno scelte.
    - Split ufficiali di FRED per sequenza (`dataset_splits/`): non si usano, lo split resta per sessione (passo 11);
      le sessioni di FRED vanno ricostruite.
    - Schema per FRED (deciso il 16/10/2026), una sequenza alla volta, in ordine di utilità per i buchi:
      0. solo le label (piccole): grandezza dei droni, si scaricano le sequenze con box nelle celle che mancano;
      1. download del solo blocco RGB + label (niente dati a eventi);
      2. ritaglio a 640×512 attorno al drone, senza il bordo nero;
      3. hash a 16 bit dentro la sequenza, **prima** di SAM e CVAT (sul ritaglio attorno al box funziona anche con
         box larghi): si rivede a mano solo ciò che resta (correzione proposta e accettata il 16/10/2026);
      4. **box non aderenti** (verificato dall'utente): SAM 2, poi **revisione manuale in CVAT** (fondamentale,
         decisione dell'utente);
      5. via lo zip, i frame tolti e le sequenze fuori scenario (indoor, senza cielo; foglio per l'utente);
      6. conteggio e buchi; si ripete finché le celle dei droni non arrivano al minimo.
    - Passo 0 (`tools/fred.py --step label`, 16/10/2026): per ogni zip l'indice e `coordinates_rgb.txt` con richieste
      a intervalli → `data/fred/0_raw/label/<seq>.txt` e `0_raw/sequenze.csv` (frame, box, modelli, data nello zip,
      ora di inizio e fine dai nomi dei frame, box per fascia sui box originali, non aderenti). Circa 10 s per
      sequenza; riparte dalle sequenze mancanti. Log in `0_raw/label.log`. Lento per la latenza del proxy (4–5
      richieste per sequenza), non per i byte: 8 sequenze in parallelo.
    - Esito del passo 0 (16/10/2026): 231 sequenze, 808.953 frame, 673.221 box (box originali, non aderenti): 16–32
      190.294, 32–64 410.528, 64–96 60.460, > 96 11.602; 126 sequenze con almeno 100 box ≥ 64 px. La data nello zip
      (22/05/2025 per 230 su 231) è quella di creazione degli zip, non della ripresa.
    - Split ufficiali: per sequenza, circa una su cinque pescata dentro ogni blocco di riprese (stesso giorno, posto e
      drone a pochi minuti). Sessioni da ricostruire dall'ora nei nomi dei frame (blocco nuovo quando l'ora torna
      indietro), poi controllate sui frame. Lo split resta per sessione (decisione dell'utente).
    - Sessioni provvisorie (ora che torna indietro o pausa > 1 h): 26. Scelta delle sequenze **bilanciata per modello
      di drone** (proposta dell'utente, 16/10/2026): sequenze con ≥ 500 box ≥ 64 px (originali) per modello: Betafpv
      air75 1 su 51 (9 sessioni), DJI Mini 2 8 su 23 (2), DJI Mini 3 12 su 16 (4), DJI Tello EDU 17 su 111 (12),
      DarwinFPV cineape20 14 su 57 (10; "cineape20ger", 2 sequenze, probabilmente lo stesso drone). Proposta: 2–3
      sequenze per modello da sessioni diverse, le più ricche di box grandi; prima 3 di prova per misurare quanto si
      stringono i box e quanti frame tiene l'hash.
    - Scenario (decisione dell'utente, 16/10/2026): si tolgono le sequenze **indoor**; si tengono cielo, cortili e
      notte, sempre bilanciando per drone. Le label non dicono dove è stata fatta la ripresa: `tools/fred.py --step
      anteprima` scarica il frame a metà di ogni sequenza (`0_raw/anteprima/<seq>.jpg`, solo quel file dallo zip) e
      scrive fogli da 24 in `0_raw/fogli/` con il numero della sequenza; l'utente dà i numeri delle indoor.
      Fatto il 16/10/2026: 231 anteprime, 10 fogli (`01.jpg`–`10.jpg`). Primo foglio (sequenze 0–23): quasi tutto
      cortile con il drone davanti ai muri, cielo solo in 9–12, la 21 quasi nera (notte).
      La 21 (DarwinFPV, 20:33, notte per FRED): su 6 frame sparsi l'RGB è nero (pixel massimo 5–6 su 255, dentro i box
      4–26): i box vengono dalla camera a eventi, il drone nell'RGB non si vede. Anteprime molto scure ma con luci:
      33–36, 111, 145, 147–149, 153; di notte si tiene solo dove il drone si vede nell'RGB (controllo in CVAT).
    - Scenari dai fogli (revisione dell'utente, 16/10/2026) in `0_raw/scenari.csv`: cielo 52–80, 91–110, 11–12, 115,
      218–219; cielo con alberi 81–90, 143; cortile con cielo 48–51 (utilizzabile); notte 21, 111; **interni 168–196
      (via)**; il resto cortile. Sequenze per drone e scenario (tra parentesi i box ≥ 64 px, originali):

      | drone | cielo | cielo con alberi | cortile con cielo | cortile | notte | interni |
      |---|---|---|---|---|---|---|
      | Betafpv air75 | 28 (1.987) | 0 | 1 (50) | 22 (4.332) | 0 | 0 |
      | DJI Mini 2 | 2 (0) | 0 | 0 | 21 (18.833) | 0 | 0 |
      | DJI Mini 3 | 9 (4.739) | 0 | 0 | 6 (7.931) | 1 (980) | 0 |
      | DJI Tello EDU | 5 (930) | 1 (162) | 4 (1.273) | 72 (22.943) | 0 | 29 (15.289) |
      | DarwinFPV cineape20 | 19 (2.347) | 10 (348) | 3 (1.223) | 26 (17.314) | 1 (28) | 0 |

      Una sequenza con due droni conta per entrambi.
    - **Compromessi** (16/10/2026, in `regole.md`): FRED si ferma alle 13 sequenze scaricate, 11 utili (107, 18, 2, 0, 198,
      209, 108, 133, 165, 19, 28; tolte 210, illeggibile, e 111, notte); box in CVAT completi solo per le sessioni che
      vanno in val e test, train a campione; sessioni con una sola revisione (fatta).
    - **Notte tolta** (decisione dell'utente, 16/10/2026, cambia quella di prima): via 21 e 111; restano cielo,
      cielo con alberi, cortile con cielo, cortile. Le sessioni si rifanno senza le sequenze di notte: la notte resta
      nel raggruppamento del modello (così proposta e correzioni non cambiano) e si toglie alla fine → **37 sessioni**;
      sessioni per drone Betafpv 16, DJI Mini 2 4, DJI Mini 3 5, Tello 15, DarwinFPV 12.
    - Prime 3 sequenze di prova (16/10/2026), una per drone, tra le più ricche di box ≥ 64 px: **107** (DJI Mini 3,
      cielo), **210** (DJI Mini 2, cortile), **18** (DarwinFPV, cortile). La 21 (nera) e le interne 168–196 escluse.
    - Cartelle di FRED **numerate in ordine di esecuzione** (richiesta dell'utente, 16/10/2026): `0_raw/` (label,
      anteprime, scenari), `1_frame/FRED_<seq>/` (frame interi 1280×720 e label originali YOLO, classe 2),
      `2_annotati/annotati.csv` (frame con almeno un box), `3_dedup/tenuti.csv`, `4_ritagli/` (640×512), `5_box/`
      (SAM, fogli, poi `manuale/` da CVAT), `6_sessioni/`, `7_conteggio/`. `tools/buchi.py` legge
      `data/<fonte>/*_conteggio/`. La deduplica si fa **prima** del ritaglio, sui frame interi (la posizione del box
      nel frame intero è stabile; nel ritaglio il drone è in posizione casuale e la geometria dell'hash non varrebbe
      più).
    - `--step frame --seq ...`: frame di `RGB/` (non `RGB_extra/`, che c'è in alcune sequenze) in ordine di tempo
      dall'ora nel nome (ultimi tre campi `HH_MM_SS.micro`; i nomi cambiano tra sequenze), label sul frame più vicino
      entro mezzo frame.
    - `--step frame --seq 107 210 18` (16/10/2026): 107 3.387 frame e 2.797 box, 210 3.500 e 3.177, 18 3.412 e
      2.476; 1, 2 e 0 label senza un frame entro mezzo frame (tolte). Su disco 262, 590 e 523 MB.
    - `--step annotati --seq ...` (passo a sé, richiesta dell'utente): via i frame senza box, prima della deduplica.
      Esito: tolti 590 frame della 107, 323 della 210, 936 della 18.
    - `--step dedup --seq ...`: `dedup_clip` di Svanström a 16 bit sui frame di `2_annotati`, interi. Esito (dopo
      aver tolto i frame senza box): 107 1.953 tenuti su 2.797 (70%), 210 721 su 3.177 (23%), 18 1.460 su 2.476
      (59%). Più alti di Svanström (17%): droni vicini, che si muovono e cambiano forma più in fretta.
    - `--step ritaglio --seq ...`: 640×512 senza ridimensionare, dentro l'area senza bordo nero (dalla mediana di
      qualche frame), box più grande in posizione casuale a seed fisso per frame; gli altri box tagliati alla
      finestra (parte visibile). Posizione dei ritagli in `4_ritagli/ritagli.csv`, con i box tagliati o fuori.
    - Il ritaglio **non garantisce** il drone intero: il box più grande resta intero solo se sta in 640×512 e fuori
      dal bordo nero. Esito (16/10/2026): frame con box tagliati 51 su 1.953 (107), 13 su 721 (210), 60 su 1.460 (18),
      nessun box del tutto fuori; dei 124: 117 con il box che entra nel bordo nero (drone già al bordo della camera),
      1 più grande di 640×512, 6 altri (secondo drone). Restano con il box sulla parte visibile (regole).
    - `--step sam` con il box originale come prompt (16/10/2026): **non funziona**. Lato SAM/originale in mediana
      0,91–0,95: SAM ridà quasi il box, maschera lo sfondo (cielo, muro), non il drone. Fogli in `5_box/fogli_sam/`
      (rosso originale, verde SAM). Dai fogli:
      - 107 (cielo): drone ben visibile, ma i box originali sono spesso **spostati** rispetto al drone, a volte con il
        drone fuori dal box (1059, 1061, 3201): label disallineate (ritardo nel tempo o calibrazione eventi → RGB?);
      - 18 (muro): drone visibile, box larghi e spostati;
      - 210 (DJI Mini 2, alberi e recinzioni in ombra): drone quasi mai visibile nell'RGB.
    - `--step yolo` (proposta dell'utente, 16/10/2026): box proposti da `data/modelli/yolo26m-intern.pt` (classi
      UAV ala rotante, UAV ala fissa, aereo, elicottero, uccello; allenata su dati anti-UAV IR e RGB) sui ritagli; per
      ogni box di FRED la detection UAV col centro nel box allargato 2 volte e la confidenza più alta (soglia 0,1),
      in `5_box/yolo/`; fogli in `5_box/fogli_yolo/`. Esito:
      - 107 (cielo): trovati 1.905 su 1.953 (98%), box aderenti a vista; lato YOLO/originale in mediana **0,25**: i
        box di FRED sono circa 4 volte il drone. Fasce con i box YOLO: < 8 69, 8–16 1.249, 16–32 551, 32–64 36,
        nessuno ≥ 64 (con gli originali: 32–64 1.418, 64–96 475, > 96 49). Mediana 13 px.
      - 18 (muro) 4 su 1.460, 210 (alberi) 9 su 721: sui cortili la YOLO non trova i droni (sfondo diverso dal suo
        training).
      - Conseguenza: le fasce del passo 0 (box originali) sovrastimano molto; le sequenze di cielo non riempiono le
        celle dei droni ≥ 64 px.
    - Decisione dell'utente (16/10/2026): **cielo con la YOLO**; per i **cortili** un'altra strada. `--step
      movimento`: la camera è su un treppiede, lo sfondo è la mediana di 60 frame sparsi della sequenza (il drone che
      si muove sparisce); nel box di FRED allargato 1,5 volte i pixel che si staccano dallo sfondo (soglia mediana + 6
      MAD della differenza, almeno 20), la componente più grande, poi SAM 2 con un punto al suo centro (tenuto solo se
      non più di 4 volte l'area della componente). In `5_box/movimento/`, fogli in `5_box/fogli_movimento/`. Esito:
      - 18 (muro): 1.445 su 1.460 trovati, box aderenti a vista, anche col drone fuori dal box di FRED (2598); lato
        su originale in mediana 0,32; fasce: < 8 8, 8–16 620, 16–32 748, 32–64 66, 64–96 1, > 96 2 (mediana 17 px);
      - 210 (alberi in ombra): 681 su 721 "trovati" ma sul rumore: il drone non si vede, la 210 si scarta.
    - Uso di FRED (decisione dell'utente, 16/10/2026): **droni sotto i 64 px** (le celle ≥ 64 restano buchi: altra
      fonte o limite dichiarato), e si aggiunge anche oltre il minimo: la scrematura si fa al bilanciamento (passo 12),
      con gli stessi criteri per tutte le sequenze. Secondo giro: 10 sequenze, 2 per drone, le più ricche di droni
      stimati in 32–64 (box di FRED × 0,25 sul cielo, × 0,32 sui cortili): Betafpv 2, 0; DJI Mini 2 198, 209; DJI Mini
      3 111 (notte), 108 (cielo); Tello 133, 165; DarwinFPV 19, 28. Stessa catena e stessi parametri delle prove: frame,
      annotati, dedup a 16 bit, ritaglio, poi YOLO sul cielo e sfondo fermo sui cortili (sulla 111 entrambi, si sceglie
      dai fogli).
    - Vincolo per gli split (richiesta dell'utente): abbastanza sessioni per drone da averlo in train, val e test,
      quindi **≥ 3 sessioni per drone**. Le coppie scelte sono spesso vicine (0 e 2, 19 e 28, 198 e 209; 107, 108 e
      111 nello stesso blocco delle 20:21–20:45): dopo il secondo giro si ricostruiscono le sessioni (posto e ora dai
      fogli) e si aggiungono sequenze di altre sessioni per i droni che non arrivano a 3 (per il DJI Mini 3, per
      esempio, il blocco 91–93 delle 13:58).
    - `--step sessioni` (passo 6, 16/10/2026): proposta dalle anteprime già scaricate, senza le interne: sequenze in
      ordine di numero, blocco nuovo quando l'ora torna indietro o dopo una pausa di oltre un'ora → 202 sequenze in
      **20 blocchi**, `6_sessioni/proposta.csv` e un foglio per blocco in `6_sessioni/fogli/BNN.jpg` (numero, ora,
      drone; in fondo in grigio la prima del blocco dopo). Ore lette dai nomi dei frame; il passo 0 le aveva scritte
      con un parser che sbagliava su alcuni formati (corretto: ultimi tre campi, solo `RGB/`); per 8, 16, 68, 106
      l'ora non è leggibile e non si taglia lì. Revisione umana come per Svanström: stesso posto e stesso giorno?
      Eccezioni come `BNN + BMM`, `BNN: via <seq>`, `<seq> con BNN`.
    - Con un modello, per ridurre il lavoro a mano (richiesta dell'utente): stesso **giorno** (blocchi orari) **o**
      stesso **posto** (embedding DINOv2 ViT-S/14 dell'anteprima senza bordo nero, legame medio sopra 0,9, come
      `propose_sessions` di Svanström), uniti per transitività: la scelta più prudente contro il leakage. Soglie
      provate: 0,8 → 30 gruppi visivi, 0,85 → 36, 0,9 → 44, 0,95 → 72; con 0,9 e i blocchi orari **14 sessioni**
      (34, 28, 27, 17, 15, 15, 14, 13, 13, 11, 6, 5, 3, 1 sequenze). `proposta.csv` ha ora la colonna `sessione`;
      fogli `6_sessioni/fogli/SNN.jpg`, uno per sessione, con la più simile fuori in grigio; `embedding.npz`.
      Su ogni miniatura numero, ora, blocco (`B`, giorno) e gruppo visivo (`g`, posto): si vede perché due sequenze
      sono insieme. Anche in `proposta.csv` (colonna `gruppo_visivo`).
    - Revisione dell'utente (16/10/2026), in `correzioni.txt` (`SNN: via a b c-d`: le sequenze escono da SNN e
      formano insieme una sessione nuova): S02 via 5 e 6; S10 via 130–139 (posto diverso da 116–129); S12 via 164–166
      (diverso da 155–163); S14 via 211–217 (211–214 diverso da 203–210; 215–217 con loro, stesso gruppo visivo). In
      tutti i casi il modello le separava già (gruppi visivi diversi), le univa solo il blocco orario: lo stesso giorno
      in un posto diverso è una sessione diversa. Esito con `--step sessioni`: **18 sessioni** (34, 25, 18, 15, 15,
      14, 13, 11, 10, 10, 10, 7, 6, 5, 3, 3, 2, 1 sequenze) in `6_sessioni/sessioni.csv` (clip `FRED_<seq>/s0`,
      sessione `FRED_SNN`), fogli rigenerati uno per sessione. Sessioni per drone: Betafpv 7, DJI Mini 2 3, DJI Mini
      3 3, Tello 9, DarwinFPV 7: tutti possono stare in train, val e test.
      Seconda revisione: S17 (della proposta S14) via 218–219, posto diverso da 203–210; S14 (della proposta S12) via
      167, posto diverso dal resto; S11 (della proposta S10) via 140–142 e via 143, ognuna diversa dal resto; S06
      (della proposta S05) via 49–51; S03 (della proposta S02) divisa in 7–8, 9–10, 11–12, 13–20 con 224–230 ("13–20
      gruppo" e "224–230 + 13–17 gruppo": uniti per transitività, nel dubbio stessa sessione), 220–223 → **27 sessioni**.
    - **Regola** (utente, 16/10/2026, anche in `regole.md`): sessione = stesso posto e stesse condizioni; posto
      diverso = sessione diversa, e lo stesso posto in condizioni diverse (altro giorno, altra luce) = due sessioni.
      Proposte di unione dell'operatore (panorama 52, 66, 91; 81 e 143; 220 e 167: stesso posto, giorni diversi) applicate
      e poi tolte per questa regola. Formato `unisci a b c` (numeri di sequenza) resta in `fred.py` per le unioni.
      Esito: **27 sessioni**.
    - Revisione dell'operatore con la regola (16/10/2026), `separa a b c-d` in `correzioni.txt` (numeri di sequenza: le
      elencate formano una sessione nuova): panorama diviso per blocco orario (66–68, 69–80, 94–111, 115); pini ed
      edificio bianco per blocco (81–86, 87–90, 112–114); 224–230 (altro giorno) da 13–17; 18–20 (parete con oblò e
      albero) da 13–17; S01 in tre pareti (0–1, 2, 3–4); 48 con 49–51 (stesso prato); 154 (altro piazzale) da
      144–153; condizioni diverse nella stessa ripresa: 111 (notte), 33–36 (crepuscolo).
    - Controllo di coerenza: con la regola automatica stesso giorno (blocco) **e** stesso posto (DINOv2, soglia 0,8)
      ogni coppia di sequenze nella stessa sessione automatica è anche nella stessa sessione manuale; le manuali
      tengono insieme in più solo cambi di luce o di inquadratura nella stessa ripresa (22–32 con 23, 81–83 con
      84–86, 144–152 con 145–153, 203–210 con 206). Esito: **39 sessioni**. Nel file
      delle correzioni i nomi sono quelli della proposta (colonna `proposta` di `proposta.csv`).
    - **Uccelli, aerei, elicotteri** (decisione dell'utente, 19/10/2026): FBD-SV-2024 per gli uccelli, DVIDS per aerei
      ed elicotteri militari (gli aerei civili li copre Svanström). Obiettivo: il minimo per cella di `buchi.py`
      (870 istanze, 3 sessioni, 12 video, fasce ≥ 16 px) **più il 30%** di margine, perché i box proposti valgono
      come buoni finché non si rivedono in CVAT; si scarica finché non si arriva, così dopo la revisione non serve
      riscaricare. Velivoli a terra: qualcuno va bene, ma la precedenza è al cielo.
    - `tools/aggiunte.py` (`--fonte fbd|dvids`): un passo `frame` per fonte, poi la catena di `fred.py` (annotati,
      dedup a 16 bit, ritaglio 640×512), le label del ritaglio come proposte, sessioni, conteggio, CVAT. `fred.py` ha
      ora il prefisso dei video in `PREFIX` (non più `FRED_` fisso) e il ritaglio tiene la classe della label.
    - **FBD-SV-2024** (Sun et al., Scientific Data 2025; Kaggle `swjtuziwei/fbd-sv-2024`, licenza "Unknown" su
      Kaggle: da dichiarare). 483 clip 1280×720 a 25 fps da telecamere di sorveglianza (campi, recinzioni,
      orizzonte), 28.694 frame, un XML per frame (VOC, track id, `difficult`). Box **aderenti** a vista (8 clip, 19/10):
      si tengono gli originali, niente YOLO né SAM. Zip intero (1,8 GB) in `data/fbd_sv/0_raw/archive.zip`;
      `--step frame` → `1_frame/FBD_<n>/` (frame i = XML i), `0_raw/clip.csv`. Sessioni: gruppi DINOv2 (legame medio,
      soglia 0,9) del frame a metà clip, come FRED (camere fisse).
      Esito (19/10/2026): 483 clip, 28.694 frame e 28.366 box (come il paper), 13 GB in `1_frame`; dedup a 16 bit
      10.831 frame tenuti; 221 sessioni. Conteggio uccelli (box originali): < 8 85, 8–16 1.489, 16–32 4.609, 32–64
      4.514, 64–96 511, > 96 158. Con Svanström: uccello 16–32 e 32–64 a posto, la classe supera le 10.000
      istanze; restano 64–96 (512, mancano 358) e > 96 (199, mancano 671): limite dichiarato (regole, celle vuote
      degli uccelli > 64 px). `--step cvat`: ora più sequenze corte in un task, fino a 1.000 ritagli, in ordine di
      sessione (`cvat_export(per_task=...)`; FRED resta un task per sequenza); 12 task `fbd_<prima>-<ultima>`
      (463–474), 10.831 ritagli, box originali come rettangoli (controllato sul task 465: box sui frame giusti).
    - **DVIDS** (dvidshub.net, Dipartimento della Difesa USA). Licenza: solo pagine con "PUBLIC DOMAIN" (opere di
      dipendenti federali); si scrivono id, autore e URL in `0_raw/video.csv`; nella tesi la frase di non-endorsement
      ("The appearance of U.S. Department of War (DoW) visual information does not imply or constitute DoW
      endorsement."). La ricerca del sito è dietro un anti-bot: URL delle pagine da ricerca web (`0_raw/candidati.csv`),
      la pagina e l'mp4 (CloudFront, con Range) si scaricano. Prova (19/10, video 921866, UH-60 a Fort Novosel,
      1440×1080): 10 s scaricati con `ffmpeg -ss` = 15,7 MB su 182. Catena: `info` (licenza, autore, mp4, durata)
      → `anteprima` (un frame ogni 5 s **dai 10 s**, letto dal file remoto) → `scegli` (YOLO `yolo26m-intern` sulle
      anteprime: tratti di ±2,5 s attorno alle anteprime con aereo o elicottero, uniti; un tratto si scarica solo se
      almeno un'anteprima ha l'oggetto sul cielo, euristica: fascia sopra il box liscia) → `frame` (solo i tratti, 10
      fps, nome `f<k>` con k = round(t·10), stabile; label = YOLO sul frame intero, classi intern → nostre). Sessione
      = video (una ripresa).
      Esito (19/10/2026): 325 URL candidati raccolti con una ricerca web (per modello di velivolo: 152
      elicotteri e 173 aerei); `info`: 324 pubblico dominio, 1 no; tenuti 185 (≥ 720p, > 15 s: 100 elicotteri,
      85 aerei, 588 min; 6 senza autore leggibile). `anteprima`: con 5 s e 8 connessioni ~30 al minuto (proxy a
      ~0,5 MB/s per connessione, ogni anteprima rilegge un pezzo di GOP), quindi 10 s, 24 connessioni,
      `-noaccurate_seek` e timeout gestiti: 3.202 anteprime. `scegli`: 417 tratti (±5 s), 412 con un'anteprima sul
      cielo (elicotteri 4.625 s, aerei 3.100 s, 134 video). L'euristica del cielo lascia passare riprese dalla cabina
      e aria-aria: `--step fogli` (un'anteprima per tratto, 21 fogli in `0_raw/fogli_tratti/`) rivisti dall'operatore
      → `0_raw/esclusi.txt` (`id inizio motivo`): 92 tratti esclusi (41 cabina, 37 aria-aria, 14 senza velivolo);
      i 54 con velivoli solo a terra restano (circa 13%: "un po' ci sta", utente). `frame` a 5 fps (non 10: 50 GB
      liberi, la dedup ne toglie comunque), JPEG `-q:v 3`, a giri: `--secondi N` = secondi nuovi per classe nel
      giro, i tratti in ordine a giri sui video (prima il primo tratto di ogni video), 8 download in parallelo.
    - Giro 1 (`--secondi 900`): 102 tratti, 8.832 frame, 12.574 box YOLO; dedup 3.317 su 7.034. Due problemi: (1) la
      YOLO intern chiama UAV molti aerei ed elicotteri (618 istanze "drone" su DVIDS): su DVIDS un UAV prende la
      classe del video (`_cls`; 2.660 box già scritti rimappati); (2) quasi tutto > 96 px (aerei 1.118, elicotteri
      1.577: i video stringono sul velivolo). `scegli` scrive ora anche `grandezza` (√(w·h) del velivolo più piccolo del tratto
      nelle anteprime, in pixel del frame = pixel del ritaglio; tratti identici a prima, controllato) e `frame
      --grandezza 96` scarica solo i tratti con un velivolo sotto i 96 px. Il ritaglio usa il frame intero quando
      l'area senza bordo nero stimata è più piccola di 640×512 (video scuri); `annotati`/`dedup` saltano i tratti
      senza frame (download fallito).
    - Giro 2 (`--grandezza 96`, tutti i tratti restanti: aerei 1.340 s, elicotteri 570 s): 81 tratti, 8.900 frame. Conteggio
      DVIDS (box proposti): aerei 16–32 1.093, 32–64 2.837, 64–96 1.980, > 96 4.995; elicotteri 16–32 219, 32–64 436,
      64–96 378, > 96 2.380. Con le altre fonti: **aerei a posto** in tutte le fasce (anche col margine del 30%) e
      sopra le 10.000 istanze; elicotteri 16–32 643 e 64–96 799 sotto il minimo, 32–64 1.137 al margine. Secondo
      gruppo di candidati (ricerca web, velivoli lontani, soprattutto elicotteri): 384 URL in più in `candidati.csv`.
      `fogli` mostra ora solo i tratti non ancora visti (`fogli_tratti2/`, …).
    - Giro 3, solo elicotteri (`--classe helicopter --grandezza 96`). `info` e `anteprima` sui nuovi: 422 video tenuti in
      tutto (234 elicotteri, 188 aerei), 6.851 anteprime. `scegli` andava in memoria piena sui 4K (YOLO a piena
      risoluzione): ingresso YOLO limitato a 1.920 px (anteprime e frame); 102 dei 417 tratti vecchi cambiano un poco
      (nessun problema: i frame hanno il nome dal tempo, un tratto sovrapposto riscrive gli stessi file). `fogli
      --classe helicopter --grandezza 96`: 88 tratti nuovi in `fogli_tratti2/`, 32 esclusi (16 cabina, 7 aria-aria, 9
      vuoti). Scaricati 59 tratti, 2.320 s, 10.840 frame.
    - **Esito DVIDS** (19/10/2026, box YOLO non ancora rivisti): 146 video, dedup 10.868 ritagli. Aerei 16–32 1.104,
      32–64 2.876, 64–96 2.080, > 96 5.299; elicotteri 16–32 1.043, 32–64 1.409, 64–96 1.009, > 96 4.724 (più
      qualche uccello: alcuni veri, alcuni errori). Con Svanström: aerei 16–32 1.419, 32–64 3.001, 64–96 2.168;
      elicotteri 16–32 1.467, 32–64 2.110, 64–96 1.430: **tutte sopra 870 + 30%** (1.131); aerei ed elicotteri
      sopra le 10.000 istanze. Buchi rimasti 4 su 16: uccelli e droni > 64 px (limite dichiarato).
    - CVAT: 13 task `dvids_<primo>-<ultimo>` (475–487), 10.868 ritagli, box YOLO come rettangoli (controllato sui
      task 479 e 487). Foglio di 24 ritagli a caso: quasi tutti cielo, box per lo più aderenti; errori da
      correggere in CVAT: box doppi (elicottero e carico al gancio, aereo + elicottero sullo stesso velivolo),
      classe sbagliata (T-6 a elica chiamato elicottero), formazioni con box sovrapposti o mancanti. **Da
      decidere**: molte riprese di airshow hanno il fumo delle pattuglie (Thunderbirds, Blue Angels), simile alle
      scie che le regole escludono.
    - **Obiettivo nuovo** (utente, 19/10/2026, `regole.md`): fascia 8–16 in tutti gli split, oggetti < 4 px via,
      2.000 istanze nel train per cella (10.000 / 5 fasce) → 2.857 in tutto con il 70/20/10; `buchi.py` aggiornato
      (fasce dalla 8–16, `MIN_INST = 2.857`): **13 buchi su 20**. Aerei ed elicotteri: mancano 8–16 (~2.100 ciascuno),
      16–32 (~1.400), 64–96 (aerei 689, elicotteri 1.427), elicotteri 32–64 (747). Uccelli 8–16: 117.
    - **Margine** (utente, 19/10/2026): prima della revisione in CVAT si scarica fino a +10% per cella (3.143);
      `buchi.py` scrive anche `mancano_col_margine`. La colonna `lato` di `tratti.csv` e l'opzione `--lato`
      diventano `grandezza` / `--grandezza`: erano già √(w·h) del box (regole), il nome traeva in inganno.
    - **Licenze** (utente, 19/10/2026): solo licenze aperte (pubblico dominio, CC0, CC BY), verificate prima del
      download e scritte in [`licenze.md`](licenze.md). Svanström: CC0 1.0 (repo GitHub). FRED: Apache 2.0. DVIDS:
      pubblico dominio per video. **FBD-SV-2024: "solo ricerca, non commerciale"** (figshare 27377595): non aperta;
      **tenuto** come eccezione dichiarata (utente: tesi = ricerca, il modello non va in un prodotto), separabile.
    - **API DVIDS**: chiave dell'utente in `~/.config/dvids.env` (`DVIDS_API_KEY`, fuori dalla repo). `search`
      restituisce per video id, titolo, durata, `hd`, autore (`credit`), unità, parole chiave, al massimo 1.000
      risultati per query (50 per pagina): al posto della ricerca web per i candidati. DVIDS ha anche video di
      esercitazioni anti-drone (counter-UAS) ripresi da terra: prova su Scarlet Dragon 26.3 (1025126), sciami di
      quadricotteri in volo, YOLO intern li trova come UAV a 20–165 px (quasi tutti 40–80): fonte per i droni
      32–64, 64–96, > 96; lì gli UAV restano droni (la rimappatura sulla classe del video vale solo per i video di
      aerei ed elicotteri).
    - `--step cerca` (19/10/2026): candidati dalla API, `sort=score` (pertinenza; senza, l'ordine è per data e le
      query senza virgolette prendono quasi tutto), frasi tra virgolette (`"counter-UAS"`: 130 risultati), 4
      pagine da 50 per query, solo `hd`, 15 s–15 min, titoli senza interviste, cabina, aerial, news, cerimonie.
      Query in `QUERY` di `aggiunte.py`: 20 per i droni (counter-UAS, sUAS, quadcopter, swarm, FPV, Skydio, Raven,
      Puma, Black Hornet…), 18 per gli elicotteri (air assault, sling load, modelli…), 18 per gli aerei. Classe
      `drone` nuova in `candidati.csv`: `scegli` cerca il bersaglio del video (UAV per i droni, aerei ed
      elicotteri per gli altri). FBD tenuto (vedi Licenze).
      Esito: 4.616 candidati nuovi (droni 743, elicotteri 2.089, aerei 2.493). Con tutte le anteprime ogni 10 s
      sarebbero servite ~2,5 ore di `info` e ~12 di anteprime: `info` e `anteprima` hanno ora `--classe` (prima
      droni, poi elicotteri, gli aerei dopo) e al più 12 anteprime per video, distribuite sulla durata (i video già
      fatti restano com'erano). **Bug corretto**: `page_info` prendeva il primo mp4 in ordine alfabetico, spesso
      una versione ridotta (`-1024x576-1769k.mp4`); ora l'originale, se no la più grande. 407 righe così in
      `video.csv`, tutte del giro nuovo e nessuna scaricata (erano sotto 720p, quindi già escluse): tolte e rifatte.
    - **Doppioni** (richiesta dell'utente, 19/10/2026): DVIDS ha lo stesso b-roll ripubblicato con id diversi (unità
      diverse, versioni sottotitolate). `--step doppioni`: stessa durata (entro 0,5 s) e anteprime allo stesso
      istante uguali (dHash 64 bit entro 6) in almeno metà degli istanti → `0_raw/doppioni.txt`, esclusi da tutti i
      passi (`_ok`); si tiene il video già scaricato, se no l'id più basso. Prima prova con somiglianza libera fra
      tutte le anteprime (pigeonhole su 4 pezzi da 16 bit): 43–58 "doppioni", quasi tutti falsi (video diversi di
      cielo con un velivolo piccolo, schermate comuni); con la durata: 12 su 765 video, tutti veri a vista dei
      titoli (stesso servizio ripubblicato, versioni IT / master / mastersub). Da rifare dopo ogni giro di anteprime.
    - Controllo dei box YOLO sui droni (anteprime dei video counter-UAS, 32 ritagli ingranditi, box esatti):
      per lo più aderenti; falsi positivi su toppe, bocche, armi, uccelli e macchie (si tolgono con la revisione
      dei fogli dei tratti e in CVAT). Il primo foglio mandato all'utente aveva i rettangoli allargati di ~12 px
      per lato solo per il disegno. Niente SAM per ora: i box YOLO sul cielo sono già stretti; si aggiunge se in
      CVAT molti risultano larghi.
    - Giro 4, droni: `info --classe drone` (743 candidati, 601 tenuti), `anteprima --classe drone` (molte fallite
      per 5XX del proxy: da ripassare, il passo salta quelle fatte), `doppioni` (13 su 963 video), `scegli`: 463
      tratti di droni in 279 video (5.350 s). `fogli --classe drone`: 432 tratti in `fogli_tratti3/` rivisti da un
      operatore: 189 esclusi (48 interni/schermi, 49 persone in primo piano, 39 droni a terra, 19 dall'alto/FPV,
      34 niente in volo). `frame --classe drone`: 243 tratti, 2.850 s, 163 video → 9.599 frame, 15.617 box YOLO.
      Conteggio DVIDS droni: 16–32 460, 32–64 1.029, 64–96 628, > 96 1.803. Con FRED e Svanström: droni 32–64
      2.159, 64–96 723, > 96 1.870 (mancano col margine 984, 2.420, 1.273). Il tetto di 12 anteprime ora vale
      sempre (prima, un video con anteprime a metà per i 5XX le avrebbe riprese tutte ogni 10 s): è un
      sottoinsieme della griglia da 10 s, quindi i video già fatti non scaricano altro.
    - Giro 5 (19/10/2026): `info --classe airplane` si fermava per un timeout di ffprobe non gestito (ora gestito;
      i file con sola versione ridotta sotto 720p non si provano). Video tenuti: elicotteri 824, aerei 677, droni
      601. Anteprime ripassate (5XX), `doppioni`: 30 su 1.659 video. Fogli: `fogli_tratti4/` elicotteri con
      grandezza < 96 (465 tratti, 257 esclusi: 121 cabina, 23 aria-aria, 56 persone, 57 vuoti), `fogli_tratti5/`
      droni nuovi (103, 37 esclusi). Esclusioni con `--step escludi --fogli fogli_tratti<N> --scarti <file>` (elenco
      `n codice` della revisione → `esclusi.txt`). Download: droni restanti e 217 tratti di elicotteri (4.100 s, 170 video).
      Esito: droni 145 tratti (5.542 frame), elicotteri 217 tratti (11.875 frame). Con tutte le fonti (box
      proposti): aerei 8–16 788, 16–32 1.506, 32–64 3.139, 64–96 2.477; elicotteri 8–16 960, 16–32 1.880, 32–64
      2.653, 64–96 1.793; droni 32–64 2.578, 64–96 1.028, > 96 2.581. **12 buchi su 20**. Le fasce piccole
      crescono poco: 4.100 s di elicotteri hanno dato +225 istanze in 8–16 e +410 in 16–32; in parte perché la YOLO
      intern trova male gli oggetti sotto i 16 px (i conteggi sono dei box proposti: gli oggetti mancati si
      aggiungono solo in CVAT). `fogli_tratti6/`: aerei con grandezza < 96, 561 tratti, 310 esclusi (cabina,
      aria-aria, persone, solo a terra, vuoti); download di 263 tratti, 7.050 s, 162 video → 16.779 frame, 36.839
      box. Con tutte le fonti: aerei 8–16 1.161, 16–32 2.866, 32–64 e 64–96 a posto; elicotteri 8–16 1.006, 16–32
      2.029, 32–64 2.856, 64–96 1.904; **10 buchi su 20**. Disco: `dvids/1_frame` 8,7 GB, anteprime 4,7 GB; 34 GB liberi.
    - Giro 7: i tratti "visti e tenuti" ma non scaricati (aerei 112, 3.400 s; elicotteri 82, 1.720 s; droni 50)
      sono in gran parte tratti già presi con i confini spostati dal cambio di `imgsz` (1.920) in `scegli`: si
      riscaricano (stessi nomi dei frame, nessun doppione, solo banda) per prendere le parti nuove.
      Esito: aerei 3.400 s, elicotteri 1.720 s, droni 810 s; DVIDS 33.369 ritagli dopo la dedup. Con tutte le fonti
      **8 buchi su 20**: aerei 8–16 (1.299), elicotteri 8–16 (1.156), 16–32 (2.317), 64–96 (2.221), droni 64–96
      (1.179), droni 32–64 e > 96 sopra il minimo ma sotto il margine, uccelli > 64 (limite). Fonti attuali
      esaurite per i tratti sotto i 96 px. I task CVAT DVIDS 475–487 non corrispondono più del tutto ai dati
      (dedup e ritagli rifatti nei video con frame aggiunti): da ricreare, se l'utente non li ha ancora rivisti.
      Proposte per le celle rimaste: YOLO a tasselli sui frame già scaricati (i 4K oggi la YOLO li vede a 1.920),
      poi altre query DVIDS mirate; oppure 8–16 come limite dichiarato. In attesa della decisione dell'utente.
    - Task CVAT DVIDS 475–487 **cancellati** (utente, 19/10/2026: nessuna revisione fatta, a parte Svanström); si
      ricreano dopo la decisione sui tasselli (che cambierebbe box e ritagli).
    - Memoria GPU (19/10/2026): un frame 4K alla risoluzione piena (`imgsz` 3.840) usa 2,8 GB (1,5 in half); il
      limite a 1.920 messo nel giro 3 veniva da 3 video 8K (7680×4320), non dai 4K. Prova dello zoom ×2 su 200
      frame 1080p (`imgsz` 3.840 invece di 1.920): 8–16 da 13 a 34, 16–32 da 33 a 59, > 96 da 149 a 138; 0,08 s
      per frame, 4,6 GB. Ma dei 40 oggetti < 32 px trovati **solo** con lo zoom quasi tutti sono falsi positivi
      (stelle delle livree, pezzi di aereo, rumore della visione notturna, sfocature): il guadagno vero è piccolo
      e caricherebbe CVAT di box da togliere. Lo zoom non si usa; resta utile la risoluzione piena sui video
      sopra il 1080p (oggi ridotti a 1.920).
    - Giro 8 (19/10/2026): query nuove in `QUERY` (droni: "UAS flight", "drone pilot", "drone competition",
      "Coyote", "Switchblade", "Anduril"…; elicotteri: "landing zone", "helocast", "fast rope", "deck landing",
      "hoist training", "Black Hawk", "Apache", "Chinook"…) e `--classe` anche in `cerca`. 1.593 candidati nuovi
      (droni 257, elicotteri 1.336), tutti pubblico dominio; 47 doppioni. Fogli `fogli_tratti7` (elicotteri,
      grandezza < 96: 136 scartati su 236) e `fogli_tratti8` (droni: 64 su 114), revisione dell'operatore, scarti in
      `scarti7.txt`/`scarti8.txt`. Scaricati elicotteri 2.160 s, droni 720 s; DVIDS 738 video. Con tutte le fonti
      **7 buchi su 20**: aerei 8–16 (1.324), elicotteri 8–16 (1.359), 16–32 (2.543), 64–96 (2.563), droni 64–96
      (1.215: +36, le query DVIDS sui droni sono esaurite), uccelli > 64 (limite); droni 32–64 e > 96 sotto il
      margine (mancano 192 e 171).
    - Giro 9 (19/10/2026): query elicotteri nuove (Huey, Kiowa, Super Stallion, Viper, Pave Hawk, Jayhawk, MH-65,
      CASEVAC, MEDEVAC, rappel, aerial firefighting, aviation brigade, rotary wing, search and rescue…; tolti
      Osprey/MV-22, convertiplano di classe ambigua, e "night vision"): 1.419 candidati, 54 doppioni, 204 tratti in
      `fogli_tratti9` (149 scartati dall'operatore, `scarti9.txt`), 1.480 s scaricati; DVIDS 783 video.
    - **Anti-UAV300 RGB** (19/10/2026, licenza MIT, `licenze.md`): Google Drive bloccato dal proxy, si usa il mirror
      Hugging Face `DestinyLove/AntiUAV300` (MIT, `yolo_rgb`: 296.901 frame 1920×1080, un drone etichettato a mano
      per frame, 16.834 frame senza drone). Prima solo `all_labels.txt` (22 MB): grandezza mediana 86 px (25–75%:
      66–103), 64–96 121.082 istanze in 254 video. `--fonte antiuav --step frame`: un frame ogni 25 di ogni video,
      label vere (classe 2) → 11.920 frame, 308 assenti sul mirror (404: 6 video interi e parte di altri 3), 11.612
      tenuti in 312 video, 1,4 GB. Scritte e mirino sovrapposti alle immagini: accettati (utente). Catena di
      `fred.py`; sessioni = registrazione (`<data>_<ora>`): 312 clip in 37 sessioni. FRED resta per i droni piccoli
      (Anti-UAV ne ha quasi nessuno sotto i 32 px).
    - Buchi dopo giro 9 e Anti-UAV: **6 su 20**; droni tutti chiusi col margine (32–64 4.770, 64–96 4.308, > 96
      5.217); restano aerei 8–16 (1.336), elicotteri 8–16 (1.444), 16–32 (2.807, mancano 50, col margine 336),
      64–96 (2.797, mancano 60, col margine 346), uccelli > 64 (limite).
    - Task CVAT (19/10/2026): DVIDS 488–528 (41 task, 38.211 ritagli), Anti-UAV 529–536 (8 task, 7.236 ritagli).
      `cvat_export` ora salta le sequenze gia' presenti in un task della stessa fonte (nomi dei frame dei task): le
      sequenze nuove dei giri successivi vanno in task nuovi senza spostare i gruppi fatti. Corretto il nome dei task
      di Anti-UAV (prima solo la data: due task con lo stesso nome, il secondo saltato).
    - Giro 10 elicotteri (19/10/2026): query MH-47, Little Bird, OH-58, UH-1N, Grey Wolf, HH-60W, MH-60R/S, CH-53K,
      VERTREP, pinnacle landing, brownout, insertion/extraction, Coast Guard e National Guard: 786 candidati, 438
      tratti in `fogli_tratti10` (292 scartati dall'operatore, `scarti10.txt`), 2.470 s scaricati. Elicotteri
      16–32 3.173 e 64–96 3.274: **chiusi col margine**.
    - Uccelli DVIDS (19/10/2026, utente: "proviamo con dvids"): classe `bird` in `QUERY` (bird strike, BASH,
      falconry, bald eagle, seagulls, pelicans, geese, vultures…; "hawk" sostituito da "red-tailed hawk" per non
      prendere i Black Hawk); nei video di uccelli il bersaglio e' l'uccello e gli UAV della YOLO diventano uccelli.
      Nuova opzione `--minimo` (fogli e frame: il velivolo piu' piccolo del tratto almeno di questa grandezza).
      442 candidati, 1.330 anteprime, 103 tratti con un uccello, 22 con `--minimo 48` (`fogli_tratti11`, visti dal
      operatore): quasi tutti falsi positivi (persone, mezzi, grafica), tenuti 3 (aquila in volo, stormo),
      100 frame. **DVIDS non serve per gli uccelli grandi.**
    - Task CVAT nuovi DVIDS 537–540 (3.476 ritagli: giro 10 e uccelli). Buchi: **4 su 20** — aerei 8–16 (1.392),
      elicotteri 8–16 (1.665), uccelli 64–96 (578) e > 96 (215).
    - **Fine della raccolta (utente, 19/10/2026: "direi che è accettabile")**: per classe tutte oltre 14.000
      istanze; restano 4 celle sotto il minimo (aerei 8–16, elicotteri 8–16, uccelli 64–96 e > 96), limite
      dichiarato: in val e test minimo più basso con l'incertezza dal bootstrap per sessione (`regole.md`). La fascia
      8–16 si rivede dopo CVAT (oggetti aggiunti a mano). Prossimo passo: revisione in CVAT.
    - Campione per stimare la revisione (19/10/2026): 120 ritagli a caso (seed 0: DVIDS 70, FRED 20, FBD 18,
      Anti-UAV 12) visti dall'operatore con i box proposti. Da correggere: DVIDS 33 su 70 (classe 9, mancati
      9, box unite nelle formazioni 6, primi piani 5, falsi positivi 5), FRED 1 su 20, FBD 0–2 dubbi, Anti-UAV 0.
      FBD: uccelli quasi tutti su campi e alberi, non sul cielo.
    - **Preselezione** (utente, 19/10/2026), `tools/preselezione.py`: (1) DVIDS, ogni box non-uccello prende la classe
      del video (7.327 box cambiate su 70.189); (2) DVIDS, via i ritagli con una box oltre metà del ritaglio (5.787);
      (3) tetto con margine (4.000 + 857) × 1,1 = 5.343 istanze per cella su tutte le fonti (Svanström conta ma non si
      tocca): via i ritagli con tutte le box in celle sopra il tetto, a turno fra i video (DVIDS 10.002, FRED 3.215,
      FBD 218, Anti-UAV 99). Esclusi in `data/<fonte>/5_box/esclusi.txt`, letti da conteggio e `cvat_export`
      (`fred.excluded`); la `catena` riscrive le proposte, quindi la preselezione va rilanciata dopo. Buchi invariati
      (4); elicotteri 64–96 da 3.274 a 3.564 grazie alla classe del video. Task vecchi 451–540 cancellati (utente) e
      ricreati: FRED 541–551 (8.996 ritagli), DVIDS 552–579 (26.047), Anti-UAV 580–587 (7.137), FBD 588–598 (10.613).
    - Controllo di qualità FBD e Anti-UAV (19/10/2026): 48 + 48 ritagli a caso con zoom 4× sulla box: 0 errori
      certi (FBD 4 dubbi), box FBD un po' larghe (~1,3×). Si rivedono comunque in CVAT (utente). Anti-UAV: 23 su 48
      di notte (solo luci).
    - Decisioni (utente, 19/10/2026), aggiunte a `tools/preselezione.py`: **niente notte** (ritagli Anti-UAV e
      DVIDS con luminosità media < 35: 2.586 + 707; FRED escluso dal filtro: i suoi ritagli bui sono cortili di
      giorno sottoesposti); **fumo degli airshow = scia**: un'anteprima a due ritagli per ognuno dei 288 tratti di
      aerei (`0_raw/fogli_fumo/`), l'operatore segna le scie (`scarti_fumo.txt`) → 86 tratti in `0_raw/fumo.txt`, 7.233
      ritagli tolti; **box sotto i 4 px²** (√(w·h) < 2 px) tolte dalle label (29, tutte FRED); **primi piani**
      più stretti (utente: "ancora aerei molto grandi"): box oltre il 25% del ritaglio o tagliata su 2+ lati
      (8.147 ritagli DVIDS). FBD tenuto come estensione del dominio all'orizzonte; elicotteri 16–32 accettato.
      Buchi: **7 su 20** — aerei 8–16 (948), 16–32 (2.176), 64–96 (2.225) (le pattuglie acrobatiche erano la fonte
      principale degli aerei medi), elicotteri 8–16 (1.535), 16–32 (2.797), uccelli > 64.
    - Giro 11 aerei (19/10/2026, utente "approvo"): query senza airshow (takeoff, landing, final approach,
      departs, arrives, flyby, low pass, elephant walk, Red Flag, bomber task force, C-5, KC-46, B-1, B-2, P-8,
      F/A-18…); airshow, Thunderbirds, Blue Angels, demonstration team e open house aggiunti alle parole escluse
      dai titoli (`NO`).
    - Giro 11 aerei (19–20/10/2026): 2.617 candidati, 781 tratti in `fogli_tratti12`, operatore 522 scartati
      (`scarti12.txt`: T 145, C 112, P 88, A 64, N 53, I 38, F 22), 5.380 s scaricati. Dopo la preselezione:
      aerei 16–32 3.353 e 64–96 4.092 (**chiusi**), 8–16 1.296. Buchi **5 su 20**: aerei 8–16, elicotteri 8–16,
      elicotteri 16–32 (2.797, accettato), uccelli 64–96 e > 96. Task CVAT ricreati (nessuno toccato): DVIDS 599–628
      (28.520 ritagli), Anti-UAV 629–633 (4.616), FBD 634–644 (10.544); FRED 541–551 invariati.
    - Notte di Anti-UAV sfuggita alla soglia di luminosità (utente in CVAT, 20/10/2026, job 978: luci della città
      forti alzano la media sopra 35). Decisione (utente, "concordo"): via le riprese dalle **19 in poi** (ora nel
      nome), tenuto il crepuscolo delle 18:34–18:39 (drone visibile). `tools/preselezione.py` le toglie **dopo il
      tetto**, così le altre fonti non cambiano: +55 ritagli `notte` (20190925_200805, 20190926_193515, _195921,
      _200510), nient'altro diverso. Segnati cancellati in CVAT: task 631 (8) e 633 (47).
    - Disco: `fbd_sv` 1_frame 13 GB + zip 1,8 GB (tenuto: niente riscaricamenti), `dvids` 1_frame 3,3 GB; 43 GB
      liberi.
11. **Split per sessione** (`data/<fonte>/6_dedup + 3_sessioni → data/aggregati/9_split`, su tutte le fonti, fuori da `svanstrom.py`): ogni sessione di clip tutta in uno split, quindi un
    video con clip di sessioni diverse può finire in split diversi. Su tutte le fonti, bilanciato su classe × fascia. Da val
    e test via le immagini con oggetti < 16 px e i quasi-duplicati del train.
    - Secondo giro finito (16/10/2026). Box dello sfondo fermo, lato su originale in mediana: 2 0,34, 0 0,33, 198
      0,49, 209 0,32, 133 0,62, 165 0,57, 19 0,37, 28 0,31 (Tello e DJI Mini 2 sono più grandi: rapporti più alti, box
      aderenti a vista sui fogli); sul Betafpv nelle zone d'ombra il drone non si vede (2: 944 su 1.064). Tolti 210,
      111 e lo zip di prova della 36 (1 GB). `--step proposte`: YOLO sul cielo (107, 108), sfondo fermo sul resto, in
      `5_box/proposte/`; `--step conta` → `7_conteggio/conteggio.csv` (FRED, 11 sequenze, box proposti): droni < 8 687,
      8–16 5.610, 16–32 4.630, 32–64 826, 64–96 75, > 96 35. Buchi con Svanström + FRED: **14 su 16**; i droni 16–32 e
      **32–64** ora sono a posto (32–64: 1.130 istanze), la classe drone supera le 10.000; restano 64–96 (95) e > 96
      (67), con sessioni e video sufficienti.
    - **Flusso unico** (decisione dell'utente, 16/10/2026, anche in `regole.md`): download → label automatiche →
      deduplica dentro la clip → **revisione in CVAT dei frame tenuti** → impronta semantica e gruppi di clip → split
      per gruppi. Le sessioni fatte a mano (Svanström e FRED) restano come controllo, la regola è il raggruppamento
      automatico. Per FRED il prossimo passo è il task CVAT dei ritagli tenuti con i box proposti.
    - `--step cvat` (16/10/2026): un task per sequenza, `fred_<seq>` (task 451–462), con i ritagli tenuti e i box
      proposti; CVAT monta solo `data/svanstrom`, quindi i ritagli si caricano dal disco, **uno zip per sequenza** (un
      task unico da 12.000 file e i task con più di ~1.000 file davano errore 500 sul server). 11 task, 12.211 frame,
      11.863 box proposti. Rientro: export "Ultralytics YOLO Detection 1.0" senza immagini in `5_box/cvat_zip/`, poi
      `--step cvat_import` → `5_box/manuale/`; i frame senza box nell'export valgono come nessun box.
    - **Procedura unica per tutte le fonti** (proposta dell'utente, 16/10/2026: un'impronta per clip e nessuna coppia
      simile in split diversi). `tools/split.py --step gruppi --soglia T` → `data/aggregati/9_split/gruppi.csv`:
      impronta = embedding DINOv2 della striscia (un frame al secondo, 640×512; per FRED i ritagli dei frame tenuti,
      `embedding_fred.npz`), gruppi per legame medio su tutte le clip di tutte le fonti, senza scie e scartate (FRED
      scartate in `fred/5_box/scartate.txt`: 210, 111). Prima prova, senza correzioni a mano (334 clip, 26.716 frame:
      Svanström 14.505, FRED 12.211):
      - soglia 0,95: 181 gruppi; il 99% delle coppie nello stesso gruppo è anche nella stessa sessione manuale, ma 38
        sessioni manuali su 119 sono divise in più gruppi (S051 in 8): il gruppo automatico è meno prudente delle
        sessioni riviste a mano;
      - 0,9: 148 gruppi; 0,97: 207; nessun gruppo mescola le fonti;
      - i gruppi più grandi in frame sono sequenze di FRED (107 + 108: 14,8%; 18 + 19: 10,9%), poi S076 (6,3%):
        dopo l'hash una sequenza di FRED tiene 700–2.000 frame, una clip di Svanström qualche decina. Si sistema con il
        tetto per sessione al bilanciamento (passo 12);
      - unione di gruppi automatici e sessioni manuali: 117 unità, le più grandi 14,8%, 10,9%, 9,4%.
    - Prova delle soglie (19/10/2026, stesse 334 clip; quota delle coppie nella stessa sessione manuale finite nello
      stesso gruppo / sessioni manuali spezzate / gruppo più grande): 0,97 36% / 43 / 10,9%; 0,95 57% / 38 / 14,8%;
      0,93 68% / 34 / 14,8%; 0,90 90% / 24 / 14,8%; 0,88 93% / 22; **0,85 94% / 17 / 14,8%**; 0,80 94% / 15 /
      14,8%. Abbassare la soglia non fa crescere il gruppo più grande; sotto 0,85 non si guadagna quasi niente.
    - **Decisione (utente, 19/10/2026)**: soglia **0,85** e **unione** con le sessioni note (manuali di Svanström e
      FRED, registrazioni di Anti-UAV, video di DVIDS, gruppi di FBD): una clip sta nello stesso split del suo gruppo
      e della sua sessione. Dopo lo split si misura il leakage rimasto: per ogni immagine di val e test la somiglianza
      DINOv2 con l'immagine più vicina del train (distribuzione nel report). `split.py` va esteso a FBD, DVIDS e
      Anti-UAV.
12. **Bilanciamento** in ogni split: val e test piatti (~571 e ~286 per cella), train con **tetto di 4.000 per cella** (utente, 19/10/2026), tetto per sessione. Niente copie ridotte a 0,75 per la fascia 8–16: lo scale di Ultralytics in training basta (decisione dell'utente). Stima sui box proposti: train ~60.900, val + test ~14.900, fuori ~32.000 (soprattutto aerei ed elicotteri > 96).
13. **Background** circa 10%.
14. **Mosaici di deploy, report, manifest**, bootstrap per sessione su val e test, **congelamento**.

## Da cambiare in `tools/svanstrom.py`

- una cartella per passo, come sopra;
- hash dentro la sequenza invece che dentro il video;
- split per sessione invece che per video;
- fasce 16/32/64/96 invece di COCO (32/96);
- regola dei 16 px;
- box corretti al posto degli originali;
- più fonti;
- duplicati fra split;
- tetti e minimi per cella;
- mosaici di deploy e sottoinsiemi per fascia di val e test;
- report con matrice a 4 fasce e video per cella.
