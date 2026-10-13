# Procedura

Ogni passo dipende dal precedente. Regole in [`regole.md`](regole.md).

Ogni passo ha la sua cartella in `data/<fonte>/`: legge solo da quelle dei passi prima, così si rifà un passo senza
ripartire da zero. Nelle cartelle di soli indici (2, 3, 5) vanno CSV, non copie delle immagini.

| cartella | contenuto | passo |
|---|---|---|
| `0_raw/` | video `.mp4` + label `.mat` | download |
| `1_frame/` | frame `.png` + label originali YOLO, per frame | 2 |
| `2_sequenze/` | `candidati/` (fogli + `candidati.csv`) e `sequenze.csv` (video, frame, sequenza) | 3 |
| `3_sessioni/` | `embedding.npz`, `fogli/`, `proposta.csv`, `correzioni.txt`, `sessioni.csv` (video, sessione) | 4 |
| `4_box/` | `labels/` corrette + `revisione/` (campione rivisto a mano) | 5 |
| `5_dedup/` | `tenuti.csv` (frame che restano) | 9 |
| `6_split/` | `images/` e `labels/{train,val,test}` in hardlink + `manifest.json` + report | 10–13 |

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
     lettura): `--step cvat --clips <file>`, un task per clip con i box attuali come traccia; frame chiave uno ogni 50
     più primo e ultimo di ogni tratto visibile, CVAT interpola in mezzo. Si correggono le chiavi e se ne aggiungono
     dove serve. Credenziali in `~/.config/cvat.env`. In alternativa si scarta la clip.
   - Decisione del 13/10: **tutte** le clip in CVAT (tranne le scie) per la revisione finale del dataset, non solo le
     bocciate; `--step cvat` senza `--clips` carica tutte le clip pronte, con il box migliore (SAM, altrimenti
     contrasto). Le label finali vengono dagli export CVAT.
   - Rientro: da CVAT si esporta il task in **Ultralytics YOLO Detection 1.0** (Actions → Export task dataset, poi
     Requests → Download) e lo zip si lascia in `4_box/cvat_zip/`; `--step cvat_import` scrive le label per frame,
     interpolate da CVAT, in `4_box/manuale/<video>_sN/`.
   - Oggetti non etichettati (stormi): da fare. Si riporta il rapporto fra √area originale e corretta, per classe e
     fascia.
6. **Contare** istanze, video e sessioni per cella, sui box corretti.
7. **Buchi**: celle sotto il minimo.
8. **Aggiunte** per i buchi, a 640×512: box proposti da un detector a vocabolario aperto, resi aderenti con SAM 2,
   rivisti a mano. Ogni fonte ha le sue cartelle `0_raw`…`4_box`.
9. **Quasi-duplicati** (`4_box + 2_sequenze → 5_dedup`): hash semantico dentro ogni clip.
10. **Split per sessione** (`5_dedup + 3_sessioni → 6_split`): ogni sessione di clip tutta in uno split, quindi un
    video con clip di sessioni diverse può finire in split diversi. Su tutte le fonti, bilanciato su classe × fascia. Da val
    e test via le immagini con oggetti < 16 px e i quasi-duplicati del train.
11. **Bilanciamento** in ogni split: tetti per cella e per sessione (i droni abbondano), minimi in val e test.
12. **Background** circa 10%.
13. **Mosaici di deploy, report, manifest**, bootstrap per sessione su val e test, **congelamento**.

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
