# Procedura di costruzione

Ogni passo dipende da quello prima. Le regole sono in [`regole.md`](regole.md).

1. **Fissare le regole, prima di toccare i dati**: classi, fasce, regola del box aderente, requisiti delle fonti, e i
   **valori minimi di istanze e di video per cella** della matrice classe × fascia (ancora da decidere, vedi
   [`da_decidere.md`](da_decidere.md)).

2. **Correggere i box di Svanström (indispensabile).** Le label originali sono larghe e incoerenti: un box di 16 px
   contiene un oggetto di 4–6 px. Con box così le fasce non dicono niente, la mAP50-95 penalizza alle IoU alte anche
   un box preciso, e mescolare fonti con box aderenti insegnerebbe al modello due modi di disegnare il box.
   - Ogni box si dà come prompt a **SAM 2**; dalla sagoma si ricava il box aderente. Sugli oggetti su cielo uniforme
     si confronta con una soglia di contrasto locale dentro il box.
   - Revisione a mano di un campione e di tutti i casi critici: sfondi complessi (edifici, alberi, pali), box che
     cambiano molto, sagome vuote o spezzate.
   - Si misura e si riporta **quanto erano larghi i box originali** (rapporto fra √area originale e aderente, per
     classe e fascia).
   - Nello stesso passaggio si cercano e si etichettano gli **oggetti visibili non etichettati** (per esempio gli
     stormi).

3. **Contare.** Matrice classe × fascia con Svanström corretto: istanze e video per cella. Va fatto **dopo** il passo
   2, perché le fasce dipendono dalla grandezza dei box.

4. **Elenco dei buchi**: le celle sotto il minimo diventano la lista di cosa cercare. Con i box originali gli uccelli
   grandi sono zero e i droni grandi 37 frame di un solo video ([`svanstrom.md`](svanstrom.md)).

5. **Riempire i buchi**, con le regole del passo 1. Fonti scelte in questo momento (candidate in
   [`ricerca_fonti.md`](ricerca_fonti.md)), a colori, da terra verso il cielo, portate a **640×512**. Box proposti da
   un detector a vocabolario aperto, resi aderenti con SAM 2 e rivisti a mano; anche qui si etichettano tutti gli
   oggetti visibili.

6. **Selezione dei frame per video**: via i quasi-duplicati con l'hash semantico (`tools/svanstrom.py`).

7. **Split per video** (70/20/10), su tutte le fonti insieme, bilanciato sulla matrice classe × fascia (schema di
   `tools/svanstrom.py`). Prima del bilanciamento finale, così nessun video finisce in due split.
   - Dai video di val e test si tolgono le immagini con oggetti sotto i 16 px (restano solo nel train).
   - Controllo fra split con lo stesso hash: i quasi-duplicati di immagini del train (video diversi della stessa
     scena) si tolgono da val e test.

8. **Bilanciamento dentro ogni split**: tetti per cella e per video (i droni sono molto più numerosi delle altre
   classi). Anche val e test con un minimo per cella: la mAP di una cella con 5 oggetti non vuol dire niente.

9. **Background**: circa il 10% di frame senza oggetti, dagli stessi video (inpainting del bersaglio su cielo
   uniforme, come già fa `tools/svanstrom.py`).

10. **Set di deploy, statistiche, congelamento.**
    - `val_480p`, `val_720p`, `val_1080p` (e gli stessi per il test) a mosaico ([`regole.md`](regole.md)).
    - Statistiche: matrice classe × fascia per split, quota di ogni fonte per classe, immagini senza oggetti, oggetti
      sotto i 16 px (solo per sapere quanti sono).
    - **Manifest** e **congelamento**: da qui il dataset non cambia più, e il test si chiude.

## Cosa cambiare in `tools/svanstrom.py`

Lo script fa già download, lettura delle label, hash semantico, background, split per video e sottoinsiemi. Va
adattato:

- fasce: oggi soglie COCO (32, 96) sul box più grande dell'immagine; servono 16–32, 32–64, 64–96, sopra 96;
- regola dei 16 px: oggi non c'è; immagini con oggetti sotto i 16 px solo nel train;
- box corretti: oggi usa i box originali; deve leggere quelli corretti del passo 2;
- più fonti: oggi solo Svanström;
- controllo dei duplicati fra split dopo lo split;
- bilanciamento con tetti e minimi per cella;
- set di deploy a mosaico e sottoinsiemi per fascia anche per val e test;
- nel report, la matrice a 4 fasce con istanze e video per cella.
