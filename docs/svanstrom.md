# Svanström

[Drone-detection-dataset](https://github.com/DroneDetectionThesis/Drone-detection-dataset) (Svanström et al., CC0):
650 video con label MATLAB, circa 10 s a 30 fps. Si usano solo i **285 video visibili**, 640×512; gli IR (320×256,
in grigio) sono esclusi. Ripreso da terra verso il cielo; tutte e quattro le classi etichettate negli stessi video e
con la stessa camera.

L'xlsx del dataset dà per ogni video classe e *distance bin* (close, medium, distant): serve solo per l'elenco dei
file. Il distance bin non separa bene le grandezze a schermo (un elicottero lontano può essere più grande di un drone
vicino), quindi le fasce si calcolano dai box.

## Label

- I `.mat` sono oggetti MCOS: né scipy né `mcos-decoder` li leggono bene (`mcos-decoder` legge come vuoti i frame con
  2 oggetti e perde quelli con 3–5). `tools/svanstrom.py` ha un parser MAT5 minimo: 4 celle per frame, una per
  classe.
- Verifica per ogni clip: numero di label = numero di frame del video, altrimenti la clip è scartata
  (V_AIRPLANE_002).
- **Box larghi e incoerenti**: un box di 16 px contiene un oggetto di 4–6 px (verificato a occhio su uccello, drone
  ed elicottero; esempi V_BIRD_013_f00050, V_DRONE_105_f00158, V_HELICOPTER_009_f00001 in
  [`../immagini/svanstrom_16px.png`](../immagini/svanstrom_16px.png)). Si correggono (procedura, passo 2).
- **Oggetti non etichettati**: per esempio stormi in alcuni video di uccelli, anche nei frame di background.

## Numeri (con i box originali, `tools/svanstrom.py`, seed 42)

284 video su 285, 88.932 frame: 25.991 selezionati dall'hash semantico (29%) più 2.462 di background.

| subset (soglie COCO) | train | val | test | calib |
|---|---|---|---|---|
| `all` | 20.234 | 5.523 | 2.696 | 512 |
| `small` | 9.695 | 2.569 | 1.260 | 512 |
| `medium` | 7.137 | 1.953 | 992 | 512 |
| `large` | 1.670 | 497 | 206 | 512 |
| `background` | 1.721 | 503 | 238 | 512 |

Video per split: 199 / 56 / 29.

Box di train + val per fascia (√area, box originali):

| sotto 8 | 8–16 | 16–32 | 32–64 | 64–96 | sopra 96 |
|---|---|---|---|---|---|
| 0 | 107 | 12.967 | 7.871 | 1.346 | 2.167 |

Con i box corretti tutti i box si rimpiccioliscono, quindi molti scenderanno di fascia, e una parte finirà sotto i
16 px: i numeri vanno rifatti dopo la correzione.

**Buchi già visibili**: le immagini large sono quasi tutte di elicotteri e aerei; i droni large sono 37 frame di un
solo video; gli uccelli large nessuno.

## Limiti

- Le immagini large sono poche (circa 10%) e concentrate in pochi video (passaggi ravvicinati).
- Video della stessa sessione di volo potrebbero finire in split diversi: l'xlsx non dice la sessione.
- Nel background ci sono solo 82 frame davvero senza oggetti (4 video): il resto si crea con l'inpainting del
  bersaglio su cielo uniforme.
