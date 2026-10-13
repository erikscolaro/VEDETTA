# Svanström

[Drone-detection-dataset](https://github.com/DroneDetectionThesis/Drone-detection-dataset), CC0. Si usano i **285
video visibili** (640×512, circa 10 s a 30 fps), da terra verso il cielo, con le 4 classi negli stessi video. IR
esclusi. Il *distance bin* dell'xlsx non separa le grandezze a schermo: le fasce si calcolano dai box.

## Label

- `.mat` MCOS: scipy e `mcos-decoder` sbagliano i frame con più oggetti; `tools/svanstrom.py` ha un parser MAT5 suo.
- Clip scartata se label ≠ frame (V_AIRPLANE_002).
- **Box larghi**: un box di 16 px contiene un oggetto di 4–6 px (V_BIRD_013_f00050, V_DRONE_105_f00158,
  V_HELICOPTER_009_f00001 in [`../immagini/svanstrom_16px.png`](../immagini/svanstrom_16px.png)).
- Oggetti non etichettati: stormi in alcuni video di uccelli.

## Numeri (box originali, seed 42)

284 video, 88.932 frame → 25.991 selezionati + 2.462 di background. Video per split 199 / 56 / 29.

| subset (fasce COCO) | train | val | test |
|---|---|---|---|
| `all` | 20.234 | 5.523 | 2.696 |
| `small` | 9.695 | 2.569 | 1.260 |
| `medium` | 7.137 | 1.953 | 992 |
| `large` | 1.670 | 497 | 206 |
| `background` | 1.721 | 503 | 238 |

Box di train + val per fascia:

| < 8 | 8–16 | 16–32 | 32–64 | 64–96 | > 96 |
|---|---|---|---|---|---|
| 0 | 107 | 12.967 | 7.871 | 1.346 | 2.167 |

Con i box corretti tutto scende di fascia: da rifare dopo il passo 5.

Buchi già visibili: large quasi solo aerei ed elicotteri; drone large 37 frame di un video; bird large zero.

## Limiti

- Large pochi e concentrati in pochi video.
- Sessione di volo non nota (V_AIRPLANE_001 e 002 stessa scena): si ricostruisce a mano, split per sessione.
- Un video può contenere più sequenze (V_AIRPLANE_002 ha un taglio a circa 6 s).
- Solo 82 frame davvero vuoti: il background si crea per inpainting.
