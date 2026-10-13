# Da decidere, rischi, limiti

## Da decidere

| punto | quando |
|---|---|
| fonti più grandi di 640×512: ritaglio o ridimensionamento | prima di cercare le aggiunte |
| fonti per i buchi | passo 8 |
| detector a vocabolario aperto per le aggiunte | passo 8 |
| frame con oggetti sotto i 16 px (64% dei box corretti, uccelli 93%): solo train o scartati | passo 6 |
| soglie dell'hash (oggi `--hash-thr 8`, `--geo-thr 0.05`) | passo 9 |

## Rischi

| rischio | piano B |
|---|---|
| SAM 2 sbaglia sugli oggetti minuscoli (confermato: col box come prompt ridà il box) | contrasto locale, SAM con un punto, revisione di tutto in CVAT |
| aggiunte non trovate o troppo lente | celle scoperte dichiarate come limite |
| dopo la correzione molti oggetti < 16 px, val e test sotto il minimo | rivedere i minimi; il train li tiene |

## Limiti per la tesi

- Val e test senza oggetti < 16 px: in deploy ci saranno.
- Giunture dei mosaici, non verificabili senza fotogrammi veri ad alta risoluzione.
- Celle scoperte (per esempio uccelli grandi).
- Niente aerei con la scia: le 12 clip con scia sono escluse.
- Sessioni di volo di Svanström non note: ricostruite a mano dalle strisce.
