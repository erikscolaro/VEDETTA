# Da decidere, rischi, limiti

## Da decidere

| punto | quando |
|---|---|
| fonti più grandi di 640×512: ritaglio o ridimensionamento | prima di cercare le aggiunte |
| correzione: quale SAM 2, quanta revisione a mano, strumento | passo 5 |
| fonti per i buchi | passo 8 |
| detector a vocabolario aperto per le aggiunte | passo 8 |
| soglie dell'hash (oggi `--hash-thr 8`, `--geo-thr 0.05`) | passo 9 |

## Rischi

| rischio | piano B |
|---|---|
| SAM 2 sbaglia sugli oggetti minuscoli | contrasto locale su cielo uniforme, revisione a mano |
| aggiunte non trovate o troppo lente | celle scoperte dichiarate come limite |
| dopo la correzione molti oggetti < 16 px, val e test sotto il minimo | rivedere i minimi; il train li tiene |

## Limiti per la tesi

- Val e test senza oggetti < 16 px: in deploy ci saranno.
- Giunture dei mosaici, non verificabili senza fotogrammi veri ad alta risoluzione.
- Celle scoperte (per esempio uccelli grandi).
- Sessioni di volo di Svanström non note: ricostruite a mano dalle strisce.
