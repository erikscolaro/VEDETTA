# Da decidere, rischi, limiti

## Da decidere

| punto | quando |
|---|---|
| minimo di istanze per cella (oggi 870 = 385 val + 385 test + 100 train): con la deduplica a 16 bit regge 1 cella su 16 | passo 9 |
| minimo per cella da rivedere (proposta 150–200 in val e test, vedi Compromessi in `regole.md`) | prima dello split |
| fonti per i buchi | passo 10 |
| detector a vocabolario aperto per le aggiunte | passo 10 |
| frame con oggetti sotto i 16 px (64% dei box corretti, uccelli 93%): solo train o scartati | passo 8 |

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
- Droni piccoli in val e test da pochi voli (8–16: 5 gruppi in val, 6 in test; 16–32: 5 in test): i droni
  piccoli stanno in 32 gruppi in tutto. Dichiarato, con la barra d'errore dal bootstrap per gruppo (utente,
  23/10/2026); altre riprese vere di droni lontani se c'è tempo.
- Niente aerei con la scia: le 12 clip con scia sono escluse.
- Sessioni di volo di Svanström non note: ricostruite a mano dalle strisce.
- Leakage dello sfondo accettato: lo stesso posto ripreso in giorni o luci diverse è in sessioni diverse, quindi può
  stare in train e in test. Motivo: in produzione la torretta è fissa e lo sfondo resta quello; cambiano i droni.
