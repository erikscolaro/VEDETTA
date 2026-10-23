# Distribuzione delle grandezze

Analisi del 22/10/2026 sulle label finali dopo la revisione CVAT, con gli stessi frame di `tools/buchi.py`:
Svanström sui frame tenuti, le aggiunte da `5_box/manuale`. Solo box di almeno 8 px in origine. Grandezza =
√(w·h) in pixel sul frame 640×512. Figure rifatte con `python tools/distribuzioni.py`; curva ideale a 14.286 box per classe in entrambe.

## Distribuzione attuale

![attuale](figure/distribuzioni.png)

| classe | box | 8–16 | 16–32 | 32–64 | 64–96 | > 96 |
|---|---|---|---|---|---|---|
| airplane | 16.976 | **865** | **2.387** | 4.651 | 4.016 | 5.057 |
| bird | 12.696 | **2.571** | 4.759 | 4.595 | **562** | **209** |
| drone | 20.753 | 4.338 | 5.084 | 3.847 | 3.422 | 4.062 |
| helicopter | 13.409 | **762** | **1.864** | 3.626 | **2.596** | 4.561 |

In grassetto le celle sotto il minimo di 2.857 (`regole.md`): 8 buchi su 20.

Aerei ed elicotteri hanno il massimo tra 40 e 200 px e poco sotto i 32. Gli uccelli si fermano quasi tutti
sotto i 64 px. I droni coprono bene tutte le fasce.

## Scale di Ultralytics

![scale](figure/distribuzioni_scale.png)

Lo `scale` di Ultralytics (`RandomPerspective`) moltiplica il lato di ogni immagine per un fattore uniforme in
[1−s, 1+s]: riduce **e** ingrandisce. Un box di grandezza x diventa quindi uniforme in [x(1−s), x(1+s)].
Le istanze attese in un intervallo [a, b) sono, in modo esatto:

  Σ_box max(0, min(b, x(1+s)) − max(a, x(1−s))) / (2·s·x)

Lo scale non aggiunge istanze: le sposta tra le fasce a ogni epoca. Il conto non include il ritaglio ai bordi
né il filtro `box_candidates` (lato < 2 px, area persa), che tolgono qualcosa in più ai due estremi.

| cella | senza | s = 0,25 | s = 0,5 |
|---|---|---|---|
| aerei 8–16 | 865 | 884 | 1.007 |
| aerei 16–32 | 2.387 | 2.426 | 2.611 |
| aerei 64–96 | 4.016 | 3.623 | 3.190 |
| uccelli 8–16 | 2.571 | 2.529 | 2.597 |
| uccelli 64–96 | 562 | 612 | 732 |
| uccelli > 96 | 209 | 214 | 259 |
| elicotteri 8–16 | 762 | 760 | 825 |
| elicotteri 16–32 | 1.864 | 1.912 | 2.044 |
| elicotteri 64–96 | 2.596 | 2.500 | 2.345 |
| droni 64–96 | 3.422 | 3.236 | 2.772 |

Con s = 0,25 (quello di `regole.md`) la distribuzione resta quasi uguale: nessun buco si chiude. Con s = 0,5
nemmeno: gli aerei 16–32 arrivano a 2.611, ancora sotto il minimo. In cambio la fascia 64–96, al centro del
picco, si svuota verso i lati e i droni 64–96 scendono sotto il minimo; circa 900 droni finiscono sotto gli 8 px.
Aerei ed elicotteri 8–16 e gli uccelli grandi restano lontani. Lo scale allarga la distribuzione, ma non
sostituisce la raccolta.
Gli oggetti piccoli ottenuti riducendo un'immagine sono anche più nitidi di quelli ripresi davvero lontani.

## Distribuzione ideale

Ragionamento, senza le regole delle fasce:

1. **Scenario.** La grandezza in pixel va come 1/distanza. Un bersaglio che si avvicina a velocità costante
   resta molto più a lungo piccolo che grande: già con la distanza uniforme la densità va come 1/s², e con
   oggetti sparsi a caso nel cielo è ancora più spostata verso il piccolo. Vederlo presto vuol dire vederlo
   piccolo.
2. **Difficoltà.** Tra 8 e 32 px un oggetto è una macchia di pochi pixel: lì il modello confonde le classi e
   perde gli oggetti. Sopra i 64 px si riconosce facilmente e servono meno esempi, anche se un po' servono.
3. **Ruolo delle classi.** Aerei, elicotteri e uccelli servono soprattutto a non dare falsi allarmi sui droni,
   e la confusione c'è dove tutti sono piccoli. Un aereo da 200 px non si scambia mai per un drone.

Forma scelta: decrescente in modo continuo, densità in scala logaritmica proporzionale a s^−α tra 8 e 320 px,
con **α = 0,3 per tutte le classi**, circa −19% per ogni raddoppio di grandezza (utente, 23/10/2026; prima 0,45
per aerei, elicotteri e uccelli).

È una **stima**, il nostro best guess: la distribuzione vera del dominio non la conosciamo e non abbiamo dati per
misurarla. Uno standard non c'è: Ultralytics chiede solo istanze e immagini per classe, COCO e AI-TOD hanno la
distribuzione che è uscita dalla raccolta. La stessa curva per tutte le classi ha due vantaggi: in ogni fascia il
rapporto fra le classi è lo stesso, quindi il modello non può legare la classe alla grandezza ("piccolo =
drone"), e c'è un solo parametro da giustificare.

Nelle figure la curva ideale (linea nera continua) ha il totale deciso, **14.286 box per classe** (10.000 nel
train con lo split 70/20/10): mostra quanto manca. Lo stesso asse verticale nei 4 grafici.

| | 8–16 | 16–32 | 32–64 | 64–96 | > 96 |
|---|---|---|---|---|---|
| ideale, α = 0,3 | 4.007 | 3.255 | 2.644 | 1.310 | 3.070 |
| mancano aerei | 3.142 | 868 | – | – | – |
| mancano uccelli | 1.436 | – | – | 748 | 2.861 |
| mancano elicotteri | 3.245 | 1.391 | – | – | – |

I droni stanno sopra la curva in ogni fascia. **Decisione dell'utente (22/10/2026)**: questa è la regola nuova
(`regole.md`). Drone e uccello sono le classi principali, aereo ed elicottero secondarie: il loro deficit sul
piccolo è un compromesso accettato. Forma e α decisi il 23/10/2026 (sopra): stessa curva per tutte.

Rispetto all'ideale:
- **Aerei ed elicotteri** sono rovesciati: troppi tra 40 e 200 px, troppo pochi sotto i 32.
- **Droni:** sono già vicini all'ideale. Mancano un po' gli 8–12 px e quelli sopra i 128; il picco intorno ai
  90 px è di troppo.
- **Uccelli:** troppi tra 20 e 60 px, mancano gli 8–16 e i grandi.

Le fasce di `regole.md` non sono ottave: 64–96 è poco più di mezza ottava e > 96 quasi un'ottava e mezza.
Chiedere lo stesso numero in ogni fascia quindi non è una distribuzione piatta e pesa troppo 64–96.

Per val e test non serve la forma ideale: meglio stratificarli per grandezza, perché la precisione per
grandezza sia affidabile, e riportare i risultati per grandezza invece di una sola mAP. Sotto gli 8 px gli oggetti
in esercizio sono tanti, ma l'annotazione è poco affidabile. Se contano, serve una testa P2 (stride 4) più che
altri dati.
