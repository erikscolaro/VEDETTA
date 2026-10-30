# Procedura per costruire un dataset di detection da più fonti

La presente procedura è ricavata a posteriori dalla costruzione di un dataset a partire da più fonti pubbliche
(video e immagini, con oggetti di piccole dimensioni su sfondo di cielo). Le fasi sono riportate nell'ordine che
sarebbe stato corretto seguire, non in quello effettivamente seguito: dove l'esperienza ha mostrato che una fase
andava anticipata o posticipata, essa compare qui nella posizione appropriata. I valori numerici sono punti di
partenza che si sono rivelati efficaci; per ogni nuovo dominio vanno ritarati sui fogli di controllo.

Convenzione: **[operatore]** indica una fase in cui una persona esamina fogli di immagini numerate e redige un file
di testo, che la fase successiva legge. Ogni decisione automatica ha valore di proposta; la decisione vincolante è
quella dell'operatore.

## 0. Principi di organizzazione

- Ogni fase dispone di una cartella dedicata, numerata secondo l'ordine di esecuzione, e legge esclusivamente dalle
  cartelle delle fasi precedenti: ciò consente di rieseguire una fase senza ripartire dall'inizio.
- Dove è sufficiente, si producono indici in formato CSV anziché copie delle immagini; quando sono necessarie
  cartelle di immagini si impiegano **hardlink**.
- I file prodotti dall'operatore sono conservati accanto ai dati: documentano le scelte effettuate e consentono di
  rigenerare l'intero dataset.
- La procedura è interamente deterministica: semi fissi e nomi dei frame stabili, derivati dall'istante temporale o
  dall'indice e non dall'ordine di elaborazione. Una fase rieseguita deve produrre un output identico.
- Prima di ogni operazione distruttiva in blocco (cancellazioni nello strumento di annotazione, riscritture di
  label) si esegue un backup e si quantifica la modifica attesa.
- Le decisioni sono registrate, con la data, in un documento di regole; alla modifica di una regola si rieseguono
  le fasi che ne dipendono.

## 1. Regole da fissare prima di accedere ai dati

Le regole seguenti vanno stabilite per iscritto:

- **Classi** e casi limite. Gli oggetti simili ma esterni alle classi non ricevono label e l'immagine è mantenuta.
  Le categorie ambigue, come i convertiplani, sono assegnate fin dall'inizio a una classe, e la scelta è riportata
  nei prompt e nelle istruzioni.
- **Regola del box**: box aderente all'oggetto; limitato alla parte visibile se l'oggetto è troncato o occluso; un
  box per oggetto, anche nei gruppi; stessa regola per tutte le fonti.
- **Formato finale**: risoluzione unica. Le fonti di dimensioni maggiori sono ritagliate attorno agli oggetti senza
  ridimensionamento, in modo che i pixel delle label coincidano con quelli osservati dal modello.
- **Grandezza**: s = √(w·h), in pixel alla risoluzione finale. Si definisce una soglia minima al di sotto della
  quale l'intera immagine è rimossa (rimuovere il solo box lascerebbe un oggetto privo di label).
- **Dominio**: ciò che è incluso (per esempio riprese diurne da terra verso il cielo) e ciò che è escluso (notte,
  riprese da bordo, aria-aria, interni, scie). A ogni clip è inoltre associato uno **scenario** (per esempio cielo,
  orizzonte, terra, muro), che determina quali clip possono essere assegnate a val (validazione) e test.
- **Distribuzione obiettivo** per classe sulla grandezza (vedi passo 12).
- **Licenze ammesse**: pubblico dominio, CC0, CC BY, Apache, MIT. Ogni eccezione è dichiarata e resta separabile.

## 2. Bilancio delle carenze prima della raccolta

- Si contano istanze, clip, video e sessioni per **cella classe × fascia di grandezza** e i conteggi sono confrontati
  con i minimi richiesti dagli split: numero di istanze in val e test e almeno 3 sessioni per cella, affinché ogni
  split ne disponga di una.
- Il bilancio è ricalcolato a ogni fonte aggiunta e orienta la ricerca delle fonti successive.
- La raccolta avviene con un margine del 10–30%, poiché il numero di box proposti diminuisce dopo la revisione ed è
  preferibile evitare nuovi download.
- Le grandezze stimate dalle label originali richiedono cautela: se i box non sono aderenti (in un caso erano pari
  a 4 volte l'oggetto), le fasce risultano sovrastimate. Prima di selezionare una fonte si misurano quindi i box
  effettivi su un campione.

## 3. Scelta e download delle fonti

- La licenza è verificata **prima** del download e registrata per ciascun elemento (id, autore, URL, licenza).
- Si scarica soltanto il materiale necessario: dapprima le sole label, di dimensioni ridotte, per selezionare le
  sequenze; poi le parti utili degli archivi, mediante richieste HTTP a intervalli. Dai video remoti si leggono
  solo i tratti richiesti.
- Dei file scaricati si calcola il checksum; se l'origine non è raggiungibile si ricorre a un mirror.
- Le fonti ripubblicano il medesimo materiale con identificativi diversi. Sono considerati **doppioni di video**
  quelli con uguale durata e anteprime identiche (dHash) allo stesso istante; la somiglianza libera tra anteprime
  produce quasi esclusivamente falsi positivi.
- Gli split ufficiali delle fonti non sono utilizzati: lo split è definito per sessione, su tutte le fonti
  congiuntamente.

## 4. Selezione delle riprese [operatore]

- Per i video lunghi si estraggono anteprime a intervalli regolari (con un limite massimo per video); un detector
  ricerca l'oggetto e si conservano tratti della durata di alcuni secondi attorno alle rilevazioni.
- I fogli riportano un'anteprima per tratto. L'operatore indica i tratti da scartare, associando a ciascuno un
  **codice di motivo** (da bordo, aria-aria, persone in primo piano, interni, solo a terra, niente oggetto, scia).
  In caso di dubbio il tratto è mantenuto.
- I filtri automatici di dominio (luminosità per la notte, saturazione per la scala di grigi, dominante verde per i
  visori notturni) **non sono sufficienti da soli**: servono a ordinare i fogli, mentre il giudizio finale spetta
  all'operatore.
- Per le fonti fotografiche si filtrano le annotazioni (solo box individuali, nessuna raffigurazione) e si impiega
  un modello zero-shot (per esempio SigLIP) per conservare solo la scena pertinente: un uccello in volo, non
  posato.
- Anche il numero di frame per video è limitato, per evitare che le riprese lunghe abbiano un peso eccessivo.

## 5. Clip e frame

- Un video può contenere più riprese. I **tagli di scena** sono individuati mediante candidati a soglia bassa
  (differenza tra istogrammi e correlazione tra frame consecutivi) e successiva revisione umana: nessuna soglia
  automatica distingue i tagli dalle panoramiche e dalle variazioni di esposizione.
- Da questo punto l'unità di analisi è la **clip** (ripresa continua), non il video.
- Si mantengono soltanto i frame con almeno un box; l'eventuale inclusione di frame vuoti intenzionali è decisa
  separatamente.

## 6. Quasi-duplicati dentro la clip, prima di annotare

- Frame ravvicinati della stessa clip sono quasi identici: aumentano artificialmente i conteggi e il carico di
  revisione, e sono quindi rimossi **prima** della revisione.
- Sono considerati quasi-duplicati i frame con dHash simile del ritaglio attorno ai box, medesimo numero di box e
  posizione e grandezza simili. Per oggetti di pochi pixel una soglia stretta (8 bit su 64) non rimuove quasi
  alcun frame; 16 bit si è rivelato un buon compromesso. La soglia si sceglie osservando fogli di frame
  consecutivi, mantenuti e rimossi.
- La deduplicazione è eseguita sul **frame intero**, prima del ritaglio: dopo il ritaglio l'oggetto occupa una
  posizione casuale e il confronto geometrico non è più possibile.

## 7. Ritaglio alla risoluzione finale

- Si ritaglia una finestra alla risoluzione finale attorno all'oggetto più grande, in posizione casuale con seme
  fisso e all'interno dell'area utile (esclusi i bordi neri di padding). Gli altri box sono troncati alla finestra.
- Si registra quali box risultano troncati, poiché non è garantito che l'oggetto sia incluso per intero.

## 8. Box proposti

- Per prima cosa si verificano le **label originali** su un campione ingrandito: se sono aderenti, si mantengono.
- In caso contrario, i metodi sperimentati, in ordine di utilità, sono:
  - **detector** già addestrato sul dominio: efficace sul cielo, inutile su sfondi diversi da quelli del suo
    training;
  - **sottrazione dello sfondo**, a camera ferma (mediana di frame sparsi), seguita da SAM con un punto al centro
    della componente;
  - **contrasto locale** all'interno del box ampliato, poi **SAM 2 con un punto** dove il contrasto fallisce;
  - SAM 2 con il **box come prompt**: non è utile, poiché restituisce il box stesso.
- Prima di applicare un metodo all'intero materiale, i metodi sono confrontati su fogli che sovrappongono originale
  e proposta.
- La **preselezione** a regole, anteriore alla revisione, assegna la classe dai metadati del video quando il
  detector confonde le classi e rimuove: box doppi (IoU ≥ 0,8), box contenuti in un altro, residui sul bordo, box
  minuscoli, primi piani (box oltre una quota del ritaglio), loghi e sovraimpressioni ricorrenti, frame notturni.
- Se il materiale è abbondante, il limite per cella è applicato già in questa fase, per evitare la revisione
  manuale di ciò che verrà comunque scartato.

## 9. Revisione delle label [operatore]

- **Tutti** i frame che possono entrare nel dataset sono rivisti nello strumento di annotazione (per esempio CVAT),
  assumendo i box proposti come punto di partenza. La revisione deve concludersi **prima dello split**: in caso
  contrario ogni nuovo split introduce in val e test frame non rivisti e si genera un ciclo di revisioni.
- Indicazioni pratiche per CVAT: task di al più ~1.000 file; export senza immagini. I frame privi di box non sono
  esportati; per le clip svuotate è quindi necessaria una cartella vuota esplicita, altrimenti vengono riletti i
  box proposti. Tramite SDK, una cancellazione con lista vuota può eliminare tutte le annotazioni del task.
- **Agenti LLM come pre-revisori**: sono utili per decidere solo classe ed eliminazione (i bordi restano di
  competenza dell'operatore) e, in caso di dubbio, rispondono "ok". Il metodo più economico prevede un solo
  agente, un frame ogni 10 e ±5 frame attorno a ciascun errore, finché il bordo è pulito; più immagini lette per
  turno (la stessa sequenza favorisce la coerenza) e salvataggio periodico del lavoro. L'operatore convalida.
- **Controlli di coerenza** eseguiti a script dopo la revisione: numero di box diverso dai frame vicini, box quasi
  uguali o contenuti l'uno nell'altro, box minuscoli. Si rivedono solo i frame segnalati.
- **Seconda revisione per candidati**: due detector indipendenti, applicati a tutti i frame, segnalano oggetti
  privi di box, box su zone vuote e classi in disaccordo. Se il dubbio riguarda un oggetto mancante o una classe,
  il frame è rimosso; se riguarda un box esistente, è mantenuto.

## 10. Sessioni e gruppi

- Una **sessione** è definita da stessa località e stesse condizioni (stesso giorno, stessa luce). La stessa
  località in un giorno diverso costituisce un'altra sessione, così come una località diversa nello stesso giorno.
- La proposta automatica deriva dai metadati (date e ore nei nomi dei file o nei video, salti all'indietro
  dell'ora) **oppure** dagli embedding delle clip (DINOv2 ViT-S/14, token CLS), con unione per transitività. Si
  adotta il legame medio e non il legame singolo: sul cielo uniforme il legame singolo unisce l'intero insieme.
- I fogli riportano una riga per clip e, in fondo, la clip più simile esterna al gruppo. L'operatore indica **solo
  le eccezioni** (rimozione, spostamento, unione). In caso di dubbio si assegna la stessa sessione.
- **Doppioni tra sessioni** (stessa ripresa in sessioni diverse, materiale ripubblicato, camere fisse):
  - si calcola l'embedding di ogni frame annotato (i modelli più grandi di ViT-S non separano meglio);
  - la somiglianza tra sessioni è il massimo sulle coppie di frame;
  - a somiglianza media il valore dipende dallo **sfondo** (il cielo) e non dalla sessione; una soglia bassa unisce
    quindi l'intero insieme;
  - per la **taratura** l'operatore giudica un campione di coppie per ciascuna fascia di somiglianza e si calcola la
    quota di doppioni con intervallo di Wilson. In questo caso, sopra 0,95 tutte le coppie erano doppioni (unione
    automatica); tra 0,92 e 0,95 circa un terzo (revisione completa manuale); sotto 0,92 quasi nessuna;
  - gli agenti impiegati come giudici si sono rivelati troppo severi rispetto all'operatore; il giudizio resta
    pertanto umano;
  - ogni sessione aggiunta in seguito va riesaminata nella fascia di revisione.
- I **gruppi** sono ottenuti con un union-find di sessioni note e doppioni. Ciascun gruppo è assegnato per intero a
  un unico split.

## 11. Scenario per clip [operatore]

- Si esaminano provini di alcuni frame per clip; lo scenario è proposto (anche da agenti) e corretto
  dall'operatore.
- Come controllo incrociato si misura la texture dello sfondo attorno al box (media del Laplaciano); le
  incoerenze sono sottoposte a revisione.
- Nelle clip miste lo scenario è assegnato per frame, sulla base della texture.

## 12. Distribuzione obiettivo

- Si adotta una curva continua sulla grandezza (per esempio densità ∝ s^−α in scala logaritmica), **identica per
  tutte le classi**, affinché il modello non associ la classe alla grandezza.
- Il controllo è condotto su intervalli fini (mezze o quarti d'ottava) e non su poche fasce larghe: all'interno di
  una fascia larga la forma può derivare liberamente (per esempio con un picco a metà fascia).
- **Val** segue la curva, poiché la scelta del checkpoint avviene su di esso e le grandezze devono pesare come in
  esercizio. **Test** è uniforme in log s, per misurare con accuratezza ogni grandezza. Per val e test sono fissati
  minimi di istanze per classe.
- Le augmentation di scala del framework ridistribuiscono le istanze tra le grandezze, ma non ne aggiungono.

## 13. Split

- Lo split è eseguito su gruppi interi. Prima dell'assegnazione si rimuovono le immagini con box sotto la soglia
  minima e, da val e test, i frame fuori dominio.
- L'assegnazione è formulata come problema di ottimizzazione: numerosi ordinamenti casuali dei gruppi,
  assegnazione greedy, rifinitura mediante spostamento di un gruppo alla volta. L'esecuzione è parallela, con seme
  fisso, e si confrontano più semi.
- Si sono rivelati necessari i seguenti termini e vincoli:
  - i gruppi molto grandi sono sempre assegnati al train;
  - un gruppo in val o test non contribuisce per più del 15% di una classe né per più del 75% di una cella: se è
    più grande, se ne estrae un sottoinsieme casuale e il resto non è impiegato (non può essere assegnato al
    train);
  - un **termine di spreco** per i box così persi, in assenza del quale lo split scarta molto materiale;
  - quote per fonte in val e test proporzionali al materiale in dominio, classe per classe;
  - minimi per classe, mai oltre una quota fissa del materiale disponibile.
- Uno split moderatamente sbilanciato è spesso preferibile alla perdita di dati nel train.

## 14. Bilanciamento del train

- I limiti si applicano in quest'ordine: quota fuori dominio uguale per tutte le classi, limite per sessione,
  limite per fonte (rispetto alla sua quota in val e test), limite per cella sulla curva.
- I **mosaici** integrano gli oggetti piccoli mancanti e sono impiegati solo nel train. Consistono in griglie 2×2,
  4×4, 8×8 di immagini ridotte (×0,5, ×0,25, ×0,125, INTER_AREA), con sfocatura, rumore e compressione JPEG di
  lieve entità, e mai con box sotto la soglia minima. Le caselle provengono dalla stessa mezza ottava di
  grandezza, alternando le sessioni: dapprima con le immagini escluse dal limite e, se necessario, con copie
  ridotte.
- **Nessuna immagine sintetica in val e test.**

## 15. Background

- Il background rappresenta circa il 10% di ciascuno split e proviene da fonti indipendenti del dominio (per
  esempio cielo e orizzonte privi di oggetti).
- I ritagli sono prodotti alla risoluzione finale senza ingrandimento. Si escludono quelli bui e quelli in cui il
  detector rileva qualcosa (anche con confidenza bassa: l'esclusione è accettabile); si rimuovono poi i
  quasi-duplicati per camera mediante gli embedding e si effettua una revisione visiva.
- Le camere sono assegnate per intero a un unico split.

## 16. Controllo del leakage

- Per ogni immagine di val e test si calcola la somiglianza con l'immagine più vicina del train, confrontandola con
  due riferimenti: le coppie di gruppi diversi (livello di fondo) e le coppie dello stesso gruppo (leakage
  effettivo).
- L'obiettivo è che val e test coincidano con il livello di fondo e che nessuna immagine superi la soglia dei
  doppioni. Il leakage dello sfondo (stesso cielo, stesso luogo in giorni diversi) è dichiarato.

## 17. Export e congelamento

- Si adotta la struttura del framework (per esempio Ultralytics), realizzata con hardlink. L'ordine delle classi è
  stabilito in funzione dell'uso finale; le label sono riscritte solo in fase di export.
- I nomi sono neutri e univoci (tipo di immagine, gruppo, hash della clip, indice) e un **manifest** consente di
  risalire a fonte, clip, frame originale, sessione e scenario.
- Il set di calibrazione per la quantizzazione comprende alcune centinaia di immagini reali del train, bilanciate
  per classe e grandezza, con ~10% di background.
- Il README del dataset riporta: classi, composizione degli split, legenda dei nomi, fonti con licenza,
  attribuzioni, eccezioni, limiti dichiarati e figure (distribuzione per split, fonti per split, scenario,
  leakage).
- Il dataset esportato è **congelato**: l'export rifiuta di sovrascriverlo.

## Errori da non ripetere

- Assumere come attendibili le grandezze delle label originali senza misurarle.
- Rivedere manualmente prima di deduplicare.
- Eseguire lo split prima di aver concluso la revisione.
- Fissare soglie di somiglianza senza una taratura su campioni giudicati manualmente.
- Bilanciare solo su fasce larghe di grandezza.
- Usare curve diverse per il limite del train e per la compensazione con i mosaici, poiché la forma del train ne
  risulta deformata. Si adotta la stessa curva, traslata se necessario.
- Aggiungere fonti o sessioni dopo la revisione dei doppioni senza riesaminarle.
