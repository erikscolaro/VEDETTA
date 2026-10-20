# Fonti

Ricerca web del 12–13/10/2026. Numeri dalle fonti citate, **non ricontrollati**. Le fonti più grandi di
640×512 vanno portate a 640×512 ([`da_decidere.md`](da_decidere.md)).

## Candidate

| fonte | classi | risoluzione | punto di vista | ID video | accesso | problemi |
|---|---|---|---|---|---|---|
| [Anti-UAV300](https://github.com/ZhaoJ9014/Anti-UAV), parte RGB (TMM 2021) | drone | 1920×1080 | da terra | sì | MIT, Google Drive / Baidu | etichettato solo il bersaglio; droni grandi (mediana 116×61 px) |
| [Drone-vs-Bird](https://github.com/wosdetc/challenge) (WOSDETC) | drone | da 720×576 a 4K | da terra, camere fisse e mobili | sì | email a wosdetc@googlegroups.com, accordo d'uso | **uccelli presenti ma non etichettati**; droni quasi tutti < 32 px; label corrette su raysonlaroca.github.io |
| [TartanAviation](https://github.com/castacks/TartanAviation) (Sci. Data 2025) | aereo, rotorcraft | 2448×2048 | da terra, **4 camere fisse in un solo aeroporto** | sì | libero, 3,3 TB | uccelli e droni non etichettati; oggetti lontani (trigger a 8 km); colore non dichiarato |
| [AirBirds](https://airbirdsdata.github.io/) (ACCV 2022) | uccello | 1920×1080 | da terra, un aeroporto | parziale | modulo, CC BY-NC-SA 4.0 | 88% degli uccelli < 10 px; aerei tolti dalle label |
| Video a licenza libera (Pexels, Pixabay) | uccello, aereo | varia | da terra | sì | libera | da etichettare noi |
| [LAA3D](https://arxiv.org/abs/2511.19057) (arXiv 2025) | MAV, eVTOL, elicottero (~6%) | 1920×1280 | — | 185 sequenze | non trovato | oggetti entro 100 m |

## Ricerca del 16/10/2026, per i buchi

Ricerca web per riempire i buchi del passo 9 (dopo la deduplica a 16 bit: tutte le celle ≥ 16 px tranne drone
16–32; per classe tutte sotto le 10.000 istanze). Numeri da paper e pagine dei dataset, spesso da riassunti:
**da verificare** sul paper e sul file di licenza prima di scaricare. Le fonti più grandi di 640×512 si ritagliano
(`regole.md`).

| fonte | classi | risoluzione | punto di vista | dimensione | accesso, licenza | per quali buchi |
|---|---|---|---|---|---|---|
| [FBD-SV-2024](https://doi.org/10.1038/s41597-025-04872-6) (Sun et al., **Scientific Data 2025**) | uccello | 1280×720, 25 fps (da riassunto) | telecamere di sorveglianza; cielo non verificato | 483 clip (`bird_N.mp4`), 28.694 frame, 28.366 uccelli; XML con track id e `difficult` | [README](https://github.com/Ziwei89/FBD-SV-2024_github): figshare, Kaggle, Baidu; **licenza non scritta** nel repo | bird 16–32, 32–64 |
| [FRED](https://arxiv.org/abs/2506.05163) (Univ. Firenze, **ACM MM 2025**) | drone (5 modelli: DJI Mini 2 e 3, Tello, 2 mini FPV) | 1280×720, 30 fps | treppiede a terra; **anche sequenze indoor e di notte**, da filtrare | oltre 7 h, box per frame con id; uno zip per sequenza, split train/test | **Apache 2.0** (README e scheda HF), [Hugging Face](https://huggingface.co/datasets/GabrieleMagrini/FRED), mirror Drive | drone 32–64, 64–96, > 96 (droni piccoli e vicini) |
| [USC Drone](https://mcl.usc.edu/?p=10642) (Wang et al., APSIPA 2019) | drone (DJI Phantom) | 1920×1080, 15 fps | campus, angoli vari | 30 video da ~1 min, label 1 frame ogni 15 | GitHub dalla pagina MCL; licenza non trovata | drone grandi; un solo modello di drone |
| [VisioDECT](https://ieee-dataport.org/documents/visiodect-dataset-aerial-dataset-scenario-based-multi-drone-detection-and-identification) (Ajakwe et al. 2022) | drone (6 modelli) | 852×480 | treppiede, 30–100 m, 12 luoghi | 20.924 immagini; ID dei video non chiari | IEEE DataPort **solo per abbonati**; licenza non visibile | drone 32–96; bassa priorità |
| Real World Quadcopter ([Pawełczyk e Wojtyra, IEEE Access 2020](https://doi.org/10.1109/ACCESS.2020.3026192)) | drone | varia | da terra, riprese reali | 56.821 immagini, 55.539 box | licenza non trovata; ID dei video da verificare | drone grandi |
| Uccelli a 4K in un parco eolico ([arXiv 2105.08253](https://arxiv.org/abs/2105.08253), [1709.04666](https://arxiv.org/abs/1709.04666)) | uccello | 3840×2160, 30 fps | camera fissa da terra | 3 giorni annotati; grandezza più frequente 55 px | download non trovato, chiedere agli autori | bird 32–64, 64–96 (uccelli grandi veri, rari altrove) |
| [LRDDv3](https://research.coe.drexel.edu/ece/imaple/lrddv3/) (Drexel, ICRA 2026) | drone | 4K RGB (+ IR 640×512) | da terra, 0–175 m | 102.532 immagini da 128 clip, 17 giorni | modulo di richiesta, controllo export USA; licenza non trovata | drone di tutte le fasce, con distanza |
| [AODTA](https://ieee-dataport.org/documents/airborne-object-detection-and-threat-analysis-aodta-dataset) (IEEE DataPort 2026) | aereo, drone, elicottero, uccello | da vedere | da vedere | da vedere | DataPort; sembra assemblato da altri dataset | da guardare: rischio come AOD4 (fonti miste, niente ID) |

Verifica del 16/10/2026 (README e pagine; la rete via proxy va a tratti): **FRED** è la più pronta (licenza aperta,
sequenze con id, download diretto); **FBD-SV-2024** è pubblicato su Scientific Data ma la licenza va letta su
figshare o Kaggle; **VisioDECT** è dietro abbonamento.

Riconfermate dalla ricerca del 12–13/10: Anti-UAV300 RGB (droni grandi), Drone-vs-Bird, TartanAviation (aerei,
elicotteri), AirBirds, video liberi (Pexels, Pixabay).

Buchi ancora senza fonte buona:

- **elicotteri** (tutte le fasce ≥ 16 px): nessun dataset da terra con elicotteri oltre Svanström e TartanAviation
  (rotorcraft, un aeroporto); restano i video liberi da etichettare;
- **aerei** grandi: TartanAviation (un aeroporto, oggetti lontani) e video liberi;
- **uccelli** sopra i 64 px: solo il dataset a 4K del parco eolico, se gli autori lo danno, e i video liberi.

### Open Images V7 (16/10/2026)

Classi con box: Airplane, Helicopter, Bird (più 17 sottoclassi: Eagle, Falcon, Goose, Duck, ...), Aircraft; **niente
drone**. Analisi sui box di validation + test (~167k immagini; il train è ~10 volte tanto), senza depiction:

| classe | box | immagini | lato del box / lato immagine, mediana | box < 10% del lato | group | troncati |
|---|---|---|---|---|---|---|
| airplane | 4.338 | 3.214 | 0,59 | 7% | 3% | 28% |
| helicopter | 593 | 529 | 0,70 | 2% | 1% | 21% |
| bird | 7.758 | 4.606 | 0,46 | 12% | 4% | 17% |

Foglio di esempi in [`immagini/open_images_esempi.jpg`](../immagini/open_images_esempi.jpg) (riga a caso e riga con
box < 10% del lato, per classe). Foto Flickr (licenza CC BY 2.0 per immagine, annotazioni CC BY 4.0), oggetti
**grandi e centrati**: aerei soprattutto a terra (aeroporti, musei), elicotteri a terra o grandi in cielo, uccelli
posati e da vicino. Box piccoli rari (in validation: 43 immagini di aerei, 1 di elicotteri, 44 di uccelli) e con
label parziali (stormi). Utile solo per le celle > 96 px di aerei ed elicotteri in volo, filtrando le immagini di
cielo; niente video né sessioni.

## Scelte del 19/10/2026

- **FBD-SV-2024** per gli uccelli: box aderenti (verificati su 8 clip), telecamere di sorveglianza su campi e
  orizzonte, in dominio perché la torretta guarda anche l'orizzonte (utente). Licenza "Unknown" su Kaggle.
- **DVIDS** (dvidshub.net) per aerei ed elicotteri militari, solo video PUBLIC DOMAIN; gli aerei civili li copre
  Svanström. Elicotteri ripresi da terra da basi e giorni diversi (molte sessioni). Condizioni d'uso in
  `procedura.md`, passo 10.
- Restano candidate: AirBirds (uccelli, modulo), il dataset a 4K del parco eolico (uccelli grandi), Wikimedia
  Commons e archive.org (aerei civili).

## Scartate

| fonte | motivo |
|---|---|
| AOT (Amazon) | scala di grigi, aria-aria |
| AOD4 | split per immagine (leakage), immagini stirate a 1024×1024, classe legata alla fonte |
| Det-Fly | aria-aria, un solo drone, niente ID |
| ARD-MAV, SMOT4SB, LRDDv1–3 | aria-aria o da drone; LRDD con accesso sospeso |
| LAT-BirdDrone | droni solo IR, DataPort a pagamento |
| VIP Cup 2025 | 320×256 |
| Anti-UAV410/600, FL-Drones, Skagen e Klim | IR o scala di grigi |
| DUT Anti-UAV, SOD4SB, sekilab, MBAOD, Anti2, AVD, AODTA, BirDrone | niente ID dei video o assemblati |
| NPS-Drones, MAV-VID, UAVDB, QUT | risoluzione mista o ignota, label automatiche, niente box |
| Yoshihashi et al. | un solo luogo, accesso non trovato |
| KI4Flight, Utah GA, Deep4Air, AVOIDDS, FGVC-Aircraft, Military Aircraft, MADAI | da drone, a terra o sintetici |
| COCO, LVIS, Open Images | quasi mai oggetti in volo |
| [EV-Flying](https://arxiv.org/abs/2506.04048) (Magrini et al., CVPRW 2025) | solo dati a eventi (`.npy`), niente RGB; i droni sono 2 video presi da FRED |

Da verificare: GOT-10k, VastTrack (single-object tracking, classi e risoluzione non confermate).

## Fonti di AOD4

Da "Introducing AOD 4" (Data in Brief 2024) e dai nomi dei file (`tools/aod4_fonti.py`):

| fonte | video | immagini | box |
|---|---|---|---|
| YouTube-8M (`yt-<id>-<n>`) | 142 | 10.182 | aereo 6.926, elicottero 5.526, uccello 2.171, drone 867 |
| Anti-UAV RGB | 23 | 5.810 | drone 5.808 |
| Svanström | 185 | 4.942 | elicottero 2.374, drone 1.735, aereo 974 |
| foto stock Pexels/Unsplash | ~400 foto | 2.094 | uccello 5.729 |

I video YouTube di AOD4 sono una possibile fonte di aerei ed elicotteri da terra, ri-estraendo i frame dagli ID.
