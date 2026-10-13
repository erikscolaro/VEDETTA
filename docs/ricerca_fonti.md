# Fonti

Ricerca web del 12–13/10/2026 (agenti). Numeri dalle fonti citate, **non ricontrollati**. Le fonti più grandi di
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
