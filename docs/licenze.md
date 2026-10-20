# Licenze delle fonti

Regola dell'utente (19/10/2026): solo fonti con licenza **aperta**: pubblico dominio, CC0, CC BY (anche CC BY-SA da
chiedere), licenze software aperte (Apache, MIT). La licenza si verifica **prima** di scaricare e si scrive qui; per
le fonti a video singoli (DVIDS, Wikimedia) anche per video, nel csv della fonte.

| fonte | licenza | dove è scritta | verificata | uso nel dataset | attribuzione richiesta |
|---|---|---|---|---|---|
| Svanström, "A dataset for multi-sensor drone detection" (Data in Brief 2021) | **CC0 1.0** (dati); articolo CC BY | campo `license` del repo [DroneDetectionThesis/Drone-detection-dataset](https://github.com/DroneDetectionThesis/Drone-detection-dataset) (API GitHub) | 19/10/2026 | base | citare il paper (buona pratica) |
| FRED (Magrini et al., ACM MM 2025) | **Apache 2.0** | README e scheda Hugging Face `GabrieleMagrini/FRED` | 16/10/2026 | droni | citare il paper, riportare la licenza |
| FBD-SV-2024 (Sun et al., Scientific Data 2025) | **solo ricerca, non commerciale** ("This dataset is for learning and research purposes only, not for commercial use"); Kaggle: "Unknown" | [figshare 27377595](https://figshare.com/articles/dataset/FBD-SV-2024/27377595), Kaggle `swjtuziwei/fbd-sv-2024` | 19/10/2026 | uccelli: **tenuto** come eccezione dichiarata (utente, 19/10/2026: la tesi è ricerca e il modello non va in un prodotto); fonte separabile (`data/fbd_sv`), da togliere se il dataset si pubblica | citare il paper |
| DVIDS (dvidshub.net, Dipartimento della Difesa USA) | **pubblico dominio** per video ("PUBLIC DOMAIN" sulla pagina: opere di dipendenti federali); i video senza questa scritta non si usano | pagina di ogni video; condizioni in [dvidshub.net/about/copyright](https://www.dvidshub.net/about/copyright) | per video, `data/dvids/0_raw/video.csv` (colonne `licenza`, `autore`, `url`) | aerei, elicotteri (e droni) | frase di non-endorsement: "The appearance of U.S. Department of War (DoW) visual information does not imply or constitute DoW endorsement."; niente insegne in usi commerciali |
| Anti-UAV300, parte RGB (Jiang et al., IEEE TMM 2021) | **MIT** (progetto: "The project of Anti-UAV is released under the MIT License"); nessun termine diverso per i dati nel README né sul sito anti-uav.github.io | file `LICENSE` e README del repo [ZhaoJ9014/Anti-UAV](https://github.com/ZhaoJ9014/Anti-UAV) (API GitHub: MIT) | 19/10/2026 | droni 32–> 96 (mirror Hugging Face `DestinyLove/AntiUAV300`, scheda MIT) | riportare copyright e licenza MIT, citare il paper |

## Candidate (licenza da verificare per file)

| fonte | licenza | note |
|---|---|---|
| Wikimedia Commons (video) | per file: CC0, PD, CC BY, CC BY-SA | CC BY va bene (utente, 19/10/2026); CC BY-SA da chiedere |
| Open Images V7 | immagini CC BY 2.0 (per immagine), annotazioni CC BY 4.0 | solo aerei ed elicotteri > 96 px |
| AirBirds | CC BY-NC-SA 4.0 | non commerciale: come FBD, da chiedere |
