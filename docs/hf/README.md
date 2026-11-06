---
pretty_name: air-detection
license: cc-by-4.0
task_categories:
  - object-detection
language:
  - en
  - it
tags:
  - drone
  - uav
  - counter-uav
  - aircraft
  - helicopter
  - bird
  - yolo
  - small-object-detection
size_categories:
  - 10K<n<100K
---

# air-detection: drones, airplanes, helicopters and birds against the sky

Object detection dataset for a ground-based anti-drone turret: an RGB camera on the ground looking at the sky and
the horizon. Four classes, 32 058 images at 640×512 (plus 511 calibration copies), built from eight public sources with a split by groups of
recording sessions, so the same footage is never in two splits. Built for a master's thesis on hardware-aware YOLO
models for edge devices (Politecnico di Torino, 2026).

Italian version of this card: [`README.it.md`](README.it.md). Code and full procedure:
[github.com/erikscolaro/air-detection](https://github.com/erikscolaro/air-detection).

## Format

Ultralytics YOLO: `images/<split>/*.jpg`, `labels/<split>/*.txt` (one row per box: class, center x, center y, width,
height, normalized), `dataset.yaml`. All images are 640×512 and never upscaled.

| class | id |
|---|---|
| drone | 0 |
| airplane | 1 |
| helicopter | 2 |
| bird | 3 |

```bash
hf download Einrich99/air-detection --repo-type dataset --local-dir air-detection
yolo train data=air-detection/dataset.yaml model=yolo26n.yaml imgsz=640
```

## Splits

| split | images | real | mosaics | background |
|---|---|---|---|---|
| train | 21 958 | 18 397 | 511 | 3 050 |
| val | 6 659 | 5 759 | 0 | 900 |
| test | 3 441 | 2 925 | 0 | 516 |
| calib | 511 | 460 | 0 | 51 |

| boxes | drone | airplane | helicopter | bird |
|---|---|---|---|---|
| train | 11 637 (3 301 from mosaics) | 11 959 | 11 927 | 3 477 |
| val | 1 964 | 2 352 | 2 165 | 908 |
| test | 945 | 1 123 | 1 087 | 433 |

- **train**: real images capped per class and quarter-octave of size, with no session above 6% of a class and no
  source above 1.5× its share in val and test; then mosaics that compensate small objects, and background.
- **val**: real images following the target size distribution, plus background. Use it to choose models.
- **test**: real images with the same number of boxes per size band (uniform on a log scale), plus background. Use it
  once.
- **calib**: about 500 train images (real and background, no mosaics) balanced by class and size, for int8
  calibration (`calib.yaml`).
- **raw/**: `images/` and `labels/` of train, val and test in one folder (no calib), to redo the splits; the split of
  each image is in the manifest.
- No box is smaller than 8 px (√(w·h)). Val and test contain only sky and horizon; the train allows up to 25% per
  class of ground and wall backgrounds.
- **Known limits**: few birds against the sky (FBD-SV-2024 is mostly vegetation, so it stays in the train only);
  fixed-wing drones excluded (too few for all splits); the Anti-UAV drones added on 26/10/2026 keep the original
  labels, not reviewed.

![size distribution](distribuzione.png)

Figures (captions in Italian, see `README.it.md`): `distribuzione.png` (instances per size and class, with the
targets), `fonti.png` (share of boxes per source), `contesto.png` (share per scenario), `leakage.png` and
`leakage.md` (DINOv2 similarity of every val/test image to its nearest train image, against true leakage and the
background level: no image above 0.95).

## File names

| prefix | type | pattern | example |
|---|---|---|---|
| `R` | real image from a recording | `R_<group>_<clip>_<n>` | `R_G0000_a7bdc852_000001.jpg` |
| `M` | compensation mosaic (train only) | `M_x<grid>_<n>` | `M_x4_000012.jpg` |
| `B` | background, no labels | `B_<camera>_<n>` | `B_f88e9c93_000033.jpg` |

`<group>` is the split unit (similar recording sessions together; `GO…` Open Images, `GN…` Anti-UAV sessions added
later); `<clip>` and `<camera>` are the first 8 characters of the SHA-1 of `source:clip`. Mosaics are 2×2, 4×4 or
8×8 train crops from the same half-octave of size, scaled ×0.5, ×0.25 or ×0.125, with light blur, noise and JPEG.

## Provenance

`manifest.csv` has one row per image: `immagine` (path), `split`, `tipo` (real, mosaic, background), `gruppo`,
`fonte` (source), `sessione`, `clip`, `frame` (index in the original video), `scenario` (sky, horizon, ground, wall),
`meteo` (empty for now), `sorgente` (file in the air-detection repo).

Labels were proposed by detectors, contrast and SAM 2, then reviewed by hand in CVAT, with a second review of
candidates flagged by two independent detectors. Every operator decision is versioned in the GitHub repo.

## Sources, licenses and attribution

This dataset as a whole is released under **CC BY 4.0**. Every image keeps the license of its source (column `fonte`
of the manifest); you must respect it and give the attribution below.

| source (`fonte`) | images | content | license | attribution |
|---|---|---|---|---|
| Svanström et al. (`svanstrom`) | 6 059 | video of the 4 classes | CC0 1.0 | cite the paper |
| DVIDS (`dvids`) | 14 818 | video of airplanes, helicopters, drones, birds | public domain | see below |
| Anti-UAV300, RGB (`anti_uav`) | 3 713 | drone sequences | MIT | [`LICENSES/MIT-Anti-UAV.txt`](LICENSES/MIT-Anti-UAV.txt), cite the paper |
| FRED (`fred`) | 777 | 10 RGB drone sequences | Apache 2.0 | [`LICENSES/Apache-2.0-FRED.txt`](LICENSES/Apache-2.0-FRED.txt), cite the paper |
| Open Images V7 (`openimages`) | 1 020 | photos of birds in the sky | annotations CC BY 4.0, images CC BY 2.0 | author and page of every photo in `fonti_openimages.csv`; cite the paper |
| FBD-SV-2024 (`fbd_sv`) | 694 | bird videos | CC BY 4.0 | cite the paper |
| Clouds-1500 (`clouds1500`) | 1 756 | empty sky (background) | CC BY 4.0 | cite the dataset |
| SkyFinder (`skyfinder`) | 2 710 | outdoor webcams (background) | CC BY 4.0 | cite Mihail et al. 2016 and Nice & Wijnands 2019 |

Counts are train, val and test images, background included; the 511 mosaics (train only) are built from tiles of the real train images.

- **DVIDS**: page, author and license of every video in `fonti_dvids.csv`. "The appearance of U.S. Department of War
  (DoW) visual information does not imply or constitute DoW endorsement."
- **Download pages**: [Svanström](https://github.com/DroneDetectionThesis/Drone-detection-dataset) ·
  [DVIDS](https://www.dvidshub.net) · [Anti-UAV](https://github.com/ZhaoJ9014/Anti-UAV) ·
  [FRED](https://huggingface.co/datasets/GabrieleMagrini/FRED) ·
  [Open Images](https://storage.googleapis.com/openimages/web/index.html) ·
  [FBD-SV-2024](https://figshare.com/articles/dataset/FBD-SV-2024/27377595) ·
  [Clouds-1500](https://data.mendeley.com/datasets/2khchjbgzr/2) · [SkyFinder](https://zenodo.org/records/2562396)

## Citation

```bibtex
@misc{scolaro2026airdetection,
  author       = {Scolaro, Erik},
  title        = {air-detection: drones, airplanes, helicopters and birds against the sky},
  year         = {2026},
  howpublished = {Hugging Face},
  url          = {https://huggingface.co/datasets/Einrich99/air-detection}
}
```

Sources:

```bibtex
@article{svanstrom2021dib,
  author  = {Svanstr{\"o}m, Fredrik and Alonso-Fernandez, Fernando and Englund, Cristofer},
  title   = {A dataset for multi-sensor drone detection},
  journal = {Data in Brief},
  volume  = {39},
  pages   = {107521},
  year    = {2021},
  doi     = {10.1016/j.dib.2021.107521}
}
@article{jiang2023antiuav,
  author  = {Jiang, Nan and Wang, Kuiran and Peng, Xiaoke and Yu, Xuehui and Wang, Qiang and Xing, Junliang and others},
  title   = {Anti-{UAV}: A Large-Scale Benchmark for Vision-Based {UAV} Tracking},
  journal = {IEEE Transactions on Multimedia},
  volume  = {25},
  pages   = {486--500},
  year    = {2023},
  doi     = {10.1109/TMM.2021.3128047}
}
@inproceedings{magrini2025fred,
  author    = {Magrini, Gabriele and Marini, Niccol{\`o} and Becattini, Federico and Berlincioni, Lorenzo and
               Biondi, Niccol{\`o} and Pala, Pietro and Del Bimbo, Alberto},
  title     = {{FRED}: The Florence {RGB}-Event Drone Dataset},
  booktitle = {Proceedings of the 33rd ACM International Conference on Multimedia},
  pages     = {13170--13176},
  year      = {2025},
  doi       = {10.1145/3746027.3758271}
}
@article{kuznetsova2020openimages,
  author  = {Kuznetsova, Alina and Rom, Hassan and Alldrin, Neil and Uijlings, Jasper and Krasin, Ivan and
             Pont-Tuset, Jordi and Kamali, Shahab and Popov, Stefan and Malloci, Matteo and Kolesnikov, Alexander and
             Duerig, Tom and Ferrari, Vittorio},
  title   = {The Open Images Dataset {V4}: Unified Image Classification, Object Detection, and Visual Relationship
             Detection at Scale},
  journal = {International Journal of Computer Vision},
  volume  = {128},
  pages   = {1956--1981},
  year    = {2020},
  doi     = {10.1007/s11263-020-01316-z}
}
@article{sun2025fbdsv,
  author  = {Sun, Zi-Wei and Hua, Ze-Xi and Li, Heng-Chao and Qi, Zhi-Peng and Li, Xiang and Li, Yan and
             Zhang, Jin-Chi},
  title   = {{FBD-SV-2024}: Flying Bird Object Detection Dataset in Surveillance Video},
  journal = {Scientific Data},
  volume  = {12},
  pages   = {530},
  year    = {2025},
  doi     = {10.1038/s41597-025-04872-6}
}
@misc{arrais2023clouds1500,
  author    = {Arrais, Juliana Marian and Cerentini, Allan and Juncklaus, Bruno and Polli, Marco and Chaves, Thiago and
               von Wangenheim, Aldo and Ricci, Gilberto and Mantelli, Sylvio and Silva, Caio and Rosa, Gabriel},
  title     = {Clouds-1500},
  year      = {2023},
  publisher = {Mendeley Data},
  version   = {2},
  doi       = {10.17632/2khchjbgzr.2}
}
@inproceedings{mihail2016sky,
  author    = {Mihail, Radu Paul and Workman, Scott and Bessinger, Zach and Jacobs, Nathan},
  title     = {Sky Segmentation in the Wild: An Empirical Study},
  booktitle = {IEEE Winter Conference on Applications of Computer Vision (WACV)},
  pages     = {1--6},
  year      = {2016},
  doi       = {10.1109/WACV.2016.7477637}
}
@misc{nice2019skyfinder,
  author    = {Nice, Kerry A. and Wijnands, Jasper S.},
  title     = {Dataset for: Sky pixel detection in outdoor imagery using an adaptive algorithm and machine learning},
  year      = {2019},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.2562396}
}
```
