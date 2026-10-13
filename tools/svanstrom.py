#!/usr/bin/env python3
"""Costruisce il dataset Svanstrom (Drone-detection-dataset, solo video visibili).

Copiato da yoloeval il 13/10/2026 come punto di partenza del dataset custom:
va adattato alle regole in docs/regole.md (fasce, 16 px, box corretti, piu'
fonti, set di deploy; elenco in docs/procedura.md). Passi attuali:

1. scarica `Data/Video_V/*` e l'xlsx con classe e distance bin da GitHub;
2. per ogni video legge le label (.mat MATLAB groundTruth) e tiene solo i frame
   "nuovi" secondo un hash semantico: dHash del ritaglio attorno al bersaglio
   piu' posizione/dimensione del box. L'hash sul frame intero non funziona qui:
   il cielo domina e il bersaglio occupa pochi pixel, tutti i frame sembrano
   uguali. Un frame per cartella video in `<src>/frames/<video>/`; ogni frame
   ha una fascia small/medium/large (soglie COCO sul box piu' grande);
3. split 70/20/10 per video (mai per frame: frame dello stesso video in split
   diversi e' lo stesso leakage di AOD4), stratificato per classe e
   bilanciato sui frame di ogni fascia;
4. `<out>/{all,small,medium,large,background}/images/{train,val,test,calib}` e
   `labels/...` come hardlink ai frame, ognuno col suo `dataset.yaml` e i
   grafici di controllo in `report/`; `<out>/manifest.json` per tutto.

Le label non si leggono con scipy (oggetti MCOS) e `mcos-decoder` sbaglia i
frame con piu' oggetti (2 box letti come vuoto, 3-5 scartati, frame
disallineati): qui c'e' un walker MAT5 minimo. Ogni clip ha la sua verifica:
numero di label == numero di frame del video, altrimenti la clip e' scartata
e finisce nel manifest.

    python tools/svanstrom.py                       # tutto, in data/svanstrom
    python tools/svanstrom.py --hash-thr 6          # selezione piu' fitta
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import struct
import sys
import time
import urllib.request
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

REPO = "DroneDetectionThesis/Drone-detection-dataset"
RAW = f"https://raw.githubusercontent.com/{REPO}/master/"
XLSX = "Data/Video_dataset_description.xlsx"

NAMES = ["airplane", "bird", "drone", "helicopter"]  # stesso ordine di AOD4
SIZES = ["small", "medium", "large"]
SIZE_EDGES = [32, 96]  # sqrt(area) in px, soglie COCO
SPLITS = {"train": 0.7, "val": 0.2, "test": 0.1}

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # tools.plots anche lanciato come script


# --- download -----------------------------------------------------------------

def _fetch(url: str, tries: int = 10, delay: float = 10) -> bytes:
    """GET con retry: il proxy cade spesso e raw.githubusercontent limita le
    richieste parallele."""
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except OSError:
            if i == tries - 1:
                raise
            time.sleep(delay)


def _get(url: str, dst: Path) -> None:
    if dst.exists() and dst.stat().st_size:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    tmp.write_bytes(_fetch(url))
    tmp.rename(dst)


def download(raw: Path) -> dict[str, str]:
    """Scarica xlsx, video e label; i nomi dei file vengono dall'xlsx."""
    _get(RAW + XLSX, raw / XLSX)
    dist = distance_bins(raw / XLSX)
    paths = [f"Data/Video_V/{v}{s}" for v in dist for s in (".mp4", "_LABELS.mat")]
    with ThreadPoolExecutor(4) as ex:
        list(ex.map(lambda p: _get(RAW + p, raw / p), paths))
    print(f"download: {len(paths)} file in {raw}")
    return dist


def distance_bins(xlsx: Path) -> dict[str, str]:
    """{V_DRONE_001: 'medium', ...} dall'xlsx, senza openpyxl."""
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    z = zipfile.ZipFile(xlsx)
    strings = ["".join(si.itertext()) for si in
               ElementTree.fromstring(z.read("xl/sharedStrings.xml")).iter(ns + "si")]
    out = {}
    for row in ElementTree.fromstring(z.read("xl/worksheets/sheet1.xml")).iter(ns + "row"):
        cells = []
        for c in row.iter(ns + "c"):
            v = c.find(ns + "v")
            v = None if v is None else v.text
            cells.append(strings[int(v)] if c.get("t") == "s" and v else v)
        if len(cells) > 3 and cells[0] == "V":
            out[f"V_{cells[1]}_{int(cells[2]):03d}"] = cells[3].strip().lower()
    return out


# --- label MATLAB --------------------------------------------------------------

DTYPES = {1: "<i1", 2: "<u1", 3: "<i2", 4: "<u2", 5: "<i4", 6: "<u4",
          7: "<f4", 9: "<f8", 12: "<i8", 13: "<u8"}


def _elements(buf: bytes):
    """(tipo, payload) degli elementi MAT5 in sequenza."""
    off = 0
    while off + 8 <= len(buf):
        t, n = struct.unpack_from("<II", buf, off)
        if t >> 16:  # small data element: dimensione nei 16 bit alti
            t, n, p, off = t & 0xFFFF, t >> 16, off + 4, off + 8
        else:
            p, off = off + 8, off + 8 + n + (-n % 8)
        if not 0 < t <= 18:
            return
        yield t, buf[p:p + n]


def _matrices(buf: bytes, parent: tuple = ()):
    """(id del padre, sotto-elementi) per ogni miMATRIX, in profondita'."""
    for i, (t, payload) in enumerate(_elements(buf)):
        if t == 14:
            yield parent, list(_elements(payload))
            yield from _matrices(payload, parent + (i,))


def _box_matrix(kids) -> np.ndarray | None:
    """Matrice double kx4 (o vuota) come array (k, 4); None per tutto il resto."""
    if len(kids) < 3 or kids[0][0] != 6 or kids[1][0] != 5:
        return None
    if np.frombuffer(kids[0][1], "<u4")[0] & 0xFF != 6:  # mxDOUBLE_CLASS
        return None
    dims = np.frombuffer(kids[1][1], "<i4")
    if dims.prod() == 0:
        return np.zeros((0, 4))
    if len(dims) != 2 or dims[1] != 4 or len(kids) < 4:
        return None
    t, data = kids[3]  # MATLAB salva double interi anche come int compressi
    return np.frombuffer(data, DTYPES[t]).astype(float).reshape(4, dims[0]).T


def read_labels(mat: Path) -> list[np.ndarray] | None:
    """Per frame un array (k, 5): classe, x, y, w, h in pixel, 1-based (MATLAB).

    Le celle della timetable sono matrici double kx4 sorelle sotto lo stesso
    padre (si prende quello che ne ha di piu'), quattro per frame, una per
    colonna: AIRPLANE, BIRD, DRONE, HELICOPTER, lo stesso ordine di NAMES.
    Le celle accanto che non sono box (es. la stringa col path del video) si
    scartano. None se le celle non sono un multiplo delle colonne.
    """
    import scipy.io

    fw = scipy.io.loadmat(mat)["__function_workspace__"].tobytes()[8:]
    groups = defaultdict(list)
    for parent, kids in _matrices(fw):
        groups[parent].append(_box_matrix(kids))
    best = max(groups.values(), key=lambda g: sum(b is not None and len(b) > 0 for b in g))
    cells = [b for b in best if b is not None]
    if len(cells) % len(NAMES):
        return None
    return [np.vstack([np.column_stack([np.full(len(b), c), b])
                       for c, b in enumerate(cells[f:f + len(NAMES)])])
            for f in range(0, len(cells), len(NAMES))]


# --- selezione frame -------------------------------------------------------------

def _dhash(gray: np.ndarray) -> np.ndarray:
    import cv2

    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA).astype(np.int16)
    return (small[:, 1:] > small[:, :-1]).ravel()


def _signature(gray: np.ndarray, boxes: np.ndarray):
    """Hash del ritaglio attorno all'unione dei box + geometria normalizzata."""
    h, w = gray.shape
    x0, y0 = boxes[:, 0].min(), boxes[:, 1].min()
    x1, y1 = (boxes[:, 0] + boxes[:, 2]).max(), (boxes[:, 1] + boxes[:, 3]).max()
    pad = max(x1 - x0, y1 - y0, 8)  # contesto: lo sfondo vicino conta
    crop = gray[int(max(0, y0 - pad)):int(min(h, y1 + pad)),
                int(max(0, x0 - pad)):int(min(w, x1 + pad))]
    geo = np.array([(x0 + x1) / 2 / w, (y0 + y1) / 2 / h,
                    np.log(max(x1 - x0, 1)), np.log(max(y1 - y0, 1))])
    return _dhash(crop), geo, len(boxes)


def _is_new(sig, kept, hash_thr: int, geo_thr: float) -> bool:
    # ponytail: confronto con tutti i frame tenuti, O(n*k) per video; basta
    # finche' i video restano da ~300 frame
    hsh, geo, k = sig
    for h2, g2, k2 in kept:
        if (k == k2 and np.count_nonzero(hsh != h2) <= hash_thr
                and (geo is None or (np.hypot(*(geo[:2] - g2[:2])) <= geo_thr
                                     and np.abs(geo[2:] - g2[2:]).max() <= 0.3))):
            return False
    return True


def _mask(shape, boxes: np.ndarray, grow: float) -> np.ndarray:
    """Maschera dei box (x, y, w, h) allargati di `grow` volte il lato, min 4 px."""
    m = np.zeros(shape, np.uint8)
    for x, y, w, h in boxes:
        g = max(4, grow * max(w, h))
        m[int(max(0, y - g)):int(y + h + g), int(max(0, x - g)):int(x + w + g)] = 255
    return m


def _uniform(gray: np.ndarray, boxes: np.ndarray, std_thr: float) -> bool:
    """Sfondo attorno ai box uniforme (cielo): l'inpainting non lascia tracce."""
    ring = (_mask(gray.shape, boxes, 1.0) > 0) & (_mask(gray.shape, boxes, 0.25) == 0)
    return ring.any() and gray[ring].std() < std_thr


def _size(boxes: np.ndarray) -> str | None:
    """Fascia COCO dell'immagine dal box piu' grande; None senza box.

    Per immagine e non per box: togliere un box dalle label insegnerebbe che
    li' non c'e' niente."""
    if not len(boxes):
        return None
    return SIZES[np.digitize(np.sqrt(boxes[:, 3] * boxes[:, 4]).max(), SIZE_EDGES)]


def process_video(args) -> dict:
    """Seleziona i frame di un video e li scrive in `frames/<video>/`.

    Background: il dataset ne ha solo 82 frame in 4 video. Per averne una quota
    fissa per video (e quindi 70/20/10 come il resto) si prendono frame non
    selezionati con il bersaglio su cielo uniforme e lo si cancella con
    l'inpainting; label vuota, suffisso `_bg`.
    """
    import cv2

    video, mat, dst, hash_thr, geo_thr, bg_frac, bg_std = args
    labels = read_labels(mat)
    if labels is None:
        return {"video": video.stem, "skipped": "celle delle label non multiple di 4"}
    shutil.rmtree(dst, ignore_errors=True)
    (dst / "images").mkdir(parents=True)
    (dst / "labels").mkdir()

    def write(stem, img, text):
        cv2.imwrite(str(dst / "images" / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        (dst / "labels" / f"{stem}.txt").write_text(text)

    cap = cv2.VideoCapture(str(video))
    kept, selected, sizes, candidates, n = [], [], [], [], 0
    while (frame := cap.read()[1]) is not None:
        if n < len(labels):  # oltre si contano solo i frame, per la verifica
            boxes = labels[n].copy()
            boxes[:, 1:3] -= 1  # MATLAB 1-based
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sig = _signature(gray, boxes[:, 1:]) if len(boxes) else (_dhash(gray), None, 0)
            if _is_new(sig, kept, hash_thr, geo_thr):
                kept.append(sig)
                write(f"{video.stem}_f{n:05d}", frame,
                      _yolo(boxes, frame.shape[1], frame.shape[0]))
                selected.append(n)
                sizes.append(_size(boxes))
            elif len(boxes) and _uniform(gray, boxes[:, 1:], bg_std):
                candidates.append(n)
        n += 1
    if n != len(labels):
        shutil.rmtree(dst)
        return {"video": video.stem, "skipped": f"{len(labels)} label per {n} frame"}

    # background distribuito nel tempo tra i candidati
    k = min(len(candidates), round(bg_frac * len(selected)))
    background = sorted({candidates[i] for i in np.linspace(0, len(candidates) - 1, k).round().astype(int)}) if k else []
    cap = cv2.VideoCapture(str(video))
    for i in range(background[-1] + 1 if background else 0):
        frame = cap.read()[1]
        if i in background:
            boxes = labels[i][:, 1:].copy()
            boxes[:, :2] -= 1
            mask = _mask(frame.shape[:2], boxes, 0.25)
            write(f"{video.stem}_f{i:05d}_bg", cv2.inpaint(frame, mask, 5, cv2.INPAINT_TELEA), "")
    return {"video": video.stem, "frames": n, "selected": selected, "sizes": sizes,
            "background": background}


def extract_frames(args) -> dict:
    """Passo 2: tutti i frame di un video in `<dst>/images/*.jpg` e le label
    originali in YOLO in `<dst>/labels/*.txt`. Clip scartata se label != frame."""
    import cv2

    video, mat, dst = args
    labels = read_labels(mat)
    if labels is None:
        return {"video": video.stem, "skipped": "celle delle label non multiple di 4"}
    shutil.rmtree(dst, ignore_errors=True)
    (dst / "images").mkdir(parents=True)
    (dst / "labels").mkdir()
    cap = cv2.VideoCapture(str(video))
    n = 0
    while (frame := cap.read()[1]) is not None:
        if n < len(labels):  # oltre si contano solo i frame, per la verifica
            boxes = labels[n].copy()
            boxes[:, 1:3] -= 1  # MATLAB 1-based
            stem = f"{video.stem}_f{n:05d}"
            cv2.imwrite(str(dst / "images" / f"{stem}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            (dst / "labels" / f"{stem}.txt").write_text(_yolo(boxes, frame.shape[1], frame.shape[0]))
        n += 1
    if n != len(labels):
        shutil.rmtree(dst)
        return {"video": video.stem, "skipped": f"{len(labels)} label per {n} frame"}
    return {"video": video.stem, "frames": n}


def _hist(img: np.ndarray) -> np.ndarray:
    import cv2
    hsv = cv2.cvtColor(cv2.resize(img, (160, 128)), cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    return cv2.normalize(h, h).flatten()


def _gray(img: np.ndarray) -> np.ndarray:
    import cv2
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (80, 64), interpolation=cv2.INTER_AREA)
    g = cv2.GaussianBlur(g.astype(np.float32), (3, 3), 0)
    g -= g.mean()
    return g / (np.linalg.norm(g) + 1e-6)


def _spike(d: list[float], w: int = 5) -> np.ndarray:
    """Valore meno la mediana dei w vicini per lato: alto solo per un salto
    isolato, non per panoramiche o cambi di esposizione lenti."""
    d = np.asarray(d)
    return np.array([d[i] - np.median(np.r_[d[max(0, i - w):i], d[i + 1:i + 1 + w]])
                     for i in range(len(d))])


def cut_scores(frames: Path) -> tuple[str, list[float]]:
    """Passo 3: punteggio di taglio fra ogni frame e il precedente (indice i ->
    fra i e i+1): picco della distanza degli istogrammi HSV + picco di
    1 - correlazione in scala di grigi."""
    import cv2

    hs, gs = [], []
    for p in sorted((frames / "images").glob("*.jpg")):
        img = cv2.imread(str(p))
        hs.append(_hist(img))
        gs.append(_gray(img))
    dh = [cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA) for a, b in zip(hs, hs[1:])]
    dn = [1 - float((a * b).sum()) for a, b in zip(gs, gs[1:])]
    return frames.name, (_spike(dh) + _spike(dn)).tolist()


def cut_sheets(frames: Path, out: Path, scores: dict, thr: float, per: int = 12) -> None:
    """Candidati sopra `thr`, a basso apposta: fogli numerati per la revisione
    a mano (sinistra il frame prima, destra quello dopo) e `candidati.csv`."""
    import csv
    import cv2

    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    cand = sorted((v, i + 1, s) for v, ss in scores.items() for i, s in enumerate(ss) if s > thr)
    rows, sheet = [], []
    for n, (v, i, s) in enumerate(cand, 1):
        a, b = (cv2.resize(cv2.imread(str(frames / v / "images" / f"{v}_f{j:05d}.jpg")), (400, 320))
                for j in (i - 1, i))
        r = np.hstack([a, np.full((320, 6, 3), 255, np.uint8), b])
        r = np.vstack([r, np.full((6, r.shape[1], 3), 255, np.uint8)])
        cv2.putText(r, f"{n}. {v} {i - 1}|{i}", (8, 26), 0, 0.75, (0, 0, 255), 2)
        sheet.append(r)
        rows.append({"n": n, "video": v, "frame": i, "punteggio": round(s, 2)})
        if len(sheet) == per or n == len(cand):
            cv2.imwrite(str(out / f"foglio_{(n - 1) // per + 1:02d}.jpg"), np.vstack(sheet),
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
            sheet = []
    with open(out / "candidati.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ["n", "video", "frame", "punteggio"])
        w.writeheader()
        w.writerows(rows)
    print(f"2_sequenze: {len(rows)} candidati in {len({r['video'] for r in rows})} video, "
          f"{-(-len(rows) // per)} fogli in {out}")


def write_sequences(frames: Path, seq: Path) -> None:
    """`sequenze.csv` (video, frame, sequenza) dai numeri dei tagli veri in
    `tagli.txt`, uno per riga, scelti a mano fra quelli di `candidati.csv`."""
    import csv

    keep = {int(x) for x in (seq / "tagli.txt").read_text().split()}
    cuts = defaultdict(set)
    for r in csv.DictReader(open(seq / "candidati" / "candidati.csv")):
        if int(r["n"]) in keep:
            cuts[r["video"]].add(int(r["frame"]))  # primo frame della nuova sequenza
    n_videos = len(cuts)  # prima del ciclo: cuts[...] aggiunge chiavi vuote
    with open(seq / "sequenze.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "frame", "sequenza"])
        for d in sorted(p for p in frames.iterdir() if p.is_dir()):
            s = 0
            for i in range(len(list((d / "images").glob("*.jpg")))):
                s += i in cuts[d.name]
                w.writerow([d.name, i, s])
    print(f"2_sequenze: {len(keep)} tagli in {n_videos} video -> {seq / 'sequenze.csv'}")


def strip_frames(seq: Path, fps: int = 30) -> dict[str, list[int]]:
    """Passo 4: per clip (`video/sN`) il primo frame e poi uno al secondo."""
    import csv

    out = defaultdict(list)
    for r in csv.DictReader(open(seq / "sequenze.csv")):
        clip, i = f"{r['video']}/s{r['sequenza']}", int(r["frame"])
        if i % fps == 0 or clip not in out:
            out[clip].append(i)
    return dict(out)


def _jpg(frames: Path, clip: str, i: int) -> str:
    v = clip.split("/")[0]
    return str(frames / v / "images" / f"{v}_f{i:05d}.jpg")


def embed_strips(frames: Path, strips: dict[str, list[int]]) -> dict[str, np.ndarray]:
    """Embedding DINOv2 ViT-S/14 (token CLS, norma 1) dei frame di ogni clip:
    descrive la scena (luogo, cielo, luce), il bersaglio e' pochi pixel."""
    import cv2
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", trust_repo=True).to(dev).eval()
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    out = {}
    with torch.no_grad():
        for v, idx in strips.items():
            x = [(cv2.resize(cv2.imread(_jpg(frames, v, i)), (280, 224))[..., ::-1]
                  / 255 - mean) / std for i in idx]  # 640x512 -> 280x224, multipli di 14
            x = torch.tensor(np.stack(x).transpose(0, 3, 1, 2), dtype=torch.float32, device=dev)
            e = torch.nn.functional.normalize(model(x), dim=1)
            out[v] = e.cpu().numpy()
    return out


def propose_sessions(emb: dict[str, np.ndarray], thr: float) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Somiglianza fra clip = massimo fra le coppie dei loro frame; gruppi per
    legame medio sopra `thr`. Il legame singolo
    incatena tutto: su cielo uniforme quasi ogni coppia supera 0,9."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    vids = sorted(emb)
    sim = np.eye(len(vids))
    for a in range(len(vids)):
        for b in range(a + 1, len(vids)):
            sim[a, b] = sim[b, a] = (emb[vids[a]] @ emb[vids[b]].T).max()
    groups = fcluster(linkage(squareform(1 - sim, checks=False), "average"), 1 - thr, "distance")
    return vids, sim, groups


def session_sheets(frames: Path, out: Path, strips: dict, vids: list[str], sim: np.ndarray,
                   groups: np.ndarray, singles_per_sheet: int = 6) -> None:
    """Un foglio per gruppo (una striscia per clip, in fondo la clip piu'
    simile fuori dal gruppo) e le clip sole a fogli da `singles_per_sheet`,
    ognuna con la sua piu' simile. `proposta.csv`: clip, gruppo, vicino, somiglianza."""
    import csv
    import cv2

    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    W, H = 160, 128

    def row(v, label, color=(0, 0, 255)):
        r = np.hstack([cv2.resize(cv2.imread(_jpg(frames, v, i)), (W, H))
                       for i in strips[v]])
        r = np.vstack([np.full((28, r.shape[1], 3), 255, np.uint8), r])
        cv2.putText(r, label, (6, 21), 0, 0.65, color, 2)
        return r

    def save(rows, name):
        w = max(r.shape[1] for r in rows)
        rows = [np.pad(r, ((0, 4), (0, w - r.shape[1]), (0, 0)), constant_values=255) for r in rows]
        cv2.imwrite(str(out / name), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])

    # nomi dei gruppi nell'ordine della prima clip; clip sole a parte
    order = sorted(set(groups), key=lambda g: min(i for i in range(len(vids)) if groups[i] == g))
    multi = [g for g in order if (groups == g).sum() > 1]
    name = {g: f"g{k:02d}" for k, g in enumerate(multi, 1)}
    recs, singles = [], []
    for i, v in enumerate(vids):
        outside = np.where(groups != groups[i])[0]
        j = outside[np.argmax(sim[i, outside])]
        recs.append({"clip": v, "gruppo": name.get(groups[i], "solo"),
                     "vicino_fuori": vids[j], "somiglianza": round(float(sim[i, j]), 3)})
    for g in multi:
        idx = np.where(groups == g)[0]
        rows = [row(vids[i], f"{name[g]}  {vids[i]}") for i in idx]
        sub_ = sim[np.ix_(idx, np.setdiff1d(np.arange(len(vids)), idx))]
        a, b = np.unravel_index(sub_.argmax(), sub_.shape)
        j = np.setdiff1d(np.arange(len(vids)), idx)[b]
        rows.append(row(vids[j], f"piu' simile fuori: {vids[j]} ({recs[j]['gruppo']}, {sub_[a, b]:.2f})", (128, 128, 128)))
        save(rows, f"{name[g]}.jpg")
    for i, v in enumerate(vids):
        if recs[i]["gruppo"] == "solo":
            j = vids.index(recs[i]["vicino_fuori"])
            singles += [row(v, f"solo  {v}"),
                        row(vids[j], f"piu' simile: {vids[j]} ({recs[j]['gruppo']}, {recs[i]['somiglianza']:.2f})", (128, 128, 128))]
    for k in range(0, len(singles), 2 * singles_per_sheet):
        save(singles[k:k + 2 * singles_per_sheet], f"soli_{k // (2 * singles_per_sheet) + 1:02d}.jpg")
    with open(out.parent / "proposta.csv", "w", newline="") as f:
        w = csv.DictWriter(f, recs[0].keys())
        w.writeheader()
        w.writerows(recs)
    n_solo = sum(r["gruppo"] == "solo" for r in recs)
    print(f"3_sessioni: {len(vids)} clip, {len(multi)} gruppi, {n_solo} clip sole, "
          f"{len(list(out.glob('*.jpg')))} fogli in {out}")


def apply_sessions(ses: Path) -> None:
    """`sessioni.csv` (clip, sessione) da `proposta.csv` + `correzioni.txt`:
    gruppi proposti meno i `via`, piu' le unioni, chiuse per transitivita'."""
    import csv

    group = {r["clip"]: r["gruppo"] for r in csv.DictReader(open(ses / "proposta.csv"))}
    lines = [ln.split("#")[0].strip() for ln in (ses / "correzioni.txt").read_text().splitlines()]
    lines = [ln for ln in lines if ln]
    for ln in lines:
        if ": via " in ln:
            group[ln.split(": via ")[1]] = "solo"
    parent = {c: c for c in group}

    def find(c):
        while parent[c] != c:
            parent[c] = c = parent[parent[c]]
        return c

    def node(x):  # clip o gruppo -> una clip del gruppo
        return x if x in group else next(c for c, g in group.items() if g == x)

    for c, g in group.items():
        if g != "solo":
            parent[find(c)] = find(node(g))
    for ln in lines:
        if " con " in ln or " + " in ln:
            a, b = ln.replace(" + ", " con ").split(" con ")
            parent[find(node(a))] = find(node(b))
    roots = sorted({find(c) for c in group}, key=lambda r: min(c for c in group if find(c) == r))
    sid = {r: f"S{k:03d}" for k, r in enumerate(roots, 1)}
    with open(ses / "sessioni.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip", "sessione"])
        w.writerows((c, sid[find(c)]) for c in sorted(group))
    sizes = Counter(sid[find(c)] for c in group)
    print(f"3_sessioni: {len(group)} clip in {len(sizes)} sessioni, {sum(n == 1 for n in sizes.values())} "
          f"da una clip, la piu' grande {max(sizes.values())} clip -> {ses / 'sessioni.csv'}")


def _crop(shape, box, ctx: float = 4, min_side: int = 96):
    """Ritaglio quadrato attorno al box, `ctx` volte il lato maggiore (almeno
    `min_side`), tagliato al frame: SAM lo porta a 1024, gli oggetti di pochi
    pixel diventano grandi abbastanza da segmentarli."""
    x, y, w, h = box
    s = max(ctx * max(w, h), min_side)
    x0, y0 = int(max(0, x + w / 2 - s / 2)), int(max(0, y + h / 2 - s / 2))
    x1, y1 = int(min(shape[1], x + w / 2 + s / 2)), int(min(shape[0], y + h / 2 + s / 2))
    return x0, y0, x1, y1


def _mask_box(m: np.ndarray):
    ys, xs = np.nonzero(m)
    return None if not len(xs) else (xs.min(), ys.min(), xs.max() + 1 - xs.min(), ys.max() + 1 - ys.min())


def refine_sam(model, img: np.ndarray, box) -> tuple | None:
    """Box aderente dalla maschera SAM 2 con il box originale come prompt, sul
    ritaglio ingrandito; None se la maschera e' vuota."""
    x0, y0, x1, y1 = _crop(img.shape, box)
    x, y, w, h = box
    r = model(img[y0:y1, x0:x1], bboxes=[[x - x0, y - y0, x - x0 + w, y - y0 + h]], verbose=False)[0]
    if r.masks is None or not len(r.masks.data):
        return None
    b = _mask_box(r.masks.data[0].cpu().numpy() > 0.5)
    return None if b is None else (b[0] + x0, b[1] + y0, b[2], b[3])


def refine_contrast(img: np.ndarray, box, k: float = 4) -> tuple[tuple | None, float]:
    """Box dai pixel che si staccano dallo sfondo locale (mediana e MAD
    dell'anello attorno al box), dentro il box allargato del 20%: controllo
    indipendente da SAM, valido su cielo uniforme. Ritorna anche la MAD dello
    sfondo: alta vuol dire sfondo non uniforme (alberi, pali, nuvole)."""
    import cv2

    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    x, y, w, h = box
    inner = _crop(g.shape, box, ctx=1.2, min_side=0)
    outer = _crop(g.shape, box, ctx=2.5, min_side=24)
    ring = np.ones((outer[3] - outer[1], outer[2] - outer[0]), bool)
    ring[inner[1] - outer[1]:inner[3] - outer[1], inner[0] - outer[0]:inner[2] - outer[0]] = False
    bg = g[outer[1]:outer[3], outer[0]:outer[2]][ring]
    if bg.size < 20:
        return None, float("nan")
    med, mad = np.median(bg), np.median(np.abs(bg - np.median(bg))) * 1.4826 + 1
    m = np.abs(g[inner[1]:inner[3], inner[0]:inner[2]] - med) > k * mad
    b = _mask_box(m)
    return (None if b is None else (b[0] + inner[0], b[1] + inner[1], b[2], b[3])), float(mad)


def contrast_video(d: Path) -> list[dict]:
    """Passo 5: per ogni box di ogni frame del video il box a contrasto e la
    MAD dello sfondo; i box originali vengono da `1_frame`."""
    import cv2

    rows = []
    for lab in sorted((d / "labels").glob("*.txt")):
        lines = lab.read_text().split("\n")
        lines = [ln for ln in lines if ln.strip()]
        if not lines:
            continue
        img = cv2.imread(str(d / "images" / f"{lab.stem}.jpg"))
        H, W = img.shape[:2]
        for k, ln in enumerate(lines):
            c, x, y, w, h = map(float, ln.split())
            box = ((x - w / 2) * W, (y - h / 2) * H, w * W, h * H)
            cb, mad = refine_contrast(img, box)
            rows.append({"video": d.name, "frame": int(lab.stem.rsplit("_f", 1)[1]), "k": k, "classe": int(c),
                         "ox": box[0], "oy": box[1], "ow": box[2], "oh": box[3],
                         **dict(zip(["cx", "cy", "cw", "ch"], cb if cb else [np.nan] * 4)), "mad": mad})
    return rows


BOX_COLS = ["x", "y", "w", "h"]


def flag_boxes(rows: list[dict], mad_thr: float = 15) -> np.ndarray:
    """Box a contrasto a rischio: vuoto, sotto i 3 px, allungato oltre 3:1
    (scie), sfondo non uniforme (MAD > `mad_thr`)."""
    cw, ch, mad = (np.array([float(r[k]) for r in rows]) for k in ("cw", "ch", "mad"))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (np.isnan(cw) | (np.sqrt(cw * ch) < 3)
                | (np.maximum(cw, ch) / np.minimum(cw, ch) > 3) | (mad > mad_thr))


def fix_clip(rows: list[dict], flags: np.ndarray) -> None:
    """Dentro una clip, per ogni oggetto (indice k), i box segnalati presi per
    interpolazione lineare dai box buoni piu' vicini prima e dopo; in `fonte`
    contrasto, interpolato o originale (nessun box buono nella clip)."""
    by_k = defaultdict(list)
    for i, r in enumerate(rows):
        by_k[r["k"]].append(i)
    for idx in by_k.values():
        idx.sort(key=lambda i: int(rows[i]["frame"]))
        good = [i for i in idx if not flags[i]]
        fr = np.array([int(rows[i]["frame"]) for i in good])
        for i in idx:
            r = rows[i]
            if not flags[i]:
                r.update(zip(BOX_COLS, (float(r[k]) for k in ("cx", "cy", "cw", "ch"))), fonte="contrasto")
            elif good:
                f = int(r["frame"])
                r.update({c: float(np.interp(f, fr, [float(rows[j][k]) for j in good]))
                          for c, k in zip(BOX_COLS, ("cx", "cy", "cw", "ch"))}, fonte="interpolato")
            else:
                r.update(zip(BOX_COLS, (float(r[k]) for k in ("ox", "oy", "ow", "oh"))), fonte="originale")


def box_sheets(frames: Path, out: Path, clips: dict[str, list[dict]], per_clip: int = 6,
               per_sheet: int = 8, tile: int = 200) -> None:
    """Un foglio ogni `per_sheet` clip numerate; per clip `per_clip` frame a
    intervalli regolari (piu' un interpolato, se c'e'), ritaglio ingrandito
    attorno all'oggetto: rosso Svanstrom, blu box corretto."""
    import cv2

    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    rows_img = []
    for n, (clip, rows) in enumerate(clips.items(), 1):
        pick = [rows[int(i)] for i in np.linspace(0, len(rows) - 1, per_clip).round()]
        interp = [r for r in rows if r["fonte"] == "interpolato"]
        if interp:
            pick[-1] = interp[len(interp) // 2]
        tiles = []
        for r in pick:
            img = cv2.imread(_jpg(frames, clip, int(r["frame"])))
            o = tuple(float(r[k]) for k in ("ox", "oy", "ow", "oh"))
            x0, y0, x1, y1 = _crop(img.shape, o, ctx=3, min_side=48)
            z = tile / max(x1 - x0, y1 - y0)
            im = cv2.resize(img[y0:y1, x0:x1], None, fx=z, fy=z, interpolation=cv2.INTER_NEAREST)
            im = cv2.copyMakeBorder(im, 0, tile - im.shape[0], 0, tile - im.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))
            for b, col in [(o, (0, 0, 255)), (tuple(r[k] for k in BOX_COLS), (255, 0, 0))]:
                cv2.rectangle(im, (int((b[0] - x0) * z), int((b[1] - y0) * z)),
                              (int((b[0] + b[2] - x0) * z), int((b[1] + b[3] - y0) * z)), col, 1)
            cv2.putText(im, f"f{r['frame']}" + (" interp" if r["fonte"] == "interpolato" else ""),
                        (3, tile - 6), 0, 0.45, (255, 255, 255), 1)
            tiles.append(cv2.copyMakeBorder(im, 0, 0, 0, 4, cv2.BORDER_CONSTANT, value=(255, 255, 255)))
        head = np.full((26, len(tiles) * (tile + 4), 3), 255, np.uint8)
        cv2.putText(head, f"{n}. {clip}", (4, 19), 0, 0.6, (0, 0, 255), 2)
        rows_img.append(np.vstack([head, np.hstack(tiles), np.full((6, head.shape[1], 3), 255, np.uint8)]))
        if len(rows_img) == per_sheet or n == len(clips):
            cv2.imwrite(str(out / f"foglio_{(n - 1) // per_sheet + 1:02d}.jpg"), np.vstack(rows_img),
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
            rows_img = []


def sam_clip(model, frames: Path, clip: str, rows: list[dict]) -> list:
    """Box SAM 2 per ogni box della clip, con un punto come prompt: centro del
    box a contrasto se c'e', altrimenti del box originale. Il box largo come
    prompt non serve: SAM ridà il box stesso."""
    import cv2

    out = []
    for r in rows:
        img = cv2.imread(_jpg(frames, clip, int(r["frame"])))
        o = tuple(float(r[k]) for k in ("ox", "oy", "ow", "oh"))
        c = [float(r[k]) for k in ("cx", "cy", "cw", "ch")]
        px, py = (c[0] + c[2] / 2, c[1] + c[3] / 2) if not np.isnan(c[0]) else (o[0] + o[2] / 2, o[1] + o[3] / 2)
        x0, y0, x1, y1 = _crop(img.shape, o)
        res = model(img[y0:y1, x0:x1], points=[[px - x0, py - y0]], labels=[1], verbose=False)[0]
        b = None
        if res.masks is not None and len(res.masks.data):
            b = _mask_box(res.masks.data[0].cpu().numpy() > 0.5)
        out.append(None if b is None else [float(b[0] + x0), float(b[1] + y0), float(b[2]), float(b[3])])
    return out


def _yolo(boxes: np.ndarray, W: int, H: int) -> str:
    """(classe, x, y, w, h) in pixel 0-based -> righe YOLO, box tagliati al frame."""
    lines = []
    for c, x, y, w, h in boxes:
        x0, y0 = max(x, 0), max(y, 0)
        x1, y1 = min(x + w, W), min(y + h, H)
        if x1 - x0 < 1 or y1 - y0 < 1:
            continue
        lines.append(f"{int(c)} {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} "
                     f"{(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}\n")
    return "".join(lines)


# --- split ---------------------------------------------------------------------

def assign_splits(videos: list[dict], seed: int) -> None:
    """Split per video, per classe, bilanciato sui frame di ogni fascia.

    Greedy in ordine casuale: ogni video va allo split piu' indietro rispetto
    al target sul suo mix di fasce (frame small/medium/large, pesati per la
    loro frazione del totale della classe). Non dal piu' pesante: i video
    grossi finirebbero tutti in train e a val/test resterebbero solo le clip
    statiche (pochi frame selezionati).
    """
    rng = random.Random(seed)
    strata = defaultdict(list)
    for v in videos:
        strata[v["class"]].append(v)
    for group in strata.values():
        rng.shuffle(group)
        counts = [Counter(v["sizes"]) for v in group]
        tot = sum(counts, Counter())
        got = {s: Counter() for s in SPLITS}
        for v, c in zip(group, counts):
            v["split"] = max(SPLITS, key=lambda s: sum(
                k * (SPLITS[s] * tot[b] - got[s][b]) / tot[b] for b, k in c.items()))
            got[v["split"]].update(c)


# --- costruzione ---------------------------------------------------------------

def build(out: Path, frames: Path, videos: list[dict], n_calib: int, seed: int,
          keep=lambda stem: True) -> None:
    """Hardlink ai frame dei video nei loro split, solo quelli con keep(stem)."""
    for d in ("images", "labels"):
        shutil.rmtree(out / d, ignore_errors=True)
    files = defaultdict(list)
    for v in videos:
        src = frames / v["video"]
        for img in sorted((src / "images").glob("*.jpg")):
            if keep(img.stem):
                files[v["split"]].append((img, src / "labels" / f"{img.stem}.txt"))
    # ponytail: calib e' un campione delle immagini di train; disgiunto da
    # val/test, che e' quello che conta per non calibrare sui dati di valutazione
    files["calib"] = random.Random(seed).sample(files["train"],
                                                min(n_calib, len(files["train"])))
    for split, items in files.items():
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)
        for img, lab in items:
            os.link(img, out / "images" / split / img.name)
            os.link(lab, out / "labels" / split / lab.name)
    (out / "dataset.yaml").write_text(
        "train: images/train\nval: images/val\ntest: images/test\n"
        "calib: images/calib    # solo calibrazione INT8, campione di train\n\n"
        f"nc: {len(NAMES)}\nnames:\n"
        + "".join(f"    {i}: {n}\n" for i, n in enumerate(NAMES)))


# --- report --------------------------------------------------------------------

def report(out: Path, videos: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from tools.plots import GRAY, HALF, STYLE, WIDTH, color, save

    plt.style.use(STYLE)
    splits = ["train", "val", "test", "calib"]
    rows = []  # (split, classe, x, y, w, h) in pixel, per ogni box
    images, empty = Counter(), Counter()
    W, H = 640, 512  # tutti i video visibili
    for split in splits:
        for lab in (out / "labels" / split).glob("*.txt"):
            lines = lab.read_text().split()
            images[split] += 1
            empty[split] += not lines
            for i in range(0, len(lines), 5):
                c, x, y, w, h = map(float, lines[i:i + 5])
                rows.append((split, NAMES[int(c)], x * W, y * H, w * W, h * H))
    rep = out / "report"
    shutil.rmtree(rep, ignore_errors=True)  # niente grafici di build precedenti
    cats = NAMES + ["background"]  # background: immagini senza oggetti
    x = np.arange(len(cats))

    # 1. immagini per split
    fig, ax = plt.subplots(figsize=(HALF * 1.4, 2.6))
    pos = [images[s] - empty[s] for s in splits]
    ax.bar(splits, pos, color=[color(s) for s in splits], width=0.6)
    ax.bar(splits, [empty[s] for s in splits], bottom=pos, color=GRAY,
           width=0.6, label="senza oggetti")
    for i, s in enumerate(splits):
        ax.text(i, images[s], f"{images[s]}", ha="center", va="bottom")
    ax.set_ylabel("immagini")
    ax.set_title("Immagini per split")
    ax.legend()
    save(fig, rep / "images_per_split")

    # 2. classi per split: assolute e quota di ogni classe per split
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(WIDTH, 2.8))
    cnt = Counter((r[0], r[1]) for r in rows)
    cnt.update({(s, "background"): empty[s] for s in splits})
    for i, s in enumerate(splits):
        a1.bar(x + (i - 1.5) * 0.2, [cnt[s, c] for c in cats], 0.18,
               color=color(s), label=s)
    a1.set_xticks(x, cats, rotation=25, ha="right")
    a1.set_ylabel("istanze (background: immagini)")
    a1.set_title("Istanze per classe")
    a1.legend()
    for i, s in enumerate(splits[:3]):
        tot = [sum(cnt[t, c] for t in splits[:3]) or 1 for c in cats]
        a2.bar(x + (i - 1) * 0.25, [100 * cnt[s, c] / t for c, t in zip(cats, tot)],
               0.23, color=color(s), label=s)
        a2.axhline(100 * SPLITS[s], color=color(s), ls="--", lw=1)
    a2.set_xticks(x, cats, rotation=25, ha="right")
    a2.set_ylabel("% della classe nello split")
    a2.set_title("Quota per split (- - target)")
    save(fig, rep / "classes_per_split")
    print(f"report: {rep}")
    for s in splits:
        print(f"  {s:5s} {images[s]:6d} immagini, {empty[s]} senza oggetti")
    if not rows:  # sottoinsieme di solo background: niente box ne' strati
        return

    # 3. box per split, come labels.jpg di Ultralytics: un campione di box
    # centrati, densita' dei centri, densita' di larghezza x altezza (log: i
    # box sono di poche decine di pixel su 640)
    from matplotlib.patches import Rectangle

    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(3, 3, figsize=(WIDTH, 0.85 * WIDTH))
    wh_bins = np.logspace(np.log10(4), np.log10(W), 40)
    for (a0, a1, a2), s in zip(axes, SPLITS):
        b = np.array([r[2:] for r in rows if r[0] == s])
        for x, y, w, h in b[rng.choice(len(b), min(400, len(b)), replace=False)]:
            a0.add_patch(Rectangle((W / 2 - w / 2, H / 2 - h / 2), w, h, fill=False,
                                   lw=0.4, alpha=0.3, color=color(s)))
        a0.set(xlim=(0, W), ylim=(H, 0), aspect="equal")
        a0.set_ylabel(f"{s}\n{len(b)} box")
        a1.hist2d(b[:, 0], b[:, 1], bins=(32, 26), range=((0, W), (0, H)),
                  cmin=1, norm="log")
        a1.set(xlim=(0, W), ylim=(H, 0), aspect="equal")
        a2.hist2d(b[:, 2], b[:, 3], bins=(wh_bins, wh_bins), cmin=1, norm="log")
        a2.plot([4, W], [4, W], "--", color=GRAY, lw=0.8)  # box quadrati
        a2.set(xscale="log", yscale="log", aspect="equal")
        for ax in (a0, a1, a2):
            ax.grid(False)
        for ax in (a0, a1):  # bordo intero: e' il frame 640x512
            ax.spines[:].set_visible(True)
    axes[0, 0].set_title("box (campione, centrati) [px]")
    axes[0, 1].set_title("centri [px]")
    axes[0, 2].set_title("larghezza × altezza [px]")
    axes[2, 0].set_xlabel("x")
    axes[2, 1].set_xlabel("x")
    axes[2, 2].set_xlabel("w")
    for r in range(3):
        axes[r, 1].set_ylabel("y")
        axes[r, 2].set_ylabel("h")
    save(fig, rep / "bbox_per_split")

    if not videos:  # sottoinsiemi per fascia: la tabella e' quella di `all`
        return

    # 4. classe x fascia: quota dei frame di ogni cella per split
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.4))
    for ax, s in zip(axes, SPLITS):
        m = np.zeros((len(NAMES), len(SIZES)))
        tot = np.zeros_like(m)
        for v in videos:
            i = NAMES.index(v["class"])
            for z in v["sizes"]:
                if z is not None:
                    tot[i, SIZES.index(z)] += 1
                    m[i, SIZES.index(z)] += v["split"] == s
        pct = 100 * m / np.maximum(tot, 1)
        ax.imshow(pct, vmin=0, vmax=100)
        for i in range(len(NAMES)):
            for j in range(len(SIZES)):
                ax.text(j, i, f"{pct[i, j]:.0f}%\n{int(m[i, j])}", ha="center",
                        va="center", fontsize=7,
                        color="white" if pct[i, j] > 55 else "#222")
        ax.set_xticks(range(len(SIZES)), SIZES, rotation=30)
        ax.set_yticks(range(len(NAMES)), NAMES if ax is axes[0] else [])
        ax.grid(False)
        ax.set_title(f"{s} (target {SPLITS[s]:.0%})")
    fig.suptitle("Frame di ogni cella classe × fascia per split (quota e numero)",
                 fontsize=10)
    save(fig, rep / "class_size_per_split")


# --- main ----------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", type=Path, default=ROOT / "data" / "svanstrom",
                   help="dataset finale (local_path in conf/dataset/svanstrom.yaml)")
    p.add_argument("--src", type=Path, default=ROOT / "data" / "svanstrom",
                   help="una cartella per passo (docs/procedura.md): 0_raw, 1_frame, ...")
    p.add_argument("--step", choices=["frames", "tagli", "sequenze", "sessioni", "unione", "box", "sam"],
                   help="solo questo passo della nuova procedura; senza, la pipeline vecchia")
    p.add_argument("--hash-thr", type=int, default=8,
                   help="bit di dHash (su 64) sotto cui due frame sono uguali")
    p.add_argument("--geo-thr", type=float, default=0.05,
                   help="spostamento del centro (frazione del frame) sotto cui "
                        "il box e' fermo")
    p.add_argument("--bg-frac", type=float, default=0.1,
                   help="frame di background per video, in proporzione ai selezionati")
    p.add_argument("--bg-std", type=float, default=8.0,
                   help="deviazione standard massima (livelli di grigio) dello "
                        "sfondo attorno al box per cancellarlo con l'inpainting")
    p.add_argument("--cut-thr", type=float, default=0.07,
                   help="punteggio sopra cui una coppia di frame e' candidata taglio")
    p.add_argument("--session-thr", type=float, default=0.95,
                   help="somiglianza media DINOv2 sopra cui due gruppi di video si uniscono")
    p.add_argument("--n-calib", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    a = p.parse_args(argv)

    raw = a.src / "0_raw" / "Data" / "Video_V"
    dist = download(a.src / "0_raw")
    if a.step == "frames":
        jobs = [(raw / f"{v}.mp4", raw / f"{v}_LABELS.mat", a.src / "1_frame" / v) for v in sorted(dist)]
        with ProcessPoolExecutor(a.workers) as ex:
            results = list(ex.map(extract_frames, jobs))
        (a.src / "1_frame" / "frame.json").write_text(json.dumps(results, indent=1))
        skipped = [r for r in results if "skipped" in r]
        print(f"1_frame: {len(results) - len(skipped)} video, "
              f"{sum(r.get('frames', 0) for r in results)} frame")
        for s in skipped:
            print(f"  scartato {s['video']}: {s['skipped']}")
        return 0
    if a.step == "box":
        import csv
        box = a.src / "4_box"
        box.mkdir(exist_ok=True)
        with ProcessPoolExecutor(a.workers) as ex:
            rows = [r for rs in ex.map(contrast_video, sorted(p for p in (a.src / "1_frame").iterdir()
                                                               if p.is_dir())) for r in rs]
        seq = {(r["video"], int(r["frame"])): r["sequenza"]
               for r in csv.DictReader(open(a.src / "2_sequenze" / "sequenze.csv"))}
        clips = defaultdict(list)
        for r, f in zip(rows, flag_boxes(rows)):
            r["clip"], r["segnalato"] = f"{r['video']}/s{seq[(r['video'], r['frame'])]}", bool(f)
            clips[r["clip"]].append(r)
        kind = {c: "pulita" if np.mean([r["segnalato"] for r in rs]) <= 0.1 else "difficile"
                for c, rs in clips.items()}
        for c, rs in clips.items():
            fix_clip(rs, np.array([r["segnalato"] for r in rs]))
            for r in rs:
                r["tipo_clip"] = kind[c]
        with open(box / "box.csv", "w", newline="") as f:
            w = csv.DictWriter(f, rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        clean = {c: rs for c, rs in sorted(clips.items()) if kind[c] == "pulita"}
        box_sheets(a.src / "1_frame", box / "fogli_pulite", clean)
        print(f"4_box: {len(rows)} box, {len(clean)} clip pulite e {len(clips) - len(clean)} difficili; "
              f"fogli in {box / 'fogli_pulite'}")
        return 0
    if a.step == "sam":
        # clip difficili + pulite segnate sbagliate in revisione; un json per clip,
        # cosi' si riprende da dove si era fermi
        import csv
        from ultralytics import SAM
        box = a.src / "4_box"
        bad = {ln.split()[1] for ln in open(box / "pulite_sbagliate.txt") if not ln.startswith("#")}
        clips = defaultdict(list)
        for r in csv.DictReader(open(box / "box.csv")):
            if r["tipo_clip"] == "difficile" or r["clip"] in bad:
                clips[r["clip"]].append(r)
        (box / "sam").mkdir(exist_ok=True)
        model = SAM(str(ROOT / "data" / "modelli" / "sam2.1_l.pt"))
        t0 = time.time()
        for n, (clip, rows) in enumerate(sorted(clips.items()), 1):
            dst = box / "sam" / f"{clip.replace('/', '_')}.json"
            if not dst.exists():
                dst.write_text(json.dumps({"frame": [int(r["frame"]) for r in rows], "k": [r["k"] for r in rows],
                                           "box": sam_clip(model, a.src / "1_frame", clip, rows)}))
            print(f"{n}/{len(clips)} {clip} {time.time() - t0:.0f}s", flush=True)
        return 0
    if a.step == "unione":
        apply_sessions(a.src / "3_sessioni")
        return 0
    if a.step == "sessioni":
        ses = a.src / "3_sessioni"
        ses.mkdir(exist_ok=True)
        strips = strip_frames(a.src / "2_sequenze")
        cache = ses / "embedding.npz"  # il modello gira una volta; la soglia si cambia al volo
        if cache.exists() and set(np.load(cache).files) != set(strips):
            cache.unlink()  # strisce cambiate (tagli diversi)
        emb = dict(np.load(cache)) if cache.exists() else embed_strips(a.src / "1_frame", strips)
        np.savez(cache, **emb)
        vids, sim, groups = propose_sessions(emb, a.session_thr)
        session_sheets(a.src / "1_frame", ses / "fogli", strips, vids, sim, groups)
        return 0
    if a.step == "sequenze":
        write_sequences(a.src / "1_frame", a.src / "2_sequenze")
        return 0
    if a.step == "tagli":
        frames = a.src / "1_frame"
        with ProcessPoolExecutor(a.workers) as ex:
            scores = dict(ex.map(cut_scores, sorted(p for p in frames.iterdir() if p.is_dir())))
        (a.src / "2_sequenze").mkdir(exist_ok=True)
        (a.src / "2_sequenze" / "punteggi.json").write_text(json.dumps(scores))
        cut_sheets(frames, a.src / "2_sequenze" / "candidati", scores, a.cut_thr)
        return 0
    jobs = [(raw / f"{v}.mp4", raw / f"{v}_LABELS.mat", a.src / "frames" / v,
             a.hash_thr, a.geo_thr, a.bg_frac, a.bg_std) for v in sorted(dist)]
    with ProcessPoolExecutor(a.workers) as ex:
        results = list(ex.map(process_video, jobs))

    skipped = [r for r in results if "skipped" in r]
    videos = [{**r, "class": r["video"].split("_")[1].lower(), "distance": dist[r["video"]]}
              for r in results if "skipped" not in r]
    assign_splits(videos, a.seed)
    # oltre a `all` (tutto), un dataset per fascia (immagini con oggetti) e uno
    # di solo background, da aggiungere a mano: lo split bilancia anche le
    # fasce e il background segue il video, quindi ognuno resta 70/20/10
    size_of = {f"{v['video']}_f{n:05d}": z
               for v in videos for n, z in zip(v["selected"], v["sizes"])}
    subsets = {"all": lambda st: True, "background": lambda st: st.endswith("_bg")} | {
        z: (lambda st, z=z: size_of.get(st) == z) for z in SIZES}
    for name, keep in subsets.items():
        build(a.out / name, a.src / "frames", videos, a.n_calib, a.seed, keep)

    sel = sum(len(v["selected"]) for v in videos)
    bg = sum(len(v["background"]) for v in videos)
    tot = sum(v["frames"] for v in videos)
    (a.out / "manifest.json").write_text(json.dumps({
        "source": f"https://github.com/{REPO}", "sensor": "visible",
        "params": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
        "selected_frames": sel, "background_frames": bg, "total_frames": tot,
        "skipped": skipped, "videos": videos}, indent=1))
    print(f"video: {len(videos)} usati, {len(skipped)} scartati; "
          f"frame: {sel} selezionati su {tot} ({sel / max(tot, 1):.1%}), "
          f"{bg} di background con inpainting")
    for s in skipped:
        print(f"  scartato {s['video']}: {s['skipped']}")
    for name in subsets:
        report(a.out / name, videos if name == "all" else [])
    return 0


if __name__ == "__main__":
    sys.exit(main())
