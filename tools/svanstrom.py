#!/usr/bin/env python3
"""Svanström (Drone-detection-dataset, CC0, solo i video visibili 640x512): un `--step` per passo di
`docs/procedura.md` (passi 2-8), una cartella per passo in `data/svanstrom/`. Elenco dei comandi nel README.

Le label non si leggono con scipy (oggetti MCOS) e `mcos-decoder` sbaglia i frame con piu' oggetti: qui c'e' un
lettore MAT5 minimo; clip scartata se il numero di label e' diverso dal numero di frame.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # anche lanciato come script
from tools.common import (BAND_EDGES, BANDS, NAMES, ROOT, cvat_client, dedup_clip, mask_box,  # noqa: E402
                          propose_sessions)

REPO = "DroneDetectionThesis/Drone-detection-dataset"
RAW = f"https://raw.githubusercontent.com/{REPO}/master/"
XLSX = "Data/Video_dataset_description.xlsx"


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


# --- meteo --------------------------------------------------------------------

WEATHER = {  # piu' frasi per classe, media degli embedding di testo; nomi = tag in CVAT
    "sole": ["a photo of a clear blue sky on a sunny day", "a photo taken on a bright sunny day"],
    "nuvole": ["a photo of an overcast grey sky", "a photo of a cloudy sky on a dull day"],
    "pioggia": ["a photo taken in the rain", "a photo of a rainy day with rain falling"],
    "neve": ["a photo taken while it is snowing", "a photo of a snowy winter day"],
    "nebbia": ["a photo taken in thick fog", "a photo of a hazy misty sky with low visibility"],
}
CLIP_MODEL = ("ViT-SO400M-14-SigLIP-384", "webli")  # open_clip


def weather_probs(frames: Path, strips: dict[str, list[int]]) -> dict[str, np.ndarray]:
    """Probabilita' per classe di WEATHER di ogni clip: SigLIP zero-shot sui
    frame della striscia, softmax sui logit, media sui frame."""
    import open_clip
    import torch
    from PIL import Image

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, prep = open_clip.create_model_and_transforms(*CLIP_MODEL, device=dev)
    tok, norm = open_clip.get_tokenizer(CLIP_MODEL[0]), torch.nn.functional.normalize
    out = {}
    with torch.no_grad():
        t = norm(model.eval().encode_text(tok(sum(WEATHER.values(), [])).to(dev)), dim=1)
        t = norm(t.view(len(WEATHER), -1, t.shape[1]).mean(1), dim=1)
        for clip, idx in strips.items():
            x = torch.stack([prep(Image.open(_jpg(frames, clip, i)).convert("RGB")) for i in idx]).to(dev)
            p = torch.softmax(model.logit_scale.exp() * norm(model.encode_image(x), dim=1) @ t.T, dim=1)
            out[clip] = p.mean(0).float().cpu().numpy()
    return out


def weather_proposal(strips: dict, probs: dict, session: dict) -> list[dict]:
    """Un frame per clip (quello a meta' striscia), ordinate per classe
    proposta e dalla meno sicura; n = posizione del frame nel task CVAT."""
    names = list(WEATHER)
    recs = [{"clip": c, "frame": strips[c][len(strips[c]) // 2], "sessione": session.get(c, ""),
             "meteo": names[int(p.argmax())], "confidenza": round(float(p.max()), 3)}
            | {k: round(float(v), 3) for k, v in zip(names, p)} for c, p in probs.items()]
    recs.sort(key=lambda r: (names.index(r["meteo"]), r["confidenza"]))
    for n, r in enumerate(recs):
        r["n"] = n
    return recs


def cvat_weather(met: Path, recs: list[dict], name: str = "meteo") -> None:
    """`proposta.csv` e task CVAT `name` per la revisione: un frame per clip
    nell'ordine della proposta, la classe proposta come tag del frame; si
    correggono i tag. Se il task c'e' gia' non tocca niente: la proposta
    deve restare quella del task."""
    import csv
    from cvat_sdk.core.proxies.tasks import ResourceType
    from cvat_sdk.api_client import models

    with cvat_client() as client:
        if any(tk.name == name for tk in client.tasks.list()):
            raise SystemExit(f"meteo: il task '{name}' esiste gia' in CVAT, cancellarlo a mano per rifarlo")
        with open(met / "proposta.csv", "w", newline="") as f:
            w = csv.DictWriter(f, ["n", "clip", "frame", "sessione", "meteo", "confidenza", *WEATHER])
            w.writeheader()
            w.writerows(recs)
        v = lambda r: r["clip"].split("/")[0]
        task = client.tasks.create_from_data(
            spec={"name": name, "labels": [{"name": c, "type": "tag"} for c in WEATHER]},
            resource_type=ResourceType.SHARE,
            resources=[f"1_frame/{v(r)}/images/{v(r)}_f{r['frame']:05d}.jpg" for r in recs],
            data_params={"sorting_method": "predefined", "image_quality": 95})
        label = {lb.name: lb.id for lb in task.get_labels()}
        task.update_annotations(models.PatchedLabeledDataRequest(
            tags=[models.LabeledImageRequest(frame=r["n"], label_id=label[r["meteo"]]) for r in recs]))
    print(f"meteo: {len(recs)} clip, " + ", ".join(f"{c} {sum(r['meteo'] == c for r in recs)}" for c in WEATHER)
          + f"; task CVAT {task.id}, proposta in {met / 'proposta.csv'}")


def weather_import(met: Path, name: str = "meteo") -> None:
    """`meteo.csv` (clip, sessione, meteo, modello) dai tag del task CVAT `name`;
    frame senza tag o con piu' tag segnalati e lasciati vuoti."""
    import csv

    recs = {int(r["n"]): r for r in csv.DictReader(open(met / "proposta.csv"))}
    with cvat_client() as client:
        task = next(tk for tk in client.tasks.list() if tk.name == name)
        label = {lb.id: lb.name for lb in task.get_labels()}
        tags = defaultdict(set)
        for t in task.get_annotations().tags:
            tags[t.frame].add(label[t.label_id])
    bad = [r["clip"] for n, r in recs.items() if len(tags[n]) != 1]
    with open(met / "meteo.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip", "sessione", "meteo", "modello"])
        w.writerows((r["clip"], r["sessione"], next(iter(tags[n])) if len(tags[n]) == 1 else "", r["meteo"])
                    for n, r in sorted(recs.items(), key=lambda x: x[1]["clip"]))
    count = Counter(next(iter(tags[n])) for n in recs if len(tags[n]) == 1)
    changed = sum(tags[n] != {r["meteo"]} for n, r in recs.items() if len(tags[n]) == 1)
    print(f"meteo: {len(recs)} clip, {changed} cambiate in CVAT; " + ", ".join(f"{c} {count[c]}" for c in WEATHER)
          + f" -> {met / 'meteo.csv'}")
    for c in bad:
        print(f"  senza un tag solo: {c}")


def _crop(shape, box, ctx: float = 4, min_side: int = 96):
    """Ritaglio quadrato attorno al box, `ctx` volte il lato maggiore (almeno
    `min_side`), tagliato al frame: SAM lo porta a 1024, gli oggetti di pochi
    pixel diventano grandi abbastanza da segmentarli."""
    x, y, w, h = box
    s = max(ctx * max(w, h), min_side)
    x0, y0 = int(max(0, x + w / 2 - s / 2)), int(max(0, y + h / 2 - s / 2))
    x1, y1 = int(min(shape[1], x + w / 2 + s / 2)), int(min(shape[0], y + h / 2 + s / 2))
    return x0, y0, x1, y1


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
    b = mask_box(m)
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
            b = mask_box(res.masks.data[0].cpu().numpy() > 0.5)
        out.append(None if b is None else [float(b[0] + x0), float(b[1] + y0), float(b[2]), float(b[3])])
    return out


def clip_boxes(box: Path, clip: str) -> dict[tuple, list]:
    """Box attuali di una clip, {(frame, k): [classe, x, y, w, h]}: quelli a
    mano se la clip e' in `manuale/` (k = riga del file), altrimenti SAM se la
    clip e' passata da SAM, altrimenti quello di `box.csv`."""
    import csv

    man = box / "manuale" / clip.replace("/", "_")
    if man.exists():
        out = {}
        for t in man.glob("*.txt"):
            f = int(t.stem.rsplit("_f", 1)[1])
            for k, ln in enumerate(t.read_text().splitlines()):
                c, xc, yc, w, h = map(float, ln.split())
                out[(f, str(k))] = [int(c), (xc - w / 2) * 640, (yc - h / 2) * 512, w * 640, h * 512]
        return out
    rows = {(int(r["frame"]), r["k"]): r for r in csv.DictReader(open(box / "box.csv")) if r["clip"] == clip}
    out = {key: [int(r["classe"])] + [float(r[c]) for c in BOX_COLS] for key, r in rows.items()}
    sam = box / "sam" / f"{clip.replace('/', '_')}.json"
    if sam.exists():
        d = json.loads(sam.read_text())
        for f, k, b in zip(d["frame"], d["k"], d["box"]):
            if b:
                out[(f, k)][1:] = b
    return out


def cvat_export(src: Path, clips: list[str], name: str = "box", job_frames: int = 10_000,
                per_patch: int = 20) -> None:
    """Un task CVAT `name`, clip una dopo l'altra in job di clip intere da al
    piu' `job_frames` frame, immagini dalla cartella condivisa (`src` montata
    in /home/django/share), box attuali come tracce, una per oggetto e clip,
    con ogni frame chiave: l'interpolazione di CVAT sbagliava sugli elicotteri."""
    import csv
    from cvat_sdk.core.proxies.tasks import ResourceType
    from cvat_sdk.api_client import models

    frames = defaultdict(list)
    for r in csv.DictReader(open(src / "2_sequenze" / "sequenze.csv")):
        frames[f"{r['video']}/s{r['sequenza']}"].append(int(r["frame"]))
    jobs, req, start = [], [], 0  # immagini per job, tracce per clip
    total = sum(len(frames[c]) for c in clips)
    for clip in clips:
        v, fr = clip.split("/")[0], sorted(frames[clip])
        if not jobs or len(jobs[-1]) + len(fr) > job_frames:
            jobs.append([])
        jobs[-1] += [f"1_frame/{v}/images/{v}_f{i:05d}.jpg" for i in fr]
        pos = {f: start + j for j, f in enumerate(fr)}
        start += len(fr)
        tracks, req = defaultdict(dict), req + [[]]
        for (f, k), (c, x, y, w, h) in clip_boxes(src / "4_box", clip).items():
            tracks[(k, c)][pos[f]] = [x, y, x + w, y + h]
        for (k, c), shapes in tracks.items():
            seq, prev = [], None
            for j in range(min(shapes), max(shapes) + 2):
                if j in shapes:
                    seq.append(models.TrackedShapeRequest(type="rectangle", frame=j, points=shapes[j],
                                                          outside=False, keyframe=True))
                elif prev is not None and j < total:  # sparito, o clip finita: fuori, se no CVAT la allunga
                    seq.append(models.TrackedShapeRequest(type="rectangle", frame=j, points=shapes[prev],
                                                          outside=True, keyframe=True))
                prev = j if j in shapes else None
            req[-1].append((NAMES[c], min(shapes), seq))
    with cvat_client() as client:
        if any(tk.name == name for tk in client.tasks.list()):
            raise SystemExit(f"cvat: il task '{name}' esiste gia', cancellarlo a mano per rifarlo")
        task = client.tasks.create_from_data(
            spec={"name": name, "labels": [{"name": n} for n in NAMES]},
            resource_type=ResourceType.SHARE, resources=sum(jobs, []),
            data_params={"image_quality": 95, "job_file_mapping": jobs})  # ordine = quello dei job
        label = {lb.name: lb.id for lb in task.get_labels()}
        # un PATCH ogni `per_patch` clip: uno solo per tutto il task va in timeout (504)
        for k in range(0, len(req), per_patch):
            task.update_annotations(models.PatchedLabeledDataRequest(tracks=[
                models.LabeledTrackRequest(frame=f, label_id=label[c], shapes=seq)
                for tr in req[k:k + per_patch] for c, f, seq in tr]))
            print(f"cvat: {name} clip {min(k + per_patch, len(req))}/{len(req)}", flush=True)
    print(f"cvat: task {name} ({task.id}), {len(clips)} clip in {len(jobs)} job, {start} frame", flush=True)


def cvat_import(src: Path) -> None:
    """Correzioni a mano: gli zip esportati da CVAT ("Ultralytics YOLO
    Detection 1.0", con o senza immagini) lasciati in `4_box/cvat_zip/`. Le
    label vanno in `4_box/manuale/<video>_sN/<frame>.txt`, la clip ricavata dal
    nome del frame; uno zip piu' recente sovrascrive i frame di uno vecchio."""
    import csv
    import zipfile

    box = src / "4_box"
    seq = {(r["video"], int(r["frame"])): r["sequenza"] for r in csv.DictReader(open(src / "2_sequenze" / "sequenze.csv"))}
    for z in sorted((box / "cvat_zip").glob("*.zip"), key=lambda p: p.stat().st_mtime):
        clips = Counter()
        with zipfile.ZipFile(z) as zf:
            for n in zf.namelist():
                if n.startswith("labels/") and n.endswith(".txt"):
                    stem = Path(n).stem
                    v, f = stem.rsplit("_f", 1)
                    d = box / "manuale" / f"{v}_s{seq[(v, int(f))]}"
                    d.mkdir(parents=True, exist_ok=True)
                    (d / f"{stem}.txt").write_bytes(zf.read(n))
                    clips[d.name] += 1
        print(f"cvat: {z.name} -> " + ", ".join(f"{c} ({n} frame)" for c, n in clips.items()))


# --- quasi-duplicati -------------------------------------------------------------

def dedup(src: Path, hash_thr: int, geo_thr: float, workers: int) -> None:
    """Passo 7: quasi-duplicati dentro ogni clip sulle label finali
    (`4_box/manuale/`, meno scie e scartate) -> `6_dedup/tenuti.csv`
    (clip, frame) e il riepilogo per classe."""
    import csv

    box = src / "4_box"
    skip = {ln.strip() for f in ("scie.txt", "scartate.txt") for ln in open(box / f)
            if ln.strip() and not ln.startswith("#")}
    idx = defaultdict(list)
    for r in csv.DictReader(open(src / "2_sequenze" / "sequenze.csv")):
        idx[f"{r['video']}/s{r['sequenza']}"].append(int(r["frame"]))
    jobs = [(src / "1_frame", box / "manuale" / c.replace("/", "_"), c, sorted(f), hash_thr, geo_thr)
            for c, f in sorted(idx.items()) if c not in skip]
    out = src / "6_dedup"
    out.mkdir(exist_ok=True)
    got, tot = Counter(), Counter()
    with ProcessPoolExecutor(workers) as ex, open(out / "tenuti.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip", "frame"])
        for clip, keep, n in ex.map(dedup_clip, jobs):
            w.writerows((clip, i) for i in keep)
            cls = clip.split("_")[1].lower()
            got[cls] += len(keep)
            tot[cls] += n
    print(f"dedup: hash-thr {hash_thr}, geo-thr {geo_thr}; {len(jobs)} clip, {sum(got.values())} frame tenuti su "
          f"{sum(tot.values())} ({sum(got.values()) / sum(tot.values()):.0%})")
    for c in sorted(tot):
        print(f"  {c:<11}{got[c]:>7} / {tot[c]:<7}{got[c] / tot[c]:.0%}")
    print(f"-> {out / 'tenuti.csv'}")


def dedup_sheets(src: Path, per_class: int = 3, n: int = 24, side: int = 96, seed: int = 0) -> None:
    """Fogli di controllo della deduplica in `6_dedup/fogli/`: per classe
    `per_class` clip a caso, `n` frame consecutivi a meta' clip, ritaglio
    attorno al primo box ingrandito a `side` px; bordo verde = tenuto,
    rosso = tolto. Un foglio per clip, `clip.txt` con i numeri."""
    import csv

    import cv2

    kept = defaultdict(set)
    for r in csv.DictReader(open(src / "6_dedup" / "tenuti.csv")):
        kept[r["clip"]].add(int(r["frame"]))
    idx = defaultdict(list)
    for r in csv.DictReader(open(src / "2_sequenze" / "sequenze.csv")):
        idx[f"{r['video']}/s{r['sequenza']}"].append(int(r["frame"]))
    rng = random.Random(seed)
    by_cls = defaultdict(list)
    for c in sorted(kept):
        by_cls[c.split("_")[1]].append(c)
    clips = [c for k in sorted(by_cls) for c in rng.sample(by_cls[k], min(per_class, len(by_cls[k])))]
    out = src / "6_dedup" / "fogli"
    out.mkdir(exist_ok=True)
    for k, clip in enumerate(clips, 1):
        v, fr = clip.split("/")[0], sorted(idx[clip])
        tiles, last = [], None
        for i in fr[max(0, len(fr) // 2 - n // 2):][:n]:
            img = cv2.imread(str(src / "1_frame" / v / "images" / f"{v}_f{i:05d}.jpg"))
            t = src / "4_box" / "manuale" / clip.replace("/", "_") / f"{v}_f{i:05d}.txt"
            ln = t.read_text().split() if t.exists() else []
            if ln:
                last = (float(ln[1]) * img.shape[1], float(ln[2]) * img.shape[0])
            cx, cy = last or (img.shape[1] / 2, img.shape[0] / 2)
            x0 = int(min(max(cx - 32, 0), img.shape[1] - 64))
            y0 = int(min(max(cy - 32, 0), img.shape[0] - 64))
            tile = cv2.resize(img[y0:y0 + 64, x0:x0 + 64], (side, side), interpolation=cv2.INTER_NEAREST)
            tile = cv2.copyMakeBorder(tile, 4, 18, 4, 4, cv2.BORDER_CONSTANT,
                                      value=(0, 200, 0) if i in kept[clip] else (0, 0, 220))
            cv2.putText(tile, str(i), (6, side + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            tiles.append(tile)
        rows = [np.hstack(tiles[j:j + 12] + [np.zeros_like(tiles[0])] * (12 - len(tiles[j:j + 12])))
                for j in range(0, len(tiles), 12)]
        cv2.imwrite(str(out / f"{k:02d}.jpg"), np.vstack(rows))
    (out / "clip.txt").write_text("".join(f"{k:02d} {c}\n" for k, c in enumerate(clips, 1)))
    print(f"fogli_dedup: {len(clips)} fogli in {out}")


# --- conteggio -------------------------------------------------------------------

def count_cells(src: Path, W: int = 640, H: int = 512) -> None:
    """Passo 8: istanze, frame, clip, video e sessioni per cella classe x
    fascia sui frame tenuti dalla deduplica (`6_dedup/tenuti.csv`, gia' senza
    scie e scartate) con le label di `4_box/manuale/` ->
    `7_conteggio/conteggio.csv`. Un frame conta in ogni cella di cui ha un box."""
    import csv

    ses = {r["clip"]: r["sessione"] for r in csv.DictReader(open(src / "3_sessioni" / "sessioni.csv"))}
    inst, sets = Counter(), defaultdict(lambda: (set(), set(), set(), set()))
    labelled, n_frames = Counter(), Counter()
    for r in csv.DictReader(open(src / "6_dedup" / "tenuti.csv")):
        clip, v = r["clip"], r["clip"].split("/")[0]
        n_frames[clip] += 1
        t = src / "4_box" / "manuale" / clip.replace("/", "_") / f"{v}_f{int(r['frame']):05d}.txt"
        rows = [ln.split() for ln in t.read_text().splitlines() if ln.strip()] if t.exists() else []
        labelled[clip] += bool(rows)
        for c, _, _, w, h in rows:
            cell = (NAMES[int(c)], BANDS[np.digitize(np.sqrt(float(w) * W * float(h) * H), BAND_EDGES)])
            inst[cell] += 1
            for st, x in zip(sets[cell], (t.stem, clip, v, ses[clip])):
                st.add(x)
    out = src / "7_conteggio"
    out.mkdir(exist_ok=True)
    with open(out / "conteggio.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["classe", "fascia", "istanze", "frame", "clip", "video", "sessioni"])
        w.writerows([*cell, inst[cell], *map(len, sets[cell])] for cell in
                    ((c, b) for c in NAMES for b in BANDS) if inst[cell])
    print(f"conta: {len(n_frames)} clip, {sum(labelled.values())} frame con box, "
          f"{sum(n_frames.values()) - sum(labelled.values())} senza box")
    for key, i in (("istanze", None), ("frame", 0), ("video", 2), ("sessioni", 3)):
        print(f"\n{key:<11}" + "".join(f"{b:>8}" for b in BANDS))
        for c in NAMES:
            print(f"{c:<11}" + "".join(f"{inst[c, b] if i is None else len(sets[c, b][i]):>8}" for b in BANDS))
    print(f"-> {out / 'conteggio.csv'}")


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

# --- costruzione ---------------------------------------------------------------

# --- report --------------------------------------------------------------------

# --- main ----------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--src", type=Path, default=ROOT / "data" / "svanstrom",
                   help="una cartella per passo (docs/procedura.md): 0_raw, 1_frame, ...")
    p.add_argument("--step", choices=["frames", "tagli", "sequenze", "sessioni", "unione", "box", "sam", "fogli_sam", "cvat", "cvat_import", "meteo", "meteo_import", "dedup", "fogli_dedup", "conta"],
                   required=True, help="passo della procedura (docs/procedura.md)")
    p.add_argument("--hash-thr", type=int, default=16,
                   help="bit di dHash (su 64) sotto cui due frame sono uguali")
    p.add_argument("--geo-thr", type=float, default=0.05,
                   help="spostamento del centro (frazione del frame) sotto cui "
                        "il box e' fermo")
    p.add_argument("--cut-thr", type=float, default=0.07,
                   help="punteggio sopra cui una coppia di frame e' candidata taglio")
    p.add_argument("--session-thr", type=float, default=0.95,
                   help="somiglianza media DINOv2 sopra cui due gruppi di video si uniscono")
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
    if a.step == "fogli_sam":
        # fogli delle clip gia' passate da SAM (anche a SAM in corso), senza le clip con scia
        import csv
        box = a.src / "4_box"
        scie = set((box / "scie.txt").read_text().split()) if (box / "scie.txt").exists() else set()
        by = defaultdict(dict)
        for r in csv.DictReader(open(box / "box.csv")):
            by[r["clip"]][(int(r["frame"]), r["k"])] = r
        clips = {}
        for p in sorted((box / "sam").glob("*.json")):
            clip = "/".join(p.stem.rsplit("_", 1))
            if clip in scie:
                continue
            d = json.loads(p.read_text())
            rows = []
            for f, k, b in zip(d["frame"], d["k"], d["box"]):
                r = dict(by[clip][(f, k)])
                r.update(zip(BOX_COLS, b) if b else zip(BOX_COLS, (float(r["ox"]), float(r["oy"]), 0.0, 0.0)),
                         fonte="sam" if b else "vuoto")
                rows.append(r)
            clips[clip] = rows
        box_sheets(a.src / "1_frame", box / "fogli_sam", clips)
        (box / "fogli_sam" / "clip.txt").write_text("".join(f"{n} {c}\n" for n, c in enumerate(clips, 1)))
        print(f"4_box: {len(clips)} clip SAM nei fogli ({len(scie)} con scia escluse)")
        return 0
    if a.step == "fogli_dedup":
        dedup_sheets(a.src)
        return 0
    if a.step == "dedup":
        dedup(a.src, a.hash_thr, a.geo_thr, a.workers)
        return 0
    if a.step == "conta":
        count_cells(a.src)
        return 0
    if a.step == "cvat_import":
        cvat_import(a.src)
        return 0
    if a.step == "cvat":
        # tutte le clip tranne le scie, in ordine di sessione, in un task con un job solo
        import csv
        box = a.src / "4_box"
        scie = set((box / "scie.txt").read_text().split())
        ses = {r["clip"]: r["sessione"] for r in csv.DictReader(open(a.src / "3_sessioni" / "sessioni.csv"))}
        cvat_export(a.src, sorted((c for c in ses if c not in scie), key=lambda c: (ses[c], c)))
        return 0
    if a.step == "meteo":
        # passo 6, non usa i box: uscite in 5_meteo/
        import csv
        met = a.src / "5_meteo"
        met.mkdir(exist_ok=True)
        strips = strip_frames(a.src / "2_sequenze")
        session = {r["clip"]: r["sessione"] for r in csv.DictReader(open(a.src / "3_sessioni" / "sessioni.csv"))}
        cvat_weather(met, weather_proposal(strips, weather_probs(a.src / "1_frame", strips), session))
        return 0
    if a.step == "meteo_import":
        weather_import(a.src / "5_meteo")
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
    return 0

if __name__ == "__main__":
    sys.exit(main())
