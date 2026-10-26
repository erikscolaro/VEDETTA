"""Whole frozen dataset (`data/output/`) in CVAT for a full re-check (user, 26/10/2026).

    python tools/cvat_output.py --step carica    # project 'output_26ott', tasks '26ott_<split>_<nn>' of up to 2000 images, jobs of 500
    python tools/cvat_output.py --step importa   # review of train and val -> intermediate/16_cvat/
    python tools/cvat_output.py --step carica_nuovi [--giro 17b]   # frames of 17_dataset not reviewed -> project 'nuovi_17'
    python tools/cvat_output.py --step importa_nuovi [--giro 17b]  # their review -> intermediate/16_cvat/nuovi_<giro>.csv

Images in name order (source and session together), labels as rectangles. Tasks already in the project are skipped,
so a run that stops halfway can be resumed. Only reads `data/output/`.

`importa` brings the review back onto the source frames (manifest `sorgente`, label of `15_dataset/frame.csv`) of the
real images of train and val (test was not corrected; mosaics and background are rebuilt). An image is removed when its
frame was deleted in CVAT or it has no box left. A box is unchanged with the same class and IoU >= 0.9 with the
original. Writes `16_cvat/label/` (corrected copies, intermediate class order), `16_cvat/frame.csv` (every real image of
the export: source frame, split, name, label to use, state) and `16_cvat/modifiche.csv` (one row per changed box)."""

from __future__ import annotations

import argparse
import csv
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from tools.common import DATA, INTERMEDIATE, ROOT, cvat_client

OUT = ROOT / "data" / "output"
PROJECT = "output_26ott"
WORK = INTERMEDIATE / "16_cvat"
INTER_NAMES = ["airplane", "bird", "drone", "helicopter"]  # order of the intermediate labels


def _task(client, proj, name: str, items: list, per_job: int = 500, per_patch: int = 2000) -> None:
    """One CVAT task of `items` [(image path, name in CVAT, [(class name, x1, y1, x2, y2)])], in name order."""
    from cvat_sdk.api_client import models
    from cvat_sdk.core.proxies.tasks import ResourceType

    label = {lb.name: lb.id for lb in proj.get_labels()}
    items = sorted(items, key=lambda x: x[1])  # as CVAT (lexicographical)
    with tempfile.TemporaryDirectory() as tmp:  # one zip: beyond ~1000 files per request CVAT gives 500
        zp = Path(tmp) / f"{name}.zip"
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_STORED) as zf:
            for p, arc, _ in items:
                zf.write(p, arc)
        task = client.tasks.create_from_data(
            spec={"name": name, "project_id": proj.id, "segment_size": per_job},
            resource_type=ResourceType.LOCAL, resources=[str(zp)],
            data_params={"image_quality": 95, "sorting_method": "lexicographical"})
    assert task.size == len(items), f"{name}: {task.size} frame invece di {len(items)}"
    shapes = [models.LabeledShapeRequest(type="rectangle", frame=frame, label_id=label[c], points=list(b))
              for frame, (_, _, boxes) in enumerate(items) for c, *b in boxes]
    for k in range(0, len(shapes), per_patch):  # in pieces: a single one times out
        task.update_annotations(models.PatchedLabeledDataRequest(shapes=shapes[k:k + per_patch]))
    n = len(task.get_annotations().shapes)
    assert n == len(shapes), f"{name}: {n} box in CVAT invece di {len(shapes)}"
    print(f"cvat: {name} ({task.id}), {len(items)} immagini, {n} box", flush=True)


def _project(client, name: str, names: list[str]):
    from cvat_sdk.api_client import models

    proj = next((p for p in client.projects.list() if p.name == name), None)
    return proj or client.projects.create(models.ProjectWriteRequest(
        name=name, labels=[models.PatchedLabelRequest(name=n) for n in names]))


def _yolo(path: Path, names: list[str], W: int, H: int) -> list:
    out = []
    for ln in (path.read_text().splitlines() if path.exists() else []):
        c, cx, cy, w, h = ln.split()
        cx, cy, w, h = float(cx) * W, float(cy) * H, float(w) * W, float(h) * H
        out.append((names[int(c)], cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    return out


def upload(per_task: int = 2000) -> None:
    names = yaml.safe_load(open(OUT / "dataset.yaml"))["names"]
    names = [names[k] for k in sorted(names)]
    with cvat_client() as client:
        proj = _project(client, PROJECT, names)
        have = {t.name for t in proj.get_tasks()}
        for split in ("train", "val", "test"):
            imgs = sorted((OUT / "images" / split).glob("*.jpg"), key=lambda p: p.name)
            for s in range(0, len(imgs), per_task):
                name = f"26ott_{split}_{s // per_task + 1:02d}"
                if name not in have:
                    _task(client, proj, name, [(p, p.name, _yolo(OUT / "labels" / split / f"{p.stem}.txt", names, *Image.open(p).size))
                                               for p in imgs[s:s + per_task]])


def upload_new(round_: str = "17", per_task: int = 2000) -> None:
    """Frames of `17_dataset` not reviewed yet (train.csv, val and test of split.csv; user, 26/10/2026: every new
    frame is reviewed): project 'nuovi_17', tasks '<round>_<split>_<nn>' (one round per new split), images named after
    the source frame. Writes `17_dataset/cvat_<round>.csv` (task, nome, immagine)."""
    work = INTERMEDIATE / "17_dataset"
    names = yaml.safe_load(open(OUT / "dataset.yaml"))["names"]
    names = [names[k] for k in sorted(names)]
    fr = {r["immagine"]: r for r in csv.DictReader(open(work / "frame.csv"))}
    use = {r["immagine"]: "train" for r in csv.DictReader(open(work / "train.csv"))}
    use |= {r["immagine"]: r["split"] for r in csv.DictReader(open(work / "split.csv")) if r["split"] in ("val", "test")}
    new = sorted((s, i) for i, s in use.items() if fr[i]["rivisto"] != "1")
    assert len({Path(i).name for _, i in new}) == len(new), "nomi doppi"
    rows = []
    with cvat_client() as client:
        proj = _project(client, "nuovi_17", names)
        have = {t.name for t in proj.get_tasks()}
        for split in ("train", "val", "test"):
            imgs = sorted((i for s, i in new if s == split), key=lambda i: Path(i).name)
            for s in range(0, len(imgs), per_task):
                name, part = f"{round_}_{split}_{s // per_task + 1:02d}", imgs[s:s + per_task]
                rows += [(name, Path(i).name, i) for i in part]
                if name not in have:
                    _task(client, proj, name, [(DATA / i, Path(i).name, [(INTER_NAMES[c], *b) for c, *b in _yolo(
                        DATA / fr[i]["label"], list(range(4)), *Image.open(DATA / i).size)]) for i in part])
    with open(work / f"cvat_{round_}.csv", "w", newline="") as fh:
        csv.writer(fh).writerows([("task", "nome", "immagine")] + rows)
    print(f"nuovi: {len(rows)} frame -> progetto nuovi_17, giro {round_}")


def _boxes(lines: list[str], W: int, H: int) -> tuple[np.ndarray, np.ndarray]:
    g = np.array([[float(v) for v in ln.split()] for ln in lines if ln.strip()]).reshape(-1, 5)
    return g[:, 0].astype(int), np.c_[(g[:, 1] - g[:, 3] / 2) * W, (g[:, 2] - g[:, 4] / 2) * H,
                                      (g[:, 1] + g[:, 3] / 2) * W, (g[:, 2] + g[:, 4] / 2) * H]


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    ar = lambda z: (z[:, 2:] - z[:, :2]).prod(1)  # noqa: E731
    return inter / (ar(a)[:, None] + ar(b)[None] - inter + 1e-9)


def import_review(splits: tuple[str, ...] = ("train", "val")) -> None:
    frames = {r["immagine"]: r for r in csv.DictReader(open(INTERMEDIATE / "15_dataset" / "frame.csv"))}
    items = {Path(r["immagine"]).stem: (r["sorgente"], r["split"], frames[r["sorgente"]]["label"])
             for r in csv.DictReader(open(INTERMEDIATE / "16_cvat" / "manifest_26ott.csv"))  # names in CVAT: the 26/10 export
             if r["tipo"] == "reale" and r["split"] != "calib"}
    _import(PROJECT, items, splits, "frame.csv", "modifiche.csv")


def import_new(round_: str = "17") -> None:
    """Review of a round of new frames of `17_dataset` (project 'nuovi_17') -> `16_cvat/nuovi_<round>.csv` and
    `16_cvat/modifiche_nuovi_<round>.csv`, corrected copies in `16_cvat/label/`; `dataset.py --step rivisti` reads
    every round with `16_cvat/frame.csv`."""
    work = INTERMEDIATE / "17_dataset"
    lab = {r["immagine"]: r["label"] for r in csv.DictReader(open(work / "frame.csv"))}
    split = {r["immagine"]: r["task"].split("_")[1] for r in csv.DictReader(open(work / f"cvat_{round_}.csv"))}
    items = {Path(i).stem: (i, s, lab[i]) for i, s in split.items()}
    _import("nuovi_17", items, ("train", "val", "test"), f"nuovi_{round_}.csv", f"modifiche_nuovi_{round_}.csv", round_)


def _import(project: str, items: dict, splits: tuple[str, ...], out_frame: str, out_changes: str,
            prefix: str = "26ott") -> None:
    """CVAT review of `project` against `items` {name in CVAT: (source frame, split, label)}: writes `out_frame`
    (immagine, split, nome, label, stato) and `out_changes` in `16_cvat/`, corrected copies in `16_cvat/label/`."""
    names = yaml.safe_load(open(OUT / "dataset.yaml"))["names"]
    to_inter = {k: INTER_NAMES.index(n) for k, n in names.items()}
    to_out = {v: k for k, v in to_inter.items()}
    review, deleted = {}, set()  # name in CVAT -> CVAT boxes (output ids, pixels); names of deleted frames
    with cvat_client() as client:
        proj = next(p for p in client.projects.list() if p.name == project)
        label = {lb.id: next(k for k, n in names.items() if n == lb.name) for lb in proj.get_labels()}
        for task in proj.get_tasks():
            if task.name.split("_")[0] != prefix or task.name.split("_")[1] not in splits:
                continue
            info, ann = task.get_frames_info(), task.get_annotations()
            assert not ann.tracks and all(s.type.value == "rectangle" for s in ann.shapes), task.name
            gone = set(task.get_meta().deleted_frames or [])
            boxes = {k: [] for k in range(len(info))}
            for s in ann.shapes:
                f = info[s.frame]
                x1, x2 = sorted(min(max(v, 0), f.width) for v in s.points[0::2])
                y1, y2 = sorted(min(max(v, 0), f.height) for v in s.points[1::2])
                if x2 - x1 >= 1 and y2 - y1 >= 1:
                    boxes[s.frame].append((label[s.label_id], x1, y1, x2, y2, f.width, f.height))
            for k, f in enumerate(info):
                if k in gone:
                    deleted.add(Path(f.name).stem)
                else:
                    review[Path(f.name).stem] = boxes[k]
    (WORK / "label").mkdir(parents=True, exist_ok=True)
    rows, changes, state_n = [], [], {}
    for name, (img, sp, src) in sorted(items.items()):
        m = {"sorgente": img, "split": sp}
        state, lab = "invariata", src
        if name in deleted:
            state = "cancellata"
        elif m["split"] in splits:
            assert name in review, f"{name} non e' in CVAT"
            W, H = Image.open(DATA / m["sorgente"]).size
            oc, ob = _boxes((DATA / src).read_text().splitlines(), W, H)
            oc = np.array([to_out[c] for c in oc], dtype=int)
            new = review[name]
            rc = np.array([b[0] for b in new], dtype=int)
            rb = np.array([b[1:5] for b in new], dtype=float).reshape(-1, 4)
            M = _iou(ob, rb) if len(ob) and len(rb) else np.zeros((len(ob), len(rb)))
            om, rm = {}, {}
            for i, j in sorted(np.ndindex(*M.shape), key=lambda ij: -M[ij]):
                if M[i, j] >= 0.3 and i not in om and j not in rm:
                    om[i], rm[j] = j, i
            side = lambda b: round(float(np.sqrt((b[2] - b[0]) * (b[3] - b[1]))), 1)  # noqa: E731
            mine = []
            for i in range(len(ob)):
                j = om.get(i)
                kind = "tolto" if j is None else "classe" if oc[i] != rc[j] else "invariato" if M[i, j] >= 0.9 else "spostato"
                if kind != "invariato":
                    mine.append((m["split"], name, m["sorgente"], kind, names[oc[i]], "" if j is None else names[rc[j]],
                                 "" if j is None else round(float(M[i, j]), 3), side(ob[i])))
            mine += [(m["split"], name, m["sorgente"], "aggiunto", "", names[rc[j]], "", side(rb[j]))
                     for j in range(len(rb)) if j not in rm]
            changes += mine
            if not len(rb):
                state = "vuota"
            elif mine:
                state = "cambiata"
                lab = WORK / "label" / f"{Path(src).parent.name}__{Path(src).name}"
                lab.write_text("".join(f"{to_inter[c]} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} "
                                       f"{(y2 - y1) / H:.6f}\n" for c, x1, y1, x2, y2, *_ in new))
                lab = str(lab.relative_to(DATA))
        state_n[m["split"], state] = state_n.get((m["split"], state), 0) + 1
        rows.append((m["sorgente"], m["split"], name, lab, state))
    with open(WORK / out_frame, "w", newline="") as fh:
        csv.writer(fh).writerows([("immagine", "split", "nome", "label", "stato")] + rows)
    with open(WORK / out_changes, "w", newline="") as fh:
        csv.writer(fh).writerows([("split", "nome", "immagine", "modifica", "classe_prima", "classe_dopo", "iou", "lato_px")]
                                 + changes)
    for s in ("train", "val", "test"):
        print(f"  {s}: " + ", ".join(f"{k} {v}" for (x, k), v in sorted(state_n.items()) if x == s))
    print(f"importa: {len(changes)} box modificati -> {WORK}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--step", choices=["carica", "importa", "carica_nuovi", "importa_nuovi"], required=True)
    p.add_argument("--giro", default="17", help="carica_nuovi, importa_nuovi: prefix of the round of new frames")
    a = p.parse_args()
    if a.step in ("carica_nuovi", "importa_nuovi"):
        {"carica_nuovi": upload_new, "importa_nuovi": import_new}[a.step](a.giro)
    else:
        {"carica": upload, "importa": import_review}[a.step]()
