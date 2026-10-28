"""Open Images V7 (annotations CC BY 4.0, Google LLC; images listed as CC BY 2.0): birds in the sky (user, 25/10/2026).

    data/input/openimages/0_raw/scarica.sh             # annotations and image lists, with sha256
    python tools/openimages.py --step candidati        # images whose birds are all small -> 1_candidati.csv
    python tools/openimages.py --step immagini         # download from the public CVDF bucket -> 2_immagini/
    python tools/openimages.py --step ritagli          # 640x512 crops, no resize, sky only -> 4_ritagli/, 5_box/manuale/
    python tools/openimages.py --step volo             # round 2: SigLIP "bird flying in the sky" >= 0.7, double boxes out

Rules: individual boxes only (images with a group-of, depiction or inside bird box are dropped, the flock would stay
without labels); airplanes and helicopters of the same image are kept as labels; birds outside 8-96 px or crops
with a textured background around a bird (mean |Laplacian| >= `sky`, `tools/scenario.py`) are dropped."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict

import numpy as np

from tools.common import INPUT

SRC = INPUT / "openimages"
RAW = SRC / "0_raw"
BIRDS = {"/m/015p6", "/m/01f8m5", "/m/09ddx", "/m/09csl", "/m/0f6wt", "/m/0dbvp", "/m/012074", "/m/09d5_",
         "/m/0gv1x", "/m/06j2d", "/m/0h23m", "/m/0dftk", "/m/01dy8n"}  # bird and its flying subclasses
SKY = {"/m/015p6", "/m/09csl", "/m/0f6wt", "/m/06j2d", "/m/012074"}  # round 2 (user, 28/10/2026): bird, eagle, falcon, raven, magpie
OTHER_BIRDS = BIRDS - SKY | {"/m/0ccs93", "/m/09b5t", "/m/05n4y", "/m/05z6w", "/m/0jly1"}  # ducks, geese, swans, parrots...: out
OTHERS = {"/m/0cmf2": 0, "/m/09ct_": 3}  # airplane, helicopter -> our classes (0 airplane, 1 bird, 2 drone, 3 helicopter)
ANN = ["oidv6-train-annotations-bbox.csv", "validation-annotations-bbox.csv", "test-annotations-bbox.csv"]
META = ["train-images-boxable-with-rotation.csv", "validation-images-with-rotation.csv", "test-images-with-rotation.csv"]


def candidates(max_side: float = 0.12) -> None:
    """Images with at least one bird, all birds individual and with normalized side <= `max_side`. Round 2
    (28/10/2026): only the `SKY` classes, images with an `OTHER_BIRDS` box out; the rows of the images already in
    `5_box/ritagli.csv` stay as they are."""
    done = {r["image"] for r in csv.DictReader(open(SRC / "5_box" / "ritagli.csv"))}
    old = [r for r in csv.DictReader(open(SRC / "1_candidati.csv")) if r["image"] in done]
    boxes, bad = defaultdict(list), set(done)
    for f in ANN:
        for r in csv.DictReader(open(RAW / f)):
            lab = r["LabelName"]
            if lab == "/m/0k5j":  # generic "aircraft" (drones, balloons...): it would stay without our label
                bad.add(r["ImageID"])
                continue
            if lab in OTHER_BIRDS:
                bad.add(r["ImageID"])
                continue
            if lab not in SKY and lab not in OTHERS:
                continue
            if r["IsGroupOf"] == "1" or r["IsDepiction"] == "1" or r["IsInside"] == "1":
                bad.add(r["ImageID"])
                continue
            boxes[r["ImageID"]].append((1 if lab in SKY else OTHERS[lab], *(float(r[k]) for k in ("XMin", "YMin", "XMax", "YMax"))))
    keep = {k: v for k, v in boxes.items() if k not in bad and any(c == 1 for c, *_ in v)
            and max(math.sqrt((x2 - x1) * (y2 - y1)) for c, x1, y1, x2, y2 in v if c == 1) <= max_side}
    meta = {}
    for f in META:
        for r in csv.DictReader(open(RAW / f)):
            if r["ImageID"] in keep:
                meta[r["ImageID"]] = r
    with open(SRC / "1_candidati.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "subset", "license", "author", "url", "rotation", "boxes"])
        w.writerows([r[k] for k in ("image", "subset", "license", "author", "url", "rotation", "boxes")] for r in old)
        for k in sorted(keep):
            m = meta.get(k)
            if m and float(m.get("Rotation") or 0) == 0:  # rotated images: the boxes refer to the rotated view, skip
                w.writerow([k, m["Subset"], m["License"], m["Author"], m["OriginalLandingURL"], m["Rotation"],
                            ";".join(f"{c} {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f}" for c, x1, y1, x2, y2 in keep[k])])
    print(f"openimages: {len(keep)} immagini candidate nuove, {len(old)} gia' ritagliate -> {SRC / '1_candidati.csv'}")


def images(workers: int = 16) -> None:
    """Download from the CVDF public bucket (https://open-images-dataset.s3.amazonaws.com/<subset>/<id>.jpg)."""
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor

    out = SRC / "2_immagini"
    out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(SRC / "1_candidati.csv")))

    def get(r):
        f = out / f"{r['image']}.jpg"
        if not f.exists():
            try:
                urllib.request.urlretrieve(f"https://open-images-dataset.s3.amazonaws.com/{r['subset']}/{r['image']}.jpg", f)
            except Exception as e:  # noqa: BLE001 - one missing image must not stop the rest
                return f"{r['image']}: {e}"
        return None

    with ThreadPoolExecutor(workers) as ex:
        err = [e for e in ex.map(get, rows) if e]
    (SRC / "2_errori.txt").write_text("\n".join(err) + "\n")
    print(f"openimages: {len(rows) - len(err)} immagini, {len(err)} errori -> {out}")


def crops(sky: float = 8.0, smin: float = 8, smax: float = 96, seed: int = 0) -> None:
    """One 640x512 crop per image around its birds, without resizing (README, Format): random position that
    contains the largest set of birds; boxes cut by the crop border keep the visible part only if >= 50% of their
    area is inside, otherwise the crop is dropped (it would leave an object without label). A crop is kept only if
    every bird in it is 8-96 px and the background around the birds is sky or horizon (texture < `sky`; < 3 = sky). Writes `4_ritagli/images`, `5_box/manuale/<id>.txt`
    (intermediate order: 0 airplane, 1 bird, 2 drone, 3 helicopter) and `5_box/ritagli.csv`."""
    import cv2

    rng = np.random.default_rng(seed)
    (SRC / "4_ritagli" / "images").mkdir(parents=True, exist_ok=True)
    (SRC / "5_box" / "manuale").mkdir(parents=True, exist_ok=True)
    out = SRC / "5_box" / "ritagli.csv"
    old = list(csv.reader(open(out)))[1:] if out.exists() else []  # crops already made stay as they are
    rows = [r for r in csv.DictReader(open(SRC / "1_candidati.csv")) if r["image"] not in {o[1] for o in old}]
    kept, why = [], defaultdict(int)
    for r in rows:
        f = SRC / "2_immagini" / f"{r['image']}.jpg"
        im = cv2.imread(str(f)) if f.exists() else None
        if im is None:
            why["mancante"] += 1
            continue
        H, W = im.shape[:2]
        if W < 640 or H < 512:
            why["piccola"] += 1
            continue
        bx = [(int(c), x1 * W, y1 * H, x2 * W, y2 * H) for c, x1, y1, x2, y2 in
              (map(float, b.split()) for b in r["boxes"].split(";"))]
        birds = [b for b in bx if b[0] == 1]
        got = None
        for b in [birds[i] for i in rng.permutation(len(birds))]:  # window around one bird, at a random position
            if b[3] - b[1] > 640 or b[4] - b[2] > 512:  # no 640x512 window holds this bird (wide wings, round 2)
                continue
            for _ in range(10):
                x0 = int(rng.uniform(max(0, b[3] - 640), min(b[1], W - 640) + 1e-6))
                y0 = int(rng.uniform(max(0, b[4] - 512), min(b[2], H - 512) + 1e-6))
                lab, drop = [], False
                for c, x1, y1, x2, y2 in bx:
                    a = (x2 - x1) * (y2 - y1)
                    cx1, cy1, cx2, cy2 = max(x1, x0), max(y1, y0), min(x2, x0 + 640), min(y2, y0 + 512)
                    inside = max(0, cx2 - cx1) * max(0, cy2 - cy1)
                    if inside == 0:
                        continue
                    if inside < 0.5 * a or (c == 1 and not smin <= math.sqrt(inside) <= smax):
                        drop = True  # cut object (it would stay half labelled) or bird outside smin-smax px
                        break
                    lab.append((c, (cx1 + cx2) / 2 - x0, (cy1 + cy2) / 2 - y0, cx2 - cx1, cy2 - cy1))
                if not drop:
                    got = (x0, y0, lab)
                    break
            if got:
                break
        if not got:
            why["nessuna_finestra"] += 1
            continue
        x0, y0, lab = got
        crop = im[y0:y0 + 512, x0:x0 + 640]
        lap = np.abs(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32), cv2.CV_32F))
        tex = []
        for c, cx, cy, w, h in lab:
            if c != 1:
                continue
            m = max(w, h, 24)
            a0, b0, a1, b1 = (int(v) for v in (max(0, cx - w / 2 - m), max(0, cy - h / 2 - m), min(640, cx + w / 2 + m), min(512, cy + h / 2 + m)))
            mask = np.ones((b1 - b0, a1 - a0), bool)
            mask[max(0, int(cy - h / 2) - b0):int(cy + h / 2) - b0, max(0, int(cx - w / 2) - a0):int(cx + w / 2) - a0] = False
            tex.append(float(lap[b0:b1, a0:a1][mask].mean()))
        if max(tex) >= sky:  # textured background (vegetation, roofs): not the domain
            why["sfondo"] += 1
            continue
        name = f"OI_{r['image']}"
        cv2.imwrite(str(SRC / "4_ritagli" / "images" / f"{name}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        (SRC / "5_box" / "manuale" / f"{name}.txt").write_text(
            "".join(f"{c} {cx / 640:.6f} {cy / 512:.6f} {w / 640:.6f} {h / 512:.6f}\n" for c, cx, cy, w, h in lab))
        kept.append((name, r["image"], x0, y0, len(lab), round(max(tex), 2), r["license"], r["author"], r["url"]))
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ritaglio", "image", "x0", "y0", "box", "tessitura_max", "license", "author", "url"])
        w.writerows(old + kept)
    print(f"openimages: {len(kept)} ritagli nuovi ({len(old)} di prima); scartati {dict(why)}")


PROMPTS = ["a photo of a bird flying in the sky", "a photo of a bird perched on a branch", "a photo of a bird standing on the ground",
           "a photo of a bird sitting on a post or a roof", "a photo of a bird swimming on water", "a painting or drawing of a bird",
           "a photo of a bird held by a person", "a close-up photo of a bird in a zoo or a cage"]


def flying(first: int = 0, thr: float = 0.7, dup: float = 0.7) -> None:
    """Round 2 (user, 28/10/2026): the texture does not separate perched and flying big birds (6 in flight out of 20 in a
    sample). SigLIP zero-shot on the crops after the first `first` (0: all, user 28/10/2026: a heron standing in water of
    round 1 was in the validation as sky; 280 of the 623 crops of round 1 are out): kept if the probability of
    `PROMPTS[0]` among `PROMPTS` is >= `thr` (all 6 in flight at >= 0.97, the other 14 at <= 0.45) or if it is the
    largest (user, 28/10/2026: the 20 of round 1 under `thr` are sensible flights), and the crop is not in
    `5_box/scarti_volo.txt` (dropped by hand from the sheet of those under `thr`). Bird boxes with
    IoU >= `dup` with another bird box are dropped (the same bird labelled Bird and Eagle). Writes `5_box/volo.csv`
    (scores) and rewrites `5_box/ritagli.csv`; the dropped crops stay on disk."""
    import open_clip
    import torch
    from PIL import Image

    out = SRC / "5_box" / "ritagli.csv"
    head, *rows = list(csv.reader(open(out)))
    m, _, pre = open_clip.create_model_and_transforms("ViT-SO400M-14-SigLIP-384", pretrained="webli", device="cuda")
    tok = open_clip.get_tokenizer("ViT-SO400M-14-SigLIP-384")
    m.eval()
    score, top = {}, {}
    with torch.no_grad(), torch.autocast("cuda"):
        t = m.encode_text(tok(PROMPTS).cuda())
        t = t / t.norm(dim=-1, keepdim=True)
        new = rows[first:]
        for k in range(0, len(new), 32):
            b = torch.stack([pre(Image.open(SRC / "4_ritagli" / "images" / f"{r[0]}.jpg").convert("RGB"))
                             for r in new[k:k + 32]]).cuda()
            f = m.encode_image(b)
            f = f / f.norm(dim=-1, keepdim=True)
            for r, p in zip(new[k:k + 32], (100 * f @ t.T).softmax(-1).float().cpu().numpy()):
                score[r[0]], top[r[0]] = round(float(p[0]), 3), int(p.argmax()) == 0
    with open(SRC / "5_box" / "volo.csv", "w", newline="") as fh:
        csv.writer(fh).writerows([("ritaglio", "volo", "volo_massima")] + [(k, v, int(top[k])) for k, v in sorted(score.items())])
    drop = {ln.strip() for ln in open(SRC / "5_box" / "scarti_volo.txt") if ln.strip() and not ln.startswith("#")}

    def iou(a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        i = max(0, min(ax + aw / 2, bx + bw / 2) - max(ax - aw / 2, bx - bw / 2)) * \
            max(0, min(ay + ah / 2, by + bh / 2) - max(ay - ah / 2, by - bh / 2))
        return i / (aw * ah + bw * bh - i)

    kept, n = rows[:first], 0
    for r in rows[first:]:
        if (score[r[0]] < thr and not top[r[0]]) or r[0] in drop:
            continue
        f = SRC / "5_box" / "manuale" / f"{r[0]}.txt"
        bs = [ln.split() for ln in f.read_text().splitlines() if ln.strip()]
        keep = []
        for b in bs:
            if b[0] == "1" and any(o[0] == "1" and iou(tuple(map(float, b[1:])), tuple(map(float, o[1:]))) >= dup for o in keep):
                n += 1
                continue
            keep.append(b)
        f.write_text("".join(" ".join(b) + "\n" for b in keep))
        kept.append(r[:4] + [str(len(keep))] + r[5:])
    with open(out, "w", newline="") as fh:
        csv.writer(fh).writerows([head] + kept)
    print(f"openimages: {len(kept) - first} ritagli in volo su {len(rows) - first} nuovi, {n} box doppi tolti")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--step", choices=["candidati", "immagini", "ritagli", "volo"], required=True)
    p.add_argument("--lato", type=float, default=0.12, help="candidati: lato massimo normalizzato degli uccelli")
    p.add_argument("--max-px", type=float, default=96, help="ritagli: grandezza massima degli uccelli nel ritaglio (px)")
    a = p.parse_args()
    if a.step == "candidati":
        candidates(a.lato)
    elif a.step == "ritagli":
        crops(smax=a.max_px)
    elif a.step == "volo":
        flying()
    else:
        images()
