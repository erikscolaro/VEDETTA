"""Main dataset (user, 25-26/10/2026, TODO.md night plan): the pipeline of `split.py` plus domain alignment.
Intermediate files in `intermediate/15_dataset/` (table) and `intermediate/17_dataset/` (after the CVAT review of
the 26/10 dataset, user 26/10/2026), the dataset in `data/output/`.

    python tools/dataset.py --step tabella    # frames, scenario, exclusions, corrected labels -> 15_dataset/frame.csv
    python tools/dataset.py --step rivisti    # CVAT review, night and dark frames out -> 17_dataset/frame.csv
    python tools/dataset.py --step split      # val/test in domain, balanced per source -> 17_dataset/split.csv
    python tools/dataset.py --step train      # scenario quota, session cap, cell cap -> 17_dataset/train.csv
    python tools/dataset.py --step mosaici    # mosaics from the same half octave -> 17_dataset/mosaici/
    python tools/dataset.py --step export     # Ultralytics dataset -> data/output/ (refuses if it exists)
    python tools/dataset.py --step figure     # docs/figure: distribuzione, fonti, contesto (it and en)

Inputs read by the steps (as written by the operator or the agents, reviewed by the user):
- `12_scenario/proposte/*.csv`: scenario of each clip and `ala_fissa` (fixed-wing drone clips are dropped);
- `13_revisione/candidati.csv` + `decisioni/*.csv`: label review;
- `16_cvat/frame.csv`: full CVAT review of the 26/10 dataset (`tools/cvat_output.py --step importa`);
- `input/openimages/5_box/` (birds on sky, CC BY 2.0) and `input/openimages/6_background/` (horizon background)."""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.common import DATA, INPUT, INTERMEDIATE, OUTPUT  # noqa: E402
from tools.split import HALF_OCTAVES, QUARTER_OCTAVES, TEST_TOT  # noqa: E402

BASE = INTERMEDIATE / "15_dataset"  # step tabella: every annotated frame (26/10/2026)
WORK = INTERMEDIATE / "17_dataset"  # from step rivisti on: frames after the CVAT review of the 26/10 dataset
OUT = OUTPUT  # the main dataset since 26/10/2026 (user): it replaced the export of split.py
NAMES = ["airplane", "bird", "drone", "helicopter"]  # intermediate label order
DOMAIN = ("cielo", "orizzonte")  # the only scenarios of val and test
OFF_DOMAIN_MAX = 0.25  # train: at most this share of each class on ground or wall (0.10 left 1,767 birds, 26/10)
SESSION_MAX = 0.06  # train: at most this share of the boxes of a class from one session (0.03 removed 5,641 frames)
# split: weight of the source mix per class (airplane, bird, drone, helicopter): 3 left Anti-UAV at 7% of its drones in
# val; 10 for every class left the val without birds (the first group always worsens the mix), 26/10/2026
SOURCE_WEIGHT = np.array([3, 3, 10, 3])
W, H = 640, 512


def _boxes(path: Path) -> list[tuple[int, float, float, float, float]]:
    return [(int(c), float(x), float(y), float(w), float(h)) for c, x, y, w, h in
            (ln.split() for ln in path.read_text().splitlines() if ln.strip())]


def _side(w: float, h: float) -> float:
    return math.sqrt(w * W * h * H)



# --- step tabella ---------------------------------------------------------------------------------------------------

def _decisions() -> dict[str, tuple[str, str]]:
    out = {}
    for f in sorted(glob.glob(str(INTERMEDIATE / "13_revisione" / "decisioni" / "*.csv"))):
        for r in csv.DictReader(open(f)):
            out[r["id"].strip()] = (r["decisione"].strip().lower(), r.get("nota", ""))
    return out


def table() -> None:
    """One row per frame: fonte, video, clip, sessione, immagine, label (corrected copy if the review changed it),
    scenario. Dropped: clips with fixed-wing drones, frames with an unresolved review (`dubbio` on a missing box or
    on a class), frames left without boxes, frames with a box under 8 px (README, Rules). Mixed clips get the
    scenario per frame from the background texture (`12_scenario/metrica.csv`): < 3 sky, < 8 horizon, otherwise
    ground. Writes `15_dataset/frame.csv`, `15_dataset/label/` and `15_dataset/esclusi.csv` (immagine, motivo)."""
    (BASE / "label").mkdir(parents=True, exist_ok=True)
    idx = {r["id"]: r for r in csv.DictReader(open(INTERMEDIATE / "12_scenario" / "indice.csv"))}
    prop = {r["id"]: r for f in sorted(glob.glob(str(INTERMEDIATE / "12_scenario" / "proposte" / "*.csv")))
            for r in csv.DictReader(open(f))}
    clip_sc = {(idx[k]["fonte"], idx[k]["clip"]): (r["scenario"].strip(), r["ala_fissa"].strip().lower() == "si", r["nota"])
               for k, r in prop.items()}
    fix = INTERMEDIATE / "12_scenario" / "correzioni.csv"  # scenario corrections by the user, they win
    if fix.exists():
        for r in csv.DictReader(open(fix)):
            k = (r["fonte"], r["clip"])
            clip_sc[k] = (r["scenario"], clip_sc[k][1], clip_sc[k][2])
    tex = defaultdict(float)
    for r in csv.DictReader(open(INTERMEDIATE / "12_scenario" / "metrica.csv")):
        if r["tessitura"]:
            tex[r["immagine"]] = max(tex[r["immagine"]], float(r["tessitura"]))
    # label review: edits per image
    dec = _decisions()
    edit = defaultdict(lambda: {"add": [], "drop": set(), "cls": {}, "kill": ""})
    nd = Counter()
    for c in csv.DictReader(open(INTERMEDIATE / "13_revisione" / "candidati.csv")):
        d = dec.get(c["id"], ("", ""))[0]
        e = edit[c["immagine"]]
        nd[c["tipo"], d.split(":")[0] or "non_visto"] += 1
        if c["tipo"] == "senza_box":
            if d.startswith("aggiungi"):
                cls = d.split(":", 1)[1].strip() if ":" in d else c["proposta"]
                if cls in NAMES:
                    x1, y1, x2, y2 = (float(c[k]) for k in ("x1", "y1", "x2", "y2"))
                    e["add"].append((NAMES.index(cls), (x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H))
            elif d in ("dubbio", ""):
                e["kill"] = "senza_box non risolto"
        elif c["tipo"] == "vuota":
            if d == "togli":
                e["drop"].add(int(c["k"]))
        else:  # classe, bird_dvids
            if d == "togli":
                e["drop"].add(int(c["k"]))
            elif d.startswith("classe:") and d.split(":", 1)[1].strip() in NAMES:
                e["cls"][int(c["k"])] = NAMES.index(d.split(":", 1)[1].strip())
            elif d == "dubbio":
                e["kill"] = f"{c['tipo']} dubbio"
    rows, gone = [], []
    src = list(csv.DictReader(open(INTERMEDIATE / "9_split" / "frame.csv")))
    oi_tex = {f"input/openimages/4_ritagli/images/{r['ritaglio']}.jpg": float(r["tessitura_max"])
              for r in csv.DictReader(open(INPUT / "openimages" / "5_box" / "ritagli.csv"))}

    def texture(img, label):
        """background texture around the boxes, as `scenario.py --step metrica` (for frames added after it)"""
        import cv2

        g = cv2.imread(str(DATA / img), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        lap = np.abs(cv2.Laplacian(g, cv2.CV_32F))
        t = 0.0
        for _, x, y, w, h in _boxes(DATA / label):
            m = max(w * W, h * H, 24)
            a0, b0 = int(max(0, (x - w / 2) * W - m)), int(max(0, (y - h / 2) * H - m))
            a1, b1 = int(min(W, (x + w / 2) * W + m)), int(min(H, (y + h / 2) * H + m))
            mask = np.ones((b1 - b0, a1 - a0), bool)
            mask[max(0, int((y - h / 2) * H) - b0):int((y + h / 2) * H) - b0, max(0, int((x - w / 2) * W) - a0):int((x + w / 2) * W) - a0] = False
            if mask.any():
                t = max(t, float(lap[b0:b1, a0:a1][mask].mean()))
        return t

    for r in src:
        img = r["immagine"]
        if r["fonte"] == "openimages":
            sc = "cielo" if oi_tex[img] < 3 else "orizzonte"
        elif (r["fonte"], r["clip"]) not in clip_sc:  # clips added after the review (Anti-UAV videos never used)
            t = texture(img, r["label"])
            sc = "cielo" if t < 3 else "orizzonte" if t < 8 else "terra"
        else:
            sc, fixed, note = clip_sc[r["fonte"], r["clip"]]
            if fixed:
                gone.append((img, "ala fissa"))
                continue
            if sc == "misto":
                t = tex[img] if img in tex else texture(img, r["label"])
                sc = "cielo" if t < 3 else "orizzonte" if t < 8 else ("muro" if "muro" in note.lower() else "terra")
        e = edit.get(img)
        if e and e["kill"]:
            gone.append((img, e["kill"]))
            continue
        bx = _boxes(DATA / r["label"])
        lab = r["label"]
        if e and (e["add"] or e["drop"] or e["cls"]):
            bx = [(e["cls"].get(k, c), *rest) for k, (c, *rest) in enumerate(bx) if k not in e["drop"]] + e["add"]
            lab = f"intermediate/15_dataset/label/{Path(r['label']).parent.name}__{Path(r['label']).name}"
            (DATA / lab).write_text("".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, x, y, w, h in bx))
        if not bx:
            gone.append((img, "senza box dopo la revisione"))
            continue
        if min(_side(w, h) for _, _, _, w, h in bx) < 8:
            gone.append((img, "box sotto 8 px"))
            continue
        rows.append(dict(fonte=r["fonte"], video=r["video"], clip=r["clip"], sessione=r["sessione"], immagine=img,
                         label=lab, scenario=sc))
    with open(BASE / "frame.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, ["fonte", "video", "clip", "sessione", "immagine", "label", "scenario"])
        w.writeheader()
        w.writerows(rows)
    with open(BASE / "esclusi.csv", "w", newline="") as fh:
        csv.writer(fh).writerows([("immagine", "motivo")] + gone)
    print(f"tabella: {len(rows)} frame, esclusi {len(gone)}: {dict(Counter(m for _, m in gone))}")
    print("  revisione: " + ", ".join(f"{t}/{d} {n}" for (t, d), n in sorted(nd.items())))
    sc = Counter()
    for r in rows:
        for c, *_ in _boxes(DATA / r["label"]):
            sc[NAMES[c], r["scenario"]] += 1
    for c in NAMES:
        print(f"  {c:<11}" + "  ".join(f"{s} {sc[c, s]}" for s in ("cielo", "orizzonte", "terra", "muro")))


# --- step rivisti ----------------------------------------------------------------------------------------------------

DARK = 30  # mean grey level (0-255) under which a frame not reviewed is night (the deleted Anti-UAV frames: 3-25)


def reviewed() -> None:
    """Frames after the full CVAT review of the 26/10 dataset (user, 26/10/2026). The reviewed real images
    (`16_cvat/frame.csv`) keep their corrected labels, without the deleted ones and those left without boxes. The other
    frames of the table stay as candidates, without: the groups of the deleted frames (night Anti-UAV), frames darker
    than `DARK`, clips where the review removed a box. Column `rivisto` (1: reviewed or approved in CVAT): split and
    train take them first, mosaics only use them. Writes `17_dataset/frame.csv` and `17_dataset/esclusi.csv`."""
    from concurrent.futures import ThreadPoolExecutor

    from PIL import Image

    WORK.mkdir(parents=True, exist_ok=True)
    grp = {r["sessione"]: r["gruppo"] for r in csv.DictReader(open(INTERMEDIATE / "11_gruppi" / "gruppi.csv"))}
    base = list(csv.DictReader(open(BASE / "frame.csv")))
    by_img = {r["immagine"]: r for r in base}
    cv = INTERMEDIATE / "16_cvat"  # the 26/10 dataset and the new frames of this folder (`nuovi_<round>.csv`, once reviewed)
    files = ["frame.csv"] + sorted(f.name for f in cv.glob("nuovi_*.csv"))  # 26/10 dataset, then the rounds of new frames
    rev = {r["immagine"]: r for f in files for r in csv.DictReader(open(cv / f))}

    def group(i):
        return grp[by_img[i]["fonte"] + ":" + by_img[i]["sessione"]]

    removed = {r["immagine"] for f in ["modifiche.csv"] + sorted(f.name for f in cv.glob("modifiche_nuovi_*.csv"))
               for r in csv.DictReader(open(cv / f)) if r["modifica"] == "tolto"}
    risky = {by_img[i]["clip"] for i in removed} | {by_img[i]["clip"] for i, r in rev.items() if r["stato"] == "vuota"}
    night = {group(i) for i, r in rev.items() if r["stato"] == "cancellata"}
    lum_f = WORK / "luminosita.csv"  # cache: mean grey level of the frames not reviewed
    lum = {r["immagine"]: float(r["lum"]) for r in csv.DictReader(open(lum_f))} if lum_f.exists() else {}
    todo = [r["immagine"] for r in base if r["immagine"] not in rev and r["immagine"] not in lum]

    def grey(i):
        im = Image.open(DATA / i)
        im.draft("L", (80, 64))
        return float(np.asarray(im.convert("L")).mean())

    with ThreadPoolExecutor(16) as ex:
        lum.update(zip(todo, ex.map(grey, todo)))
    with open(lum_f, "w", newline="") as fh:
        csv.writer(fh).writerows([("immagine", "lum")] + [(i, round(v, 1)) for i, v in sorted(lum.items())])
    rows, gone = [], []
    for r in base:
        i = r["immagine"]
        if i in rev:
            if rev[i]["stato"] in ("cancellata", "vuota"):
                gone.append((i, f"revisione: {rev[i]['stato']}"))
            else:
                rows.append(dict(r, label=rev[i]["label"], rivisto=1))
        elif group(i) in night:
            gone.append((i, "gruppo notturno"))
        elif lum[i] < DARK:
            gone.append((i, "buio"))
        elif r["clip"] in risky:
            gone.append((i, "clip con box tolti in revisione"))
        else:
            rows.append(dict(r, rivisto=0))
    with open(WORK / "frame.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, list(base[0]) + ["rivisto"])
        w.writeheader()
        w.writerows(rows)
    with open(WORK / "esclusi.csv", "w", newline="") as fh:
        csv.writer(fh).writerows([("immagine", "motivo")] + gone)
    print(f"rivisti: {len(rows)} frame ({sum(r['rivisto'] for r in rows)} rivisti), esclusi "
          + ", ".join(f"{k} {v}" for k, v in Counter(m for _, m in gone).most_common()) + f"; gruppi notturni {sorted(night)}")


# --- step split ------------------------------------------------------------------------------------------------------

_ST: dict = {}


def _obj(s: str, x, t, xs, ps, xh) -> float:
    """Error of a split: cells (relative, drones x4 as in `split.py`), minimum per class, source mix (per class,
    against the share of each source in the in-domain pool) and share of horizon boxes per class (against the
    in-domain pool: a fixed 20% sent almost every sky bird to val and test, user 26/10/2026)."""
    from tools.split import _err

    mn = _ST["MIN"][s]  # per class: the frozen minimum, but never more than 20% of what is in domain
    short = np.maximum(0, mn * 1.02 - x.sum(1)) / mn
    n = x.sum(1, keepdims=True)
    src = SOURCE_WEIGHT @ (((xs - ps * n) / np.maximum(ps * n, 1)) ** 2 * (ps > 0.02)).sum(1)
    ph = _ST["PH"]
    hor = (((xh / np.maximum(n[:, 0], 1)) - ph) / np.maximum(ph, 0.01)) ** 2
    return _err(x, t) + 100 * (short ** 2).sum() + src + 0.5 * hor.sum()


def _trial(seed: int) -> tuple:
    free, tg, ok, X, XS, XH, PS = (_ST[k] for k in ("free", "tg", "ok", "X", "XS", "XH", "PS"))
    rng = np.random.default_rng(seed)
    cur = {s: [np.zeros_like(tg[s]), np.zeros_like(PS), np.zeros(4)] for s in tg}

    def tot(s, c):
        return _obj(s, c[0], tg[s], c[1], PS, c[2])

    def add(c, s, g, k):
        return [c[0] + k * X[s][g], c[1] + k * XS[s][g], c[2] + k * XH[s][g]]

    lab = {}
    for g in rng.permutation(free):
        opts = [(0.0, "train")]
        for s in tg:
            if g in ok[s]:
                opts.append((tot(s, add(cur[s], s, g, 1)) - tot(s, cur[s]), s))
        d, name = min(opts)
        lab[g] = name
        if name in cur:
            cur[name] = add(cur[name], name, g, 1)
    moved = True
    while moved:
        moved = False
        for g in free:
            a = lab[g]
            base = sum(tot(s, cur[s]) for s in tg)
            for b in ("train", "val", "test"):
                if b == a or (b in tg and g not in ok[b]):
                    continue
                new = {s: list(cur[s]) for s in tg}
                if a in new:
                    new[a] = add(new[a], a, g, -1)
                if b in new:
                    new[b] = add(new[b], b, g, 1)
                if sum(tot(s, new[s]) for s in tg) < base - 1e-9:
                    lab[g], cur, moved = b, new, True
                    break
    return sum(tot(s, cur[s]) for s in tg), lab


def split(trials: int = 400, share: float = 0.75, seed: int = 0) -> None:
    """Split by groups as `split.py` (targets: val 20% of the curve over half octaves, test uniform in log s, at
    most 20%/25% of what is in domain), with these differences:
    - val and test count and keep only the in-domain frames (`DOMAIN`);
    - a group sent to val or test gives only a random subset of its frames (in turn across its sessions), as large as
      possible while no cell gets more than `share` of its target (variety); the rest of the group is dropped, never
      moved to the train (leakage). Large groups (the Anti-UAV recordings of the same afternoon, merged by `gruppi`)
      can then go to val and test too;
    - extra terms: source mix per class and horizon share (both against the in-domain pool).
    Writes `17_dataset/split.csv` (immagine, split: train, val, test, via_fuori_dominio, via_eccesso)."""
    import multiprocessing as mp

    from tools.distributions import ALPHA, CUT, TOT, ideal, uniform
    from tools.split import MIN_SPLIT

    rng = np.random.default_rng(seed)
    rows = list(csv.DictReader(open(WORK / "frame.csv")))
    grp = {r["sessione"]: r["gruppo"] for r in csv.DictReader(open(INTERMEDIATE / "11_gruppi" / "gruppi.csv"))}
    gids = sorted({grp[r["fonte"] + ":" + r["sessione"]] for r in rows})
    gi = {g: k for k, g in enumerate(gids)}
    srcs = sorted({r["fonte"] for r in rows})
    si = {s: k for k, s in enumerate(srcs)}
    nb = len(HALF_OCTAVES) - 1
    fg = [gi[grp[r["fonte"] + ":" + r["sessione"]]] for r in rows]
    cells = []  # per frame: [(class, half octave)] if in domain, else None
    for r in rows:
        cells.append([(c, int(np.digitize(_side(w, h), HALF_OCTAVES[1:-1]))) for c, _, _, w, h in _boxes(DATA / r["label"])]
                     if r["scenario"] in DOMAIN else None)
    avail = np.zeros((4, nb))
    hor = np.zeros(4)  # horizon boxes per class in the in-domain pool
    for r, cs in zip(rows, cells):
        for c, b in cs or []:
            avail[c, b] += 1
            hor[c] += r["scenario"] == "orizzonte"
    tv = np.minimum(0.2 * np.array([ideal(TOT[c], ALPHA[c], HALF_OCTAVES, CUT[c]) for c in NAMES]), 0.2 * avail)
    tt = np.minimum(np.array([uniform(TEST_TOT, HALF_OCTAVES, CUT[c]) for c in NAMES]), 0.25 * avail)
    # test at most half of the val per class, same shape (user, 26/10/2026: about 70/20/10 also for birds)
    tt *= np.minimum(1, 0.5 * tv.sum(1) / np.maximum(tt.sum(1), 1))[:, None]
    tg = {"val": tv, "test": tt}
    mins = {s: np.minimum(np.minimum(MIN_SPLIT[s], 0.2 * avail.sum(1)), tg[s].sum(1)) for s in ("val", "test")}
    # per group and split: the subset of in-domain frames that fits under `share` of every target cell
    by_g = defaultdict(lambda: defaultdict(list))
    for i in rng.permutation(len(rows)):
        if cells[i]:
            by_g[fg[i]][rows[i]["sessione"]].append(int(i))
    sub = {s: {} for s in tg}
    X = {s: np.zeros((len(gids), 4, nb)) for s in tg}
    XS = {s: np.zeros((len(gids), 4, len(srcs))) for s in tg}
    XH = {s: np.zeros((len(gids), 4)) for s in tg}
    for g, by_s in by_g.items():
        # reviewed frames last in each list: `pop` takes them first
        order, q = [], [sorted(v, key=lambda i: rows[i]["rivisto"] == "1") for v in by_s.values()]
        while q:
            q = [v for v in q if v]
            order += [v.pop() for v in q]
        for s in tg:
            lim = share * tg[s]
            take = []
            for i in order:
                cs = cells[i]
                if any(tg[s][c, b] == 0 for c, b in cs):
                    continue
                if all(X[s][g][c, b] + sum(1 for x in cs if x == (c, b)) <= lim[c, b] for c, b in set(cs)):
                    take.append(i)
                    for c, b in cs:
                        X[s][g][c, b] += 1
                        XS[s][g][c, si[rows[i]["fonte"]]] += 1
                        XH[s][g][c] += rows[i]["scenario"] == "orizzonte"
            sub[s][g] = set(take)
    XSall = sum(XS["val"][g] for g in range(len(gids)))  # source share of the in-domain pool (val view)
    PS = XSall / np.maximum(XSall.sum(1, keepdims=True), 1)
    free = [g for g in range(len(gids)) if g in by_g]
    ok = {s: {g for g in free if X[s][g].sum() > 0} for s in tg}
    _ST.update(free=free, tg=tg, ok=ok, X=X, XS=XS, XH=XH, PS=PS, MIN=mins, PH=hor / np.maximum(avail.sum(1), 1))
    with mp.get_context("fork").Pool(20) as pool:
        best = min(pool.map(_trial, [seed * 100_000 + k for k in range(trials)]), key=lambda x: x[0])
    lab = {g: "train" for g in range(len(gids))} | best[1]
    out = []
    for i, (r, g) in enumerate(zip(rows, fg)):
        s = lab[g]
        if s != "train":
            s = s if i in sub[s][g] else ("via_fuori_dominio" if r["scenario"] not in DOMAIN else "via_eccesso")
        out.append((r["immagine"], s))
    with open(WORK / "split.csv", "w", newline="") as fh:
        csv.writer(fh).writerows([("immagine", "split")] + out)
    c = Counter(s for _, s in out)
    print(f"split: errore {best[0]:.2f}; " + ", ".join(f"{k} {v}" for k, v in c.most_common()))
    for s in ("val", "test"):
        S = sum(X[s][g] for g, x in lab.items() if x == s)
        SS = sum(XS[s][g] for g, x in lab.items() if x == s)
        SH = sum(XH[s][g] for g, x in lab.items() if x == s)
        for k, n in enumerate(NAMES):
            print(f"  {s:<5}{n:<11}{int(S[k].sum()):>6} (obiettivo {round(tg[s][k].sum())}), orizzonte "
                  f"{SH[k] / max(1, S[k].sum()):.0%}, fonti " + " ".join(f"{srcs[j]} {SS[k, j] / max(1, S[k].sum()):.0%}"
                                                                      for j in np.argsort(-SS[k]) if SS[k, j]))


# --- step train ------------------------------------------------------------------------------------------------------

def train(seed: int = 0) -> None:
    """Train selection: (1) off-domain (ground, wall) boxes at most `OFF_DOMAIN_MAX` of each class, (2) no session
    above `SESSION_MAX` of the boxes of a class, (3) cell cap at 70% of the curve over quarter octaves (`split.py`);
    the frames are removed in random order, in turn across sessions. Writes `17_dataset/train.csv` (immagine, label) and
    `17_dataset/tolti.csv` (immagine, motivo)."""
    from tools.distributions import ALPHA, CUT, TOT, ideal

    rng = np.random.default_rng(seed)
    rows = {r["immagine"]: r for r in csv.DictReader(open(WORK / "frame.csv"))}
    tr = [rows[r["immagine"]] for r in csv.DictReader(open(WORK / "split.csv")) if r["split"] == "train"]
    bx = {r["immagine"]: _boxes(DATA / r["label"]) for r in tr}
    kept = {r["immagine"] for r in tr}
    gone = []

    # running counts, updated at every removal (a recount per removal was quadratic)
    n_c, off_c, src_c, ses_c = np.zeros(4), np.zeros(4), Counter(), Counter()

    def ses_of(i):
        return rows[i]["fonte"] + ":" + rows[i]["sessione"]

    def unreviewed(i):
        return rows[i]["rivisto"] != "1"

    def account(i, k):
        for c, *_ in bx[i]:
            n_c[c] += k
            off_c[c] += k * (rows[i]["scenario"] not in DOMAIN)
            src_c[c, rows[i]["fonte"]] += k
            ses_c[c, ses_of(i)] += k

    for i in kept:
        account(i, 1)

    def per_class(sel):
        n = np.zeros(4)
        for i in sel:
            for c, *_ in bx[i]:
                n[c] += 1
        return n

    def drop(cands, why, ok_after):
        """Remove frames of `cands` (in turn across sessions, random order) while `ok_after()` is False."""
        by = defaultdict(list)
        for i in rng.permutation(sorted(cands)):
            by[rows[i]["sessione"]].append(i)
        q = [sorted(v, key=unreviewed) for v in by.values()]  # `pop` takes the frames not reviewed first
        while q and not ok_after():
            q = [v for v in q if v]
            for v in q:
                if ok_after():
                    break
                i = v.pop()
                if i in kept:
                    kept.discard(i)
                    account(i, -1)
                    gone.append((i, why))

    # (1) off-domain share per class
    for c in range(4):
        off = [i for i in kept if rows[i]["scenario"] not in DOMAIN and any(b[0] == c for b in bx[i])]
        drop(off, "fuori dominio oltre quota", lambda c=c: off_c[c] <= OFF_DOMAIN_MAX * n_c[c])
    # (2) session cap per class (on the class total before this step)
    for c in range(4):
        tot = n_c[c]
        ses = defaultdict(list)
        for i in kept:
            if any(b[0] == c for b in bx[i]):
                ses[ses_of(i)].append(i)
        for s_, frames in ses.items():
            drop(frames, "sessione oltre il tetto", lambda c=c, s_=s_, tot=tot: ses_c[c, s_] <= SESSION_MAX * tot)
    # (2b) source cap per class: no source above 1.5x its share in val+test (at least 5%), so that the three
    # splits have a similar mix (user, 26/10/2026: drones, FRED 28% in train and 0% in val)
    ev = Counter()
    for r in csv.DictReader(open(WORK / "split.csv")):
        if r["split"] in ("val", "test"):
            for c, *_ in _boxes(DATA / rows[r["immagine"]]["label"]):
                ev[c, rows[r["immagine"]]["fonte"]] += 1
    for c in range(4):
        tot_ev = sum(v for (cc, _), v in ev.items() if cc == c)
        for src in sorted({rows[i]["fonte"] for i in kept}):
            cap = max(0.05, 1.5 * ev[c, src] / max(1, tot_ev))
            frames = [i for i in kept if rows[i]["fonte"] == src and any(b[0] == c for b in bx[i])]
            drop(frames, f"fonte oltre il tetto ({src})", lambda c=c, src=src, cap=cap: src_c[c, src] <= cap * n_c[c])
    # (3) cell cap (quarter octaves, 70% of the curve), as split.train_mosaics
    edges = QUARTER_OCTAVES
    nb = len(edges) - 1
    target = 0.7 * np.array([ideal(TOT[c], ALPHA[c], edges, CUT[c]) for c in NAMES])
    target[target == 0] = np.inf
    cnt = np.zeros((4, nb))
    cells = {i: [(c, int(np.digitize(_side(w, h), edges[1:-1]))) for c, _, _, w, h in bx[i]] for i in bx}
    for i in kept:
        for c, b in cells[i]:
            cnt[c, b] += 1
    moved = True
    while moved:
        moved = False
        by = defaultdict(list)
        for i in rng.permutation(sorted(kept)):
            by[rows[i]["sessione"]].append(i)
        order = []
        q = [sorted(v, key=unreviewed) for v in by.values()]  # `pop` takes the frames not reviewed first
        while q:
            q = [v for v in q if v]
            order += [v.pop() for v in q]
        for i in order:
            cs = cells[i]
            if all(cnt[c, b] - sum(1 for x in cs if x == (c, b)) >= target[c, b] for c, b in set(cs)):
                for c, b in cs:
                    cnt[c, b] -= 1
                kept.discard(i)
                gone.append((i, "tetto di cella"))
                moved = True
    with open(WORK / "train.csv", "w", newline="") as fh:
        csv.writer(fh).writerows([("immagine", "label")] + [(i, rows[i]["label"]) for i in sorted(kept)])
    with open(WORK / "tolti.csv", "w", newline="") as fh:
        csv.writer(fh).writerows([("immagine", "motivo")] + gone)
    n = per_class(kept)
    print(f"train: {len(kept)} immagini reali su {len(tr)}; tolte " + ", ".join(f"{k} {v}" for k, v in Counter(m for _, m in gone).items()))
    print("  box per classe: " + ", ".join(f"{NAMES[c]} {int(n[c])}" for c in range(4)))


# --- step mosaici ----------------------------------------------------------------------------------------------------

def mosaics(seed: int = 0, factors=(0.5, 0.25, 0.125)) -> None:
    """Mosaics for the train cells under the curve (user, 24/10/2026: tiles taken at random from the same half
    octave, the "drawer"). Only in-domain train frames whose boxes all fall in one half octave are drawers'
    content. For each class and destination half octave under target, the factor with the most candidates is
    used: a mosaic of g x g tiles (g = 1/f) all from the source drawer (destination + 2 log2(1/f) half octaves),
    chosen at random in turn across sessions, reduced with INTER_AREA, light blur, noise and JPEG as `split.py`.
    Each frame is used at most once. Writes `17_dataset/mosaici/{images,labels}` and `17_dataset/mosaici.csv`."""
    import cv2

    from tools.distributions import ALPHA, CUT, TOT, ideal

    rng = np.random.default_rng(seed)
    out = WORK / "mosaici"
    for d in ("images", "labels"):
        (out / d).mkdir(parents=True, exist_ok=True)
        for f in (out / d).iterdir():
            f.unlink()
    rows = {r["immagine"]: r for r in csv.DictReader(open(WORK / "frame.csv"))}
    # frames removed by the review of the new frames are no longer in frame.csv
    train = [r["immagine"] for r in csv.DictReader(open(WORK / "train.csv")) if r["immagine"] in rows]
    allowed = {r["immagine"] for r in csv.DictReader(open(WORK / "split.csv"))  # tiles only from reviewed frames
               if r["split"] == "train" and r["immagine"] in rows and rows[r["immagine"]]["rivisto"] == "1"}
    edges = HALF_OCTAVES
    nb = len(edges) - 1
    fill = 0.7 * np.array([ideal(TOT[c], ALPHA[c], edges, CUT[c]) for c in NAMES])
    cnt = np.zeros((4, nb))
    for i in train:
        for c, _, _, w, h in _boxes(DATA / rows[i]["label"]):
            cnt[c, np.digitize(_side(w, h), edges[1:-1])] += 1
    # source cap of the train step on real train + tiles (user, 26/10/2026: the tiles were 71% Anti-UAV for drones)
    ev, src_n = Counter(), Counter()
    for r in csv.DictReader(open(WORK / "split.csv")):
        if r["split"] in ("val", "test") and r["immagine"] in rows:
            for c, *_ in _boxes(DATA / rows[r["immagine"]]["label"]):
                ev[c, rows[r["immagine"]]["fonte"]] += 1
    for i in train:
        for c, *_ in _boxes(DATA / rows[i]["label"]):
            src_n[c, rows[i]["fonte"]] += 1

    def src_ok(c, f):
        cap = max(0.05, 1.5 * ev[c, f] / max(1, sum(v for (x, _), v in ev.items() if x == c)))
        return src_n[c, f] < cap * cnt[c].sum()

    # drawers: (class set, half octave) of in-domain train frames (also those removed by the cap: the excess is part
    # of the drawer, not its only content) with every box in the same half octave
    drawer = defaultdict(lambda: defaultdict(list))  # (class, half octave) -> session -> frames
    for i in sorted(allowed):
        r = rows[i]
        if r["scenario"] not in DOMAIN:
            continue
        b = _boxes(DATA / r["label"])
        ho = {int(np.digitize(_side(w, h), edges[1:-1])) for _, _, _, w, h in b}
        if len(ho) == 1:
            h0 = next(iter(ho))
            for c in {x[0] for x in b}:
                drawer[c, h0][r["fonte"] + ":" + r["sessione"]].append(i)
    used, recs, n_m = set(), [], 0
    for c in range(4):
        for dst in range(nb):
            while cnt[c, dst] < fill[c, dst]:
                best = None
                for f in factors:
                    srcb = dst + 2 * int(round(math.log2(1 / f)))
                    if srcb >= nb:
                        continue
                    pool = {s: [i for i in v if i not in used] for s, v in drawer[c, srcb].items() if src_ok(c, s.split(":")[0])}
                    n = sum(len(v) for v in pool.values())
                    if n >= round(1 / f) ** 2 and (best is None or n > best[2]):
                        best = (f, pool, n)
                if best is None:
                    break
                f, pool, _ = best
                g = round(1 / f)
                q = [list(rng.permutation(v)) for v in pool.values() if v]
                rng.shuffle(q)
                pick = []
                while len(pick) < g * g and any(q):
                    for v in q:
                        if v and len(pick) < g * g:
                            pick.append(v.pop())
                tw, th = W // g, H // g
                canvas = np.zeros((H, W, 3), np.uint8)
                lab = []
                q_ = int(rng.integers(75, 93))
                name = f"M{n_m:05d}_x{g}"
                for k, i in enumerate(pick):
                    used.add(i)
                    r_, c_ = divmod(k, g)
                    im = cv2.resize(cv2.imread(str(DATA / i)), (tw, th), interpolation=cv2.INTER_AREA)
                    sig, noise = float(rng.uniform(.3, .8)), float(rng.uniform(1, 3))
                    im = np.clip(cv2.GaussianBlur(im, (0, 0), sig) + rng.normal(0, noise, (th, tw, 3)), 0, 255).astype(np.uint8)
                    canvas[r_ * th:(r_ + 1) * th, c_ * tw:(c_ + 1) * tw] = im
                    for cc, x, y, w, h in _boxes(DATA / rows[i]["label"]):
                        lab.append(f"{cc} {(c_ + x) / g:.6f} {(r_ + y) / g:.6f} {w / g:.6f} {h / g:.6f}")
                        cnt[cc, np.digitize(_side(w / g, h / g), edges[1:-1])] += 1
                        src_n[cc, rows[i]["fonte"]] += 1
                    recs.append((name, k, i, f, round(sig, 2), round(noise, 2), q_))
                cv2.imwrite(str(out / "images" / f"{name}.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, q_])
                (out / "labels" / f"{name}.txt").write_text("\n".join(lab) + "\n")
                n_m += 1
    with open(WORK / "mosaici.csv", "w", newline="") as fh:
        csv.writer(fh).writerows([("mosaico", "casella", "sorgente", "fattore", "sigma", "rumore", "jpeg")] + recs)
    src = Counter(rows[r[2]]["fonte"] for r in recs)
    print(f"mosaici: {n_m} da {len(recs)} caselle; fonti " + ", ".join(f"{k} {v}" for k, v in src.most_common()))
    for c in range(4):
        print(f"  {NAMES[c]:<11}" + " ".join(f"{int(cnt[c, b])}/{round(fill[c, b])}" for b in range(nb)) + f"  totale {int(cnt[c].sum())}")


# --- step export -----------------------------------------------------------------------------------------------------

def export(bg_share: float = 0.10, seed: int = 0) -> None:
    """Ultralytics dataset in `data/output/` (frozen: the step refuses if the folder exists, move it by hand):
    names, class ids and layout as `split.export` (0 drone, 1 airplane, 2 helicopter, 3 bird); background: the
    frozen list (`input/background/0_raw/background.csv`) plus the Open Images horizon crops in the split of their
    photo; calib from the real train. `manifest.csv` with the scenario."""
    import hashlib
    import re
    from tools.split import OUT_ID, OUT_NAMES

    rng = np.random.default_rng(seed)
    if OUT.exists():
        sys.exit(f"{OUT} esiste ed e' congelato: non lo sovrascrivo")
    rows_f = {r["immagine"]: r for r in csv.DictReader(open(WORK / "frame.csv"))}
    grp = {r["sessione"]: r["gruppo"] for r in csv.DictReader(open(INTERMEDIATE / "11_gruppi" / "gruppi.csv"))}
    sp = {r["immagine"]: r["split"] for r in csv.DictReader(open(WORK / "split.csv"))}
    rows = []

    def real(s, img, lab):
        r = rows_f[img]
        m = re.search(r"_f(\d+)$", Path(img).stem)
        return dict(split=s, img=img, lab=lab, tipo="reale", gruppo=grp[r["fonte"] + ":" + r["sessione"]], fonte=r["fonte"],
                    sessione=r["sessione"], clip=r["clip"], frame=int(m.group(1)) if m else "", scenario=r["scenario"])

    for r in csv.DictReader(open(WORK / "train.csv")):  # labels of frame.csv: corrected by the last review
        if r["immagine"] in rows_f:
            rows.append(real("train", r["immagine"], rows_f[r["immagine"]]["label"]))
    for f in sorted((WORK / "mosaici" / "images").glob("*.jpg")):
        rows.append(dict(split="train", img=str(f.relative_to(DATA)), lab=str((WORK / "mosaici" / "labels" / f"{f.stem}.txt").relative_to(DATA)),
                         tipo="mosaico", gruppo="", fonte="", sessione="", clip="", frame="", scenario=""))
    for img, s in sp.items():
        if s in ("val", "test") and img in rows_f:
            rows.append(real(s, img, rows_f[img]["label"]))
    for r in csv.DictReader(open(INPUT / "background" / "0_raw" / "background.csv")):
        if (DATA / r["immagine"]).exists():
            rows.append(dict(split=r["split"], img=r["immagine"], lab=None, tipo="background", gruppo="", fonte=r["fonte"],
                             sessione="", clip=r["camera"], frame=Path(r["sorgente"]).stem, scenario="cielo"))
    bgf = INPUT / "openimages" / "6_background" / "background.csv"
    if bgf.exists():  # horizon backgrounds: same split as the bird crops of the same photo, otherwise by hash
        photo_split = {rows_f[i]["sessione"]: s for i, s in sp.items() if i in rows_f and rows_f[i]["fonte"] == "openimages"}
        for r in csv.DictReader(open(bgf)):
            s = photo_split.get(r["image"])
            if s is None or s.startswith("via"):
                h = int(hashlib.sha1(r["image"].encode()).hexdigest(), 16) % 10
                s = "train" if h < 7 else "val" if h < 9 else "test"
            rows.append(dict(split=s, img=r["immagine"], lab=None, tipo="background", gruppo="", fonte="openimages",
                             sessione="", clip=r["image"], frame="", scenario="orizzonte"))

    def h8(s):
        return hashlib.sha1(s.encode()).hexdigest()[:8]

    by_key = defaultdict(list)
    for r in rows:
        by_key[r["tipo"], r["fonte"], r["clip"]].append(r)
    for (kind, f, c), rs in by_key.items():
        for n, r in enumerate(sorted(rs, key=lambda r: (str(r["frame"]).zfill(12), r["img"]))):
            if kind == "reale":
                r["nome"] = f"R_{r['gruppo']}_{h8(f'{f}:{c}')}_{n:06d}"
            elif kind == "background":
                r["nome"] = f"B_{h8(f'{f}:{c}')}_{n:06d}"
            else:
                k, g = re.match(r"M(\d+)_x(\d+)", Path(r["img"]).stem).groups()
                r["nome"] = f"M_x{g}_{int(k):06d}"
    assert len({r["nome"] for r in rows}) == len(rows), "nomi doppi"
    # calib: 512 real train images, 10% background, as split.calib (cells class x 5 bands, in turn across sessions)
    tr = [r for r in rows if r["split"] == "train" and r["tipo"] == "reale"]
    bands = [16, 32, 64, 96]
    by_cell = defaultdict(lambda: defaultdict(list))
    for r in tr:
        for c, _, _, w, h in _boxes(DATA / r["lab"]):
            by_cell[c, int(np.digitize(_side(w, h), bands))][r["sessione"]].append(r)
    n_bg = 51
    per = (512 - n_bg) // len(by_cell)
    chosen = {}
    for cell in sorted(by_cell):
        q = [[v[k] for k in rng.permutation(len(v))] for v in by_cell[cell].values()]
        got = 0
        while got < per and any(q):
            for v in q:
                while v and v[-1]["nome"] in chosen:
                    v.pop()
                if v and got < per:
                    r_ = v.pop()
                    chosen[r_["nome"]] = r_
                    got += 1
    bgs = [r for r in rows if r["split"] == "train" and r["tipo"] == "background"]
    calib = list(chosen.values()) + [bgs[k] for k in rng.choice(len(bgs), n_bg, replace=False)]
    rows += [dict(r, split="calib") for r in calib]
    for s in ("train", "val", "test", "calib"):
        (OUT / "images" / s).mkdir(parents=True)
        (OUT / "labels" / s).mkdir(parents=True)
    for r in rows:
        os.link(DATA / r["img"], OUT / "images" / r["split"] / f"{r['nome']}.jpg")
        if r["lab"]:
            (OUT / "labels" / r["split"] / f"{r['nome']}.txt").write_text("".join(
                f"{OUT_ID[int(c)]} {rest}\n" for c, rest in (ln.split(" ", 1) for ln in (DATA / r["lab"]).read_text().splitlines() if ln.strip())))
    (OUT / "dataset.yaml").write_text(f"# air-detection, tools/dataset.py --step export\npath: {OUT}\ntrain: images/train\n"
                                      "val: images/val\ntest: images/test\nnames:\n" + "".join(f"  {k}: {c}\n" for k, c in enumerate(OUT_NAMES)))
    (OUT / "calib.yaml").write_text(f"# calibrazione int8\npath: {OUT}\ntrain: images/calib\nval: images/calib\nnames:\n"
                                    + "".join(f"  {k}: {c}\n" for k, c in enumerate(OUT_NAMES)))
    with open(OUT / "manifest.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        # `meteo` is empty for now (user, 26/10/2026: the weather of each frame will be added later)
        w.writerow(["immagine", "split", "tipo", "gruppo", "fonte", "sessione", "clip", "frame", "scenario", "meteo", "sorgente"])
        w.writerows((f"images/{r['split']}/{r['nome']}.jpg", r["split"], r["tipo"], r["gruppo"], r["fonte"], r["sessione"],
                     r["clip"], r["frame"], r["scenario"], "", r["img"]) for r in rows)
    (OUT / "raw" / "images").mkdir(parents=True)  # train, val and test together, for whoever redoes the splits
    (OUT / "raw" / "labels").mkdir(parents=True)
    for r in rows:
        if r["split"] != "calib":
            os.link(OUT / "images" / r["split"] / f"{r['nome']}.jpg", OUT / "raw" / "images" / f"{r['nome']}.jpg")
            if r["lab"]:
                os.link(OUT / "labels" / r["split"] / f"{r['nome']}.txt", OUT / "raw" / "labels" / f"{r['nome']}.txt")
    used = {r["clip"].split("/")[0].removeprefix("DVIDS_") for r in rows if r["fonte"] == "dvids"}
    with open(OUT / "fonti_dvids.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "titolo", "autore", "licenza", "url"])
        w.writerows((v["id"], v["titolo"], v["autore"], v["licenza"], v["url"])
                    for v in csv.DictReader(open(INPUT / "dvids" / "0_raw" / "video.csv")) if v["id"] in used)
    oi = {r["clip"] for r in rows if r["fonte"] == "openimages"}
    with open(OUT / "fonti_openimages.csv", "w", newline="") as fh:  # CC BY 2.0: author and link of every photo
        w = csv.writer(fh)
        w.writerow(["image", "autore", "licenza", "url"])
        w.writerows(sorted({(r["image"], r["author"], r["license"], r["url"]) for r in
                            csv.DictReader(open(INPUT / "openimages" / "5_box" / "ritagli.csv")) if r["image"] in oi}))
    c = Counter((r["split"], r["tipo"]) for r in rows)
    tab = "| split | immagini | reali | mosaici | background |\n|---|---|---|---|---|\n" + "".join(
        f"| {s} | {sum(v for (x, _), v in c.items() if x == s):,} | {c[s, 'reale']:,} | {c[s, 'mosaico']:,} | "
        f"{c[s, 'background']:,} |\n".replace(",", ".") for s in ("train", "val", "test", "calib"))
    import datetime

    (OUT / "README.md").write_text((Path(__file__).resolve().parents[1] / "docs" / "dataset_readme.md").read_text()
                                   .replace("{conteggi}", tab.strip()).replace("{data}", f"{datetime.date.today():%d/%m/%Y}"))
    for s in ("train", "val", "test", "calib"):
        print(f"  {s}: {sum(v for (x, _), v in c.items() if x == s)} immagini (" + ", ".join(f"{k} {v}" for (x, k), v in sorted(c.items()) if x == s) + ")")
    box = Counter()
    for r in rows:
        if r["lab"] and r["split"] != "calib":
            for cc, *_ in _boxes(DATA / r["lab"]):
                box[r["split"], NAMES[cc]] += 1
    for s in ("train", "val", "test"):
        print(f"  box {s}: " + ", ".join(f"{n} {box[s, n]}" for n in NAMES))
    print(f"export -> {OUT}")


# --- step figure -----------------------------------------------------------------------------------------------------

SOURCES = [("svanstrom", "Svanström"), ("dvids", "DVIDS"), ("anti_uav", "Anti-UAV"), ("fred", "FRED"),
           ("fbd_sv", "FBD-SV-2024"), ("openimages", "Open Images")]
CLASS_TXT = {"it": {"drone": "drone", "airplane": "aereo", "helicopter": "elicottero", "bird": "uccello"},
             "en": {"drone": "drone", "airplane": "airplane", "helicopter": "helicopter", "bird": "bird"}}


SCENARIOS = {"it": [("cielo", "Cielo"), ("orizzonte", "Orizzonte"), ("terra", "Terra"), ("muro", "Muro")],
             "en": [("cielo", "Sky"), ("orizzonte", "Horizon"), ("terra", "Ground"), ("muro", "Wall")]}
SPLIT_TXT = {"it": {"train": "Addestramento", "val": "Validazione", "test": "Test"},
             "en": {"train": "Training", "val": "Validation", "test": "Test"}}


def _share_plot(rows: list, lang: str, path: Path) -> None:
    """One row of three panels (train, val, test) per attribute in `rows` [(name, [(key, label)], Counter((split,
    class, key) -> boxes))]: four horizontal bars (classes), each split in the share of every key, percentages
    written on the segments of at least 12%, one legend per row. Width of the thesis text, for side-by-side reading."""
    import matplotlib.pyplot as plt

    from tools.plots import WIDTH, _P, save, use
    from tools.split import OUT_NAMES

    use(lang)
    f, axs = plt.subplots(len(rows), 3, figsize=(WIDTH, 2.3 * len(rows)), sharey=True, squeeze=False)
    letter = 0
    for (name, keys, cnt), arow in zip(rows, axs):
        colors = {k: _P[i % len(_P)] for i, (k, _) in enumerate(keys)}
        for ax, sp in zip(arow, ("train", "val", "test")):
            for y, c in enumerate(OUT_NAMES):
                tot = sum(cnt[sp, c, k] for k, _ in keys)
                left = 0.0
                for k, _ in keys:
                    v = cnt[sp, c, k] / tot if tot else 0
                    if v:
                        ax.barh(y, v, left=left, height=0.62, color=colors[k], edgecolor="white", linewidth=.8)
                        if v >= 0.12:
                            ax.text(left + v / 2, y, f"{v * 100:.0f}", ha="center", va="center", fontsize=7, color="white")
                    left += v
            ax.set_title(f"({'abcdefghi'[letter]}) {SPLIT_TXT[lang][sp]}")
            letter += 1
            ax.set_xlim(0, 1)
            ax.set_xticks([0, .5, 1], ["0", "50%", "100%"])
            ax.grid(False)
        arow[0].set_yticks(range(4), [CLASS_TXT[lang][c].capitalize() for c in OUT_NAMES])
        arow[0].invert_yaxis()
        handles = [plt.Rectangle((0, 0), 1, 1, color=colors[k]) for k, _ in keys]
        if len(rows) == 1:  # one legend under the figure (a legend tied to an axis collapses the layout)
            f.legend(handles, [t for _, t in keys], loc="outside lower center", ncol=len(keys))
        else:  # one legend per row, on its right
            arow[2].legend(handles, [t for _, t in keys], loc="center left", bbox_to_anchor=(1.02, 0.5))
    save(f, path)


def figures() -> None:
    """Standard figures of the dataset (user, 26/10/2026), Italian and English, in `docs/figure/` and next to the
    dataset: `distribuzione` (sizes per class: train real and mosaics, val, test, with the targets), `fonti` (share
    of each source per split and class) and `contesto` (share of each scenario; weather as a second row once the
    manifest has it) and `leakage` (DINOv2 similarity of val and test to the nearest training image, with
    `leakage.md` next to the dataset)."""
    import matplotlib.pyplot as plt

    from matplotlib.ticker import NullLocator

    from tools.distributions import ALPHA, BINS, CUT, TOT, curve
    from tools.plots import WIDTH, _P, save, use
    from tools.split import OUT_NAMES

    lab_of = {r["immagine"]: r["label"] for r in csv.DictReader(open(WORK / "frame.csv"))}
    src_of = {r["immagine"]: r["fonte"] for r in csv.DictReader(open(WORK / "frame.csv"))}
    tiles = defaultdict(list)
    for r in csv.DictReader(open(WORK / "mosaici.csv")):
        tiles[r["mosaico"]].append(r["sorgente"])
    size = defaultdict(list)  # (split, class) -> sizes; ("mosaico", class)
    share = Counter()  # (split, class, source) -> boxes
    ctx = Counter()  # (split, class, scenario) -> boxes
    weather = Counter()  # (split, class, weather) -> boxes, empty until the manifest has `meteo`
    sc_of = {r["immagine"]: r["scenario"] for r in csv.DictReader(open(WORK / "frame.csv"))}
    for r in csv.DictReader(open(OUT / "manifest.csv")):
        if r["tipo"] == "background" or r["split"] == "calib":
            continue
        if r["tipo"] == "mosaico":
            name = Path(r["sorgente"]).stem
            for c, _, _, w, h in _boxes(DATA / r["sorgente"].replace("/images/", "/labels/").replace(".jpg", ".txt")):
                size["train", NAMES[c]].append(_side(w, h))
                size["mosaico", NAMES[c]].append(_side(w, h))
            for src in tiles[name]:
                for c, *_ in _boxes(DATA / lab_of[src]):
                    share["train", NAMES[c], src_of[src]] += 1
                    ctx["train", NAMES[c], sc_of[src]] += 1
            continue
        for c, _, _, w, h in _boxes(DATA / lab_of[r["sorgente"]]):
            size[r["split"], NAMES[c]].append(_side(w, h))
            share[r["split"], NAMES[c], r["fonte"]] += 1
            ctx[r["split"], NAMES[c], r["scenario"]] += 1
            if r.get("meteo"):
                weather[r["split"], NAMES[c], r["meteo"]] += 1
    fig_dir = Path(__file__).resolve().parents[1] / "docs" / "figure"
    T = {"it": dict(real="Addestramento, reali", mos="Addestramento, mosaici", val="Validazione", test="Test",
                    ttr="Obiettivo addestramento", tval="Obiettivo validazione", tte="Obiettivo test (uniforme)",
                    x="Lato della box √(w·h) [px]", y="Istanze", sp={"train": "Addestramento", "val": "Validazione", "test": "Test"},
                    share="quota delle box"),
         "en": dict(real="Training, real", mos="Training, mosaics", val="Validation", test="Test",
                    ttr="Training target", tval="Validation target", tte="Test target (uniform)",
                    x="Box side √(w·h) [px]", y="Instances", sp={"train": "Training", "val": "Validation", "test": "Test"},
                    share="share of boxes")}
    for lang, t in T.items():
        use(lang)
        f, axs = plt.subplots(2, 2, figsize=(WIDTH, 4.6), sharex=True, sharey=True)
        dlog = np.log(BINS[1] / BINS[0])
        for k, (ax, c) in enumerate(zip(axs.flat, OUT_NAMES)):
            h_tr = np.histogram(size["train", c], BINS)[0]
            h_mo = np.histogram(size["mosaico", c], BINS)[0]
            ax.stairs(h_tr - h_mo, BINS, fill=True, color=_P[0], alpha=.35, lw=0, label=t["real"])
            ax.stairs(h_tr, BINS, baseline=h_tr - h_mo, fill=True, color=_P[3], alpha=.75, lw=0, label=t["mos"])
            ax.stairs(h_tr, BINS, color=_P[0], lw=.5)
            for sp, col in (("val", _P[1]), ("test", _P[2])):
                ax.stairs(np.histogram(size[sp, c], BINS)[0], BINS, color=col, label=t[sp])
            ax.plot(*curve(TOT[c] * .7, ALPHA[c], CUT[c]), color="0.1", lw=.9, label=t["ttr"])
            ax.plot(*curve(TOT[c] * .2, ALPHA[c], CUT[c]), color="0.1", lw=.9, ls=(0, (4, 2)), label=t["tval"])
            n_t = len(size["test", c])
            ax.plot([8, CUT[c]], [n_t * dlog / np.log(CUT[c] / 8)] * 2, color="0.1", lw=.9, ls=(0, (1, 1.5)), label=t["tte"])
            ax.set_xscale("log")
            ax.set_xlim(7, 400)  # the boxes start at 8 px
            ax.set_xticks([8, 16, 32, 64, 128, 256], ["8", "16", "32", "64", "128", "256"])
            ax.xaxis.set_minor_locator(NullLocator())  # the base-10 minor ticks do not fall on the powers of 2
            ax.set_title(f"({'abcd'[k]}) {CLASS_TXT[lang][c].capitalize()}")
            ax.tick_params(labelbottom=True)
        axs[0, 0].set_yscale("log")
        axs[0, 0].set_ylim(3, max(np.histogram(size["train", c], BINS)[0].max() for c in OUT_NAMES) * 1.5)
        for ax in axs[1]:
            ax.set_xlabel(t["x"])
        for ax in axs[:, 0]:
            ax.set_ylabel(t["y"])
        hs, ls = axs[0, 0].get_legend_handles_labels()
        f.legend(hs, ls, loc="outside lower center", ncol=4)
        save(f, fig_dir / "distribution")
        # standard figures of the suite (user, 26/10/2026): share of the boxes per split and class by source,
        # and by context (scenario; weather as a second row when `meteo` is filled in the manifest)
        _share_plot([("fonte", SOURCES, share)], lang, fig_dir / "sources")
        rows_ctx = [("scenario", SCENARIOS[lang], ctx)]
        if any(v for k, v in weather.items()):
            rows_ctx.append(("meteo", sorted({(k[2], k[2]) for k in weather}), weather))
        _share_plot(rows_ctx, lang, fig_dir / "context")
    # leakage: DINOv2 of every real image, nearest training image (as split._leakage_plot)
    import torch

    from tools.split import _leakage_plot, embed

    real = [r for r in csv.DictReader(open(OUT / "manifest.csv")) if r["tipo"] == "reale" and r["split"] != "calib"]
    e = torch.from_numpy(embed([str(OUT / r["immagine"]) for r in real])).cuda()
    gi = {x: k for k, x in enumerate(sorted({r["gruppo"] for r in real}))}
    ci = {x: k for k, x in enumerate(sorted({r["fonte"] + ":" + r["clip"] for r in real}))}
    g = torch.tensor([gi[r["gruppo"]] for r in real], device="cuda")
    c = torch.tensor([ci[r["fonte"] + ":" + r["clip"]] for r in real], device="cuda")
    data = (e, g, c, [r["split"] for r in real])
    _leakage_plot(OUT, lang="it", data=data)
    _leakage_plot(fig_dir, lang="en", table=False, data=data)
    import shutil

    # the Italian figures also next to the dataset, with the names of its README
    for src, dst in (("distribution", "distribuzione"), ("sources", "fonti"), ("context", "contesto")):
        for ext in ("png", "pdf"):
            shutil.copy(fig_dir / f"it_{src}.{ext}", OUT / f"{dst}.{ext}")
    print(f"figure -> {fig_dir} e {OUT}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--step", choices=["tabella", "rivisti", "split", "train", "mosaici", "export", "figure"], required=True)
    a = p.parse_args()
    {"tabella": table, "rivisti": reviewed, "split": split, "train": train, "mosaici": mosaics, "export": export, "figure": figures}[a.step]()
