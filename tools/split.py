#!/usr/bin/env python3
"""Steps 11-14 of the procedure, on all sources together (`data/intermediate/`).

Step 11, groups (23/10/2026): DINOv2 embedding of every annotated frame of all sources; the video or
recording stays the minimum unit, the embeddings only serve to merge. One file per item, in `intermediate/9_split/`.

    python tools/split.py --step frame        # frame.csv: kept annotated frames, all sources
    python tools/split.py --step embedding    # embedding_s.npy: DINOv2 ViT-S per frame
    python tools/split.py --step somiglianze  # somiglianze_s.png on a sample
    python tools/split.py --step somiglianze_tutte  # somiglianze_tutte.png on all frames
    python tools/split.py --step anteprima [--fasce 0.95 0.99 1.01]  # pairs of different sessions per band
    python tools/split.py --step doppioni     # numbered sheets of session pairs >= 0.95, to review
    python tools/split.py --step gruppi       # 11_gruppi/gruppi.csv: known sessions + unire.txt
    python tools/split.py --step somiglianze_gruppi  # like somiglianze_tutte, with the group curves
    python tools/split.py --step split        # 12_split: groups into train, val, test with the per-cell targets
    python tools/split.py --step controllo    # groups per val and test cell, most similar neighbour in train
    python tools/split.py --step mosaici      # 13_train_compensation: cap = curve and reduced mosaics from the excess
    python tools/split.py --step calib        # 15_calib: 512 train images for int8, balanced by class and band
    python tools/split.py --step export       # data/output: Ultralytics dataset, hardlinks (train, val, test, calib, background)
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.common import DATA, INPUT, INTERMEDIATE, OUTPUT  # noqa: E402


def discarded() -> set[str]:
    """Clips removed from the dataset in every source: Svanstrom trails and rejects,
    FRED rejects (`*/*_box/scie.txt`, `*/*_box/scartate.txt`)."""
    out = set()
    for f in list(INPUT.glob("*/*_box/scie.txt")) + list(INPUT.glob("*/*_box/scartate.txt")):
        out |= {ln.split("#")[0].strip() for ln in open(f) if ln.split("#")[0].strip()}
    return out


def frame_table() -> None:
    """Step 11, table of the kept annotated frames (with at least one box) of all sources ->
    `intermediate/9_split/frame.csv` (fonte, video, clip, sessione, immagine, label, box). Same frames as the
    counts: Svanstrom from `6_dedup/tenuti.csv` with the labels of `4_box/manuale/`, the other sources as
    `pipeline.count` (CVAT labels for the reviewed sequences, otherwise preselection, without the excluded)."""
    from tools.pipeline import boxes, excluded

    rows, dropped = [], discarded()
    sv = INPUT / "svanstrom"
    ses = {r["clip"]: r["sessione"] for r in csv.DictReader(open(sv / "3_sessioni" / "sessioni.csv"))}
    for r in csv.DictReader(open(sv / "6_dedup" / "tenuti.csv")):
        clip, v = r["clip"], r["clip"].split("/")[0]
        stem = f"{v}_f{int(r['frame']):05d}"
        rows.append(("svanstrom", v, clip, ses[clip], sv / "1_frame" / v / "images" / f"{stem}.jpg",
                     sv / "4_box" / "manuale" / clip.replace("/", "_") / f"{stem}.txt"))
    for s in ("fred", "dvids", "anti_uav", "fbd_sv"):
        src = INPUT / s
        ses = {r["clip"]: r["sessione"] for r in csv.DictReader(open(src / "6_sessioni" / "sessioni.csv"))}
        skip = excluded(src)
        for d in sorted((src / "5_box" / "proposte").iterdir()):
            clip = f"{d.name}/s0"
            if clip not in ses:
                continue
            manual = src / "5_box" / "manuale" / d.name
            for t in sorted(d.glob("*.txt")):
                if t.stem not in skip:
                    rows.append((s, d.name, clip, ses[clip], src / "4_ritagli" / d.name / "images" / f"{t.stem}.jpg",
                                 (manual if manual.exists() else boxes(src) / d.name) / t.name))
    out = INTERMEDIATE / "9_split"
    out.mkdir(parents=True, exist_ok=True)
    n, kept = Counter(), []
    for s, v, clip, se, img, lab in rows:
        nb = sum(1 for ln in lab.read_text().splitlines() if ln.strip()) if lab.exists() else 0
        if nb and clip not in dropped:
            assert img.exists(), img
            kept.append((s, v, clip, se, img.relative_to(DATA), lab.relative_to(DATA), nb))
            n[s] += 1
    with open(out / "frame.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["fonte", "video", "clip", "sessione", "immagine", "label", "box"])
        w.writerows(kept)
    print(f"frame: {len(kept)} frame annotati ({', '.join(f'{s} {k}' for s, k in n.items())}), "
          f"{sum(r[6] for r in kept)} box -> {out / 'frame.csv'}")


MODELS = {"s": "dinov2_vits14"}  # B and L tried on 23/10/2026: they do not separate sessions better


def embed(paths: list, batch: int = 512, workers: int = 16) -> np.ndarray:
    """DINOv2 ViT-S/14 embedding (CLS, norm 1) of the JPEGs `paths`: decoded and resized on the GPU (-> 280x224),
    float16, N x 384, same order."""
    import torch
    import torchvision.io as tio

    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)

    class Files(torch.utils.data.Dataset):  # bytes only: decoding is on the GPU
        def __len__(self):
            return len(paths)

        def __getitem__(self, i):  # bytes, not tensors: 512 tensors per batch exhaust the shared fds
            return Path(paths[i]).read_bytes()

    model = torch.hub.load("facebookresearch/dinov2", MODELS["s"], trust_repo=True).cuda().eval().half()
    emb = np.empty((len(paths), model.embed_dim), np.float16)
    i = 0
    dl = torch.utils.data.DataLoader(Files(), batch_size=batch, num_workers=workers, collate_fn=list, prefetch_factor=4)
    with torch.no_grad():
        for k, raw in enumerate(dl):
            imgs = tio.decode_jpeg([torch.frombuffer(bytearray(r), dtype=torch.uint8) for r in raw], device="cuda")
            x = torch.cat([torch.nn.functional.interpolate(im[None].float(), size=(224, 280), mode="bilinear",
                                                           antialias=True) for im in imgs])
            emb[i:i + len(x)] = torch.nn.functional.normalize(model(((x / 255 - mean) / std).half()).float(), dim=1).cpu().numpy()
            i += len(x)
            if k % 20 == 0:
                print(f"  {i}/{len(paths)}", flush=True)
    assert i == len(paths)
    return emb


def frame_embedding() -> None:
    """Step 11, `embed` of every row of `frame.csv` -> `intermediate/9_split/embedding_s.npy` (same order)."""
    out = INTERMEDIATE / "9_split"
    e = embed([str(DATA / r["immagine"]) for r in csv.DictReader(open(out / "frame.csv"))])
    np.save(out / "embedding_s.npy", e)
    print(f"embedding: {e.shape} -> {out / 'embedding_s.npy'}")


def similarity_plot(per_clip: int = 10, n_pairs: int = 100_000, seed: int = 0) -> None:
    """Cosine similarity (norm-1 embeddings: cosine = dot product), one plot per model on the same frames:
    for each source two sessions with two clips each (DVIDS: four videos), `per_clip` evenly spaced frames per
    clip. Matrix on the left, on the right the distribution over `n_pairs` random pairs per type
    (same clip, same session and other clip, different sessions) with the AUC same session against different
    sessions -> `intermediate/9_split/somiglianze_s.png`."""
    import matplotlib.pyplot as plt

    from tools.plots import STYLE, WIDTH, _P, save

    out = INTERMEDIATE / "9_split"
    rows = list(csv.DictReader(open(out / "frame.csv")))
    rng = np.random.default_rng(seed)
    by_clip = defaultdict(list)
    for i, r in enumerate(rows):
        by_clip[r["clip"]].append(i)
    info = {r["clip"]: (r["fonte"], r["sessione"]) for r in rows}
    clips = []
    for s in ("svanstrom", "fred", "anti_uav", "fbd_sv", "dvids"):
        ses = defaultdict(list)
        for c, (f, se) in info.items():
            if f == s and len(by_clip[c]) >= per_clip:
                ses[se].append(c)
        multi = sorted(se for se, cs in ses.items() if len(cs) >= 2)
        if multi:
            for se in rng.choice(multi, 2, replace=False):
                clips += sorted(rng.choice(sorted(ses[se]), 2, replace=False))
        else:
            clips += [ses[se][0] for se in rng.choice(sorted(ses), 4, replace=False)]
    idx = [by_clip[c][j] for c in clips for j in np.linspace(0, len(by_clip[c]) - 1, per_clip).astype(int)]

    clip = np.array([r["clip"] for r in rows])
    ses = np.array([r["fonte"] + ":" + r["sessione"] for r in rows])
    a, b = rng.integers(0, len(rows), (2, 5_000_000))
    a, b = a[a != b], b[a != b]
    kind = {"stessa clip": clip[a] == clip[b], "stessa sessione, altra clip": (ses[a] == ses[b]) & (clip[a] != clip[b]),
            "sessioni diverse": ses[a] != ses[b]}
    pairs = {k: (a[m][:n_pairs], b[m][:n_pairs]) for k, m in kind.items()}
    labs = [f"{info[c][0]} {info[c][1].split('_')[-1]} · {c.split('/')[0].split('_')[-1]}" for c in clips]
    ticks = [k * per_clip + per_clip / 2 - .5 for k in range(len(clips))]
    plt.style.use(STYLE)
    for name in MODELS:
        emb = np.load(out / f"embedding_{name}.npy").astype(np.float32)
        S = emb[idx] @ emb[idx].T
        f, (ax, ax2) = plt.subplots(1, 2, figsize=(WIDTH * 1.4, WIDTH * 0.75))
        im = ax.imshow(S, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
        for k in range(1, len(clips)):
            ax.axhline(k * per_clip - .5, color="w", lw=.4)
            ax.axvline(k * per_clip - .5, color="w", lw=.4)
        ax.set_xticks(ticks, [str(k + 1) for k in range(len(clips))], fontsize=5)
        ax.set_yticks(ticks, labs, fontsize=5)
        ax.tick_params(length=0)
        ax.grid(False)
        f.colorbar(im, ax=ax, fraction=.046, pad=.02).ax.tick_params(labelsize=6)
        ax.set_title(f"{len(clips)} clip × {per_clip} frame (fonte sessione · video)", fontsize=7)
        sims = {k: (emb[pa] * emb[pb]).sum(1) for k, (pa, pb) in pairs.items()}
        pos = np.concatenate([sims["stessa clip"], sims["stessa sessione, altra clip"]])
        auc = np.searchsorted(np.sort(sims["sessioni diverse"]), pos).mean() / len(sims["sessioni diverse"])
        bins = np.linspace(-.1, 1, 111)
        for (k, v), col in zip(sims.items(), (_P[0], _P[1], _P[2])):
            ax2.hist(v, bins, density=True, histtype="step", lw=1.2, color=col,
                     label=f"{k} (mediana {np.median(v):.2f})".replace(".", ","))
        ax2.set_xlabel("somiglianza coseno", fontsize=7)
        ax2.set_ylabel("densità", fontsize=7)
        ax2.set_yticks([])
        ax2.legend(fontsize=5.5, loc="upper left", frameon=False)
        ax2.set_title(f"{n_pairs:,}".replace(",", ".") + " coppie a caso per tipo; AUC stessa sessione / diverse "
                      + f"{auc:.3f}".replace(".", ","), fontsize=7)
        f.suptitle(f"DINOv2 {MODELS[name].split('_')[1]}, {emb.shape[1]} dimensioni", fontsize=8)
        print(f"{name}: AUC {auc:.4f}, mediane " + ", ".join(f"{k} {np.median(v):.3f}" for k, v in sims.items()),
              save(f, out / f"somiglianze_{name}", pdf=False))


def similarity_all(by_group: bool = False, lo: float = -0.2, bins: int = 240, block: int = 4096, px: int = 2000) -> None:
    """Similarities on all frames, in source, session, clip order. On the left the matrix of the mean similarity
    between blocks of N/px consecutive frames (exact: the mean of the cosines between two blocks is the product of
    the mean embeddings), so the area of each source is proportional to its frames.
    On the right the exact histograms of all frame pairs (i < j) per type, in blocks on the GPU, with the AUC
    same session against different sessions -> `intermediate/9_split/somiglianze_tutte.png`. With `by_group`
    (`11_gruppi/gruppi.csv`) two more curves, same group and other session (the hand-merged pairs) and different
    groups, and an inset on the tail in log scale -> `intermediate/11_gruppi/somiglianze_gruppi.png`."""
    import matplotlib.pyplot as plt
    import torch

    from tools.plots import STYLE, WIDTH, _P, save

    out = INTERMEDIATE / "9_split"
    rows = list(csv.DictReader(open(out / "frame.csv")))
    order = sorted(range(len(rows)), key=lambda i: (rows[i]["fonte"], rows[i]["sessione"], rows[i]["clip"]))
    rows = [rows[i] for i in order]
    e = torch.from_numpy(np.load(out / "embedding_s.npy")[order]).cuda()
    clips = list(dict.fromkeys(r["clip"] for r in rows))
    sess = list(dict.fromkeys(r["fonte"] + ":" + r["sessione"] for r in rows))
    ci = {c: k for k, c in enumerate(clips)}
    si = {s: k for k, s in enumerate(sess)}
    cid = torch.tensor([ci[r["clip"]] for r in rows], device="cuda")
    sid = torch.tensor([si[r["fonte"] + ":" + r["sessione"]] for r in rows], device="cuda")
    grp = {r["sessione"]: r["gruppo"] for r in csv.DictReader(open(INTERMEDIATE / "11_gruppi" / "gruppi.csv"))}\
        if by_group else {s: s for s in sess}
    gi = {g: k for k, g in enumerate(dict.fromkeys(grp.values()))}
    pid = torch.tensor([gi[grp[r["fonte"] + ":" + r["sessione"]]] for r in rows], device="cuda")
    kinds = ["stessa clip", "stessa sessione, altra clip", "sessioni diverse"]\
        + ["stesso gruppo, altra sessione", "gruppi diversi"] * by_group

    # matrix rows: blocks of consecutive frames (same number of frames), so the area is proportional to the frames
    n = len(rows)
    gid = torch.arange(n, device="cuda") * px // n
    m = torch.zeros(px, e.shape[1], device="cuda").index_add_(0, gid, e.float())
    m /= torch.bincount(gid, minlength=px).unsqueeze(1)
    Sc = (m @ m.T).cpu().numpy()

    h = torch.zeros(len(kinds), bins, dtype=torch.long, device="cuda")
    j = torch.arange(n, device="cuda")
    with torch.no_grad():
        for s in range(0, n, block):
            i = j[s:s + block]
            k = (((e[i] @ e.T).float() - lo) / (1 - lo) * bins).long().clamp_(0, bins - 1)
            keep = j[None] > i[:, None]
            same_c = cid[i][:, None] == cid[None]
            same_s = sid[i][:, None] == sid[None]
            same_g = pid[i][:, None] == pid[None]
            masks = (same_c, same_s & ~same_c, ~same_s, same_g & ~same_s, ~same_g)
            for t, mk in enumerate(masks[:len(kinds)]):
                h[t] += torch.bincount(k[mk & keep], minlength=bins)
    h = h.cpu().numpy().astype(float)
    pos, neg = h[0] + h[1], h[2]
    auc = (pos * (np.cumsum(neg) - neg / 2)).sum() / (pos.sum() * neg.sum())
    edges = np.linspace(lo, 1, bins + 1)
    mid = (edges[1:] + edges[:-1]) / 2
    med = [mid[np.searchsorted(np.cumsum(x), x.sum() / 2)] for x in h]

    plt.style.use(STYLE)
    f, (ax, ax2) = plt.subplots(1, 2, figsize=(WIDTH * 1.4, WIDTH * 0.75))
    im = ax.imshow(Sc, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    first = {}
    for i, r in enumerate(rows):
        first.setdefault(r["fonte"], i * px / n)
    starts = sorted(first.values()) + [px]
    for b in starts[1:-1]:
        ax.axhline(b - .5, color=_P[7], lw=.5)
        ax.axvline(b - .5, color=_P[7], lw=.5)
    names = sorted(first, key=first.get)
    ticks = [(a + b) / 2 for a, b in zip(starts, starts[1:])]
    ax.set_xticks(ticks, names, fontsize=5.5, rotation=30)
    ax.set_yticks(ticks, names, fontsize=5.5)
    ax.tick_params(length=0)
    ax.grid(False)
    f.colorbar(im, ax=ax, fraction=.046, pad=.02).ax.tick_params(labelsize=6)
    ax.set_title(f"somiglianza media, una riga ogni ~{n // px} frame (ordine fonte, sessione)", fontsize=7)
    top = mid >= 0.95
    for x, k, col, md in zip(h, kinds, _P, med):
        ax2.stairs(x / x.sum() / (edges[1] - edges[0]), edges, lw=1.2, color=col, ls="--" if k == "gruppi diversi" else "-",
                   label=f"{k}: " + f"{x.sum():,.0f}".replace(",", ".") + " coppie, mediana " + f"{md:.2f}".replace(".", ",")
                   + ", sopra 0,95: " + f"{x[top].sum():,.0f}".replace(",", "."))
    if by_group:  # the tail of different sessions and different groups, in log scale
        ins = ax2.inset_axes([.06, .28, .42, .32])
        for x, col, ls in ((h[2], _P[2], "-"), (h[4], _P[4], "--")):
            ins.stairs(np.maximum(x, .5), edges, color=col, lw=1, ls=ls)
        ins.set_yscale("log")
        ins.set_xlim(.85, 1)
        ins.tick_params(labelsize=5)
        ins.set_title("coda: coppie per bin (log)", fontsize=5.5)
    ax2.set_xlabel("somiglianza coseno", fontsize=7)
    ax2.set_ylabel("densità", fontsize=7)
    ax2.set_yticks([])
    ax2.legend(fontsize=5.5, loc="upper left", bbox_to_anchor=(0, -.12), frameon=False)  # below: it covers the curves
    ax2.set_title(f"tutte le coppie di frame ({n:,})".replace(",", ".") + "; AUC stessa sessione / diverse "
                  + f"{auc:.3f}".replace(".", ","), fontsize=7)
    f.suptitle("DINOv2 ViT-S/14, tutti i frame annotati", fontsize=8)
    print(f"AUC {auc:.4f}, coppie {', '.join(f'{x.sum():,.0f}' for x in h)}, mediane {[round(float(x), 3) for x in med]}",
          save(f, INTERMEDIATE / "11_gruppi" / "somiglianze_gruppi" if by_group else out / "somiglianze_tutte", pdf=False))


TILE = (256, 205, 16)  # thumbnail 640x512 -> 256x205, text strip on top


def _who(r: dict) -> str:
    return f"{r['fonte']} {r['sessione'].split('_')[-1]} {r['video'].split('_')[-1]}"


def _tile(r: dict, text: str | None, lw: int = 2) -> np.ndarray:
    """Thumbnail of a `frame.csv` frame with the boxes in red (thickness `lw` at 640x512) and `text` on top (None: no
    strip)."""
    import cv2

    W, H, TH = TILE
    im = cv2.imread(str(DATA / r["immagine"]))
    for ln in (DATA / r["label"]).read_text().splitlines():
        if ln.strip():
            _, x, y, w, h = map(float, ln.split())
            cv2.rectangle(im, (int((x - w / 2) * 640) - 3, int((y - h / 2) * 512) - 3),
                          (int((x + w / 2) * 640) + 3, int((y + h / 2) * 512) + 3), (0, 0, 255), lw)
    im = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
    if text is None:
        return im
    bar = np.full((TH, W, 3), 255, np.uint8)
    cv2.putText(bar, text, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, .38, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([bar, im])


def _pairs_by_band(edges, seed: int = 0, block: int = 4096, same_clip: bool = False):
    """Frame pairs of `frame.csv` per DINOv2 similarity band (cosine, `embedding_s.npy`): from different sessions,
    or from the same clip with `same_clip`. Per band up to 50 random candidates per block (sim, i, j), the
    number of frame pairs and the set of session pairs -> (rows, cand, n_pairs, ses_pairs)."""
    import torch

    out = INTERMEDIATE / "9_split"
    rows = list(csv.DictReader(open(out / "frame.csv")))
    e = torch.from_numpy(np.load(out / "embedding_s.npy")).cuda()
    si = {s: k for k, s in enumerate(dict.fromkeys(r["fonte"] + ":" + r["sessione"] for r in rows))}
    sid = torch.tensor([si[r["fonte"] + ":" + r["sessione"]] for r in rows], device="cuda")
    ci = {c: k for k, c in enumerate(dict.fromkeys(r["clip"] for r in rows))}
    cid = torch.tensor([ci[r["clip"]] for r in rows], device="cuda")
    n, nb = len(rows), len(edges) - 1
    g = torch.Generator(device="cuda").manual_seed(seed)
    cand = [[] for _ in range(nb)]  # random (sim, i, j), up to 50 per block and band
    n_pairs, ses_pairs = [0] * nb, [set() for _ in range(nb)]
    j = torch.arange(n, device="cuda")
    with torch.no_grad():
        for s in range(0, n, block):
            i = j[s:s + block]
            S = (e[i] @ e.T).float()
            ok = (j[None] > i[:, None]) & ((cid[i][:, None] == cid[None]) if same_clip else (sid[i][:, None] != sid[None]))
            for b in range(nb):
                ii, jj = torch.nonzero(ok & (S >= edges[b]) & (S < edges[b + 1]), as_tuple=True)
                n_pairs[b] += len(ii)
                if len(ii):
                    a, c = sid[i[ii]], sid[jj]
                    ses_pairs[b] |= set(zip(torch.minimum(a, c).tolist(), torch.maximum(a, c).tolist()))
                    k = torch.randperm(len(ii), generator=g, device="cuda")[:50]
                    cand[b] += zip(S[ii[k], jj[k]].tolist(), i[ii[k]].tolist(), jj[k].tolist())
    return rows, cand, n_pairs, ses_pairs


def pair_preview(edges=(0.6, 0.7, 0.8, 0.9, 1.01), per: int = 6, seed: int = 0, block: int = 4096) -> None:
    """Preview of frame pairs from different sessions, one row per similarity band (0.6-0.7, 0.7-0.8, ...):
    `per` random pairs per band, boxes drawn, similarity and source/session/video above each image. Prints per
    band the frame pairs and the session pairs involved
    -> `intermediate/9_split/anteprima_sessioni_diverse_<first threshold>.jpg`."""
    import cv2

    out = INTERMEDIATE / "9_split"
    rows, cand, n_pairs, ses_pairs = _pairs_by_band(edges, seed, block)
    nb = len(edges) - 1
    rng = np.random.default_rng(seed)
    W, H, TH = TILE

    blank = np.full((H + TH, W, 3), 230, np.uint8)
    lines = []
    for b in range(nb):
        c = cand[b]
        pick = [c[x] for x in rng.choice(len(c), min(per, len(c)), replace=False)] if c else []
        pairs = [np.vstack([_tile(rows[i], f"{sv:.2f} {_who(rows[i])}"), _tile(rows[jj], f"{sv:.2f} {_who(rows[jj])}")])
                 for sv, i, jj in pick]
        pairs += [np.vstack([blank, blank])] * (per - len(pairs))
        head = np.full((24, (W + 8) * per, 3), 255, np.uint8)
        txt = (f"{edges[b]:g}-{min(edges[b + 1], 1):g}: " + f"{n_pairs[b]:,}".replace(",", ".") + " coppie di frame; "
               + f"{len(ses_pairs[b]):,}".replace(",", ".") + " coppie di sessioni")
        cv2.putText(head, txt, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 1, cv2.LINE_AA)
        lines += [head, np.hstack([np.pad(x, ((0, 0), (0, 8), (0, 0)), constant_values=255) for x in pairs]),
                  np.full((12, (W + 8) * per, 3), 255, np.uint8)]
        print(txt)
    f = out / f"anteprima_sessioni_diverse_{edges[0]:g}.jpg"
    cv2.imwrite(str(f), np.vstack(lines), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"-> {f}")


def _session_pairs(lo: float, block: int = 4096):
    """Pairs of different sessions with at least one frame pair >= `lo` (DINOv2, `embedding_s.npy`): per pair the
    most similar frame pair (sim, i, j) and how many frame pairs exceed `lo`."""
    import torch

    out = INTERMEDIATE / "9_split"
    rows = list(csv.DictReader(open(out / "frame.csv")))
    e = torch.from_numpy(np.load(out / "embedding_s.npy")).cuda()
    keys = [r["fonte"] + ":" + r["sessione"] for r in rows]
    sess = list(dict.fromkeys(keys))
    si = {s: k for k, s in enumerate(sess)}
    sid = torch.tensor([si[k] for k in keys], device="cuda")
    by_ses = defaultdict(list)
    for i, k in enumerate(keys):
        by_ses[si[k]].append(i)
    best, cnt = {}, Counter()  # (sa, sb) -> (sim, i, j)
    n = len(rows)
    j = torch.arange(n, device="cuda")
    with torch.no_grad():
        for s in range(0, n, block):
            i = j[s:s + block]
            S = (e[i] @ e.T).float()
            ii, jj = torch.nonzero((j[None] > i[:, None]) & (sid[i][:, None] != sid[None]) & (S >= lo), as_tuple=True)
            for sv, a, b in zip(S[ii, jj].tolist(), i[ii].tolist(), jj.tolist()):
                key = tuple(sorted((int(sid[a]), int(sid[b]))))
                if key[0] != int(sid[a]):
                    a, b = b, a
                cnt[key] += 1
                if sv > best.get(key, (0,))[0]:
                    best[key] = (sv, a, b)
    return rows, e, sess, by_ses, best, cnt


def _pair_sheets(dst: Path, order: list, rows, e, sess, by_ses, best, cnt, thr: float, prefix: str,
                 per_sheet: int = 12, seed: int = 0) -> None:
    """`dst/coppie.csv` and numbered sheets `dst/fogli/<prefix>NN.jpg`, one row per session pair: the most similar
    frame pair, then a random frame of A with the frame of B most similar to it."""
    import cv2
    import torch

    (dst / "fogli").mkdir(parents=True, exist_ok=True)
    with open(dst / "coppie.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["n", "sessione_a", "sessione_b", "massimo", "coppie_frame", "frame_a", "frame_b"])
        w.writerows((k + 1, sess[a], sess[b], f"{best[a, b][0]:.3f}", cnt[a, b], rows[best[a, b][1]]["immagine"],
                     rows[best[a, b][2]]["immagine"]) for k, (a, b) in enumerate(order))
    rng = np.random.default_rng(seed)
    W, H, TH = TILE
    for f in (dst / "fogli").glob(f"{prefix}*.jpg"):
        f.unlink()
    for p0 in range(0, len(order), per_sheet):
        lines = []
        for k in range(p0, min(p0 + per_sheet, len(order))):
            a, b = order[k]
            sv, fa, fb = best[a, b]
            oa = rng.choice([x for x in by_ses[a] if x != fa] or [fa])
            cb = torch.tensor(by_ses[b], device="cuda")
            sb = (e[cb] @ e[oa]).float()
            ob, so = int(cb[sb.argmax()]), float(sb.max())
            head = np.full((26, (W + 8) * 4, 3), 255, np.uint8)
            cv2.putText(head, f"{k + 1}.  {sess[a]}  /  {sess[b]}   max {sv:.3f}, {cnt[a, b]} coppie di frame >= {thr:g}",
                        (6, 19), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 0), 1, cv2.LINE_AA)
            ims = [_tile(rows[fa], f"A piu' simile  {_who(rows[fa])}"), _tile(rows[fb], f"B piu' simile  {_who(rows[fb])}"),
                   _tile(rows[oa], f"A a caso  {_who(rows[oa])}"), _tile(rows[ob], f"B piu' simile ad A {so:.2f}")]
            lines += [head, np.hstack([np.pad(x, ((0, 0), (0, 8), (0, 0)), constant_values=255) for x in ims]),
                      np.full((10, (W + 8) * 4, 3), 255, np.uint8)]
        cv2.imwrite(str(dst / "fogli" / f"{prefix}{p0 // per_sheet + 1:02d}.jpg"), np.vstack(lines),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])


def duplicate_sheets(thr: float = 0.95, per_sheet: int = 12, seed: int = 0) -> None:
    """Step 11, duplicates between sessions: every pair of different sessions with at least one frame pair >= `thr`
    -> `intermediate/10_doppioni/coppie.csv` (n, sessions, sources, maximum, frame pairs above threshold, frames
    of the best pair) and numbered sheets `fogli/DNN.jpg` (user, 23/10/2026). The user answers with the numbers to
    merge, which go in `10_doppioni/unire.txt` (one number per line, `#` for comments), read by the groups step."""
    dst = INTERMEDIATE / "10_doppioni"
    rows, e, sess, by_ses, best, cnt = _session_pairs(thr)
    order = sorted(best, key=lambda k: (sess[k[0]].split(":")[0], sess[k[1]].split(":")[0], -best[k][0]))
    _pair_sheets(dst, order, rows, e, sess, by_ses, best, cnt, thr, "D", per_sheet, seed)
    print(f"doppioni: {len(order)} coppie di sessioni con frame >= {thr} ("
          + ", ".join(f"{k} {v}" for k, v in Counter(sess[a].split(":")[0] + "-" + sess[b].split(":")[0]
                                                    for a, b in order).most_common())
          + f"), {-(-len(order) // per_sheet)} fogli -> {dst}")


TUNING = (0.85, 0.90, 0.92, 0.94, 0.95)  # bands of the maximum per session pair, below the duplicates threshold


def threshold_sheets(per: int = 40, seed: int = 0) -> None:
    """Tuning of the 0.95 threshold (user, 23/10/2026): `per` random session pairs per band of `TUNING`
    (maximum similarity between their frames), below the threshold hence never reviewed, in decreasing order of
    similarity (user, 23/10/2026) -> `10_doppioni/taratura/coppie.csv` (with the `fascia` column) and sheets
    `fogli/TNN.jpg` like those of the duplicates. The operator writes in `taratura/stessa_sessione.txt` the numbers
    of the pairs that are the same session (same place, same conditions, same flight or footage); the same place on
    another day does not count. Then `--step stima_soglia`."""
    dst = INTERMEDIATE / "10_doppioni" / "taratura"
    rows, e, sess, by_ses, best, cnt = _session_pairs(TUNING[0])
    rng = np.random.default_rng(seed)
    pick = []
    for lo, hi in zip(TUNING, TUNING[1:]):
        band = sorted(k for k in best if lo <= best[k][0] < hi)
        pick += [band[i] for i in rng.choice(len(band), min(per, len(band)), replace=False)]
        print(f"taratura {lo:.2f}-{hi:.2f}: {len(band)} coppie di sessioni, {min(per, len(band))} nei fogli")
    order = sorted(pick, key=lambda k: -best[k][0])  # decreasing: oddities on top are seen first (user)
    _pair_sheets(dst, order, rows, e, sess, by_ses, best, cnt, TUNING[0], "T", seed=seed)
    print(f"{len(order)} coppie, {-(-len(order) // 12)} fogli -> {dst}; risposte in stessa_sessione.txt")


def band_review_sheets(lo: float = 0.92, hi: float = 0.95) -> None:
    """Full review of the session pairs whose maximum similarity is in [lo, hi) (user, 23/10/2026): the tuning showed
    ~36% true duplicates there and almost none below 0.92. Pairs already judged in `taratura/` are left out (their
    answers are reused). Sheets in descending similarity -> `10_doppioni/fascia_092/` (`coppie.csv`, `fogli/RNN.jpg`);
    the operator writes the numbers of the duplicates in `fascia_092/stessa_sessione.txt` (same recording or same
    footage; same place on another day, or different light or saturation, is not a duplicate)."""
    dst = INTERMEDIATE / "10_doppioni" / "fascia_092"
    rows, e, sess, by_ses, best, cnt = _session_pairs(lo)
    done = {tuple(sorted((r["sessione_a"], r["sessione_b"])))
            for r in csv.DictReader(open(INTERMEDIATE / "10_doppioni" / "taratura" / "coppie.csv"))}
    order = sorted((k for k in best if best[k][0] < hi and tuple(sorted((sess[k[0]], sess[k[1]]))) not in done),
                   key=lambda k: -best[k][0])
    _pair_sheets(dst, order, rows, e, sess, by_ses, best, cnt, lo, "R")
    print(f"{lo}-{hi}: {len(order)} session pairs to review ({len(done)} already judged in taratura), "
          f"{-(-len(order) // 12)} sheets -> {dst}")


def threshold_marks(per_sheet: int = 12) -> None:
    """Operators' judgements on the tuning sheets, at a glance (user, 23/10/2026): reads
    `taratura/giudizi_*.txt` (lines `N S|D[?] reason`) and redraws the sheets in `taratura/fogli_esito/` with a frame
    per pair: green = same session, red = different, orange = uncertain (`?`), with the letter in the header."""
    import cv2

    dst = INTERMEDIATE / "10_doppioni" / "taratura"
    vote = {}
    for f in sorted(dst.glob("giudizi_*.txt")):
        for ln in f.read_text().splitlines():
            t = ln.split()
            if len(t) >= 2 and t[0].isdigit():
                vote[int(t[0])] = t[1]
    W, H, TH = TILE
    step = 26 + TH + H + 10  # header, text strip, thumbnail, gap (like `_pair_sheets`)
    (dst / "fogli_esito").mkdir(exist_ok=True)
    n = Counter()
    for f in sorted((dst / "fogli").glob("T*.jpg")):
        im = cv2.imread(str(f))
        first = (int(f.stem[1:]) - 1) * per_sheet + 1
        for k in range(im.shape[0] // step):
            v = vote.get(first + k, "")
            col = (0, 165, 255) if "?" in v else (0, 170, 0) if v.startswith("S") else (0, 0, 230) if v else (128, 128, 128)
            n["?" if "?" in v else v[:1] or "-"] += 1
            y = k * step
            cv2.rectangle(im, (2, y + 2), (im.shape[1] - 3, y + step - 9), col, 5)
            cv2.putText(im, v or "-", (im.shape[1] - 70, y + 20), cv2.FONT_HERSHEY_SIMPLEX, .7, col, 2, cv2.LINE_AA)
        cv2.imwrite(str(dst / "fogli_esito" / f.name), im, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"esito: stessa {n['S']}, diversa {n['D']}, incerte {n['?']}, senza giudizio {n['-']} -> {dst / 'fogli_esito'}")


def threshold_estimate() -> None:
    """Share of "same session" pairs per band from the review of `threshold_sheets`, with the 95% Wilson
    interval -> `10_doppioni/taratura/stima.csv`. Above 0.95 the share is 1 (146 pairs out of 146, all merged)."""
    dst = INTERMEDIATE / "10_doppioni" / "taratura"
    same = set(_merges(dst / "stessa_sessione.txt"))
    rows = list(csv.DictReader(open(dst / "coppie.csv")))
    out = []
    for lo, hi in zip(TUNING, TUNING[1:]):
        rs = [r for r in rows if lo <= float(r["massimo"]) < hi or hi == TUNING[-1] == float(r["massimo"])]  # 0.950 rounded
        n, k = len(rs), sum(int(r["n"]) in same for r in rs)
        z = 1.96
        c, h = (k + z * z / 2) / (n + z * z), z * np.sqrt(k * (n - k) / n + z * z / 4) / (n + z * z)
        out.append((f"{lo:.2f}-{hi:.2f}", n, k, f"{k / n:.3f}", f"{c - h:.3f}", f"{c + h:.3f}"))
    with open(dst / "stima.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["fascia", "coppie", "stessa_sessione", "quota", "ic95_basso", "ic95_alto"])
        w.writerows(out)
    for r in out:
        print(f"{r[0]}: {r[2]}/{r[1]} stessa sessione ({float(r[3]):.0%}, IC 95% {float(r[4]):.0%}-{float(r[5]):.0%})")


def _merges(f: Path) -> list[int]:
    """Numbers of `unire.txt`: `N`, `N-M`, `N via`; `#` comment."""
    out = []
    for ln in f.read_text().splitlines():
        ln = ln.split("#")[0].split()
        if ln:
            a, _, b = ln[0].partition("-")
            out += range(int(a), int(b or a) + 1)
    return out


def groups() -> None:
    """Step 11, groups -> `intermediate/11_gruppi/`. Union (union-find) of the known sessions of each source with the
    pairs of `10_doppioni/unire.txt`: `gruppi.csv` (sessione, gruppo, frame). No frame deduplication between
    sessions (23/10/2026): the true duplicates (same boxes in the same place) are 17 pairs above 0.95."""
    out = INTERMEDIATE / "11_gruppi"
    out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(INTERMEDIATE / "9_split" / "frame.csv")))
    keys = [r["fonte"] + ":" + r["sessione"] for r in rows]
    sess = list(dict.fromkeys(keys))
    parent = {s: s for s in sess}

    def find(s):
        while parent[s] != s:
            parent[s] = parent[parent[s]]
            s = parent[s]
        return s

    dup = INTERMEDIATE / "10_doppioni"
    pairs = {int(r["n"]): (r["sessione_a"], r["sessione_b"]) for r in csv.DictReader(open(dup / "coppie.csv"))}
    for k in _merges(dup / "unire.txt"):
        a, b = pairs[k]
        parent[find(a)] = find(b)
    gid = {r: f"G{k:04d}" for k, r in enumerate(dict.fromkeys(find(s) for s in sess))}
    nf = Counter(keys)
    with open(out / "gruppi.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sessione", "gruppo", "frame"])
        w.writerows((s, gid[find(s)], nf[s]) for s in sess)
    size = Counter()
    for s in sess:
        size[gid[find(s)]] += nf[s]
    print(f"gruppi: {len(sess)} sessioni -> {len(gid)} gruppi; i piu' grandi "
          + ", ".join(f"{g} {n} ({n / len(rows):.1%})" for g, n in size.most_common(5)))



# half-octave intervals to follow the curve inside the bands (user, 23/10/2026); the last one covers 320 px and up
HALF_OCTAVES = [8, 2 ** 3.5, 16, 2 ** 4.5, 32, 2 ** 5.5, 64, 2 ** 6.5, 128, 2 ** 7.5, 256, 1e9]
QUARTER_OCTAVES = [8 * 2 ** (k / 4) for k in range(21)] + [1e9]  # quarter octaves for the train, up to 256 px+
TEST_TOT = 1430  # boxes per class in the test, uniform in log s (286 x 5 bands as before)


_ST: dict = {}  # split state for the child processes (fork)


CLASS_WEIGHT = np.array([1, 1, 4, 1])[:, None]  # airplane, bird, drone, helicopter: drones x4 (user, 23/10/2026: ~5% error)


def _err(x, t):
    m = t > 0
    return ((CLASS_WEIGHT * ((x - t) / np.where(m, t, 1)) ** 2)[m]).sum()


MIN_SPLIT = {"val": 2000, "test": 1000}  # minimum instances per class (user, 23/10/2026; the train has 10,000)


def _obj(s: str, x, t) -> float:
    """Error of a split: cells (`_err`) plus a strong penalty for every class below the minimum `MIN_SPLIT`."""
    short = np.maximum(0, MIN_SPLIT[s] * 1.02 - x.sum(1)) / MIN_SPLIT[s]
    return _err(x, t) + 100 * (short ** 2).sum()


def _split_trial(seed: int) -> tuple:
    """One attempt of `split_groups`: greedy on a random order of the groups, then refinement moving one group
    at a time while the error decreases -> (error, assignment)."""
    free, tg, ok, X = _ST["free"], _ST["tg"], _ST["ok"], _ST["X"]
    rng = np.random.default_rng(seed)
    cur = {s: np.zeros_like(tg[s]) for s in tg}
    lab = {}
    for g in rng.permutation(free):
        opts = [(0.0, "train")] + [(_obj(s, cur[s] + X[s][g], tg[s]) - _obj(s, cur[s], tg[s]), s) for s in tg if g in ok[s]]
        d, name = min(opts)
        lab[g] = name
        if name in cur:
            cur[name] += X[name][g]
    moved = True
    while moved:
        moved = False
        for g in free:
            a_ = lab[g]
            base = sum(_obj(s, cur[s], tg[s]) for s in tg)
            for b_ in ("train", "val", "test"):
                if b_ == a_ or (b_ in tg and g not in ok[b_]):
                    continue
                if a_ in cur:
                    cur[a_] -= X[a_][g]
                if b_ in cur:
                    cur[b_] += X[b_][g]
                if sum(_obj(s, cur[s], tg[s]) for s in tg) < base - 1e-9:
                    lab[g], moved = b_, True
                    break
                if b_ in cur:
                    cur[b_] -= X[b_][g]
                if a_ in cur:
                    cur[a_] += X[a_][g]
    return sum(_obj(s, cur[s], tg[s]) for s in tg), dict(lab)


def split_groups(trials: int = 1000, big: float = 0.02, share: float = 0.75, seed: int = 0) -> None:
    """Step 11, split by groups -> `intermediate/12_split/`. Images with a box under 8 px are dropped (user,
    23/10/2026: in no split). Each group of `11_gruppi/gruppi.csv` goes entirely into one split. Targets per
    class: val = 20% of the curve (`distributions.ideal`, `TOT` per class, raised curve) over half-octave intervals
    (`HALF_OCTAVES`, user 23/10/2026: with the 5 bands the shape inside the band was free), at most 20% of what exists;
    test flat per band (5 bands), 286 per cell, at most 30% of what exists; the rest in train.
    Greedy over `trials` random orders of the groups: each group goes to val or test if it gets closer to the
    targets (squared relative error), otherwise to train, then refinement moving one group at a time while the
    error decreases; groups above `big` of the frames go to train, and a group cannot give more than `share` of the
    target of a val or test cell (variety). The best order is kept.
    Writes `split.csv` (gruppo, split), `frame_split.csv` (immagine, split) and `report.csv` (cell, targets and
    instances per split)."""
    from tools.distributions import ALPHA, CUT, TOT, ideal, uniform

    edges = [8, 16, 32, 64, 96, 1e9]
    out = INTERMEDIATE / "12_split"
    out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(INTERMEDIATE / "9_split" / "frame.csv")))
    grp = {r["sessione"]: r["gruppo"] for r in csv.DictReader(open(INTERMEDIATE / "11_gruppi" / "gruppi.csv"))}
    names = ["airplane", "bird", "drone", "helicopter"]
    gids = sorted(set(grp.values()))
    gi = {g: k for k, g in enumerate(gids)}
    C = np.zeros((len(gids), 4, 5))  # bands, for the test
    F = np.zeros((len(gids), 4, len(HALF_OCTAVES) - 1))  # half octaves, for the val
    nf = np.zeros(len(gids), int)
    fg, small = [], 0
    for r in rows:
        g = gi[grp[r["fonte"] + ":" + r["sessione"]]]
        b = [ln.split() for ln in (DATA / r["label"]).read_text().splitlines() if ln.strip()]
        s = [np.sqrt(float(w) * 640 * float(h) * 512) for _, _, _, w, h in b]
        if min(s) < 8:
            fg.append(-1)
            small += 1
            continue
        fg.append(g)
        nf[g] += 1
        for (c, *_), v in zip(b, s):
            C[g, int(c), np.digitize(v, edges[1:-1])] += 1
            F[g, int(c), np.digitize(v, HALF_OCTAVES[1:-1])] += 1
    tot = C.sum(0)
    tv = np.minimum(0.2 * np.array([ideal(TOT[c], ALPHA[c], HALF_OCTAVES, CUT[c]) for c in names]), 0.2 * F.sum(0))
    # test uniform in log s (user, 23/10/2026: the test ideal is flat), on the half octaves up to the end
    # of the curve (birds 96 px), 1,430 per class (before 286 per band x 5), at most 30% of what exists
    tt = np.minimum(np.array([uniform(TEST_TOT, HALF_OCTAVES, CUT[c]) for c in names]), 0.3 * F.sum(0))
    # at least 1,000 per class in the test (user, 23/10/2026): if the poor bands are not enough, the target of
    # the others is raised proportionally, always within 30% of what exists
    for c in range(4):
        while tt[c].sum() < 1050 and (room := (tt[c] < 0.3 * F.sum(0)[c]) & (tt[c] > 0)).any():
            tt[c, room] = np.minimum(tt[c, room] * 1.02, 0.3 * F.sum(0)[c, room])
    X = {"val": F, "test": F}
    ttb = np.array([uniform(tt[c].sum(), edges, CUT[n]) for c, n in enumerate(names)])  # per band, report only
    big_bird = {g for g in range(len(gids)) if C[g, names.index("bird"), 4]}  # groups with birds > 96 px: train only
    N = nf.sum()

    def err(x, t):
        m = t > 0
        return (((x[m] - t[m]) / t[m]) ** 2).sum()

    free = [g for g in range(len(gids)) if nf[g] and nf[g] <= big * N]
    tg = {"val": tv, "test": tt}
    ok = {s: {g for g in free if g not in big_bird and np.all(X[s][g][tg[s] > 0] <= share * tg[s][tg[s] > 0])
              and not X[s][g][tg[s] == 0].any()} for s in tg}
    # the trials are independent: in parallel on all cores (user, 23/10/2026), state shared via fork
    import multiprocessing as mp

    _ST.update(free=free, tg=tg, ok=ok, X=X)
    with mp.get_context("fork").Pool(20) as pool:  # 20 cores out of 24: headroom for the rest of the machine
        best = min(pool.map(_split_trial, [seed * 100_000 + k for k in range(trials)]), key=lambda x: x[0])
    lab = {g: "train" for g in range(len(gids)) if nf[g]} | best[1]
    with open(out / "split.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["gruppo", "split", "frame"])
        w.writerows((gids[g], s, nf[g]) for g, s in sorted(lab.items()))
    with open(out / "frame_split.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["immagine", "split"])
        w.writerows((r["immagine"], lab[g] if g >= 0 else "via_sotto_8px") for r, g in zip(rows, fg))
    S = {s: sum(C[g] for g, x in lab.items() if x == s) for s in ("train", "val", "test")}
    SF = {s: sum(F[g] for g, x in lab.items() if x == s) for s in ("train", "val", "test")}
    tvb = np.minimum(0.2 * np.array([ideal(TOT[c], ALPHA[c], edges, CUT[c]) for c in names]), 0.2 * tot)  # per band, report only
    bands = ["8-16", "16-32", "32-64", "64-96", ">96"]
    with open(out / "report.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["classe", "fascia", "totale", "obiettivo_val", "val", "obiettivo_test", "test", "train"])
        for c in range(4):
            for b in range(5):
                w.writerow([names[c], bands[b], int(tot[c, b]), round(tvb[c, b]), int(S["val"][c, b]), round(ttb[c, b]),
                            int(S["test"][c, b]), int(S["train"][c, b])])
    fr = Counter(lab[g] if g >= 0 else "via" for g in fg)
    print(f"split: via {small} immagini con box < 8 px; {N} immagini in {len(lab)} gruppi; errore {best[0]:.2f}")
    print("  " + ", ".join(f"{s} {fr[s]} immagini ({fr[s] / N:.1%}), {sum(x == s for x in lab.values())} gruppi"
                           for s in ("train", "val", "test")))
    print(f"  {'':<11}{'fascia':>7}{'tot':>7}{'ob.val':>8}{'val':>6}{'ob.test':>8}{'test':>6}{'train':>7}")
    for c in range(4):
        for b in range(5):
            print(f"  {names[c]:<11}{bands[b]:>7}{int(tot[c, b]):>7}{round(tvb[c, b]):>8}{int(S['val'][c, b]):>6}"
                  f"{round(ttb[c, b]):>8}{int(S['test'][c, b]):>6}{int(S['train'][c, b]):>7}")
    print("  val per mezza ottava (val/obiettivo): " + "".join(f"{HALF_OCTAVES[k]:.0f} " for k in range(len(HALF_OCTAVES) - 1)))
    for c in range(4):
        print(f"  {names[c]:<11}" + " ".join(f"{int(SF['val'][c, k])}/{round(tv[c, k])}" for k in range(len(HALF_OCTAVES) - 1)))
    print("  test per mezza ottava (test/obiettivo):")
    for c in range(4):
        print(f"  {names[c]:<11}" + " ".join(f"{int(SF['test'][c, k])}/{round(tt[c, k])}" for k in range(len(HALF_OCTAVES) - 1)))
    print(f"-> {out}")


def split_check(block: int = 4096) -> None:
    """Split check (`12_split/`): for each val and test cell how many groups and sources fill it, and for
    each val and test image the DINOv2 similarity with the nearest train image (remaining leakage, to
    compare with 0.95, the merge threshold) -> `vicini.png` and the printout."""
    import matplotlib.pyplot as plt
    import torch

    from tools.plots import STYLE, WIDTH, _P, save

    out = INTERMEDIATE / "12_split"
    rows = list(csv.DictReader(open(INTERMEDIATE / "9_split" / "frame.csv")))
    fs = [r["split"] for r in csv.DictReader(open(out / "frame_split.csv"))]
    grp = {r["sessione"]: r["gruppo"] for r in csv.DictReader(open(INTERMEDIATE / "11_gruppi" / "gruppi.csv"))}
    names, edges = ["airplane", "bird", "drone", "helicopter"], [16, 32, 64, 96]
    bands = ["8-16", "16-32", "32-64", "64-96", ">96"]
    cell = defaultdict(lambda: (set(), set()))
    for r, s in zip(rows, fs):
        if s in ("val", "test"):
            for ln in (DATA / r["label"]).read_text().splitlines():
                c, _, _, w, h = ln.split()
                k = (s, names[int(c)], bands[np.digitize(np.sqrt(float(w) * 640 * float(h) * 512), edges)])
                cell[k][0].add(grp[r["fonte"] + ":" + r["sessione"]])
                cell[k][1].add(r["fonte"])
    print(f"  {'':<11}{'fascia':>7}   gruppi (fonti) val   gruppi (fonti) test")
    for c in names:
        for b in bands:
            v, te = cell["val", c, b], cell["test", c, b]
            print(f"  {c:<11}{b:>7}   {len(v[0]):>6} ({len(v[1])}){'':>10}{len(te[0]):>6} ({len(te[1])})")
    e = torch.from_numpy(np.load(INTERMEDIATE / "9_split" / "embedding_s.npy")).cuda()
    tr = torch.tensor([i for i, s in enumerate(fs) if s == "train"], device="cuda")
    near = {}
    for s in ("val", "test"):
        q = [i for i, x in enumerate(fs) if x == s]
        near[s] = torch.cat([(e[q[k:k + block]] @ e[tr].T).float().max(1).values for k in range(0, len(q), block)]).cpu().numpy()
    plt.style.use(STYLE)
    f, ax = plt.subplots(figsize=(WIDTH * .7, WIDTH * .45))
    bins = np.linspace(0, 1, 101)
    for (s, v), col in zip(near.items(), _P):
        ax.hist(v, bins, histtype="step", lw=1.2, color=col,
                label=f"{s}: mediana {np.median(v):.2f}, massimo {v.max():.2f}".replace(".", ","))
    ax.axvline(.95, color="0.5", lw=.7, ls="--")
    ax.set_xlabel("somiglianza con l'immagine piu' vicina del train", fontsize=7)
    ax.set_ylabel("immagini", fontsize=7)
    ax.legend(fontsize=6, frameon=False, loc="upper left")
    print("vicini:", ", ".join(f"{s} mediana {np.median(v):.3f}, 99° {np.percentile(v, 99):.3f}, max {v.max():.3f}, "
                               f">= 0,95: {(v >= .95).sum()}" for s, v in near.items()), save(f, out / "vicini", pdf=False))


def train_mosaics(seed: int = 0, factors=(0.5, 0.25, 0.125)) -> None:
    """Step 12, train cap and reduced mosaics -> `intermediate/13_train_compensation/` (user, 23/10/2026, README, Rules).
    Cap = curve: train target per class and half octave (`HALF_OCTAVES`) = 70% of the curve (10,000 per class); removed, in
    turn across videos, the train images with all boxes in cells above the target (`tetto.csv`). With those images,
    640x512 mosaics for the cells below the target: for each image the factor (0.5, 0.25, 0.125 -> grid 2x2, 4x4,
    8x8) that brings more boxes into missing cells, only if all boxes stay >= 8 px; each image at most once,
    in random order in turn across videos; first the images out of the cap, then (if still short) reduced copies
    of images left in the train. Each tile: INTER_AREA reduction, light Gaussian blur
    (sigma 0.3-0.8), noise (sigma 1-3); the mosaic as JPEG quality 75-92. Writes `images/`, `labels/`,
    `mosaici.csv` (mosaico, casella, sorgente, fattore, sigma, rumore, jpeg) and `train.csv` (real images kept
    plus mosaics, with the label)."""
    import cv2

    from tools.distributions import ALPHA, CUT, TOT, ideal

    edges, names = QUARTER_OCTAVES, ["airplane", "bird", "drone", "helicopter"]  # quarter octaves (user, 23/10/2026: drones 8-12 px)
    nb = len(edges) - 1
    bands = [f"{edges[k]:.0f}-{edges[k + 1]:.0f}" if k < nb - 1 else f">{edges[k]:.0f}" for k in range(nb)]
    out = INTERMEDIATE / "13_train_compensation"
    for d in ("images", "labels"):
        (out / d).mkdir(parents=True, exist_ok=True)
        for f in (out / d).iterdir():
            f.unlink()
    rows = list(csv.DictReader(open(INTERMEDIATE / "9_split" / "frame.csv")))
    fs = [r["split"] for r in csv.DictReader(open(INTERMEDIATE / "12_split" / "frame_split.csv"))]
    target = 0.7 * np.array([ideal(TOT[c], ALPHA[c], edges, CUT[c]) for c in names])  # cap: raised curve
    target[target == 0] = np.inf  # beyond the end of the curve (birds > 96 px) no cap
    fill = 0.7 * np.array([ideal(TOT[c], ALPHA[c], edges, CUT[c]) for c in names])  # mosaics: the same raised curve
    rng = np.random.default_rng(seed)

    def boxes(r):
        return [(int(c), float(x), float(y), float(w), float(h)) for c, x, y, w, h in
                (ln.split() for ln in (DATA / r["label"]).read_text().splitlines() if ln.strip())]

    def cells(bx, f=1.0):
        return [(c, int(np.digitize(np.sqrt(w * 640 * h * 512) * f, edges[1:-1]))) for c, _, _, w, h in bx]

    def round_robin(idx):
        by_v = defaultdict(list)
        for i in rng.permutation(idx):
            by_v[rows[i]["video"]].append(int(i))
        q = list(by_v.values())
        rng.shuffle(q)
        out_ = []
        while q:
            q = [v for v in q if v]
            out_ += [v.pop() for v in q]
        return out_

    train = [i for i, s in enumerate(fs) if s == "train"]
    bx = {i: boxes(rows[i]) for i in train}
    cnt = np.zeros((4, nb))
    for i in train:
        for c, b in cells(bx[i]):
            cnt[c, b] += 1
    # cap: drop the images with all boxes in cells above the target, in turn across videos, as long as possible
    kept, out_cap = set(train), []
    moved = True
    while moved:
        moved = False
        for i in round_robin(sorted(kept)):
            cs = cells(bx[i])
            if all(cnt[c, b] - sum(1 for x in cs if x == (c, b)) >= target[c, b] for c, b in set(cs)):
                for c, b in cs:
                    cnt[c, b] -= 1
                kept.discard(i)
                out_cap.append(i)
                moved = True
    with open(out / "tetto.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["immagine"])
        w.writerows([rows[i]["immagine"]] for i in out_cap)
    print(f"tetto: via {len(out_cap)} immagini del train su {len(train)}; mancano "
          + ", ".join(f"{names[c]} {bands[b]} {round(fill[c, b] - cnt[c, b])}"
                      for c in range(4) for b in range(nb) if fill[c, b] - cnt[c, b] >= 1) + " (rispetto alla curva alzata)")
    # mosaics: for each image out of the cap the factor that brings more boxes into missing cells
    tiles = defaultdict(list)  # factor -> [(i, bx)]
    # first the excess of the cap, then reduced copies of images left in the train (user, 23/10/2026: it is fine if
    # an image appears more than once in the train, as long as the copies are reduced). Mosaic target: the raised
    # curve, the same as the cap; if a class stays under 10,000, raise it only below 64 px, by 0.25 at a time up to x3
    pool, used, m = round_robin(out_cap) + round_robin(sorted(kept)), set(), np.ones(4)
    small = np.array(edges[1:]) <= 64.1
    while True:
        goal = np.where(small[None, :], fill * m[:, None], fill)
        for i in pool:
            if i in used:
                continue
            best = (0, None)
            for f in factors:
                if min(np.sqrt(w * 640 * h * 512) * f for _, _, _, w, h in bx[i]) < 8:
                    continue
                gain = sum(1 for c, b in cells(bx[i], f) if cnt[c, b] < goal[c, b])
                if gain > best[0]:
                    best = (gain, f)
            if best[1]:
                used.add(i)
                for c, b in cells(bx[i], best[1]):
                    cnt[c, b] += 1
                tiles[best[1]].append(i)
        low = cnt.sum(1) < 10_000
        if not low.any() or m[low].min() >= 3:
            break
        m[low] += 0.25
    print("mosaici: moltiplicatore sotto i 64 px " + ", ".join(f"{n} {k:g}" for n, k in zip(names, m)))
    recs, n_m = [], 0
    for f, idx in sorted(tiles.items(), reverse=True):
        g = round(1 / f)
        tw, th = 640 // g, 512 // g
        for m0 in range(0, len(idx), g * g):
            canvas = np.zeros((512, 640, 3), np.uint8)
            lab = []
            group = idx[m0:m0 + g * g]
            while len(group) < g * g:  # the last mosaic is completed with tiles already used in this same mosaic
                group = group + group[:g * g - len(group)]
            q = int(rng.integers(75, 93))
            name = f"M{n_m:05d}_x{g}"
            for k, i in enumerate(group):
                r_, c_ = divmod(k, g)
                im = cv2.resize(cv2.imread(str(DATA / rows[i]["immagine"])), (tw, th), interpolation=cv2.INTER_AREA)
                sig, noise = float(rng.uniform(.3, .8)), float(rng.uniform(1, 3))
                im = cv2.GaussianBlur(im, (0, 0), sig)
                im = np.clip(im + rng.normal(0, noise, im.shape), 0, 255).astype(np.uint8)
                canvas[r_ * th:(r_ + 1) * th, c_ * tw:(c_ + 1) * tw] = im
                if k < len(idx[m0:m0 + g * g]):  # repeated tiles are not counted twice in the csv
                    recs.append((name, k, rows[i]["immagine"], f, round(sig, 2), round(noise, 2), q))
                lab += [f"{c} {(c_ + x) / g:.6f} {(r_ + y) / g:.6f} {w / g:.6f} {h / g:.6f}" for c, x, y, w, h in bx[i]]
            cv2.imwrite(str(out / "images" / f"{name}.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, q])
            (out / "labels" / f"{name}.txt").write_text("\n".join(lab) + "\n")
            n_m += 1
    with open(out / "mosaici.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["mosaico", "casella", "sorgente", "fattore", "sigma", "rumore", "jpeg"])
        w.writerows(recs)
    with open(out / "train.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["immagine", "label", "tipo"])
        w.writerows((rows[i]["immagine"], rows[i]["label"], "reale") for i in sorted(kept))
        w.writerows((f"intermediate/13_train_compensation/images/{p.stem}.jpg", f"intermediate/13_train_compensation/labels/{p.stem}.txt", "mosaico")
                    for p in sorted((out / "images").glob("*.jpg")))
    print(f"mosaici: {n_m} ({', '.join(f'x{round(1 / f)} {-(-len(v) // round(1 / f) ** 2)}' for f, v in sorted(tiles.items(), reverse=True))}), "
          f"{len(recs)} caselle da {sum(len(v) for v in tiles.values())} immagini; train {len(kept)} reali + {n_m} mosaici")
    print(f"  {'':<11}" + "".join(f"{b:>11}" for b in bands) + "   (train con i mosaici / curva alzata)")
    for c in range(4):
        print(f"  {names[c]:<11}" + "".join(f"{f'{int(cnt[c, b])}/{round(fill[c, b])}':>11}" for b in range(nb))
              + f"   totale {int(cnt[c].sum())}")
    print(f"-> {out}")


def calib(n: int = 512, bg: float = 0.1, seed: int = 0) -> None:
    """Step 14, int8 calibration images -> `intermediate/15_calib/calib.csv` (immagine, label, tipo, cella).
    Only from the train and only real images (no mosaics): `bg` of background, the rest in equal parts across the 20
    class x band cells; for each cell in turn across videos, an image counts for the cell it is chosen for."""
    edges, names = [16, 32, 64, 96], ["airplane", "bird", "drone", "helicopter"]
    bands = ["8-16", "16-32", "32-64", "64-96", ">96"]
    rng = np.random.default_rng(seed)
    out = INTERMEDIATE / "15_calib"
    out.mkdir(parents=True, exist_ok=True)
    train = [r for r in csv.DictReader(open(INTERMEDIATE / "13_train_compensation" / "train.csv")) if r["tipo"] == "reale"]
    video = {r["immagine"]: r["video"] for r in csv.DictReader(open(INTERMEDIATE / "9_split" / "frame.csv"))}
    by_cell = defaultdict(lambda: defaultdict(list))  # cell -> video -> images
    for r in train:
        for ln in (DATA / r["label"]).read_text().splitlines():
            c, _, _, w, h = ln.split()
            cell = (names[int(c)], bands[np.digitize(np.sqrt(float(w) * 640 * float(h) * 512), edges)])
            by_cell[cell][video[r["immagine"]]].append(r)
    n_bg = round(n * bg)
    per = (n - n_bg) // len(by_cell)
    extra = (n - n_bg) - per * len(by_cell)
    chosen, rows = set(), []
    for k, cell in enumerate(sorted(by_cell)):
        want = per + (k < extra)
        vids = [list(rng.permutation(v)) for _, v in sorted(by_cell[cell].items())]
        rng.shuffle(vids)
        got = 0
        while got < want and any(vids):
            for v in vids:
                while v and v[-1]["immagine"] in chosen:
                    v.pop()
                if v and got < want:
                    r = v.pop()
                    chosen.add(r["immagine"])
                    rows.append((r["immagine"], r["label"], "reale", f"{cell[0]} {cell[1]}"))
                    got += 1
    bgs = [r for r in csv.DictReader(open(INPUT / "background" / "0_raw" / "background.csv"))
           if r["split"] == "train" and (DATA / r["immagine"]).exists()]
    rows += [(r["immagine"], "", "background", "") for r in rng.choice(bgs, n_bg, replace=False)]
    with open(out / "calib.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["immagine", "label", "tipo", "cella"])
        w.writerows(rows)
    c = Counter(r[3] for r in rows if r[2] == "reale")
    print(f"calib: {len(rows)} immagini ({len(rows) - n_bg} con oggetti, {n_bg} background), per cella min {min(c.values())}, "
          f"max {max(c.values())}, {len({video[r[0]] for r in rows if r[2] == 'reale'})} video -> {out / 'calib.csv'}")


# class order in the exported dataset (user, 23/10/2026): the source labels stay in AOD4 order
# (airplane, bird, drone, helicopter), the export rewrites them with these ids
OUT_NAMES = ["drone", "airplane", "helicopter", "bird"]
OUT_ID = [OUT_NAMES.index(c) for c in ("airplane", "bird", "drone", "helicopter")]


# figure texts per language (user, 23/10/2026: also in English, US audience)
_TXT = {
    "it": {"classes": {"drone": "Drone", "airplane": "Aereo", "helicopter": "Elicottero", "bird": "Uccello"},
           "orig": "Dati originali (prima della selezione)", "real": "Addestramento, immagini reali",
           "mosaics": "Addestramento, mosaici sintetici", "val": "Validazione", "test": "Test",
           "tgt_train": "Obiettivo addestramento", "tgt_val": "Obiettivo validazione", "tgt_test": "Obiettivo test (uniforme)",
           "xsize": r"Dimensione del box $\sqrt{w\,h}$ [px]", "ysize": "Istanze per intervallo",
           "short": ["Stesso gruppo,\naltra clip (leakage)", "Gruppi diversi\n(fondo)", "Validazione", "Test"],
           "xsim": "Somiglianza con l'immagine\npiù vicina dell'addestramento", "a": "(a) Distribuzioni", "b": "(b) Coda",
           "baseline": "Fondo", "leak": "Leakage", "xthr": "Soglia di somiglianza\n", "ythr": "Quota di immagini sopra la soglia",
           "dec": ","},
    "en": {"classes": {"drone": "Drone", "airplane": "Airplane", "helicopter": "Helicopter", "bird": "Bird"},
           "orig": "Original data (before selection)", "real": "Training, real images",
           "mosaics": "Training, synthetic mosaics", "val": "Validation", "test": "Test",
           "tgt_train": "Training target", "tgt_val": "Validation target", "tgt_test": "Test target (uniform)",
           "xsize": r"Box size $\sqrt{w\,h}$ [px]", "ysize": "Instances per bin",
           "short": ["Same group,\nother clip (leakage)", "Different groups\n(baseline)", "Validation", "Test"],
           "xsim": "Similarity to the nearest\ntraining image", "a": "(a) Distributions", "b": "(b) Tail",
           "baseline": "Baseline", "leak": "Leakage", "xthr": "Similarity threshold\n", "ythr": "Share of images above threshold",
           "dec": "."},
}


def _output_plot(rows: list, out: Path, lang: str = "it", src_names=("airplane", "bird", "drone", "helicopter"),
                 table: bool = True) -> None:
    """`distribuzione.{png,pdf}` of the export, for the thesis (Italian texts, no title: it goes in the caption):
    instances per size sqrt(w*h) and class in training (real in blue, mosaics in yellow), validation and test,
    the grey shadow of the original data (all annotated images, before any selection), with the targets
    (curve raised to 70% and 20%, test uniform in log s). Instances and error with respect to the target,
    sum|observed - target| / sum(target) over the half octaves (`HALF_OCTAVES`), in `distribuzione.md` (if `table`).
    `lang` it or en (`_TXT`; in English the file is `distribution_en`); `src_names`: class order in the labels
    of `rows` (sources in the export, `OUT_NAMES` if the labels come from `data/output`)."""
    import matplotlib.pyplot as plt

    from tools.distributions import ALPHA, BINS, CUT, SMIN, TOT, _axes, curve, ideal, uniform
    from tools.plots import STYLE, WIDTH, _P, save

    T = _TXT[lang]
    cname = T["classes"]
    s = defaultdict(list)
    rep = list(csv.DictReader(open(INTERMEDIATE / "12_split" / "report.csv")))

    def sizes(lab, names=("airplane", "bird", "drone", "helicopter")):  # source label ids
        for ln in (DATA / lab).read_text().splitlines():
            c, _, _, w, h = ln.split()
            yield names[int(c)], np.sqrt(float(w) * 640 * float(h) * 512)

    for r in rows:
        if r["lab"] and r["split"] in ("train", "val", "test"):
            for c, v in sizes(r["lab"], src_names):
                s[r["split"], c].append(v)
                if r["tipo"] == "mosaico":
                    s["mosaico", c].append(v)
    # original data: all annotated images, before the < 8 px filter, split, cap and mosaics
    for r in csv.DictReader(open(INTERMEDIATE / "9_split" / "frame.csv")):
        for c, v in sizes(r["label"]):
            s["orig", c].append(v)

    def num(v):
        return f"{v:,}".replace(",", ".")

    plt.style.use(STYLE)
    f, axs = plt.subplots(2, 2, figsize=(WIDTH, WIDTH * 0.8), sharex=True, sharey=True)
    dlog = np.log(BINS[1] / BINS[0])
    md = []
    for k_ax, (ax, c) in enumerate(zip(axs.flat, OUT_NAMES)):
        h_or = np.histogram(s["orig", c], BINS)[0]
        h_tr = np.histogram(s["train", c], BINS)[0]
        h_mo = np.histogram(s["mosaico", c], BINS)[0]
        ax.stairs(h_or, BINS, fill=True, color="0.6", alpha=.22, lw=0, zorder=0, label=T["orig"])
        ax.stairs(h_or, BINS, color="0.5", lw=.5, zorder=1.5)  # in front of the < 8 px zone: what was removed shows too
        ax.stairs(h_tr - h_mo, BINS, fill=True, color=_P[0], alpha=.35, lw=0, label=T["real"])
        ax.stairs(h_tr, BINS, baseline=h_tr - h_mo, fill=True, color=_P[3], alpha=.75, lw=0,
                  label=T["mosaics"])
        ax.stairs(h_tr, BINS, color=_P[0], lw=.6, alpha=.8)
        for sp, col in (("val", _P[1]), ("test", _P[2])):
            ax.stairs(np.histogram(s[sp, c], BINS)[0], BINS, color=col, lw=1.1, label=T[sp])
        ax.plot(*curve(TOT[c] * .7, ALPHA[c], CUT[c]), color="0.1", lw=.9, label=T["tgt_train"])
        ax.plot(*curve(TOT[c] * .2, ALPHA[c], CUT[c]), color="0.1", lw=.9, ls=(0, (4, 2)), label=T["tgt_val"])
        n_t = sum(float(r["obiettivo_test"]) for r in rep if r["classe"] == c)  # test: uniforme in log s
        ax.plot([SMIN, CUT[c]], [n_t * dlog / np.log(CUT[c] / SMIN)] * 2, color="0.1", lw=.9, ls=(0, (1, 1.5)),
                label=T["tgt_test"])
        _axes(ax, c, 0)
        ax.set_title("")  # the centred title of `_axes` (in English) is not needed
        ax.set_title(f"({'abcd'[k_ax]}) {cname[c]}", loc="left", fontsize=9)
        ax.tick_params(labelsize=7.5)
        # error with respect to the target, on the half octaves
        err = {}
        for sp, k in (("train", .7), ("val", .2)):
            ide = k * ideal(TOT[c], ALPHA[c], HALF_OCTAVES, CUT[c])
            err[sp] = np.abs(np.histogram(s[sp, c], HALF_OCTAVES)[0] - ide).sum() / ide.sum()
        ide = uniform(n_t, HALF_OCTAVES, CUT[c])
        err["test"] = np.abs(np.histogram(s["test", c], HALF_OCTAVES)[0] - ide).sum() / ide.sum()
        ax.tick_params(labelbottom=True)  # x-axis numbers in the top panels too
        md.append((c, err, {sp: len(s[sp, c]) for sp in ("train", "val", "test")},
                   {sp: np.histogram(s[sp, c], HALF_OCTAVES)[0] for sp in ("train", "val", "test")},
                   {"train": .7 * ideal(TOT[c], ALPHA[c], HALF_OCTAVES, CUT[c]), "val": .2 * ideal(TOT[c], ALPHA[c], HALF_OCTAVES, CUT[c]),
                    "test": uniform(n_t, HALF_OCTAVES, CUT[c])}, len(s["mosaico", c])))
    top = max(np.histogram(s[k, c], BINS)[0].max() for c in OUT_NAMES for k in ("train", "orig"))
    axs[0, 0].set_yscale("log")  # originals (hundreds), training, val and test (tens) readable together (user)
    axs[0, 0].set_ylim(3, top * 1.5)
    for ax in axs[1]:
        ax.set_xlabel(T["xsize"])
    for ax in axs[:, 0]:
        ax.set_ylabel(T["ysize"])
    hs, ls = axs[0, 0].get_legend_handles_labels()
    # handles: 0 originals, 1 real, 2 mosaics, 3 validation, 4 test, 5 training tgt, 6 validation tgt, 7 test tgt
    order = [0, 5, 1, 6, 2, 7, 3, 4]  # four columns of two: the data above, the targets below
    f.legend([hs[k] for k in order], [ls[k] for k in order], loc="outside lower center", ncol=4, fontsize=7, frameon=False)
    save(f, out / ("distribuzione" if lang == "it" else "distribution_en"), pdf=True)
    if not table:
        return
    # the table that goes with the plot (user, 23/10/2026): same numbers, outside the figure
    lab = [f"{HALF_OCTAVES[k]:.0f}–{HALF_OCTAVES[k + 1]:.0f}" if k < len(HALF_OCTAVES) - 2 else f"> {HALF_OCTAVES[k]:.0f}" for k in range(len(HALF_OCTAVES) - 1)]
    L = ["# Distribuzione delle grandezze", "",
         "Generata con `distribuzione.png` da `tools/split.py --step export`. Errore = Σ|osservato − obiettivo| / Σ obiettivo",
         "sulle mezze ottave; obiettivi: addestramento e validazione la curva s^−0,3 alzata ×1,2 (70% e 20%), test uniforme",
         "in scala logaritmica. Uccelli fino a 96 px.", "",
         "| classe | addestramento | di cui mosaici | errore | validazione | errore | test | errore |", "|---|---|---|---|---|---|---|---|"]
    L += [f"| {cname[c]} | {num(n['train'])} | {num(nm)} | {e['train']:.0%} | {num(n['val'])} | {e['val']:.0%} | {num(n['test'])} | {e['test']:.0%} |"
          for c, e, n, _, _, nm in md]
    for c, e, n, h, ide, _ in md:
        L += ["", f"## {cname[c]}", "", "| √(w·h) [px] | addestramento | obiettivo | validazione | obiettivo | test | obiettivo |",
              "|---|---|---|---|---|---|---|"]
        L += [f"| {lab[k]} | {num(int(h['train'][k]))} | {num(round(ide['train'][k]))} | {num(int(h['val'][k]))} | "
              f"{num(round(ide['val'][k]))} | {num(int(h['test'][k]))} | {num(round(ide['test'][k]))} |" for k in range(len(lab))]
    (out / "distribuzione.md").write_text("\n".join(L).replace("%", " %") + "\n")

def _leakage_plot(out: Path, block: int = 2048, lang: str = "it", table: bool = True) -> None:
    """`leakage.{png,pdf,md}` of the export, for the thesis: for each real validation and test image the DINOv2
    similarity (ViT-S, cosine) with the nearest real training image, against two references inside the
    training set: same group and other clip (what a true leakage looks like) and different groups (the baseline:
    sky resembles sky). (a) density, (b) share of images above each threshold, in log scale; 0.95 is the
    threshold of the hand-reviewed duplicates. The table with median, percentiles and shares in `leakage.md` (if
    `table`). Reads only `intermediate/` (frame, split, groups, embedding), writes to `out`; `lang` it or en
    (`leakage_en`)."""
    import matplotlib.pyplot as plt
    import torch

    from tools.plots import STYLE, WIDTH, _P, save

    rows = list(csv.DictReader(open(INTERMEDIATE / "9_split" / "frame.csv")))
    sp = [r["split"] for r in csv.DictReader(open(INTERMEDIATE / "12_split" / "frame_split.csv"))]
    grp = {r["sessione"]: r["gruppo"] for r in csv.DictReader(open(INTERMEDIATE / "11_gruppi" / "gruppi.csv"))}
    e = torch.from_numpy(np.load(INTERMEDIATE / "9_split" / "embedding_s.npy")).cuda()
    gi = {g: k for k, g in enumerate(sorted(set(grp.values())))}
    ci = {c: k for k, c in enumerate(sorted({r["clip"] for r in rows}))}
    g = torch.tensor([gi[grp[r["fonte"] + ":" + r["sessione"]]] for r in rows], device="cuda")
    c = torch.tensor([ci[r["clip"]] for r in rows], device="cuda")
    tr = torch.tensor([i for i, s in enumerate(sp) if s == "train"], device="cuda")

    def nearest(q, mode):
        """max cosine of each row of q towards the train, with the mask `mode`"""
        outv = []
        for k in range(0, len(q), block):
            qq = q[k:k + block]
            S = (e[qq] @ e[tr].T).float()
            if mode == "same":  # same group, other clip
                S[(g[qq][:, None] != g[tr][None]) | (c[qq][:, None] == c[tr][None])] = -2
            elif mode == "other":
                S[g[qq][:, None] == g[tr][None]] = -2
            outv.append(S.max(1).values)
        v = torch.cat(outv).cpu().numpy()
        return v[v > -2]

    res = {
        "Validazione → addestramento": nearest(torch.tensor([i for i, s in enumerate(sp) if s == "val"], device="cuda"), "all"),
        "Test → addestramento": nearest(torch.tensor([i for i, s in enumerate(sp) if s == "test"], device="cuda"), "all"),
        "Addestramento, stesso gruppo e altra clip (leakage)": nearest(tr, "same"),
        "Addestramento, gruppi diversi (fondo)": nearest(tr, "other"),
    }
    # (a) one horizontal violin per comparison, from the top: leakage, baseline, validation, test
    keys = ["Addestramento, stesso gruppo e altra clip (leakage)", "Addestramento, gruppi diversi (fondo)",
            "Validazione → addestramento", "Test → addestramento"]
    T = _TXT[lang]
    short = T["short"]
    cols = ["0.2", "0.6", _P[1], _P[2]]
    plt.style.use(STYLE)
    f, (a1, a2) = plt.subplots(1, 2, figsize=(WIDTH, WIDTH * 0.45))  # same width for the two panels
    pos = np.arange(len(keys))[::-1]
    vp = a1.violinplot([res[k] for k in keys], positions=pos, vert=False, widths=.85, showextrema=False)
    for body, col in zip(vp["bodies"], cols):
        body.set_facecolor(col)
        body.set_edgecolor(col)
        body.set_alpha(.45)
    for y, k, col in zip(pos, keys, cols):
        q1, med, q3 = np.percentile(res[k], [25, 50, 75])
        a1.plot([q1, q3], [y, y], color=col, lw=3, solid_capstyle="butt")
        a1.plot([med], [y], marker="o", ms=4.5, color="white", mec=col, mew=1.2, zorder=5)
        a1.text(1.01, y, f"{med:.2f}".replace(".", T["dec"]), va="center", ha="left", fontsize=7, color="0.2",
                transform=a1.get_yaxis_transform())
    a1.set_yticks(pos, short, fontsize=7.5)
    a1.set_xlim(.3, .999)
    a1.axvline(.95, color="0.4", lw=.7)
    a1.text(.95, pos[-1] - .5, f" 0{T['dec']}95", ha="left", va="center", fontsize=7, color="0.3")
    a1.set_xlabel(T["xsim"], fontsize=8)
    a1.set_title(T["a"], loc="left", fontsize=9)
    a1.grid(axis="y", visible=False)
    # (b) tail: share of images above each threshold
    x = 1 - np.geomspace(.5, .002, 600)  # dense near 1, where the axis widens
    tail = {k: np.array([(res[k] >= s).mean() for s in x]) for k in keys}
    a2.fill_between(x, 1e-5, np.maximum(tail[keys[1]], 1e-5), color="0.6", alpha=.3, lw=0, label=T["baseline"])
    a2.plot(x, tail[keys[0]], color=cols[0], lw=1.6, ls=(0, (4, 2)), label=T["leak"])
    for k, col, lab in zip(keys[2:], cols[2:], short[2:]):
        a2.plot(x, tail[k], color=col, lw=1.6, label=lab)
    a2.axvline(.95, color="0.4", lw=.7)
    a2.set_yscale("log")
    a2.set_xlim(.5, .999)
    a2.set_ylim(1e-4, 1.3)
    a2.set_xlabel(T["xthr"], fontsize=8)  # two lines like (a), aligned axes
    a2.set_ylabel(T["ythr"], fontsize=8)
    a2.set_title(T["b"], loc="left", fontsize=9)
    a2.text(.95, 1.5e-4, f" 0{T['dec']}95", ha="left", va="bottom", fontsize=7, color="0.3")
    f.legend(*a2.get_legend_handles_labels(), loc="outside lower center", ncol=4, fontsize=7, frameon=False)
    # x axis that widens the zone near 1 (user, 23/10/2026): -log10(1 - s), ticks in similarity
    for a in (a1, a2):
        a.set_xscale("function", functions=(lambda v: -np.log10(np.clip(1 - v, 1e-4, None)),
                                             lambda u: 1 - 10 ** -u))
        a.set_xticks([.5, .9, .99, .999])
        a.xaxis.set_minor_locator(plt.NullLocator())
        a.tick_params(labelsize=7.5)
        a.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}".replace(".", T["dec"])))
    save(f, out / ("leakage" if lang == "it" else "leakage_en"), pdf=True)
    if not table:
        return

    def fmt(v):
        return f"{v:.3f}".replace(".", ",")

    L = ["# Leakage fra gli split", "",
         "Generato con `leakage.png` da `tools/split.py --step export`. Per ogni immagine reale, la somiglianza coseno",
         "DINOv2 ViT-S/14 con l'immagine reale più vicina dell'addestramento. Riferimenti dentro l'addestramento: stesso",
         "gruppo e altra clip (com'è fatto un leakage vero) e gruppi diversi (il fondo: il cielo somiglia al cielo).",
         "0,95 è la soglia dei doppioni fra sessioni rivisti a mano e uniti nello stesso gruppo: che validazione e test",
         "non la superino è in parte per costruzione. La prova è che le loro curve coincidono con il fondo anche fra 0,85",
         "e 0,95, cioè non somigliano all'addestramento più di quanto due gruppi qualsiasi dell'addestramento si",
         "somiglino fra loro. Nel grafico (a): violino = distribuzione, pallino = mediana (scritta a destra), barra =",
         "primo e terzo quartile; (b): quota di immagini con la somiglianza sopra ogni soglia, in scala logaritmica.", "",
         "| confronto | immagini | mediana | 95° perc. | 99° perc. | massimo | ≥ 0,90 | ≥ 0,95 |", "|---|---|---|---|---|---|---|---|"]
    def pct(x, d):
        return f"{x * 100:.{d}f}".replace(".", ",") + " %"

    L += [f"| {k} | " + f"{len(v):,}".replace(",", ".") + f" | {fmt(np.median(v))} | {fmt(np.percentile(v, 95))} | "
          f"{fmt(np.percentile(v, 99))} | {fmt(v.max())} | {pct((v >= .9).mean(), 1)} | {pct((v >= .95).mean(), 2)} |"
          for k, v in res.items()]
    (out / "leakage.md").write_text("\n".join(L) + "\n")


FIG = ROOT / "docs" / "figure"  # README figures, in Italian and English


def figures() -> None:
    """Figures of the finished dataset for the README, in `docs/figure/`, in Italian and English (`distribuzione`,
    `distribution_en`, `leakage`, `leakage_en`): same functions as the export, but `data/output` (frozen) is only
    read (manifest and labels, with the ids of `OUT_NAMES`); no .md tables."""
    out = OUTPUT
    rows = [dict(split=r["split"], tipo=r["tipo"],
                 lab=None if r["tipo"] == "background" else
                 str(Path("output") / "labels" / r["split"] / (Path(r["immagine"]).stem + ".txt")))
            for r in csv.DictReader(open(out / "manifest.csv"))]
    for lang in ("it", "en"):
        _output_plot(rows, FIG, lang, src_names=OUT_NAMES, table=False)
        _leakage_plot(FIG, lang=lang, table=False)
    print(f"figure -> {FIG}")


def threshold_panel(edges=(0.80, 0.85, 0.90, 0.95, 0.98, 1.01), per: int = 3, seed: int = 0) -> None:
    """Panel for the thesis on the 0.95 threshold of duplicates between sessions (`docs/figure/soglia.{png,pdf}` and
    `soglia_en`): one row per DINOv2 similarity band with `per` random pairs (fixed seed) of frames from
    different sessions, boxes in red, and at the bottom same-clip pairs as reference. Below 0.95 different shots
    with the same kind of sky; above, the same shot or the same footage republished (the 146 session pairs
    above 0.95 reviewed by hand and merged, step 10). Reference: the same-clip pairs at the 25th, 50th and 75th
    percentile of their similarity (random ones would be too scattered: a DVIDS clip changes framing)."""
    import matplotlib.pyplot as plt

    from tools.plots import STYLE, WIDTH, save

    rows, cand, n_pairs, ses_pairs = _pairs_by_band(edges, seed)
    _, ref, n_ref, _ = _pairs_by_band((-1, 1.01), seed, same_clip=True)
    rng = np.random.default_rng(seed)
    picks = [[c[x] for x in rng.choice(len(c), min(per, len(c)), replace=False)] for c in cand]
    r0 = sorted(ref[0])
    picks.append([r0[int(q * (len(r0) - 1))] for q in np.linspace(.25, .75, per)])
    tiles = [[(sv, _tile(rows[i], None, lw=6)[:, :, ::-1], _tile(rows[j], None, lw=6)[:, :, ::-1]) for sv, i, j in pk]
             for pk in picks]
    nb = len(edges) - 1
    txt = {"it": dict(dec=",", band="Sessioni diverse", ref="Stessa clip\n(riferimento)", thr="soglia 0,95",
                      pairs="coppie di sessioni"),
           "en": dict(dec=".", band="Different sessions", ref="Same clip\n(reference)", thr="threshold 0.95",
                      pairs="session pairs")}
    plt.style.use(STYLE)
    for lang, T in txt.items():
        def num(v, d=2):
            return f"{v:.{d}f}".replace(".", T["dec"])

        wr = sum(([1, 1, .12] for _ in range(per)), [])[:-1]
        left, top, row = .2, .25, (1 - .2) * WIDTH / sum(wr) * .8 + .28  # in inches: 5:4 image + title
        H = (nb + 1) * row + top
        f, axs = plt.subplots(nb + 1, len(wr), figsize=(WIDTH, H), gridspec_kw=dict(width_ratios=wr, wspace=.03))
        f.set_layout_engine("none")  # fixed positions: needed for the lines between rows
        f.subplots_adjust(left=left, right=1, top=1 - top / H, bottom=.005, hspace=.28 / (row - .28))
        for r in range(nb + 1):
            for k in range(len(wr)):
                axs[r, k].axis("off")
            for k, (sv, a, b) in enumerate(tiles[r]):
                axs[r, 3 * k].imshow(a)
                axs[r, 3 * k + 1].imshow(b)
                axs[r, 3 * k].set_title(f"s = {num(sv, 3)}", fontsize=6.5, loc="left", pad=2)
            if r < nb:
                lab = (f"{T['band']}\n{num(edges[r])}–{num(min(edges[r + 1], 1))}\n"
                       + f"{len(ses_pairs[r]):,}".replace(",", "." if T["dec"] == "," else ",") + f" {T['pairs']}")
            else:
                lab = T["ref"]
            axs[r, 0].text(-.08, .5, lab, transform=axs[r, 0].transAxes, ha="right", va="center", fontsize=6.5)
        # threshold line between the 0.90-0.95 row and the 0.95-0.98 one
        gap = .03 / H  # just below the row above, before the titles of the row below
        y = axs[list(edges).index(0.95) - 1, 0].get_position().y0 - gap
        f.add_artist(plt.Line2D([.02, 1], [y, y], transform=f.transFigure, color="0.2", lw=.8, ls=(0, (4, 2))))
        f.text(1, y - .004, T["thr"], ha="right", va="top", fontsize=6.5, color="0.2")
        y = axs[nb - 1, 0].get_position().y0 - gap
        f.add_artist(plt.Line2D([.02, 1], [y, y], transform=f.transFigure, color="0.6", lw=.5))
        print(save(f, FIG / ("soglia" if lang == "it" else "soglia_en"), pdf=True))
    for b in range(nb):
        print(f"{edges[b]:g}-{min(edges[b + 1], 1):g}: {n_pairs[b]:,} coppie di frame, {len(ses_pairs[b])} coppie di sessioni")
    print(f"stessa clip: {n_ref[0]:,} coppie di frame")


def export() -> None:
    """Step 14, the Ultralytics dataset in `data/output/` (deleted and rebuilt every time), images as hardlinks,
    labels rewritten with the ids of `OUT_NAMES` (0 drone, 1 airplane, 2 helicopter, 3 bird):
    `images/{train,val,test,calib}/` and `labels/...` with the same name, and `raw/{images,labels}/` with train, val
    and test together (without calib), for whoever wants to redo the splits; train = real images kept by the cap and
    mosaics (`13_train_compensation/train.csv`), val and test from `12_split/frame_split.csv`, plus the background of
    `data/input/background/0_raw/background.csv` (no label: for Ultralytics it is a background-only image), calib
    from `15_calib/calib.csv` (same name as the train image). Names (user, 23/10/2026):
    real `R_<group>_<hash8 of source:clip>_<n>`, mosaic `M_x<grid>_<n>`, background `B_<hash8 of
    source:camera>_<n>`; n with 6 digits, in frame order inside the clip or camera. `dataset.yaml`, `calib.yaml` and
    `manifest.csv` (immagine, split, tipo, gruppo, fonte, clip or camera, original frame, sorgente)."""
    import hashlib
    import os
    import re

    agg = INTERMEDIATE
    out = OUTPUT
    if out.exists():  # frozen (user, 23/10/2026): to rebuild it, move or delete it by hand
        sys.exit(f"{out} esiste ed e' congelato: non lo sovrascrivo")
    grp = {r["sessione"]: r["gruppo"] for r in csv.DictReader(open(agg / "11_gruppi" / "gruppi.csv"))}
    info = {r["immagine"]: r for r in csv.DictReader(open(agg / "9_split" / "frame.csv"))}
    bgi = {r["immagine"]: r for r in csv.DictReader(open(INPUT / "background" / "0_raw" / "background.csv"))}
    rows = []  # dict: split, img, lab, tipo, gruppo, fonte, clip, frame

    def real(s, img, lab):
        r = info[img]
        m = re.search(r"_f(\d+)$", Path(img).stem)
        return dict(split=s, img=img, lab=lab, tipo="reale", gruppo=grp[r["fonte"] + ":" + r["sessione"]],
                    fonte=r["fonte"], sessione=r["sessione"], clip=r["clip"], frame=int(m.group(1)) if m else "")

    for r in csv.DictReader(open(agg / "13_train_compensation" / "train.csv")):
        rows.append(real("train", r["immagine"], r["label"]) if r["tipo"] == "reale" else
                    dict(split="train", img=r["immagine"], lab=r["label"], tipo="mosaico", gruppo="", fonte="", sessione="",
                         clip="", frame=""))
    for r, s in zip(info.values(), csv.DictReader(open(agg / "12_split" / "frame_split.csv"))):
        assert r["immagine"] == s["immagine"]
        if s["split"] in ("val", "test"):
            rows.append(real(s["split"], r["immagine"], r["label"]))
    rows += [dict(split=r["split"], img=r["immagine"], lab=None, tipo="background", gruppo="", fonte=r["fonte"],
                  sessione="", clip=r["camera"], frame=Path(r["sorgente"]).stem) for r in bgi.values() if (DATA / r["immagine"]).exists()]

    def h8(s):
        return hashlib.sha1(s.encode()).hexdigest()[:8]

    keys = {(r["fonte"], r["clip"]) for r in rows if r["tipo"] != "mosaico"}
    assert len({h8(f"{f}:{c}") for f, c in keys}) == len(keys), "hash doppi"
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
    f = agg / "15_calib" / "calib.csv"
    if f.exists():  # int8 calibration: extra copies (hardlink) of train images, with the same name
        tr = {r["img"]: r for r in rows if r["split"] == "train"}
        rows += [dict(tr[r["immagine"]], split="calib") for r in csv.DictReader(open(f))]
    for s in ("train", "val", "test", "calib"):
        (out / "images" / s).mkdir(parents=True)
        (out / "labels" / s).mkdir(parents=True)
    (out / "raw" / "images").mkdir(parents=True)
    (out / "raw" / "labels").mkdir(parents=True)
    for r in rows:
        os.link(DATA / r["img"], out / "images" / r["split"] / f"{r['nome']}.jpg")
        lab = "".join(f"{OUT_ID[int(c)]} {rest}\n" for c, rest in
                      (ln.split(" ", 1) for ln in (DATA / r["lab"]).read_text().splitlines() if ln.strip())) if r["lab"] else ""
        if r["lab"]:
            (out / "labels" / r["split"] / f"{r['nome']}.txt").write_text(lab)
        if r["split"] != "calib":  # raw/: the whole dataset in a single folder, no split (user, 23/10/2026)
            os.link(DATA / r["img"], out / "raw" / "images" / f"{r['nome']}.jpg")
            if r["lab"]:
                (out / "raw" / "labels" / f"{r['nome']}.txt").write_text(lab)
    (out / "dataset.yaml").write_text(
        f"# air-detection, tools/split.py --step export\npath: {out}\ntrain: images/train\nval: images/val\n"
        "test: images/test\nnames:\n" + "".join(f"  {k}: {c}\n" for k, c in enumerate(OUT_NAMES)))
    (out / "calib.yaml").write_text(  # for the Ultralytics int8 export, which calibrates on the val split
        f"# calibrazione int8 (intermediate/15_calib), tools/split.py --step export\npath: {out}\ntrain: images/calib\n"
        "val: images/calib\nnames:\n" + "".join(f"  {k}: {c}\n" for k, c in enumerate(OUT_NAMES)))
    with open(out / "manifest.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["immagine", "split", "tipo", "gruppo", "fonte", "sessione", "clip", "frame", "sorgente"])
        w.writerows((f"images/{r['split']}/{r['nome']}.jpg", r["split"], r["tipo"], r["gruppo"], r["fonte"],
                     r["sessione"], r["clip"], r["frame"], r["img"]) for r in rows)
    # DVIDS videos used: page, author and licence (attribution, public domain video by video)
    used = {r["clip"].split("/")[0].removeprefix("DVIDS_") for r in rows if r["fonte"] == "dvids"}
    with open(out / "fonti_dvids.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "titolo", "autore", "licenza", "url"])
        w.writerows((v["id"], v["titolo"], v["autore"], v["licenza"], v["url"])
                    for v in csv.DictReader(open(INPUT / "dvids" / "0_raw" / "video.csv")) if v["id"] in used)
    _output_plot(rows, out)
    _leakage_plot(out)
    c = Counter((r["split"], r["tipo"]) for r in rows)
    tab = "| split | immagini | reali | mosaici | background |\n|---|---|---|---|---|\n" + "".join(
        f"| {s} | {sum(v for (x, _), v in c.items() if x == s):,} | {c[s, 'reale']:,} | {c[s, 'mosaico']:,} | "
        f"{c[s, 'background']:,} |\n".replace(",", ".") for s in ("train", "val", "test", "calib"))
    import datetime
    (out / "README.md").write_text((ROOT / "templates" / "dataset_readme.md").read_text()
                                   .replace("{conteggi}", tab.strip()).replace("{data}", f"{datetime.date.today():%d/%m/%Y}"))
    c = Counter((r["split"], r["tipo"]) for r in rows)
    for s in ("train", "val", "test", "calib"):
        print(f"  {s}: {sum(v for (x, _), v in c.items() if x == s)} immagini ("
              + ", ".join(f"{k} {v}" for (x, k), v in sorted(c.items()) if x == s) + ")")
    print(f"export -> {out}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fasce", type=float, nargs="+", default=[0.6, 0.7, 0.8, 0.9, 1.01], help="anteprima: soglie")
    p.add_argument("--step", choices=["frame", "embedding", "somiglianze", "somiglianze_tutte", "anteprima", "doppioni", "gruppi", "somiglianze_gruppi", "split", "controllo", "mosaici", "calib", "export", "figure", "soglia", "taratura", "esito_taratura", "stima_soglia", "revisione_fascia"], required=True)
    a = p.parse_args(argv)
    if a.step == "frame":
        frame_table()
    elif a.step == "embedding":
        frame_embedding()
    elif a.step == "somiglianze":
        similarity_plot()
    elif a.step in ("somiglianze_tutte", "somiglianze_gruppi"):
        similarity_all(a.step == "somiglianze_gruppi")
    elif a.step == "anteprima":
        pair_preview(a.fasce)
    elif a.step == "doppioni":
        duplicate_sheets()
    elif a.step == "taratura":
        threshold_sheets()
    elif a.step == "esito_taratura":
        threshold_marks()
    elif a.step == "revisione_fascia":
        band_review_sheets()
    elif a.step == "stima_soglia":
        threshold_estimate()
    elif a.step == "gruppi":
        groups()
    elif a.step == "split":
        split_groups()
    elif a.step == "controllo":
        split_check()
    elif a.step == "mosaici":
        train_mosaics()
    elif a.step == "calib":
        calib()
    elif a.step == "figure":
        figures()
    elif a.step == "soglia":
        threshold_panel()
    else:
        export()
    return 0


if __name__ == "__main__":
    sys.exit(main())
