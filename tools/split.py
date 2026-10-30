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
from tools.common import DATA, INPUT, INTERMEDIATE  # noqa: E402


def discarded() -> set[str]:
    """Clips removed from the dataset in every source: Svanstrom trails and rejects,
    FRED rejects (`*/*_box/scie.txt`, `*/*_box/scartate.txt`)."""
    out = set()
    for f in list(INPUT.glob("*/*_box/scie.txt")) + list(INPUT.glob("*/*_box/scartate.txt")):
        out |= {ln.split("#")[0].strip() for ln in open(f) if ln.split("#")[0].strip()}
    return out


def frame_table() -> None:
    """Step 9, table of the kept annotated frames (with at least one box) of all sources ->
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
    # added on 26/10/2026 (TODO.md): Anti-UAV frames between those already used (extra_sources.py --step frame_extra)
    # and the birds on sky of Open Images (openimages.py), one session per photo
    au = INPUT / "anti_uav" / "8_extra" / "extra.csv"
    for r in csv.DictReader(open(au)) if au.exists() else []:
        rows.append(("anti_uav", r["video"], f"{r['video']}/s0", r["sessione"], DATA / r["immagine"], DATA / r["label"]))
    oi = INPUT / "openimages" / "5_box" / "ritagli.csv"
    for r in csv.DictReader(open(oi)) if oi.exists() else []:
        rows.append(("openimages", r["image"], r["image"], r["image"],
                     INPUT / "openimages" / "4_ritagli" / "images" / f"{r['ritaglio']}.jpg",
                     INPUT / "openimages" / "5_box" / "manuale" / f"{r['ritaglio']}.txt"))
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

    from tools.plots import WIDTH, _P, save, use

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
    pretty = {"svanstrom": "Svanström", "dvids": "DVIDS", "anti_uav": "Anti-UAV", "fred": "FRED", "fbd_sv": "FBD-SV"}
    labs = [f"{pretty[info[c][0]]} {info[c][1].split('_')[-1]} · {c.split('/')[0].split('_')[-1]}" for c in clips]
    ticks = [k * per_clip + per_clip / 2 - .5 for k in range(len(clips))]
    use("it")
    for name in MODELS:
        emb = np.load(out / f"embedding_{name}.npy").astype(np.float32)
        S = emb[idx] @ emb[idx].T
        f, (ax, ax2) = plt.subplots(1, 2, figsize=(WIDTH, 3.9), width_ratios=[1.15, 1])
        im = ax.imshow(S, cmap="Blues", vmin=0, vmax=1, interpolation="nearest", aspect="equal")  # square (user, 28/10/2026)
        for k in range(1, len(clips)):
            ax.axhline(k * per_clip - .5, color="w", lw=.4)
            ax.axvline(k * per_clip - .5, color="w", lw=.4)
        ax.set_xticks([])
        ax.set_yticks(ticks, labs)  # source, session · video
        ax.tick_params(length=0)
        ax.grid(False)
        f.colorbar(im, ax=ax, fraction=.046, pad=.02).outline.set_linewidth(.6)
        ax.set_title(f"(a) {len(clips)} clip × {per_clip} frame")
        sims = {k: (emb[pa] * emb[pb]).sum(1) for k, (pa, pb) in pairs.items()}
        pos = np.concatenate([sims["stessa clip"], sims["stessa sessione, altra clip"]])
        auc = np.searchsorted(np.sort(sims["sessioni diverse"]), pos).mean() / len(sims["sessioni diverse"])
        bins = np.linspace(-.1, 1, 111)
        for (k, v), col in zip(sims.items(), (_P[0], _P[1], _P[2])):
            ax2.stairs(np.histogram(v, bins, density=True)[0], bins, color=col, label=k.capitalize())  # medians: log
        ax2.set_xlabel("Somiglianza coseno")
        ax2.set_ylabel("Densità")
        ax2.set_yticks([])
        ax2.legend(loc="upper left")
        ax2.set_title("(b) Coppie di frame a caso")
        ax2.text(.02, .7, "AUC stessa sessione\ncontro diverse: " + f"{auc:.3f}".replace(".", ","), transform=ax2.transAxes,
                 fontsize=7, color="0.3", va="top")
        f.align_titles()  # the square matrix is shorter than the other panel: titles on one line
        print(f"{name}: AUC {auc:.4f}, mediane " + ", ".join(f"{k} {np.median(v):.3f}" for k, v in sims.items()),
              save(f, out / f"somiglianze_{name}", pdf=False, name="similarity_sample"))


def similarity_all(by_group: bool = False, lo: float = -0.2, bins: int = 240, block: int = 4096, px: int = 2000) -> None:
    """Similarities on all frames, in source, session, clip order. On the left the matrix of the mean similarity
    between blocks of N/px consecutive frames (exact: the mean of the cosines between two blocks is the product of
    the mean embeddings), so the area of each source is proportional to its frames.
    On the right the exact histograms of all frame pairs (i < j) per type, in blocks on the GPU, with the AUC
    same session against different sessions -> `intermediate/9_split/somiglianze_tutte.png`. With `by_group`
    (`11_gruppi/gruppi.csv`) two more curves, same group and other session (the hand-merged pairs) and different
    groups, and an inset on the tail in log scale -> `docs/figure/{it,en}_similarity`."""
    import matplotlib.pyplot as plt
    import torch

    from tools.plots import WIDTH, _P, save, use

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

    # matrix rows: each source gets rows in proportion to the square root of its frames (user, 26/10/2026: every
    # source visible, the relative weight still readable; the true counts are written next to the names), and inside
    # a source blocks of consecutive frames
    n = len(rows)
    nsrc = Counter(r["fonte"] for r in rows)
    srcs = list(dict.fromkeys(r["fonte"] for r in rows))
    w = np.sqrt([nsrc[x] for x in srcs])
    alloc = np.maximum(1, np.round(px * w / w.sum()).astype(int))
    start = dict(zip(srcs, np.concatenate([[0], np.cumsum(alloc)[:-1]])))
    px = int(alloc.sum())
    local, seen = [], Counter()
    for r in rows:
        x = r["fonte"]
        local.append(start[x] + seen[x] * alloc[srcs.index(x)] // nsrc[x])
        seen[x] += 1
    gid = torch.tensor(local, device="cuda")
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

    # short legend labels (user, 26/10/2026: legends too large); the caption says "other clip", "other session"
    T = {"it": dict(kinds=["Stessa clip", "Stessa sessione", "Sessioni diverse", "Stesso gruppo", "Gruppi diversi"],
                    block="(a) Somiglianza media", all="(b) Coppie di frame", tail="(c) Coda",
                    x="Somiglianza coseno", y="Densità", ytail="Coppie", auc="AUC stessa sessione\ncontro diverse: ",
                    dec=","),
         "en": dict(kinds=["Same clip", "Same session", "Different sessions", "Same group", "Different groups"],
                    block="(a) Mean similarity", all="(b) Frame pairs", tail="(c) Tail",
                    x="Cosine similarity", y="Density", ytail="Pairs", auc="AUC same session\nagainst different: ",
                    dec=".")}
    for lang in ("it", "en") if by_group else ("it",):
        t = T[lang]
        use(lang)
        # three panels (user, 26/10/2026: the inset and the side legend squeezed the histogram): matrix, all the
        # pairs, the tail above 0.85 in log scale; one legend below, on one row
        f, axs = plt.subplots(1, 3 if by_group else 2, figsize=(WIDTH, 2.7),
                              width_ratios=[1.2, 1.25, .85] if by_group else [1, 1.2])
        ax, ax2 = axs[0], axs[1]
        im = ax.imshow(Sc, cmap="Blues", vmin=0, vmax=1, interpolation="nearest", aspect="equal")  # square (user, 28/10/2026)
        first = {x: float(start[x]) for x in srcs}
        starts = sorted(first.values()) + [px]
        for b in starts[1:-1]:
            ax.axhline(b - .5, color=_P[7], lw=.5)
            ax.axvline(b - .5, color=_P[7], lw=.5)
        names = sorted(first, key=first.get)
        ticks = [(a + b) / 2 for a, b in zip(starts, starts[1:])]
        pretty = {"svanstrom": "Svanström", "dvids": "DVIDS", "anti_uav": "Anti-UAV", "fred": "FRED", "fbd_sv": "FBD-SV",
                  "openimages": "Open Images"}
        sep = "." if lang == "it" else ","
        ax.set_yticks(ticks, [f"{pretty.get(x, x)} ({nsrc[x]:,})".replace(",", sep) for x in names])
        ax.set_xticks([])
        ax.tick_params(length=0)
        ax.grid(False)
        cb = f.colorbar(im, ax=ax, fraction=.05, pad=.03, aspect=30)
        cb.outline.set_linewidth(.6)
        cb.set_ticks([0, .5, 1])
        ax.set_title(t["block"])
        top = 0
        for q, (x, k, col) in enumerate(zip(h, t["kinds"], _P)):
            d = x / x.sum() / (edges[1] - edges[0])
            top = max(top, d.max())
            ax2.stairs(d, edges, color=col, ls="--" if q == 4 else "-", lw=1,
                       label=k)  # qualitative legend (user, 26/10/2026): the numbers are in the log
        ax2.set_ylim(0, top * 1.05)
        ax2.set_xlim(lo, 1.02)
        ax2.set_xticks([0, .5, 1])
        ax2.set_xlabel(t["x"])
        ax2.set_ylabel(t["y"])
        ax2.set_yticks([])
        ax2.set_title(t["all"])
        ax2.text(.03, .97, t["auc"] + f"{auc:.3f}".replace(".", t["dec"]), transform=ax2.transAxes, ha="left", va="top",
                 fontsize=7, color="0.3")  # top left, above the peak of the different sessions
        if by_group:  # the tail of different sessions and different groups, with the 0.95 threshold
            ax3 = axs[2]
            for x, col, ls in ((h[2], _P[2], "-"), (h[4], _P[4], "--")):
                ax3.stairs(np.maximum(x, .5), edges, color=col, lw=1, ls=ls)
            ax3.axvline(.95, color="0.4", lw=.7)
            ax3.text(.952, .97, f"0{t['dec']}95", transform=ax3.get_xaxis_transform(), fontsize=7, color="0.3", va="top")
            ax3.set_yscale("log")
            ax3.set_ylim(.5, 3e6)
            ax3.set_yticks([1, 1e2, 1e4, 1e6])
            ax3.yaxis.set_minor_locator(plt.NullLocator())
            ax3.set_xlim(.85, 1)
            ax3.set_xticks([.85, .9, .95, 1])
            ax3.set_xlabel(t["x"])
            ax3.set_ylabel(t["ytail"])
            ax3.set_title(t["tail"])
        f.legend(*ax2.get_legend_handles_labels(), loc="outside lower center", ncol=len(t["kinds"][:len(h)]))
        f.align_titles()  # the square matrix is shorter than the other panels: titles on one line
        if by_group:  # summary figure of the README (user, 23/10/2026)
            save(f, FIG / "similarity")
        else:
            save(f, out / "somiglianze_tutte", pdf=False, name="similarity_all")
    print(f"AUC {auc:.4f}, coppie {', '.join(f'{x.sum():,.0f}' for x in h)}, mediane {[round(float(x), 3) for x in med]}")

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


def band_review_new(lo: float = 0.92, hi: float = 0.95) -> None:
    """The session pairs in [lo, hi) never reviewed (sessions added after `fascia_092`, 26/10/2026; user 26/10/2026):
    not in `taratura/` nor `fascia_092/`, not already in the same group. Own folder, so the numbers of `fascia_092`
    stay valid: `10_doppioni/fascia_092_b/` (`coppie.csv`, `fogli/RNN.jpg`); the operator writes the numbers of the
    duplicates in `fascia_092_b/stessa_sessione.txt`, then `--step gruppi`."""
    dup = INTERMEDIATE / "10_doppioni"
    dst = dup / "fascia_092_b"
    rows, e, sess, by_ses, best, cnt = _session_pairs(lo)
    done = {tuple(sorted((r["sessione_a"], r["sessione_b"]))) for d in ("taratura", "fascia_092")
            for r in csv.DictReader(open(dup / d / "coppie.csv"))}
    grp = {r["sessione"]: r["gruppo"] for r in csv.DictReader(open(INTERMEDIATE / "11_gruppi" / "gruppi.csv"))}
    order = sorted((k for k in best if best[k][0] < hi and tuple(sorted((sess[k[0]], sess[k[1]]))) not in done
                    and grp[sess[k[0]]] != grp[sess[k[1]]]), key=lambda k: -best[k][0])
    _pair_sheets(dst, order, rows, e, sess, by_ses, best, cnt, lo, "R")
    print(f"{lo}-{hi}: {len(order)} session pairs never reviewed, {-(-len(order) // 12)} sheets -> {dst}")


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
    pairs of `10_doppioni/unire.txt`, every other session pair >= 0.95 (26/10/2026) and the duplicates confirmed below 0.95
    (`taratura/` and `fascia_092/stessa_sessione.txt`): `gruppi.csv` (sessione, gruppo, frame). No frame deduplication between
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
    # duplicates confirmed by hand below 0.95 (user, 23/10/2026): tuning sample and full review of 0.92-0.95
    # fascia_092_b: sessions added after the review (26/10/2026); fascia_092_c: DVIDS round 12 (28/10/2026)
    for d in ("taratura", "fascia_092", "fascia_092_b", "fascia_092_c"):
        if (dup / d / "stessa_sessione.txt").exists():
            pairs = {int(r["n"]): (r["sessione_a"], r["sessione_b"]) for r in csv.DictReader(open(dup / d / "coppie.csv"))}
            for k in _merges(dup / d / "stessa_sessione.txt"):
                a, b = pairs[k]
                parent[find(a)] = find(b)
    # every pair of sessions with two frames >= 0.95 is the same shoot (rule calibrated on 23/10/2026: 146 out of 146):
    # the pairs never reviewed (sessions and sources added on 26/10/2026) are merged too
    _, _, ss, _, best, _ = _session_pairs(0.95)
    auto = 0
    for a, b in best:
        if find(ss[a]) != find(ss[b]):
            parent[find(ss[a])] = find(ss[b])
            auto += 1
    print(f"gruppi: {auto} unioni automatiche >= 0,95 (coppie mai riviste)")
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


CLASS_WEIGHT = np.array([1, 1, 4, 1])[:, None]  # airplane, bird, drone, helicopter: drones x4 (user, 23/10/2026: ~5% error)


def _err(x, t):
    m = t > 0
    return ((CLASS_WEIGHT * ((x - t) / np.where(m, t, 1)) ** 2)[m]).sum()


MIN_SPLIT = {"val": 2000, "test": 1000}  # minimum instances per class (user, 23/10/2026; the train has 10,000)


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
           "baseline": "Fondo", "leak": "Leakage", "median": "Mediana", "xthr": "Soglia di somiglianza\n", "ythr": "Quota di immagini sopra la soglia",
           "dec": ","},
    "en": {"classes": {"drone": "Drone", "airplane": "Airplane", "helicopter": "Helicopter", "bird": "Bird"},
           "orig": "Original data (before selection)", "real": "Training, real images",
           "mosaics": "Training, synthetic mosaics", "val": "Validation", "test": "Test",
           "tgt_train": "Training target", "tgt_val": "Validation target", "tgt_test": "Test target (uniform)",
           "xsize": r"Box size $\sqrt{w\,h}$ [px]", "ysize": "Instances per bin",
           "short": ["Same group,\nother clip (leakage)", "Different groups\n(baseline)", "Validation", "Test"],
           "xsim": "Similarity to the nearest\ntraining image", "a": "(a) Distributions", "b": "(b) Tail",
           "baseline": "Baseline", "leak": "Leakage", "median": "Median", "xthr": "Similarity threshold\n", "ythr": "Share of images above threshold",
           "dec": "."},
}


def _leakage_plot(out: Path, block: int = 2048, lang: str = "it", table: bool = True, data=None) -> None:
    """`leakage.{png,pdf,md}` of the export, for the thesis: for each real validation and test image the DINOv2
    similarity (ViT-S, cosine) with the nearest real training image, against two references inside the
    training set: same group and other clip (what a true leakage looks like) and different groups (the baseline:
    sky resembles sky). (a) density, (b) share of images above each threshold, in log scale; 0.95 is the
    threshold of the hand-reviewed duplicates. The table with median, percentiles and shares in `leakage.md` (if
    `table`). Reads only `intermediate/` (frame, split, groups, embedding), writes to `out`; `lang` it or en
    (`leakage_en`)."""
    import matplotlib.pyplot as plt
    import torch

    from tools.plots import WIDTH, _P, save, use

    if data is not None:  # (embeddings, group ids, clip ids, splits) of another dataset (tools/dataset.py)
        e, g, c, sp = data
    else:
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
    use(lang)
    f, (a1, a2) = plt.subplots(1, 2, figsize=(WIDTH, 2.9))  # same width for the two panels
    pos = np.arange(len(keys))[::-1]
    vp = a1.violinplot([res[k] for k in keys], positions=pos, vert=False, widths=.85, showextrema=False)
    for body, col in zip(vp["bodies"], cols):
        body.set_facecolor(col)
        body.set_edgecolor(col)
        body.set_alpha(.45)
    for y, k, col in zip(pos, keys, cols):
        q1, med, q3 = np.percentile(res[k], [25, 50, 75])
        a1.plot([q1, q3], [y, y], color=col, lw=3, solid_capstyle="butt")
        a1.plot([med], [y], marker="o", ms=4, color="white", mec=col, mew=1, zorder=5)
        a1.text(1.01, y, f"{med:.2f}".replace(".", T["dec"]), va="center", ha="left", fontsize=7, color="0.2",
                transform=a1.get_yaxis_transform())
    a1.text(1.01, pos[0] + .55, T["median"], va="bottom", ha="left", fontsize=7, color="0.2",
            transform=a1.get_yaxis_transform())
    a1.set_yticks(pos, short)
    a1.set_xlim(.3, .999)
    a1.axvline(.95, color="0.4", lw=.7)
    a1.text(.95, pos[-1] - .5, f" 0{T['dec']}95", ha="left", va="center", fontsize=7, color="0.3")
    a1.set_xlabel(T["xsim"])
    a1.set_title(T["a"])
    a1.grid(axis="y", visible=False)
    # (b) tail: share of images above each threshold
    x = 1 - np.geomspace(.5, .002, 600)  # dense near 1, where the axis widens
    tail = {k: np.array([(res[k] >= s).mean() for s in x]) for k in keys}
    a2.fill_between(x, 1e-5, np.maximum(tail[keys[1]], 1e-5), color="0.6", alpha=.3, lw=0, label=T["baseline"])
    a2.plot(x, tail[keys[0]], color=cols[0], ls=(0, (4, 2)), label=T["leak"])
    for k, col, lab in zip(keys[2:], cols[2:], short[2:]):
        a2.plot(x, tail[k], color=col, label=lab)
    a2.axvline(.95, color="0.4", lw=.7)
    a2.set_yscale("log")
    a2.set_xlim(.5, .999)
    a2.set_ylim(1e-4, 1.3)
    a2.set_xlabel(T["xthr"])  # two lines like (a), aligned axes
    a2.set_ylabel(T["ythr"])
    a2.set_title(T["b"])
    a2.text(.95, 1.5e-4, f" 0{T['dec']}95", ha="left", va="bottom", fontsize=7, color="0.3")
    f.legend(*a2.get_legend_handles_labels(), loc="outside lower center", ncol=4)
    # x axis that widens the zone near 1 (user, 23/10/2026): -log10(1 - s), ticks in similarity
    for a in (a1, a2):
        a.set_xscale("function", functions=(lambda v: -np.log10(np.clip(1 - v, 1e-4, None)),
                                             lambda u: 1 - 10 ** -u))
        a.set_xticks([.5, .9, .99, .999])
        a.xaxis.set_minor_locator(plt.NullLocator())
        a.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}".replace(".", T["dec"])))
    save(f, out / "leakage", pdf=True)  # in docs/figure it_leakage, en_leakage
    if not table:
        return

    def fmt(v):
        return f"{v:.3f}".replace(".", ",")

    L = ["# Leakage fra gli split", "",
         "Generato con `leakage.png`. Per ogni immagine reale, la somiglianza coseno",
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


def _threshold_plot(dst: Path, lang: str = "it") -> None:
    """Duplicates threshold, one panel (user, 30/10/2026: the upper panel, the session pairs per band in log scale, was
    removed from the thesis figure): per band of maximum similarity, the share of duplicates among the pairs judged by
    hand (tuning sample `taratura/`, full review of 0.92-0.95 `fascia_092/`, the 146 pairs >= 0.95 of step 10), without
    error bars (user, 30/10/2026: Wilson interval removed), over the three zones of the rule (ignored, review by hand, automatic merge), colours as
    `threshold_examples` -> `{it,en}_threshold_tuning`."""
    import matplotlib.pyplot as plt

    from tools.plots import WIDTH, _P, save, use

    dup = INTERMEDIATE / "10_doppioni"
    sim, same = [], []
    for d in ("taratura", "fascia_092"):
        yes = set(_merges(dup / d / "stessa_sessione.txt"))
        for r in csv.DictReader(open(dup / d / "coppie.csv")):
            sim.append(float(r["massimo"]))
            same.append(int(r["n"]) in yes)
    top = sum(1 for _ in csv.DictReader(open(dup / "coppie.csv")))
    sim, same = np.array(sim + [2.0] * top), np.array(same + [True] * top)  # 2.0: the reviewed pairs of step 10
    bands = [(0.85, 0.90), (0.90, 0.92), (0.92, 0.93), (0.93, 0.94), (0.94, 0.95), (0.95, 1.01)]
    it = lang == "it"
    use(lang)
    f, a2 = plt.subplots(figsize=(WIDTH, 2.4))
    x = np.arange(len(bands))
    rev, k_ = [], []
    for lo, hi in bands:
        m = (sim >= lo) & ((sim < hi) | ((hi == 0.95) & (sim <= 0.9504))) if hi < 1 else sim > 1.5  # 0.950 rounded
        n, k = int(m.sum()), int(same[m].sum())
        rev.append(n)
        k_.append(k)
    zones = [(-.5, 1.5, "0.6", "si ignora" if it else "ignored"),
             (1.5, 4.5, _P[3], "revisione a mano" if it else "review by hand"),
             (4.5, 5.5, _P[0], "unione automatica" if it else "automatic merge")]
    for lo, hi, col, name in zones:
        a2.axvspan(lo, hi, color=col, alpha=.12, lw=0)
        a2.text((lo + hi) / 2, 1.13, name, ha="center", va="top", fontsize=7, color="0.25")
    a2.plot(x, [k / n for k, n in zip(k_, rev)], color=_P[0], lw=1, zorder=2)
    a2.scatter(x, [k / n for k, n in zip(k_, rev)], s=18, color=_P[0], zorder=4)
    for i in x:  # the last point is at 100%, under the zone name: its label goes to the left, below
        last = i == x[-1]
        a2.text(i + (-.09 if last else .09), k_[i] / rev[i] + (-.05 if last else .03), f"{k_[i]}/{rev[i]}", fontsize=7,
                va="top" if last else "bottom", ha="right" if last else "left")
    a2.set_ylim(-.04, 1.16)
    a2.set_xlim(-.5, len(bands) - .5)
    a2.set_ylabel("Quota di doppioni" if it else "Share of duplicates")
    a2.set_yticks([0, .25, .5, .75, 1], ["0", "25%", "50%", "75%", "100%"])
    a2.grid(axis="x", visible=False)
    a2.set_title("Doppioni fra le coppie riviste" if it else "Duplicates among the reviewed pairs", loc="left")
    lab = [f"{lo:.2f}–{hi:.2f}" if hi < 1 else "≥ 0.95" for lo, hi in bands]
    a2.set_xticks(x, [v.replace(".", ",") for v in lab] if it else lab)
    a2.set_xlabel("Somiglianza massima fra due sessioni (DINOv2)" if it else
                  "Maximum similarity between two sessions (DINOv2)")
    f.align_ylabels()
    save(f, dst / "threshold_tuning")


def figures() -> None:
    """Figure of the duplicates threshold tuning, in Italian and English (`{it,en}_threshold_tuning`);
    the figures of the dataset are made by `tools/dataset.py --step figure`."""
    for lang in ("it", "en"):
        _threshold_plot(FIG, lang)
    print(f"figure -> {FIG}")


def threshold_panel(edges=(0.85, 0.90, 0.92, 0.95, 0.98, 1.01), per: int = 3, seed: int = 0, dark: float = 60) -> None:
    """Panel for the thesis on the 0.95 threshold of duplicates between sessions (`docs/figure/{it,en}_threshold_examples`):
    one row per DINOv2 similarity band with `per` random pairs (fixed seed) of frames from
    different sessions, boxes in red. Below 0.95 different shots with the same kind of sky; above, the same shot or
    the same footage republished (the 146 session pairs above 0.95 reviewed by hand and merged, step 10). The
    same-clip reference row was removed (user, 28/10/2026). Highlighted (user, 26/10/2026): the band reviewed by hand (0.92-0.95, every pair) and the automatic merge (>= 0.95);
    pairs with a frame darker than `dark` (mean gray, 0-255: black backgrounds) are not shown."""
    import matplotlib.pyplot as plt

    from tools.plots import WIDTH, _P, save, use

    rows, cand, n_pairs, _ = _pairs_by_band(edges, seed)
    # session pairs per band by their maximum similarity, as in the review (10_doppioni) and the tuning figure
    n_ses = np.histogram([v[0] for v in _session_pairs(edges[0])[4].values()], edges)[0]
    import cv2

    rng = np.random.default_rng(seed)

    def bright(i):
        return cv2.imread(str(DATA / rows[i]["immagine"]), cv2.IMREAD_GRAYSCALE).mean() >= dark

    picks = []
    for c in cand:  # random order, only pairs with two bright frames
        pk = []
        for x in rng.permutation(len(c)):
            if len(pk) == per:
                break
            if bright(c[x][1]) and bright(c[x][2]):
                pk.append(c[x])
        picks.append(pk)
    tiles = [[(sv, _tile(rows[i], None, lw=6)[:, :, ::-1], _tile(rows[j], None, lw=6)[:, :, ::-1]) for sv, i, j in pk]
             for pk in picks]
    nb = len(edges) - 1
    txt = {"it": dict(dec=",", band="Sessioni diverse", pairs="coppie di sessioni",
                      thr="Unione automatica", rev="Rivista a mano"),
           "en": dict(dec=".", band="Different sessions", pairs="session pairs",
                      thr="Merged automatically", rev="Reviewed by hand")}
    from matplotlib.patches import Rectangle

    # fixed layout in inches, full page width (user, 26/10/2026): label column, then `per` pairs of images
    lab_w, g_in, g_out, t_h, r_gap = 1.42, .04, .14, .17, .07
    img_w = (WIDTH - lab_w - per * g_in - (per - 1) * g_out) / (2 * per)
    img_h = img_w * tiles[0][0][1].shape[0] / tiles[0][0][1].shape[1]
    row_h = t_h + img_h + r_gap
    H = nb * row_h
    tops = [H - r * row_h for r in range(nb)]  # top of each row (title included)
    lab_col = {"rev": "#9a6a00", "thr": _P[0]}
    for lang, T in txt.items():
        use(lang)

        def num(v, d=2):
            return f"{v:.{d}f}".replace(".", T["dec"])

        f = plt.figure(figsize=(WIDTH, H))
        f.set_layout_engine("none")
        # highlighted bands (user, 26/10/2026): reviewed by hand 0.92-0.95, merged automatically >= 0.95; a light
        # background behind the rows, adjacent bands touching, the meaning also written in the row label
        bands = [(e_ := list(edges)).index(0.92), e_.index(0.95), nb]
        for (r0, r1), color in zip(zip(bands, bands[1:]), (_P[3], _P[0])):
            y0 = 0 if r1 == nb else tops[r1] + r_gap / 2
            y1 = tops[r0] + r_gap / 2 if r0 else H
            f.add_artist(Rectangle((0, y0 / H), 1, (y1 - y0) / H, transform=f.transFigure, color=color, alpha=.12,
                                   lw=0, zorder=-5))
        for r in range(nb):
            y_img = tops[r] - t_h - img_h
            for k, (sv, a, b) in enumerate(tiles[r]):
                x = lab_w + k * (2 * img_w + g_in + g_out)
                for q, im in enumerate((a, b)):
                    ax = f.add_axes(((x + q * (img_w + g_in)) / WIDTH, y_img / H, img_w / WIDTH, img_h / H))
                    ax.imshow(im, interpolation="lanczos")
                    ax.set_axis_off()
                f.text(x / WIDTH, (y_img + img_h + .03) / H, f"s = {num(sv, 3)}", fontsize=7, va="bottom")
            col = "0.13"
            lab = (f"{T['band']}\n{num(edges[r])}–{num(min(edges[r + 1], 1))}\n"
                   + f"{n_ses[r]:,}".replace(",", "." if T["dec"] == "," else ",") + f" {T['pairs']}")
            kind = "rev" if 0.92 <= edges[r] < 0.95 else "thr" if edges[r] >= 0.95 else None
            if kind:
                lab, col = f"{T[kind]}\n" + lab, lab_col[kind]
            f.text((lab_w - .1) / WIDTH, (y_img + img_h / 2) / H, lab, ha="right", va="center", fontsize=7, color=col,
                   linespacing=1.3)
        print(save(f, FIG / "threshold_examples", pdf=True))
    for b in range(nb):
        print(f"{edges[b]:g}-{min(edges[b + 1], 1):g}: {n_pairs[b]:,} coppie di frame, {n_ses[b]} coppie di sessioni (per massimo)")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fasce", type=float, nargs="+", default=[0.6, 0.7, 0.8, 0.9, 1.01], help="anteprima: soglie")
    p.add_argument("--step", choices=["frame", "embedding", "somiglianze", "somiglianze_tutte", "anteprima", "doppioni", "gruppi", "somiglianze_gruppi", "figure", "soglia", "taratura", "esito_taratura", "stima_soglia", "revisione_fascia", "revisione_nuove"], required=True)
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
    elif a.step == "revisione_nuove":
        band_review_new()
    elif a.step == "stima_soglia":
        threshold_estimate()
    elif a.step == "gruppi":
        groups()
    elif a.step == "figure":
        figures()
    else:
        threshold_panel()
    return 0


if __name__ == "__main__":
    sys.exit(main())
