#!/usr/bin/env python3
"""Passi 11-14 della procedura, su tutte le fonti insieme (`data/intermediate/`).

Passo 11, gruppi (23/10/2026): embedding DINOv2 di ogni frame annotato di tutte le fonti; il video o la
registrazione resta l'unita' minima, gli embedding servono solo a unire. Un file per cosa, in `intermediate/9_split/`.

    python tools/split.py --step frame        # frame.csv: frame annotati tenuti, tutte le fonti
    python tools/split.py --step embedding    # embedding_s.npy: DINOv2 ViT-S per frame
    python tools/split.py --step somiglianze  # somiglianze_s.png su un campione
    python tools/split.py --step somiglianze_tutte  # somiglianze_tutte.png su tutti i frame
    python tools/split.py --step anteprima [--fasce 0.95 0.99 1.01]  # coppie di sessioni diverse per fascia
    python tools/split.py --step doppioni     # fogli numerati delle coppie di sessioni >= 0,95, da rivedere
    python tools/split.py --step gruppi       # 11_gruppi/gruppi.csv: sessioni note + unire.txt
    python tools/split.py --step somiglianze_gruppi  # come somiglianze_tutte, con le curve dei gruppi
    python tools/split.py --step split        # 12_split: gruppi in train, val, test con gli obiettivi per cella
    python tools/split.py --step controllo    # gruppi per cella di val e test, vicino piu' simile nel train
    python tools/split.py --step mosaici      # 13_train_compensation: tetto = curva e mosaici ridotti con l'eccesso
    python tools/split.py --step calib        # 15_calib: 512 immagini del train per l'int8, bilanciate per classe e fascia
    python tools/split.py --step export       # data/output: dataset Ultralytics in hardlink (train, val, test, calib, background)
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
    """Clip tolte dal dataset in ogni fonte: scie e scartate di Svanstrom,
    scartate di FRED (`*/*_box/scie.txt`, `*/*_box/scartate.txt`)."""
    out = set()
    for f in list(INPUT.glob("*/*_box/scie.txt")) + list(INPUT.glob("*/*_box/scartate.txt")):
        out |= {ln.split("#")[0].strip() for ln in open(f) if ln.split("#")[0].strip()}
    return out


def frame_table() -> None:
    """Passo 11, tabella dei frame annotati tenuti (con almeno un box) di tutte le fonti ->
    `intermediate/9_split/frame.csv` (fonte, video, clip, sessione, immagine, label, box). Stessi frame dei
    conteggi: Svanstrom da `6_dedup/tenuti.csv` con le label di `4_box/manuale/`, le altre fonti come
    `pipeline.count` (label di CVAT per le sequenze riviste, altrimenti preselezione, senza gli esclusi)."""
    from tools.pipeline import boxes, excluded

    rows, via = [], discarded()
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
        esclusi = excluded(src)
        for d in sorted((src / "5_box" / "proposte").iterdir()):
            clip = f"{d.name}/s0"
            if clip not in ses:
                continue
            man = src / "5_box" / "manuale" / d.name
            for t in sorted(d.glob("*.txt")):
                if t.stem not in esclusi:
                    rows.append((s, d.name, clip, ses[clip], src / "4_ritagli" / d.name / "images" / f"{t.stem}.jpg",
                                 (man if man.exists() else boxes(src) / d.name) / t.name))
    out = INTERMEDIATE / "9_split"
    out.mkdir(parents=True, exist_ok=True)
    n, kept = Counter(), []
    for s, v, clip, se, img, lab in rows:
        nb = sum(1 for ln in lab.read_text().splitlines() if ln.strip()) if lab.exists() else 0
        if nb and clip not in via:
            assert img.exists(), img
            kept.append((s, v, clip, se, img.relative_to(DATA), lab.relative_to(DATA), nb))
            n[s] += 1
    with open(out / "frame.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["fonte", "video", "clip", "sessione", "immagine", "label", "box"])
        w.writerows(kept)
    print(f"frame: {len(kept)} frame annotati ({', '.join(f'{s} {k}' for s, k in n.items())}), "
          f"{sum(r[6] for r in kept)} box -> {out / 'frame.csv'}")


MODELS = {"s": "dinov2_vits14"}  # B e L provati il 23/10/2026: non separano meglio le sessioni


def embed(paths: list, batch: int = 512, workers: int = 16) -> np.ndarray:
    """Embedding DINOv2 ViT-S/14 (CLS, norma 1) dei JPEG `paths`: decodificati e ridotti sulla GPU (-> 280x224),
    float16, N x 384, stesso ordine."""
    import torch
    import torchvision.io as tio

    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)

    class Files(torch.utils.data.Dataset):  # solo i byte: la decodifica e' sulla GPU
        def __len__(self):
            return len(paths)

        def __getitem__(self, i):  # bytes, non tensori: 512 tensori per batch esauriscono i fd condivisi
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
    """Passo 11, `embed` di ogni riga di `frame.csv` -> `intermediate/9_split/embedding_s.npy` (stesso ordine)."""
    out = INTERMEDIATE / "9_split"
    e = embed([str(DATA / r["immagine"]) for r in csv.DictReader(open(out / "frame.csv"))])
    np.save(out / "embedding_s.npy", e)
    print(f"embedding: {e.shape} -> {out / 'embedding_s.npy'}")


def similarity_plot(per_clip: int = 10, n_pairs: int = 100_000, seed: int = 0) -> None:
    """Somiglianza coseno (embedding a norma 1: coseno = prodotto scalare), un grafico per modello sugli stessi
    frame: per ogni fonte due sessioni con due clip ciascuna (DVIDS: quattro video), `per_clip` frame a passo
    regolare per clip. A sinistra la matrice, a destra la distribuzione su `n_pairs` coppie a caso per tipo
    (stessa clip, stessa sessione e altra clip, sessioni diverse) con l'AUC stessa sessione contro sessioni
    diverse -> `intermediate/9_split/somiglianze_s.png`."""
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


def similarity_all(gruppi: bool = False, lo: float = -0.2, bins: int = 240, block: int = 4096, px: int = 2000) -> None:
    """Somiglianze su tutti i frame, in ordine fonte, sessione, clip. A sinistra la matrice della somiglianza media
    fra gruppi di N/px frame consecutivi (esatta: la media dei coseni fra due gruppi e' il prodotto delle medie
    degli embedding), quindi l'area di ogni fonte e' proporzionale ai suoi frame.
    A destra gli istogrammi esatti di tutte le coppie di frame (i < j) per tipo, a blocchi sulla GPU, con l'AUC
    stessa sessione contro sessioni diverse -> `intermediate/9_split/somiglianze_tutte.png`. Con `gruppi`
    (`11_gruppi/gruppi.csv`) altre due curve, stesso gruppo e altra sessione (le coppie unite a mano) e gruppi
    diversi, e un riquadro sulla coda in scala log -> `intermediate/11_gruppi/somiglianze_gruppi.png`."""
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
        if gruppi else {s: s for s in sess}
    gi = {g: k for k, g in enumerate(dict.fromkeys(grp.values()))}
    pid = torch.tensor([gi[grp[r["fonte"] + ":" + r["sessione"]]] for r in rows], device="cuda")
    kinds = ["stessa clip", "stessa sessione, altra clip", "sessioni diverse"]\
        + ["stesso gruppo, altra sessione", "gruppi diversi"] * gruppi

    # righe della matrice: gruppi di frame consecutivi (stesso numero di frame), cosi' l'area e' proporzionale ai frame
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
    if gruppi:  # la coda di sessioni diverse e gruppi diversi, in scala log
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
    ax2.legend(fontsize=5.5, loc="upper left", bbox_to_anchor=(0, -.12), frameon=False)  # sotto: copre le curve
    ax2.set_title(f"tutte le coppie di frame ({n:,})".replace(",", ".") + "; AUC stessa sessione / diverse "
                  + f"{auc:.3f}".replace(".", ","), fontsize=7)
    f.suptitle("DINOv2 ViT-S/14, tutti i frame annotati", fontsize=8)
    print(f"AUC {auc:.4f}, coppie {', '.join(f'{x.sum():,.0f}' for x in h)}, mediane {[round(float(x), 3) for x in med]}",
          save(f, INTERMEDIATE / "11_gruppi" / "somiglianze_gruppi" if gruppi else out / "somiglianze_tutte", pdf=False))


TILE = (256, 205, 16)  # miniatura 640x512 -> 256x205, striscia di testo sopra


def _who(r: dict) -> str:
    return f"{r['fonte']} {r['sessione'].split('_')[-1]} {r['video'].split('_')[-1]}"


def _tile(r: dict, text: str | None, lw: int = 2) -> np.ndarray:
    """Miniatura di un frame di `frame.csv` con i box in rosso (spessore `lw` a 640x512) e `text` sopra (None: senza
    striscia)."""
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
    """Coppie di frame di `frame.csv` per fascia di somiglianza DINOv2 (coseno, `embedding_s.npy`): di sessioni
    diverse, o della stessa clip con `same_clip`. Per fascia fino a 50 candidati a caso per blocco (sim, i, j), il
    numero di coppie di frame e l'insieme delle coppie di sessioni -> (rows, cand, n_pairs, ses_pairs)."""
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
    cand = [[] for _ in range(nb)]  # (sim, i, j) a caso, fino a 50 per blocco e fascia
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
    """Anteprima di coppie di frame di sessioni diverse, una riga per fascia di somiglianza (0,6-0,7, 0,7-0,8, ...):
    `per` coppie a caso per fascia, box disegnati, somiglianza e fonte/sessione/video sopra ogni immagine. Stampa per
    fascia le coppie di frame e le coppie di sessioni coinvolte -> `intermediate/9_split/anteprima_sessioni_diverse_<prima soglia>.jpg`."""
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
    """Coppie di sessioni diverse con almeno una coppia di frame >= `lo` (DINOv2, `embedding_s.npy`): per coppia la
    coppia di frame piu' simile (sim, i, j) e quante coppie di frame superano `lo`."""
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
    """`dst/coppie.csv` e fogli numerati `dst/fogli/<prefix>NN.jpg`, una riga per coppia di sessioni: la coppia di
    frame piu' simile, poi un frame a caso di A con il frame di B piu' simile a lui."""
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
    """Passo 11, doppioni fra sessioni: ogni coppia di sessioni diverse con almeno una coppia di frame >= `thr`
    -> `intermediate/10_doppioni/coppie.csv` (n, sessioni, fonti, massimo, coppie di frame sopra soglia, frame
    della coppia migliore) e fogli numerati `fogli/DNN.jpg` (utente, 23/10/2026). L'utente risponde con i numeri da
    unire, che vanno in `10_doppioni/unire.txt` (un numero per riga, `#` per i commenti), letto dal passo dei gruppi."""
    dst = INTERMEDIATE / "10_doppioni"
    rows, e, sess, by_ses, best, cnt = _session_pairs(thr)
    order = sorted(best, key=lambda k: (sess[k[0]].split(":")[0], sess[k[1]].split(":")[0], -best[k][0]))
    _pair_sheets(dst, order, rows, e, sess, by_ses, best, cnt, thr, "D", per_sheet, seed)
    print(f"doppioni: {len(order)} coppie di sessioni con frame >= {thr} ("
          + ", ".join(f"{k} {v}" for k, v in Counter(sess[a].split(":")[0] + "-" + sess[b].split(":")[0]
                                                    for a, b in order).most_common())
          + f"), {-(-len(order) // per_sheet)} fogli -> {dst}")


TARATURA = (0.85, 0.90, 0.92, 0.94, 0.95)  # fasce del massimo per coppia di sessioni, sotto la soglia dei doppioni


def threshold_sheets(per: int = 40, seed: int = 0) -> None:
    """Taratura della soglia 0,95 (utente, 23/10/2026): `per` coppie di sessioni a caso per fascia di `TARATURA`
    (somiglianza massima fra i loro frame), sotto la soglia quindi mai riviste, in ordine di somiglianza
    decrescente (utente, 23/10/2026) -> `10_doppioni/taratura/coppie.csv` (con la colonna `fascia`) e fogli
    `fogli/TNN.jpg` come quelli dei doppioni. L'operatore scrive in `taratura/stessa_sessione.txt` i numeri delle
    coppie che sono la stessa sessione (stesso posto, stesse condizioni, stesso volo o girato); lo stesso posto in
    un altro giorno non conta. Poi `--step stima_soglia`."""
    dst = INTERMEDIATE / "10_doppioni" / "taratura"
    rows, e, sess, by_ses, best, cnt = _session_pairs(TARATURA[0])
    rng = np.random.default_rng(seed)
    pick = []
    for lo, hi in zip(TARATURA, TARATURA[1:]):
        band = sorted(k for k in best if lo <= best[k][0] < hi)
        pick += [band[i] for i in rng.choice(len(band), min(per, len(band)), replace=False)]
        print(f"taratura {lo:.2f}-{hi:.2f}: {len(band)} coppie di sessioni, {min(per, len(band))} nei fogli")
    order = sorted(pick, key=lambda k: -best[k][0])  # decrescente: le stranezze in alto si vedono prima (utente)
    _pair_sheets(dst, order, rows, e, sess, by_ses, best, cnt, TARATURA[0], "T", seed=seed)
    print(f"{len(order)} coppie, {-(-len(order) // 12)} fogli -> {dst}; risposte in stessa_sessione.txt")


def threshold_marks(per_sheet: int = 12) -> None:
    """Giudizi degli operatori sui fogli di taratura, a colpo d'occhio (utente, 23/10/2026): legge
    `taratura/giudizi_*.txt` (righe `N S|D[?] motivo`) e ridisegna i fogli in `taratura/fogli_esito/` con un riquadro
    per coppia: verde = stessa sessione, rosso = diversa, arancio = incerto (`?`), con la lettera nell'intestazione."""
    import cv2

    dst = INTERMEDIATE / "10_doppioni" / "taratura"
    vote = {}
    for f in sorted(dst.glob("giudizi_*.txt")):
        for ln in f.read_text().splitlines():
            t = ln.split()
            if len(t) >= 2 and t[0].isdigit():
                vote[int(t[0])] = t[1]
    W, H, TH = TILE
    step = 26 + TH + H + 10  # intestazione, striscia di testo, miniatura, spazio (come `_pair_sheets`)
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
    """Quota di coppie "stessa sessione" per fascia dalla revisione di `threshold_sheets`, con l'intervallo di
    Wilson al 95% -> `10_doppioni/taratura/stima.csv`. Sopra 0,95 la quota e' 1 (146 coppie su 146, tutte unite)."""
    dst = INTERMEDIATE / "10_doppioni" / "taratura"
    same = set(_merges(dst / "stessa_sessione.txt"))
    rows = list(csv.DictReader(open(dst / "coppie.csv")))
    out = []
    for lo, hi in zip(TARATURA, TARATURA[1:]):
        rs = [r for r in rows if lo <= float(r["massimo"]) < hi or hi == TARATURA[-1] == float(r["massimo"])]  # 0,950 arrotondato
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
    """Numeri di `unire.txt`: `N`, `N-M`, `N via`; `#` commento."""
    out = []
    for ln in f.read_text().splitlines():
        ln = ln.split("#")[0].split()
        if ln:
            a, _, b = ln[0].partition("-")
            out += range(int(a), int(b or a) + 1)
    return out


def groups() -> None:
    """Passo 11, gruppi -> `intermediate/11_gruppi/`. Unione (union-find) delle sessioni note di ogni fonte con le
    coppie di `10_doppioni/unire.txt`: `gruppi.csv` (sessione, gruppo, frame). Niente deduplica dei frame fra
    sessioni (23/10/2026): i doppioni veri (stessi box nello stesso punto) sono 17 coppie sopra 0,95."""
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

    dop = INTERMEDIATE / "10_doppioni"
    pairs = {int(r["n"]): (r["sessione_a"], r["sessione_b"]) for r in csv.DictReader(open(dop / "coppie.csv"))}
    for k in _merges(dop / "unire.txt"):
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



# intervalli di mezza ottava per seguire la curva dentro le fasce (utente, 23/10/2026); l'ultimo comprende fino a 320 px e oltre
HALF_OCTAVES = [8, 2 ** 3.5, 16, 2 ** 4.5, 32, 2 ** 5.5, 64, 2 ** 6.5, 128, 2 ** 7.5, 256, 1e9]
QUARTER_OCTAVES = [8 * 2 ** (k / 4) for k in range(21)] + [1e9]  # quarti d'ottava per il train, fino a 256 px e oltre
TEST_TOT = 1430  # box per classe nel test, uniformi in log s (286 x 5 fasce di prima)


_ST: dict = {}  # stato dello split per i processi figli (fork)


CLASS_WEIGHT = np.array([1, 1, 4, 1])[:, None]  # airplane, bird, drone, helicopter: droni x4 (utente, 23/10/2026: errore ~5%)


def _err(x, t):
    m = t > 0
    return ((CLASS_WEIGHT * ((x - t) / np.where(m, t, 1)) ** 2)[m]).sum()


MIN_SPLIT = {"val": 2000, "test": 1000}  # istanze minime per classe (utente, 23/10/2026; il train ne ha 10.000)


def _obj(s: str, x, t) -> float:
    """Errore di uno split: celle (`_err`) piu' una penalita' forte per ogni classe sotto il minimo `MIN_SPLIT`."""
    short = np.maximum(0, MIN_SPLIT[s] * 1.02 - x.sum(1)) / MIN_SPLIT[s]
    return _err(x, t) + 100 * (short ** 2).sum()


def _split_trial(seed: int) -> tuple:
    """Un tentativo di `split_groups`: greedy su un ordine a caso dei gruppi, poi rifinitura spostando un gruppo
    alla volta finche' l'errore scende -> (errore, assegnazione)."""
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
    """Passo 11, split per gruppi -> `intermediate/12_split/`. Via le immagini con un box sotto gli 8 px (utente,
    23/10/2026: in nessuno split). Ogni gruppo di `11_gruppi/gruppi.csv` va tutto in uno split. Obiettivi per
    classe: val = 20% della curva (`distribuzioni.ideale`, `TOT` per classe, curva alzata) su intervalli di mezza ottava (`FINE`,
    utente 23/10/2026: con le 5 fasce la forma dentro la fascia era libera), al piu' il 20% di quello che c'e';
    test piatto per fascia (5 fasce), 286 per cella, al piu' il 30% di quello che c'e'; il resto in train.
    Greedy su `trials` ordini a caso dei gruppi: ogni gruppo va in val o test se avvicina gli obiettivi (errore
    relativo al quadrato), altrimenti in train, poi rifinitura spostando un gruppo alla volta finche' l'errore scende; i gruppi sopra `big` dei frame vanno in train, e un gruppo non
    puo' dare piu' di `share` dell'obiettivo di una cella di val o test (varieta'). Si tiene l'ordine migliore.
    Scrive `split.csv` (gruppo, split), `frame_split.csv` (immagine, split) e `report.csv` (cella, obiettivi e
    istanze per split)."""
    from tools.distributions import ALPHA, CUT, TOT, ideal, uniform

    edges = [8, 16, 32, 64, 96, 1e9]
    out = INTERMEDIATE / "12_split"
    out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(INTERMEDIATE / "9_split" / "frame.csv")))
    grp = {r["sessione"]: r["gruppo"] for r in csv.DictReader(open(INTERMEDIATE / "11_gruppi" / "gruppi.csv"))}
    names = ["airplane", "bird", "drone", "helicopter"]
    gids = sorted(set(grp.values()))
    gi = {g: k for k, g in enumerate(gids)}
    C = np.zeros((len(gids), 4, 5))  # fasce, per il test
    F = np.zeros((len(gids), 4, len(HALF_OCTAVES) - 1))  # mezze ottave, per il val
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
    # test uniforme in log s (utente, 23/10/2026: l'ideale del test e' piatto), sulle mezze ottave fino alla fine
    # della curva (uccelli 96 px), 1.430 per classe (prima 286 per fascia x 5), al piu' il 30% di quello che c'e'
    tt = np.minimum(np.array([uniform(TEST_TOT, HALF_OCTAVES, CUT[c]) for c in names]), 0.3 * F.sum(0))
    # almeno 1.000 per classe nel test (utente, 23/10/2026): se le fasce povere non bastano, si alza l'obiettivo
    # delle altre in proporzione, sempre entro il 30% di quello che c'e'
    for c in range(4):
        while tt[c].sum() < 1050 and (room := (tt[c] < 0.3 * F.sum(0)[c]) & (tt[c] > 0)).any():
            tt[c, room] = np.minimum(tt[c, room] * 1.02, 0.3 * F.sum(0)[c, room])
    X = {"val": F, "test": F}
    ttb = np.array([uniform(tt[c].sum(), edges, CUT[n]) for c, n in enumerate(names)])  # per fascia, solo report
    big_bird = {g for g in range(len(gids)) if C[g, names.index("bird"), 4]}  # gruppi con uccelli > 96 px: solo train
    N = nf.sum()

    def err(x, t):
        m = t > 0
        return (((x[m] - t[m]) / t[m]) ** 2).sum()

    free = [g for g in range(len(gids)) if nf[g] and nf[g] <= big * N]
    tg = {"val": tv, "test": tt}
    ok = {s: {g for g in free if g not in big_bird and np.all(X[s][g][tg[s] > 0] <= share * tg[s][tg[s] > 0])
              and not X[s][g][tg[s] == 0].any()} for s in tg}
    # i tentativi sono indipendenti: in parallelo su tutti i core (utente, 23/10/2026), stato condiviso via fork
    import multiprocessing as mp

    _ST.update(free=free, tg=tg, ok=ok, X=X)
    with mp.get_context("fork").Pool(20) as pool:  # 20 core su 24: margine per il resto della macchina
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
    tvb = np.minimum(0.2 * np.array([ideal(TOT[c], ALPHA[c], edges, CUT[c]) for c in names]), 0.2 * tot)  # per fascia, solo report
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
    """Controllo dello split (`12_split/`): per ogni cella di val e test quanti gruppi e fonti la riempiono, e per
    ogni immagine di val e test la somiglianza DINOv2 con l'immagine piu' vicina del train (leakage rimasto, da
    confrontare con 0,95, la soglia delle unioni) -> `vicini.png` e la stampa."""
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
    """Passo 12, tetto del train e mosaici ridotti -> `intermediate/13_train_compensation/` (utente, 23/10/2026, README.md (Regole)).
    Tetto = curva: obiettivo del train per classe e mezza ottava (`FINE`) = 70% della curva (10.000 per classe); via, a turno fra i video,
    le immagini del train con tutti i box in celle sopra l'obiettivo (`tetto.csv`). Con quelle immagini, mosaici
    640x512 per le celle sotto l'obiettivo: per ogni immagine il fattore (0,5, 0,25, 0,125 -> griglia 2x2, 4x4,
    8x8) che porta piu' box in celle mancanti, solo se tutti i box restano >= 8 px; ogni immagine al massimo una
    volta, in ordine a caso a turno fra i video; prima le immagini fuori dal tetto, poi (se manca ancora) copie ridotte
    di immagini rimaste nel train. Ogni casella: riduzione INTER_AREA, sfocatura gaussiana leggera
    (sigma 0,3-0,8), rumore (sigma 1-3); il mosaico in JPEG qualita' 75-92. Scrive `images/`, `labels/`,
    `mosaici.csv` (mosaico, casella, sorgente, fattore, sigma, rumore, jpeg) e `train.csv` (immagini reali tenute
    piu' mosaici, con la label)."""
    import cv2

    from tools.distributions import ALPHA, CUT, TOT, ideal

    edges, names = QUARTER_OCTAVES, ["airplane", "bird", "drone", "helicopter"]  # quarti d'ottava (utente, 23/10/2026: droni 8-12 px)
    nb = len(edges) - 1
    bands = [f"{edges[k]:.0f}-{edges[k + 1]:.0f}" if k < nb - 1 else f">{edges[k]:.0f}" for k in range(nb)]
    out = INTERMEDIATE / "13_train_compensation"
    for d in ("images", "labels"):
        (out / d).mkdir(parents=True, exist_ok=True)
        for f in (out / d).iterdir():
            f.unlink()
    rows = list(csv.DictReader(open(INTERMEDIATE / "9_split" / "frame.csv")))
    fs = [r["split"] for r in csv.DictReader(open(INTERMEDIATE / "12_split" / "frame_split.csv"))]
    target = 0.7 * np.array([ideal(TOT[c], ALPHA[c], edges, CUT[c]) for c in names])  # tetto: curva alzata
    target[target == 0] = np.inf  # oltre la fine della curva (uccelli > 96 px) nessun tetto
    fill = 0.7 * np.array([ideal(TOT[c], ALPHA[c], edges, CUT[c]) for c in names])  # mosaici: la stessa curva alzata
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
    # tetto: via le immagini con tutti i box in celle sopra l'obiettivo, a turno fra i video, finche' si puo'
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
    # mosaici: per ogni immagine fuori dal tetto il fattore che porta piu' box in celle mancanti
    tiles = defaultdict(list)  # fattore -> [(i, bx)]
    # prima l'eccesso del tetto, poi copie ridotte di immagini rimaste nel train (utente, 23/10/2026: va bene se
    # un'immagine compare piu' volte nel train, purche' le copie siano ridotte). Obiettivo dei mosaici: la curva alzata,
    # la stessa del tetto; se una classe resta sotto 10.000, si alza solo sotto i 64 px, di 0,25 alla volta fino a x3
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
            while len(group) < g * g:  # l'ultimo mosaico si completa con caselle gia' usate in questo stesso mosaico
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
                if k < len(idx[m0:m0 + g * g]):  # le caselle ripetute non si contano due volte nel csv
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
    """Passo 14, immagini di calibrazione int8 -> `intermediate/15_calib/calib.csv` (immagine, label, tipo, cella).
    Solo dal train e solo immagini vere (niente mosaici): `bg` di background, il resto in parti uguali fra le 20
    celle classe x fascia; per ogni cella a turno fra i video, un'immagine conta per la cella per cui e' scelta."""
    edges, names = [16, 32, 64, 96], ["airplane", "bird", "drone", "helicopter"]
    bands = ["8-16", "16-32", "32-64", "64-96", ">96"]
    rng = np.random.default_rng(seed)
    out = INTERMEDIATE / "15_calib"
    out.mkdir(parents=True, exist_ok=True)
    train = [r for r in csv.DictReader(open(INTERMEDIATE / "13_train_compensation" / "train.csv")) if r["tipo"] == "reale"]
    video = {r["immagine"]: r["video"] for r in csv.DictReader(open(INTERMEDIATE / "9_split" / "frame.csv"))}
    by_cell = defaultdict(lambda: defaultdict(list))  # cella -> video -> immagini
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


# ordine delle classi nel dataset esportato (utente, 23/10/2026): le label sorgente restano nell'ordine di AOD4
# (airplane, bird, drone, helicopter), l'export le riscrive con questi id
OUT_NAMES = ["drone", "airplane", "helicopter", "bird"]
OUT_ID = [OUT_NAMES.index(c) for c in ("airplane", "bird", "drone", "helicopter")]


# testi delle figure per lingua (utente, 23/10/2026: anche in inglese, pubblico USA)
_TXT = {
    "it": {"classi": {"drone": "Drone", "airplane": "Aereo", "helicopter": "Elicottero", "bird": "Uccello"},
           "orig": "Dati originali (prima della selezione)", "reali": "Addestramento, immagini reali",
           "mosaici": "Addestramento, mosaici sintetici", "val": "Validazione", "test": "Test",
           "ob_train": "Obiettivo addestramento", "ob_val": "Obiettivo validazione", "ob_test": "Obiettivo test (uniforme)",
           "xsize": r"Dimensione del box $\sqrt{w\,h}$ [px]", "ysize": "Istanze per intervallo",
           "short": ["Stesso gruppo,\naltra clip (leakage)", "Gruppi diversi\n(fondo)", "Validazione", "Test"],
           "xsim": "Somiglianza con l'immagine\npiù vicina dell'addestramento", "a": "(a) Distribuzioni", "b": "(b) Coda",
           "fondo": "Fondo", "leak": "Leakage", "xthr": "Soglia di somiglianza\n", "ythr": "Quota di immagini sopra la soglia",
           "dec": ","},
    "en": {"classi": {"drone": "Drone", "airplane": "Airplane", "helicopter": "Helicopter", "bird": "Bird"},
           "orig": "Original data (before selection)", "reali": "Training, real images",
           "mosaici": "Training, synthetic mosaics", "val": "Validation", "test": "Test",
           "ob_train": "Training target", "ob_val": "Validation target", "ob_test": "Test target (uniform)",
           "xsize": r"Box size $\sqrt{w\,h}$ [px]", "ysize": "Instances per bin",
           "short": ["Same group,\nother clip (leakage)", "Different groups\n(baseline)", "Validation", "Test"],
           "xsim": "Similarity to the nearest\ntraining image", "a": "(a) Distributions", "b": "(b) Tail",
           "fondo": "Baseline", "leak": "Leakage", "xthr": "Similarity threshold\n", "ythr": "Share of images above threshold",
           "dec": "."},
}


def _output_plot(rows: list, out: Path, lang: str = "it", src_names=("airplane", "bird", "drone", "helicopter"),
                 table: bool = True) -> None:
    """`distribuzione.{png,pdf}` dell'export, per la tesi (testi in italiano, senza titolo: va nella didascalia):
    istanze per grandezza sqrt(w*h) e classe in addestramento (reali in blu, mosaici in giallo), validazione e test,
    l'ombra grigia dei dati originali (tutte le immagini annotate, prima di ogni selezione), con gli obiettivi
    (curva alzata al 70% e al 20%, test uniforme in log s). Istanze ed errore rispetto all'obiettivo,
    sum|osservato - obiettivo| / sum(obiettivo) sulle mezze ottave (`FINE`), in `distribuzione.md` (se `table`).
    `lang` it o en (`_TXT`; in inglese il file e' `distribution_en`); `src_names`: ordine delle classi nelle label
    di `rows` (sorgenti nell'export, `OUT_NAMES` se le label vengono da `data/output`)."""
    import matplotlib.pyplot as plt

    from tools.distributions import ALPHA, BINS, CUT, SMIN, TOT, _axes, curve, ideal, uniform
    from tools.plots import STYLE, WIDTH, _P, save

    T = _TXT[lang]
    it = T["classi"]
    s = defaultdict(list)
    rep = list(csv.DictReader(open(INTERMEDIATE / "12_split" / "report.csv")))

    def sizes(lab, names=("airplane", "bird", "drone", "helicopter")):  # id delle label sorgente
        for ln in (DATA / lab).read_text().splitlines():
            c, _, _, w, h = ln.split()
            yield names[int(c)], np.sqrt(float(w) * 640 * float(h) * 512)

    for r in rows:
        if r["lab"] and r["split"] in ("train", "val", "test"):
            for c, v in sizes(r["lab"], src_names):
                s[r["split"], c].append(v)
                if r["tipo"] == "mosaico":
                    s["mosaico", c].append(v)
    # dati originali: tutte le immagini annotate, prima di filtro < 8 px, split, tetto e mosaici
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
        ax.stairs(h_or, BINS, color="0.5", lw=.5, zorder=1.5)  # davanti alla zona < 8 px: si vede anche cosa e' stato tolto
        ax.stairs(h_tr - h_mo, BINS, fill=True, color=_P[0], alpha=.35, lw=0, label=T["reali"])
        ax.stairs(h_tr, BINS, baseline=h_tr - h_mo, fill=True, color=_P[3], alpha=.75, lw=0,
                  label=T["mosaici"])
        ax.stairs(h_tr, BINS, color=_P[0], lw=.6, alpha=.8)
        for sp, col in (("val", _P[1]), ("test", _P[2])):
            ax.stairs(np.histogram(s[sp, c], BINS)[0], BINS, color=col, lw=1.1, label=T[sp])
        ax.plot(*curve(TOT[c] * .7, ALPHA[c], CUT[c]), color="0.1", lw=.9, label=T["ob_train"])
        ax.plot(*curve(TOT[c] * .2, ALPHA[c], CUT[c]), color="0.1", lw=.9, ls=(0, (4, 2)), label=T["ob_val"])
        n_t = sum(float(r["obiettivo_test"]) for r in rep if r["classe"] == c)  # test: uniforme in log s
        ax.plot([SMIN, CUT[c]], [n_t * dlog / np.log(CUT[c] / SMIN)] * 2, color="0.1", lw=.9, ls=(0, (1, 1.5)),
                label=T["ob_test"])
        _axes(ax, c, 0)
        ax.set_title("")  # il titolo centrato di `_assi` (in inglese) non serve
        ax.set_title(f"({'abcd'[k_ax]}) {it[c]}", loc="left", fontsize=9)
        ax.tick_params(labelsize=7.5)
        # errore rispetto all'obiettivo, sulle mezze ottave
        err = {}
        for sp, k in (("train", .7), ("val", .2)):
            ide = k * ideal(TOT[c], ALPHA[c], HALF_OCTAVES, CUT[c])
            err[sp] = np.abs(np.histogram(s[sp, c], HALF_OCTAVES)[0] - ide).sum() / ide.sum()
        ide = uniform(n_t, HALF_OCTAVES, CUT[c])
        err["test"] = np.abs(np.histogram(s["test", c], HALF_OCTAVES)[0] - ide).sum() / ide.sum()
        ax.tick_params(labelbottom=True)  # numeri dell'asse x anche nei pannelli in alto
        md.append((c, err, {sp: len(s[sp, c]) for sp in ("train", "val", "test")},
                   {sp: np.histogram(s[sp, c], HALF_OCTAVES)[0] for sp in ("train", "val", "test")},
                   {"train": .7 * ideal(TOT[c], ALPHA[c], HALF_OCTAVES, CUT[c]), "val": .2 * ideal(TOT[c], ALPHA[c], HALF_OCTAVES, CUT[c]),
                    "test": uniform(n_t, HALF_OCTAVES, CUT[c])}, len(s["mosaico", c])))
    top = max(np.histogram(s[k, c], BINS)[0].max() for c in OUT_NAMES for k in ("train", "orig"))
    axs[0, 0].set_yscale("log")  # originali (centinaia), addestramento, val e test (decine) leggibili insieme (utente)
    axs[0, 0].set_ylim(3, top * 1.5)
    for ax in axs[1]:
        ax.set_xlabel(T["xsize"])
    for ax in axs[:, 0]:
        ax.set_ylabel(T["ysize"])
    hs, ls = axs[0, 0].get_legend_handles_labels()
    # handle: 0 originali, 1 reali, 2 mosaici, 3 validazione, 4 test, 5 ob. addestramento, 6 ob. validazione, 7 ob. test
    order = [0, 5, 1, 6, 2, 7, 3, 4]  # quattro colonne di due: i dati sopra, gli obiettivi sotto
    f.legend([hs[k] for k in order], [ls[k] for k in order], loc="outside lower center", ncol=4, fontsize=7, frameon=False)
    save(f, out / ("distribuzione" if lang == "it" else "distribution_en"), pdf=True)
    if not table:
        return
    # la tabella che accompagna il grafico (utente, 23/10/2026): stessi numeri, fuori dalla figura
    lab = [f"{HALF_OCTAVES[k]:.0f}–{HALF_OCTAVES[k + 1]:.0f}" if k < len(HALF_OCTAVES) - 2 else f"> {HALF_OCTAVES[k]:.0f}" for k in range(len(HALF_OCTAVES) - 1)]
    L = ["# Distribuzione delle grandezze", "",
         "Generata con `distribuzione.png` da `tools/split.py --step export`. Errore = Σ|osservato − obiettivo| / Σ obiettivo",
         "sulle mezze ottave; obiettivi: addestramento e validazione la curva s^−0,3 alzata ×1,2 (70% e 20%), test uniforme",
         "in scala logaritmica. Uccelli fino a 96 px.", "",
         "| classe | addestramento | di cui mosaici | errore | validazione | errore | test | errore |", "|---|---|---|---|---|---|---|---|"]
    L += [f"| {it[c]} | {num(n['train'])} | {num(nm)} | {e['train']:.0%} | {num(n['val'])} | {e['val']:.0%} | {num(n['test'])} | {e['test']:.0%} |"
          for c, e, n, _, _, nm in md]
    for c, e, n, h, ide, _ in md:
        L += ["", f"## {it[c]}", "", "| √(w·h) [px] | addestramento | obiettivo | validazione | obiettivo | test | obiettivo |",
              "|---|---|---|---|---|---|---|"]
        L += [f"| {lab[k]} | {num(int(h['train'][k]))} | {num(round(ide['train'][k]))} | {num(int(h['val'][k]))} | "
              f"{num(round(ide['val'][k]))} | {num(int(h['test'][k]))} | {num(round(ide['test'][k]))} |" for k in range(len(lab))]
    (out / "distribuzione.md").write_text("\n".join(L).replace("%", " %") + "\n")

def _leakage_plot(out: Path, block: int = 2048, lang: str = "it", table: bool = True) -> None:
    """`leakage.{png,pdf,md}` dell'export, per la tesi: per ogni immagine reale di validazione e test la somiglianza
    DINOv2 (ViT-S, coseno) con l'immagine reale piu' vicina dell'addestramento, contro due riferimenti dentro
    l'addestramento: stesso gruppo e altra clip (com'e' fatto un leakage vero) e gruppi diversi (il fondo: il cielo
    somiglia al cielo). (a) densita', (b) quota di immagini sopra ogni soglia, in scala logaritmica; 0,95 e' la
    soglia dei doppioni rivisti a mano. La tabella con mediana, percentili e quote in `leakage.md` (se `table`).
    Legge solo `intermediate/` (frame, split, gruppi, embedding), scrive in `out`; `lang` it o en (`leakage_en`)."""
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
        """massimo coseno di ogni riga di q verso il train, con la maschera `mode`"""
        outv = []
        for k in range(0, len(q), block):
            qq = q[k:k + block]
            S = (e[qq] @ e[tr].T).float()
            if mode == "stesso":  # stesso gruppo, altra clip
                S[(g[qq][:, None] != g[tr][None]) | (c[qq][:, None] == c[tr][None])] = -2
            elif mode == "diversi":
                S[g[qq][:, None] == g[tr][None]] = -2
            outv.append(S.max(1).values)
        v = torch.cat(outv).cpu().numpy()
        return v[v > -2]

    res = {
        "Validazione → addestramento": nearest(torch.tensor([i for i, s in enumerate(sp) if s == "val"], device="cuda"), "tutti"),
        "Test → addestramento": nearest(torch.tensor([i for i, s in enumerate(sp) if s == "test"], device="cuda"), "tutti"),
        "Addestramento, stesso gruppo e altra clip (leakage)": nearest(tr, "stesso"),
        "Addestramento, gruppi diversi (fondo)": nearest(tr, "diversi"),
    }
    # (a) un violino orizzontale per confronto, dall'alto: leakage, fondo, validazione, test
    keys = ["Addestramento, stesso gruppo e altra clip (leakage)", "Addestramento, gruppi diversi (fondo)",
            "Validazione → addestramento", "Test → addestramento"]
    T = _TXT[lang]
    short = T["short"]
    cols = ["0.2", "0.6", _P[1], _P[2]]
    plt.style.use(STYLE)
    f, (a1, a2) = plt.subplots(1, 2, figsize=(WIDTH, WIDTH * 0.45))  # stessa larghezza per i due pannelli
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
    # (b) coda: quota di immagini sopra ogni soglia
    x = 1 - np.geomspace(.5, .002, 600)  # fitto vicino a 1, dove l'asse si allarga
    tail = {k: np.array([(res[k] >= s).mean() for s in x]) for k in keys}
    a2.fill_between(x, 1e-5, np.maximum(tail[keys[1]], 1e-5), color="0.6", alpha=.3, lw=0, label=T["fondo"])
    a2.plot(x, tail[keys[0]], color=cols[0], lw=1.6, ls=(0, (4, 2)), label=T["leak"])
    for k, col, lab in zip(keys[2:], cols[2:], short[2:]):
        a2.plot(x, tail[k], color=col, lw=1.6, label=lab)
    a2.axvline(.95, color="0.4", lw=.7)
    a2.set_yscale("log")
    a2.set_xlim(.5, .999)
    a2.set_ylim(1e-4, 1.3)
    a2.set_xlabel(T["xthr"], fontsize=8)  # due righe come (a), assi allineati
    a2.set_ylabel(T["ythr"], fontsize=8)
    a2.set_title(T["b"], loc="left", fontsize=9)
    a2.text(.95, 1.5e-4, f" 0{T['dec']}95", ha="left", va="bottom", fontsize=7, color="0.3")
    f.legend(*a2.get_legend_handles_labels(), loc="outside lower center", ncol=4, fontsize=7, frameon=False)
    # asse x che allarga la zona vicino a 1 (utente, 23/10/2026): -log10(1 - s), tacche in somiglianza
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


FIG = ROOT / "docs" / "figure"  # figure del README, in italiano e in inglese


def figures() -> None:
    """Figure del dataset finito per il README, in `docs/figure/`, in italiano e in inglese (`distribuzione`,
    `distribution_en`, `leakage`, `leakage_en`): stesse funzioni dell'export, ma `data/output` (congelato) si legge
    soltanto (manifest e label, con gli id di `OUT_NAMES`); niente tabelle .md."""
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
    """Pannello per la tesi sulla soglia 0,95 dei doppioni fra sessioni (`docs/figure/soglia.{png,pdf}` e
    `soglia_en`): una riga per fascia di somiglianza DINOv2 con `per` coppie a caso (seed fisso) di frame di
    sessioni diverse, box in rosso, e in fondo coppie della stessa clip come riferimento. Sotto 0,95 riprese diverse
    con lo stesso tipo di cielo; sopra, la stessa ripresa o lo stesso girato ripubblicato (le 146 coppie di sessioni
    sopra 0,95 rivisti a mano e unite, passo 10). Riferimento: le coppie della stessa clip al 25°, 50° e 75°
    percentile della loro somiglianza (a caso sarebbero troppo disperse: una clip DVIDS cambia inquadratura)."""
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
        left, top, row = .2, .25, (1 - .2) * WIDTH / sum(wr) * .8 + .28  # in pollici: immagine 5:4 + titolo
        H = (nb + 1) * row + top
        f, axs = plt.subplots(nb + 1, len(wr), figsize=(WIDTH, H), gridspec_kw=dict(width_ratios=wr, wspace=.03))
        f.set_layout_engine("none")  # posizioni fisse: servono per le linee fra le righe
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
        # linea della soglia fra la riga 0,90-0,95 e quella 0,95-0,98
        gap = .03 / H  # appena sotto la riga sopra, prima dei titoli della riga sotto
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
    """Passo 14, il dataset per Ultralytics in `data/output/` (cancellato e rifatto ogni volta), immagini in hardlink,
    label riscritte con gli id di `OUT_NAMES` (0 drone, 1 airplane, 2 helicopter, 3 bird):
    `images/{train,val,test,calib}/` e `labels/...` con lo stesso nome, e `raw/{images,labels}/` con train, val e test
    insieme (senza calib), per chi vuole rifare gli split; train = reali tenuti dal tetto e mosaici
    (`13_train_compensation/train.csv`), val e test da `12_split/frame_split.csv`, piu' il background di
    `data/input/background/0_raw/background.csv` (senza label: per Ultralytics e' un'immagine di solo sfondo), calib da
    `15_calib/calib.csv` (stesso nome dell'immagine del train). Nomi (utente, 23/10/2026):
    reale `R_<gruppo>_<hash8 di fonte:clip>_<n>`, mosaico `M_x<griglia>_<n>`, background `B_<hash8 di
    fonte:camera>_<n>`; n a 6 cifre, in ordine di frame dentro la clip o la camera. `dataset.yaml`, `calib.yaml` e
    `manifest.csv` (immagine, split, tipo, gruppo, fonte, clip o camera, frame originale, sorgente)."""
    import hashlib
    import os
    import re

    agg = INTERMEDIATE
    out = OUTPUT
    if out.exists():  # congelato (utente, 23/10/2026): per rifarlo va spostato o cancellato a mano
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
    for (tipo, f, c), rs in by_key.items():
        for n, r in enumerate(sorted(rs, key=lambda r: (str(r["frame"]).zfill(12), r["img"]))):
            if tipo == "reale":
                r["nome"] = f"R_{r['gruppo']}_{h8(f'{f}:{c}')}_{n:06d}"
            elif tipo == "background":
                r["nome"] = f"B_{h8(f'{f}:{c}')}_{n:06d}"
            else:
                k, g = re.match(r"M(\d+)_x(\d+)", Path(r["img"]).stem).groups()
                r["nome"] = f"M_x{g}_{int(k):06d}"
    assert len({r["nome"] for r in rows}) == len(rows), "nomi doppi"
    f = agg / "15_calib" / "calib.csv"
    if f.exists():  # calibrazione int8: copie in piu' (hardlink) di immagini del train, con lo stesso nome
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
        if r["split"] != "calib":  # raw/: tutto il dataset in una cartella sola, senza split (utente, 23/10/2026)
            os.link(DATA / r["img"], out / "raw" / "images" / f"{r['nome']}.jpg")
            if r["lab"]:
                (out / "raw" / "labels" / f"{r['nome']}.txt").write_text(lab)
    (out / "dataset.yaml").write_text(
        f"# air-detection, tools/split.py --step export\npath: {out}\ntrain: images/train\nval: images/val\n"
        "test: images/test\nnames:\n" + "".join(f"  {k}: {c}\n" for k, c in enumerate(OUT_NAMES)))
    (out / "calib.yaml").write_text(  # per l'export int8 di Ultralytics, che calibra sullo split val
        f"# calibrazione int8 (intermediate/15_calib), tools/split.py --step export\npath: {out}\ntrain: images/calib\n"
        "val: images/calib\nnames:\n" + "".join(f"  {k}: {c}\n" for k, c in enumerate(OUT_NAMES)))
    with open(out / "manifest.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["immagine", "split", "tipo", "gruppo", "fonte", "sessione", "clip", "frame", "sorgente"])
        w.writerows((f"images/{r['split']}/{r['nome']}.jpg", r["split"], r["tipo"], r["gruppo"], r["fonte"],
                     r["sessione"], r["clip"], r["frame"], r["img"]) for r in rows)
    # video DVIDS usati: pagina, autore e licenza (attribuzione, pubblico dominio video per video)
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
    p.add_argument("--step", choices=["frame", "embedding", "somiglianze", "somiglianze_tutte", "anteprima", "doppioni", "gruppi", "somiglianze_gruppi", "split", "controllo", "mosaici", "calib", "export", "figure", "soglia", "taratura", "esito_taratura", "stima_soglia"], required=True)
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
