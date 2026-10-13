"""Risale alle fonti di AOD4 (YouTube-8M, Anti-UAV, Svanström, foto stock) dai nomi dei file e conta video,
immagini e box per fonte. Solo riferimento: AOD4 non si usa. Da lanciare dalla radice di yoloeval (legge data/aod4)."""
import os, re, collections, sys
from PIL import Image
root = "data/aod4"
names = ["airplane", "bird", "drone", "helicopter"]
rows = []
for split in os.listdir(f"{root}/images"):
    for f in os.listdir(f"{root}/images/{split}"):
        if f.endswith(".npy"):
            continue
        stem = re.sub(r"_(jpg|jpeg|png)\.rf\..*", "", f)
        stem = re.sub(r"_(jpg|jpeg|png)$", "", stem)
        if stem.startswith("yt-"):
            src, vid = "youtube", stem[:14]
        elif re.match(r"^\d{8}_\d{6}_", stem):
            src, vid = "antiuav", "_".join(stem.split("_")[:4])
        elif re.match(r"^[VI]R?_[A-Z]+_\d+", stem):
            src, vid = "svanstrom", "_".join(stem.split("_")[:3])
        else:
            src, vid = "altro", stem
        lab = f"{root}/labels/{split}/" + os.path.splitext(f)[0] + ".txt"
        cls = []
        if os.path.exists(lab):
            cls = [int(l.split()[0]) for l in open(lab) if l.strip()]
        rows.append((split, src, vid, f, cls))
print("immagini:", len(rows))
by = collections.defaultdict(lambda: collections.Counter())
vids = collections.defaultdict(set)
for split, src, vid, f, cls in rows:
    by[src]["img"] += 1
    for c in cls:
        by[src][names[c]] += 1
    vids[src].add(vid)
for s in by:
    print(s, "video:", len(vids[s]), dict(by[s]))
# leakage: video in più split
vs = collections.defaultdict(set)
for split, src, vid, *_ in rows:
    if src != "altro":
        vs[vid].add(split)
print("video in più split:", sum(len(v) > 1 for v in vs.values()), "su", len(vs))
# esempi "altro"
alt = [r[3] for r in rows if r[1] == "altro"]
print("esempi altro:", alt[:8])
# risoluzioni per fonte (campione)
import random
random.seed(0)
for s in by:
    fs = [r for r in rows if r[1] == s]
    smp = random.sample(fs, min(150, len(fs)))
    c = collections.Counter(Image.open(f"{root}/images/{r[0]}/{r[3]}").size for r in smp)
    print(s, "risoluzioni:", c.most_common(5))
