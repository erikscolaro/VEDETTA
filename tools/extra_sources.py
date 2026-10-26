#!/usr/bin/env python3
"""Additions for the bird, airplane and helicopter gaps (README.md, Procedure,
step 10). Each source has its own `frame` step (-> `1_frame/<P>_<seq>/`, YOLO
labels on the whole frame); the rest is the `tools/pipeline.py` chain with the source
prefix: annotati, dedup, crop, then the crop labels become the
proposals, sessions, count, CVAT.

FBD-SV-2024 (birds, Kaggle `swjtuziwei/fbd-sv-2024`): original boxes, already
tight (seen on 19/10/2026), class 1.

    python tools/extra_sources.py --fonte fbd --step frame        # zip in data/fbd_sv/0_raw -> 1_frame
    python tools/extra_sources.py --fonte fbd --step catena       # annotati, dedup, crop, proposals
    python tools/extra_sources.py --fonte fbd --step sessioni     # DINOv2 clip groups -> 6_sessioni
    python tools/extra_sources.py --fonte fbd --step conta        # -> 7_conteggio
    python tools/extra_sources.py --fonte fbd --step cvat

DVIDS (military airplanes and helicopters, PUBLIC DOMAIN videos only): page URLs
in `data/dvids/0_raw/candidati.csv` (url, classe, titolo).

    python tools/extra_sources.py --fonte dvids --step cerca      # DVIDS API (key in ~/.config/dvids.env) -> candidati.csv
    python tools/extra_sources.py --fonte dvids --step info       # page: license, author, mp4, duration -> video.csv
    python tools/extra_sources.py --fonte dvids --step anteprima  # a frame every 10 s, from 10 s, read remotely
    python tools/extra_sources.py --fonte dvids --step doppioni   # videos republished under another id -> doppioni.txt
    python tools/extra_sources.py --fonte dvids --step scegli     # YOLO on the previews -> tratti.csv
    python tools/extra_sources.py --fonte dvids --step fogli      # one preview per segment -> fogli_tratti, esclusi.txt
    python tools/extra_sources.py --fonte dvids --step escludi --fogli fogli_tratti4 --scarti scarti.txt  # review -> esclusi.txt
    python tools/extra_sources.py --fonte dvids --step frame --secondi 600  # rounds of segments up to 600 new s/class, 5 fps, + YOLO
    python tools/extra_sources.py --fonte dvids --step catena / sessioni / conta / cvat
    python tools/extra_sources.py --fonte dvids --step fogli_grigi  # low-color segments -> fogli_grigi, then grigi.txt
    python tools/extra_sources.py --fonte dvids --step grigi --fogli fogli_grigi --scarti data/dvids/0_raw/scarti_grigi.txt
    python tools/extra_sources.py --fonte dvids --step fogli_verdi  # green night vision -> fogli_verdi, then verdi.txt (N)
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.common import INPUT, DETECTOR_CLASSES, detector  # noqa: E402
from tools import pipeline  # noqa: E402

SOURCES = {"fbd": ("FBD", INPUT / "fbd_sv"), "dvids": ("DVIDS", INPUT / "dvids"),
         "antiuav": ("AUAV", INPUT / "anti_uav")}
FPS = 5  # frames extracted per second from the DVIDS videos (disk: ~50 GB free; dedup removes many anyway)


def seqs(src: Path) -> list[str]:
    p = pipeline.PREFIX + "_"
    return sorted((d.name[len(p):] for d in (src / "1_frame").iterdir() if d.name.startswith(p)), key=lambda x: (len(x), x))


# --- FBD-SV-2024 -----------------------------------------------------------------


def xml_video_frames(src: Path) -> None:
    """Videos and VID labels of the zip -> `1_frame/FBD_<n>/` (frame i = XML i) and
    `0_raw/clip.csv` (sequenza, split, frame, box)."""
    import tempfile
    import xml.etree.ElementTree as ET
    import zipfile

    import cv2

    z = zipfile.ZipFile(src / "0_raw" / "archive.zip")
    names = z.namelist()
    vids = sorted(n for n in names if re.match(r"FBD-SV-2024/videos/\w+/bird_\d+\.mp4$", n))
    rows = []
    for n in vids:
        split, num = n.split("/")[2], re.search(r"bird_(\d+)", n).group(1)
        v = f"FBD_{num}"
        dst = src / "1_frame" / v
        if (dst / "labels").exists():
            continue
        (dst / "images").mkdir(parents=True, exist_ok=True)
        xmls = sorted(x for x in names if x.startswith(f"FBD-SV-2024/VID/labels/{split}/bird_{num}/") and x.endswith(".xml"))
        with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
            tmp.write(z.read(n))
            tmp.flush()
            cap = cv2.VideoCapture(tmp.name)
            imgs = []
            while True:
                ok, im = cap.read()
                if not ok:
                    break
                imgs.append(im)
        (dst / "labels").mkdir()
        nbox = 0
        for x in xmls:
            i = int(Path(x).stem)
            if i >= len(imgs):
                continue
            H, W = imgs[i].shape[:2]
            lines = []
            for o in ET.fromstring(z.read(x)).iter("object"):
                x1, y1, x2, y2 = (float(o.find(f"bndbox/{k}").text) for k in ("xmin", "ymin", "xmax", "ymax"))
                x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, W), min(y2, H)
                if x2 - x1 >= 1 and y2 - y1 >= 1:
                    lines.append(f"1 {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}\n")
            cv2.imwrite(str(dst / "images" / f"{v}_f{i:05d}.jpg"), imgs[i], [cv2.IMWRITE_JPEG_QUALITY, 95])
            (dst / "labels" / f"{v}_f{i:05d}.txt").write_text("".join(lines))
            nbox += len(lines)
        rows.append((num, split, len(imgs), nbox))
        print(f"fbd: {v} ({split}) {len(imgs)} frame, {len(xmls)} xml, {nbox} box", flush=True)
    f = src / "0_raw" / "clip.csv"
    old = list(csv.reader(open(f)))[1:] if f.exists() else []
    with open(f, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sequenza", "split", "frame", "box"])
        w.writerows(sorted(old + rows, key=lambda r: int(r[0])))


# --- Anti-UAV300 RGB --------------------------------------------------------------

ANTIUAV_URL = "https://huggingface.co/datasets/DestinyLove/AntiUAV300/resolve/main/yolo_rgb/images/"


def _get_image(args) -> bool:
    import time

    import requests

    url, dst = args
    for k in range(6):
        try:
            r = requests.get(url, timeout=60)
            if r.ok:
                dst.write_bytes(r.content)
                return True
        except requests.RequestException:
            pass
        time.sleep(5 * (k + 1))
    return False


def download_image_frames(src: Path, every: int = 25, workers: int = 16) -> None:
    """One frame every `every` of each video from `0_raw/all_labels.txt` (Hugging
    Face mirror, MIT license) -> `1_frame/AUAV_<video>/{images,labels}`, frame `f<i>`
    with i the index in the video, real labels (one drone, class 2)."""
    import shutil

    jobs, n = [], 0
    for ln in open(src / "0_raw" / "all_labels.txt"):
        f, lab = ln.rstrip("\n").split("|||")
        name = Path(f).stem
        vid, i = name.rsplit("_", 1)
        if int(i) % every:
            continue
        v = f"AUAV_{vid}"
        d = src / "1_frame" / v
        (d / "images").mkdir(parents=True, exist_ok=True)
        (d / "labels").mkdir(exist_ok=True)
        stem = f"{v}_f{int(i):05d}"
        p = lab.split()
        (d / "labels" / f"{stem}.txt").write_text(f"2 {' '.join(p[1:5])}\n" if len(p) >= 5 else "")
        if not (d / "images" / f"{stem}.jpg").exists():
            jobs.append((ANTIUAV_URL + f.replace(".txt", ".jpg"), d / "images" / f"{stem}.jpg"))
        n += 1
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(_get_image, jobs))
    for (_, img), ok in zip(jobs, res):  # missing on the mirror (404): drop the label too
        if not ok:
            (img.parent.parent / "labels" / f"{img.stem}.txt").unlink()
    for d in (src / "1_frame").glob("AUAV_*"):
        if not any((d / "labels").iterdir()):
            shutil.rmtree(d)
    print(f"auav: {n} frame, scaricati {sum(res)} su {len(jobs)}")


def antiuav_extra(offset: int = 12, every: int = 25, smin: float = 32, per_video: int = 30, seed: int = 0) -> None:
    """More drones in the sky (user, 26/10/2026: the drone class must be complete): Anti-UAV300 frames with index
    = `offset` mod `every` (between the frames already used), a drone above `smin` px, at most `per_video` per video
    (evenly spread; all the frames above 128 px first, they are rare), from the Hugging Face
    mirror (MIT), cropped 640x512 around the drone without resizing. Original Anti-UAV labels (hand-made, one drone
    per frame), not reviewed in CVAT: to be checked on a sample. Writes `input/anti_uav/8_extra/{images,labels}`
    and `extra.csv` (video, sessione, immagine, label)."""
    import cv2

    import math
    from collections import defaultdict

    rng = np.random.default_rng(seed)
    src = INPUT / "anti_uav"
    W, H = 640, 512
    out = src / "8_extra"
    for d in ("full", "images", "labels"):
        (out / d).mkdir(parents=True, exist_ok=True)
    ses = {r["clip"].split("/")[0]: r["sessione"] for r in csv.DictReader(open(src / "6_sessioni" / "sessioni.csv"))}
    jobs, lab = [], {}
    for ln in open(src / "0_raw" / "all_labels.txt"):
        f, b = ln.rstrip("\n").split("|||")
        p = b.split()
        vid, i = Path(f).stem.rsplit("_", 1)
        if len(p) < 5 or int(i) % every != offset or f"AUAV_{vid}" not in ses:
            continue
        x, y, w, h = (float(v) for v in p[1:5])
        if math.sqrt(w * 1920 * h * 1080) <= smin:
            continue
        stem = f"AUAV_{vid}_f{int(i):05d}"
        lab[stem] = (vid, x, y, w, h, f, math.sqrt(w * 1920 * h * 1080))
    by_v = defaultdict(list)
    for stem, v in sorted(lab.items()):
        by_v[v[0]].append(stem)
    keep = set()
    for vid, stems in by_v.items():
        big = [t for t in stems if lab[t][6] > 128]
        rest = [t for t in stems if lab[t][6] <= 128]
        k = max(0, per_video - len(big))
        keep |= set(big) | {rest[j] for j in np.linspace(0, len(rest) - 1, min(k, len(rest))).astype(int)} if rest else set(big)
    lab = {t: v[:5] + (v[5],) for t, v in lab.items() if t in keep}
    for stem, v in lab.items():
        if not (out / "full" / f"{stem}.jpg").exists():
            jobs.append((ANTIUAV_URL + v[5].replace(".txt", ".jpg"), out / "full" / f"{stem}.jpg"))
    with ThreadPoolExecutor(16) as ex:
        list(ex.map(_get_image, jobs))
    rows = []
    for stem, (vid, x, y, w, h, _) in sorted(lab.items()):
        im = cv2.imread(str(out / "full" / f"{stem}.jpg"))
        if im is None:
            continue
        Hh, Ww = im.shape[:2]
        x1, y1 = max(0.0, (x - w / 2) * Ww), max(0.0, (y - h / 2) * Hh)  # boxes cut by the frame border: visible part
        x2, y2 = min(float(Ww), (x + w / 2) * Ww), min(float(Hh), (y + h / 2) * Hh)
        if x2 - x1 > W or y2 - y1 > H or Ww < W or Hh < H:
            continue
        x0 = int(rng.uniform(max(0, x2 - W), min(x1, Ww - W) + 1e-6))
        y0 = int(rng.uniform(max(0, y2 - H), min(y1, Hh - H) + 1e-6))
        cv2.imwrite(str(out / "images" / f"{stem}.jpg"), im[y0:y0 + H, x0:x0 + W], [cv2.IMWRITE_JPEG_QUALITY, 95])
        (out / "labels" / f"{stem}.txt").write_text(f"2 {((x1 + x2) / 2 - x0) / W:.6f} {((y1 + y2) / 2 - y0) / H:.6f} "
                                                    f"{(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}\n")
        rows.append((f"AUAV_{vid}", ses[f"AUAV_{vid}"], f"input/anti_uav/8_extra/images/{stem}.jpg",
                     f"input/anti_uav/8_extra/labels/{stem}.txt"))
    with open(out / "extra.csv", "w", newline="") as fh:
        csv.writer(fh).writerows([("video", "sessione", "immagine", "label")] + rows)
    print(f"antiuav: {len(rows)} frame con drone > {smin:g} px -> {out}")


# --- DVIDS -----------------------------------------------------------------------


def _get(url: str) -> str:
    import requests

    for k in range(6):
        try:
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
            if r.status_code == 200 and r.text:
                return r.text
        except requests.RequestException:
            pass
        import time
        time.sleep(5 * (k + 1))
    return ""


def _probe(url: str) -> tuple[float, int, int, float]:
    m = re.search(r"-(\d+)x(\d+)-\d+k\.mp4$", url)
    if m and int(m.group(2)) < 720:  # only a reduced version: discarded anyway, no ffprobe
        return 0.0, int(m.group(1)), int(m.group(2)), 0.0
    for _ in range(4):
        try:
            r = subprocess.run(["ffprobe", "-v", "error", "-rw_timeout", "30000000", "-select_streams", "v:0",
                                "-show_entries", "format=duration:stream=width,height,r_frame_rate", "-of",
                                "csv=p=0", url], capture_output=True, text=True, timeout=120)
        except subprocess.TimeoutExpired:
            continue
        lines = r.stdout.split()
        if len(lines) >= 2:
            w, h, fr = lines[0].split(",")[:3]
            a, b = fr.split("/")
            return float(lines[1].strip(",")), int(w), int(h), float(a) / float(b)
    return 0.0, 0, 0, 0.0


def page_info(row: dict) -> list:
    """A DVIDS page -> id, class, title, license, author, mp4, duration, w, h, fps."""
    vid = re.search(r"/video/(\d+)", row["url"]).group(1)
    html = _get(row["url"])
    text = re.sub(r"<[^>]+>", " ", html)
    pd = "PUBLIC DOMAIN" in text
    by = re.search(r"Video by\s+([^&<\n]{2,80}?)\s{2,}", text)
    # the page also has reduced versions (`-1024x576-1769k.mp4`): the original first, else the largest
    mp4 = sorted(set(re.findall(r"https://[\w.-]+\.cloudfront\.net/video/[^\"' ]+?\.mp4", html)),
                 key=lambda u: (bool(m := re.search(r"-(\d+)x(\d+)-\d+k\.mp4$", u)),
                                -int(m.group(1)) * int(m.group(2)) if m else 0))
    dur, w, h, fps = _probe(mp4[0]) if pd and mp4 else (0.0, 0, 0, 0.0)
    return [vid, row["classe"], row["titolo"], "pubblico dominio" if pd else "NO", by.group(1).strip() if by else "",
            row["url"], mp4[0] if mp4 else "", round(dur, 1), w, h, round(fps, 3)]


# queries per class (quoted phrases, sorted by relevance); chosen for distant aircraft and drones in flight
QUERY = {
    "drone": ['"counter-UAS"', '"C-UAS"', '"counter-drone"', '"counter unmanned"', '"small UAS"', '"sUAS"',
              "quadcopter", '"drone swarm"', '"FPV drone"', "Skydio", '"Raven UAS"', '"Puma UAS"',
              '"Black Hornet"', '"drone training"', '"UAS operations"', '"unmanned aircraft system"',
              "Dronebuster", '"drone defense"', '"drone demonstration"', '"one-way attack drone"',
              '"UAS demonstration"', '"drone range"', '"UAS flight"', '"Group 1 UAS"', '"Group 2 UAS"',
              '"drone pilot"', '"drone course"', '"drone competition"', '"drone flight"', '"quadcopters"',
              '"drone lethality"', '"drone strike"', '"Anduril"', '"Coyote"', '"Switchblade"'],
    "helicopter": ['"air assault"', '"helicopter formation"', '"sling load"', '"UH-60"', '"AH-64"', '"CH-47"',
                   '"HH-60"', '"MH-60"', '"UH-72"', '"CH-53"', '"AH-1Z"', '"UH-1Y"', "medevac hoist",
                   '"aerial gunnery"', "FARP", '"air movement"', '"helicopter flyover"', '"helicopters"',
                   '"helicopter landing"', "hover", '"landing zone"', "helocast", '"fast rope"', '"SPIE"',
                   '"helicopter fly-by"', '"Black Hawk"', '"Apache"', '"Chinook"', '"Lakota"', '"Seahawk"',
                   '"deck landing"', '"hoist training"', '"water bucket"', '"helicopter rescue"',
                   '"Huey"', '"Kiowa"', '"Super Stallion"', '"Viper"', '"Pave Hawk"',
                   '"Jayhawk"', '"MH-65"', '"helicopter takeoff"', '"aerial firefighting"', '"rappel"',
                   '"CASEVAC"', '"MEDEVAC"', '"aviation brigade"', '"helicopter training"', '"rotary wing"',
                   '"air ambulance"', '"search and rescue"',
                   '"MH-47"', '"Little Bird"', '"OH-58"', '"UH-1N"', '"Grey Wolf"', '"HH-60W"', '"MH-60R"',
                   '"MH-60S"', '"CH-53K"', '"VERTREP"', '"vertical replenishment"', '"pinnacle landing"', '"brownout"',
                   '"helicopter insertion"', '"helicopter extraction"', '"Coast Guard helicopter"',
                   '"National Guard helicopter"', '"helicopter support team"'],
    "bird": ['"bird strike"', '"BASH"', '"bird aircraft strike hazard"', "falconry", "falconer", '"bald eagle"',
             "seagulls", "pelicans", '"migratory birds"', "geese", '"red-tailed hawk"', "vultures", '"wildlife biologist"',
             '"bird watching"', "birds", '"bird survey"', "albatross", "egrets", "cranes migration"],
    "airplane": ['"flyover"', '"air show"', '"airdrop"', '"touch and go"', '"F-16"', '"F-35"', '"A-10"', '"C-17"',
                 '"KC-135"', '"B-52"', "Thunderbirds", '"Blue Angels"', '"formation flight"', '"flypast"',
                 '"C-130"', '"F-22"', '"F-15"', '"T-38"',
                 # round 11 (19/10/2026): no airshow (smoke = trail)
                 '"takeoff"', '"takes off"', '"landing"', '"final approach"', '"departs"', '"arrives"', '"flyby"',
                 '"low pass"', '"elephant walk"', '"Red Flag"', '"bomber task force"', '"short field landing"',
                 '"assault landing"', '"C-5"', '"KC-46"', '"KC-10"', '"B-1"', '"B-2"', '"F-15E"', '"E-3"',
                 '"P-8"', '"F/A-18"', '"EA-18G"', '"F-35C"', '"T-6"', '"refueling exercise"', '"fighter jets"',
                 '"aircraft arrival"', '"aircraft departure"'],
}
NO = re.compile(r"interview|a-roll|cockpit|aerial|air-to-air|ride along|news|update|podcast|ceremony|speech|"
                r"maintenance|briefing|explained|portrait|air ?show|thunderbirds|blue angels|demonstration team|demo team|"
                r"open house", re.I)  # airshow: smoke = trail (user, 19/10/2026)


def search_videos(src: Path, pages: int = 4, cls: str = "") -> None:
    """Candidates from the DVIDS API (key in `~/.config/dvids.env`): for each
    query of `QUERY`, `pages` pages of 50 by relevance, only HD videos from 15 s
    to 15 min with no interview or cockpit words in the title -> new rows in
    `0_raw/candidati.csv` (url, classe, titolo). The license is checked later,
    on the page (`info`)."""
    import time

    import requests

    key = dict(ln.strip().split("=", 1) for ln in open(Path.home() / ".config" / "dvids.env") if "=" in ln)["DVIDS_API_KEY"]
    f = src / "0_raw" / "candidati.csv"
    have = {re.search(r"/video/(\d+)", r["url"]).group(1) for r in csv.DictReader(open(f))}
    new = []
    for cl, qs in QUERY.items():
        if cls and cl != cls:
            continue
        for q in qs:
            n = 0
            for page in range(1, pages + 1):
                d = None
                for k in range(6):
                    try:
                        d = requests.get("https://api.dvidshub.net/search", timeout=60, params={
                            "api_key": key, "q": q, "type": "video", "sort": "score", "max_results": 50,
                            "page": page}).json()
                        break
                    except (requests.RequestException, ValueError):
                        time.sleep(5 * (k + 1))
                res = (d or {}).get("results", [])
                for r in res:
                    vid = r["id"].split(":")[1]
                    if (vid not in have and r.get("hd") and 15 < (r.get("duration") or 0) < 900
                            and not NO.search(r["title"])):
                        have.add(vid)
                        new.append({"url": r["url"], "classe": cl, "titolo": r["title"].replace(",", " ")})
                        n += 1
                if len(res) < 50:
                    break
            print(f"dvids: cerca {cl} {q}: {n} nuovi", flush=True)
    with open(f, "a", newline="") as fh:
        csv.DictWriter(fh, ["url", "classe", "titolo"]).writerows(new)
    print(f"dvids: {len(new)} candidati nuovi -> {f}")


def video_info(src: Path, cls: str = "", workers: int = 12) -> None:
    """`candidati.csv` (only `cls`, if given) -> `0_raw/video.csv`; only the pages
    with "PUBLIC DOMAIN" are kept (the others stay written with license NO)."""
    raw = src / "0_raw"
    f = raw / "video.csv"
    done = {r["url"] for r in csv.DictReader(open(f))} if f.exists() else set()
    todo = [r for r in csv.DictReader(open(raw / "candidati.csv"))
            if r["url"] not in done and (not cls or r["classe"] == cls)]
    new = not f.exists()
    with open(f, "a", newline="") as fh, ThreadPoolExecutor(workers) as ex:
        w = csv.writer(fh)
        if new:
            w.writerow(["id", "classe", "titolo", "licenza", "autore", "url", "mp4", "durata", "w", "h", "fps"])
        for r in ex.map(page_info, todo):
            w.writerow(r)
            fh.flush()
            print(f"dvids: {r[0]} {r[1]} {r[3]} {r[7]} s {r[8]}x{r[9]} {r[4]}", flush=True)


_DUP: set | None = None


def _ok(r: dict) -> bool:
    global _DUP
    if _DUP is None:
        f = SOURCES["dvids"][1] / "0_raw" / "doppioni.txt"
        _DUP = {ln.split()[0] for ln in f.read_text().splitlines() if ln.strip()} if f.exists() else set()
    return (r["licenza"] == "pubblico dominio" and float(r["durata"] or 0) > 15 and int(r["h"] or 0) >= 720
            and r["id"] not in _DUP)


def video_duplicates(src: Path, bits: int = 6) -> None:
    """Videos republished under another id (same b-roll uploaded by different
    units): same duration (within 0.5 s) and equal previews at the same instant
    (64-bit dHash within `bits`) in at least half of the common instants. Free
    similarity matching does not work: two sky videos with a small aircraft
    look alike even if different. The already downloaded video is kept, else
    the lowest id -> `0_raw/doppioni.txt` (duplicate, kept, equal instants,
    common instants); `_ok` excludes them from all steps."""
    import cv2

    from tools.common import dhash

    dur = {r["id"]: float(r["durata"] or 0) for r in csv.DictReader(open(src / "0_raw" / "video.csv"))}
    hs = {}  # id -> {istante: hash}
    for d in sorted((src / "0_raw" / "anteprima").iterdir()):
        for p in d.glob("*.jpg"):
            g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if g is not None:
                hs.setdefault(d.name, {})[int(p.stem)] = int.from_bytes(np.packbits(dhash(g)).tobytes(), "big")
    vids = sorted((v for v in hs if dur.get(v)), key=lambda v: dur[v])
    got = {d.name.split("_")[1] for d in (src / "1_frame").iterdir()}
    out = []
    for i, a in enumerate(vids):
        for b in vids[i + 1:]:
            if dur[b] - dur[a] > 0.5:
                break
            common = set(hs[a]) & set(hs[b])
            n = sum(bin(hs[a][t] ^ hs[b][t]).count("1") <= bits for t in common)
            if common and n >= max(2, len(common) / 2):
                keep = a if (a in got) or (b not in got and int(a) < int(b)) else b
                out.append((b if keep == a else a, keep, n, len(common)))
    (src / "0_raw" / "doppioni.txt").write_text("".join(f"{x} {y} {n} {c}\n" for x, y, n, c in out))
    print(f"dvids: {len(hs)} video con anteprime, {len(out)} doppioni -> 0_raw/doppioni.txt")


def _thumb(args) -> None:
    """The keyframe near `t` (`-noaccurate_seek`: no decoding up to `t`, fewer
    bytes); the proxy is intermittent: retries, and a failed one is left
    without a preview."""
    mp4, t, out = args
    if not out.exists():
        for _ in range(3):
            try:
                if subprocess.run(["ffmpeg", "-v", "error", "-y", "-rw_timeout", "30000000", "-noaccurate_seek",
                                   "-ss", str(t), "-i", mp4, "-frames:v", "1", "-q:v", "2", str(out)],
                                  timeout=120).returncode == 0:
                    break
            except subprocess.TimeoutExpired:
                pass
            out.unlink(missing_ok=True)


def video_previews(src: Path, cls: str = "", step: int = 10, skip: int = 10, per_video: int = 12,
                   workers: int = 24) -> None:
    """One frame every `step` s (10: with 5 s and 8 connections ~30 per minute, too slow), from `skip` s to the end,
    read from the remote file (only the nearby bytes) -> `0_raw/anteprima/<id>/<t>.jpg` at full
    resolution. At most `per_video` previews per video, spread over the duration (from round 4, 4,616
    candidates: with all of them it would be ~12 hours)."""
    jobs = []
    for r in csv.DictReader(open(src / "0_raw" / "video.csv")):
        if _ok(r) and (not cls or r["classe"] == cls):
            d = src / "0_raw" / "anteprima" / r["id"]
            d.mkdir(parents=True, exist_ok=True)
            ts = list(range(skip, int(float(r["durata"])) - 2, step))
            if len(ts) > per_video:  # subset of the 10 s grid: videos already done download nothing more
                ts = [ts[round(k * (len(ts) - 1) / (per_video - 1))] for k in range(per_video)]
            jobs += [(r["mp4"], t, d / f"{t:05d}.jpg") for t in ts]
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(_thumb, jobs))
    print(f"dvids: {len(jobs)} anteprime")


def _sky(gray: np.ndarray, x1, y1, x2, y2) -> bool:
    """Object in flight: the band above the box (as tall as the box) is smooth.
    ponytail: crude heuristic (sky = little texture); smooth asphalt passes,
    the sky above trees does not. Enough to give priority to the sky."""
    h = max(int(y2 - y1), 8)
    band = gray[max(0, int(y1) - h):int(y1), max(0, int(x1)):int(x2)]
    return band.size > 0 and float(np.abs(np.diff(band.astype(np.int16), axis=1)).mean()) < 4


def choose_segments(src: Path, conf: float = 0.3, half: float = 5.0) -> None:
    """YOLO on the previews: a preview counts if it has an airplane or helicopter (conf
    >= `conf`); segment = +-`half` s around it, nearby segments merged -> `0_raw/tratti.csv`
    (id, inizio, fine, cielo = previews with the object on the sky, valid previews, grandezza = the smallest
    sqrt(w*h) of the aircraft in the segment previews, in frame pixels: it is the size in the crop)."""
    import cv2
    model = detector()
    rows = []
    for r in csv.DictReader(open(src / "0_raw" / "video.csv")):
        d = src / "0_raw" / "anteprima" / r["id"]
        if not _ok(r) or not d.exists():
            continue
        hit = []
        for p in sorted(d.glob("*.jpg")):
            im = cv2.imread(str(p))
            if im is None:
                continue
            res = model(im, conf=conf, imgsz=min(max(im.shape[:2]), 1920) // 32 * 32, verbose=False)[0]
            want = {"drone": (2,), "bird": (1,)}.get(r["classe"], (0, 3))  # the video target
            boxes = [b.xyxy[0].tolist() for b in res.boxes if _cls(b.cls, r["classe"]) in want]
            if boxes:
                g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                side = min(((b[2] - b[0]) * (b[3] - b[1])) ** 0.5 for b in boxes)
                hit.append((int(p.stem), any(_sky(g, *b) for b in boxes), side))
        segs = []
        for t, sky, side in hit:
            if segs and t - half <= segs[-1][1]:
                s = segs[-1]
                s[1], s[2], s[3], s[4] = t + half, s[2] + sky, s[3] + 1, min(s[4], round(side))
            else:
                segs.append([max(t - half, 0), t + half, int(sky), 1, round(side)])
        rows += [(r["id"], *s) for s in segs]
        print(f"dvids: {r['id']} {r['classe']} {len(hit)} anteprime con velivolo, {sum(s for _, s, _ in hit)} sul cielo, "
              f"{len(segs)} tratti", flush=True)
    with open(src / "0_raw" / "tratti.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "inizio", "fine", "cielo", "anteprime", "grandezza"])
        w.writerows(rows)


def _done(src: Path, r: dict) -> bool:
    v = f"DVIDS_{r['id']}"
    return (src / "1_frame" / v / "labels" / f"{v}_f{round(float(r['inizio']) * FPS):05d}.txt").exists()


def _cls(c, cls: str) -> int:
    """Our class of a detector detection on DVIDS: in airplane and helicopter videos
    the UAVs are almost always mistaken airplanes or helicopters, so they
    take the video class (also in bird ones); in drone videos they stay drones."""
    k = DETECTOR_CLASSES[int(c)]
    return {"airplane": 0, "helicopter": 3, "bird": 1}.get(cls, k) if k == 2 else k


def _segments(src: Path) -> list[dict]:
    """Segments with at least one preview on the sky, minus those in `0_raw/esclusi.txt`
    (rows `id inizio`: cockpit, shot from another aircraft, no aircraft)."""
    f = src / "0_raw" / "esclusi.txt"
    out = {tuple(ln.split()[:2]) for ln in f.read_text().splitlines() if ln.strip()} if f.exists() else set()
    return [r for r in csv.DictReader(open(src / "0_raw" / "tratti.csv"))
            if int(r["cielo"]) > 0 and (r["id"], r["inizio"]) not in out]


def segment_sheets(src: Path, per: int = 20, cls: str = "", size: float = 0, min_size: float = 0) -> None:
    """One preview per segment (the closest to the center), numbered, in sheets of
    `per` in `0_raw/fogli_tratti/NN.jpg`, with `indice.csv` (n, id, inizio): to
    pick by eye the segments to put in `esclusi.txt`."""
    import cv2

    seen = {(r["id"], r["inizio"]) for f in (src / "0_raw").glob("fogli_tratti*/indice.csv") for r in csv.DictReader(open(f))}
    out = src / "0_raw" / f"fogli_tratti{len(list((src / '0_raw').glob('fogli_tratti*'))) + 1 if seen else ''}"
    out.mkdir(exist_ok=True)
    info = {r["id"]: r for r in csv.DictReader(open(src / "0_raw" / "video.csv"))}
    segs = [r for r in _segments(src) if (r["id"], r["inizio"]) not in seen  # only the segments not yet seen
            and (not cls or info[r["id"]]["classe"] == cls) and (not size or float(r["grandezza"]) < size)
            and float(r["grandezza"]) >= min_size and not _done(src, r)]
    tiles = []
    for n, r in enumerate(segs, 1):
        d = src / "0_raw" / "anteprima" / r["id"]
        mid = (float(r["inizio"]) + float(r["fine"])) / 2
        ps = [p for p in d.glob("*.jpg") if cv2.imread(str(p)) is not None] if d.exists() else []
        im = cv2.imread(str(min(ps, key=lambda p: abs(int(p.stem) - mid)))) if ps else np.zeros((360, 640, 3), np.uint8)
        t = cv2.resize(im, (384, 216))
        cv2.rectangle(t, (0, 0), (60, 26), (0, 0, 0), -1)
        cv2.putText(t, str(n), (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        tiles.append(cv2.copyMakeBorder(t, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(255, 255, 255)))
    for k in range(0, len(tiles), per):
        g = tiles[k:k + per] + [np.zeros_like(tiles[0])] * (-len(tiles[k:k + per]) % 4)
        cv2.imwrite(str(out / f"{k // per + 1:02d}.jpg"), np.vstack([np.hstack(g[j:j + 4]) for j in range(0, len(g), 4)]))
    with open(out / "indice.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["n", "id", "inizio"])
        w.writerows((n, r["id"], r["inizio"]) for n, r in enumerate(segs, 1))
    print(f"dvids: {len(segs)} tratti in {-(-len(segs) // per)} fogli -> {out}")


def exclude_segments(src: Path, sheets: str, rejects: Path) -> None:
    """Rows `n code` of a review of `0_raw/<sheets>/` (C cockpit, A air-to-air
    or from above, P people, I interiors, G on the ground, N nothing in flight, F smoke
    or condensation trail) -> rows `id inizio code` in `0_raw/esclusi.txt`."""
    d = src / "0_raw" / sheets
    idx = {r["n"]: r for r in csv.DictReader(open(d / "indice.csv"))}
    rej = [ln.split()[:2] for ln in open(rejects) if ln.strip()]
    with open(src / "0_raw" / "esclusi.txt", "a") as f:
        f.write(f"# {sheets}\n")
        f.writelines(f"{idx[n]['id']} {idx[n]['inizio']} {c}\n" for n, c in rej)
    print(f"dvids: {len(rej)} tratti esclusi su {len(idx)} di {sheets}")


def _gray(p: Path) -> float:
    """Crop color: 99.5th percentile of the maximum difference between channels
    (0-3 in black and white, due to JPEG noise; a color overcast sky stays above)."""
    import cv2

    im = cv2.imread(str(p)).astype(np.int16)
    b, g, r = im[..., 0], im[..., 1], im[..., 2]
    return float(np.percentile(np.maximum(np.maximum(abs(r - g), abs(g - b)), abs(r - b)), 99.5))


def _green(p: Path) -> float:
    """Night-vision green: 20th percentile of G - max(R, B), high only if almost the whole
    image is green (daytime grass is not, the sky stays blue or gray)."""
    import cv2

    im = cv2.imread(str(p))[::2, ::2].astype(np.int16)
    return float(np.percentile(im[..., 1] - np.maximum(im[..., 0], im[..., 2]), 20))


def smoke_sheets(src: Path, per: int = 16, name: str = "fumo", gray_thr: float = 20, green_thr: float = 5) -> None:
    """Already downloaded airplane segments, never seen in a `0_raw/fogli_fumo*/`: two
    crops per segment (at 1/4 and 3/4), numbered sheets in `0_raw/fogli_fumoN/` with
    `indice.csv` (n, id, inizio_k = segment start in frames, ritagli). The operator
    writes the rows `n F` of the segments with a smoke or condensation trail (README, Rules) and
    passes them to `--step fumo`. With `name="grigi"` (user, 20/10/2026: grayscale videos
    and out-of-distribution archive footage): segments of all classes with
    at least one crop kept by the preselection with color (`_gray`) up to `gray_thr`,
    in `0_raw/fogli_grigiN/`; the operator marks the black-and-white or archive segments
    and passes them to `--step grigi`. With `name="verdi"` (user, 20/10/2026: green night
    vision to remove) the same with `_green` at least `green_thr`, the greenest on the left."""
    import cv2

    raw = src / "0_raw"
    seen = {(r["id"], float(r["inizio_k"])) for f in raw.glob(f"fogli_{name}*/indice.csv") for r in csv.DictReader(open(f))}
    kept = None
    if name in ("grigi", "verdi"):
        ex = pipeline.excluded(src)
        kept = {p.stem for p in (src / "5_box" / "preselezione").glob("DVIDS_*/*.txt") if p.stem not in ex}
    info = {r["id"]: r["classe"] for r in csv.DictReader(open(raw / "video.csv"))}
    T = {}
    for r in csv.DictReader(open(raw / "tratti.csv")):
        T.setdefault(r["id"], []).append((float(r["inizio"]) * FPS, float(r["fine"]) * FPS))
    seg = {}
    for d in sorted((src / "4_ritagli").glob("DVIDS_*")):
        i = d.name.split("_")[1]
        if kept is None and info.get(i) != "airplane":
            continue
        for p in sorted((d / "images").glob("*.jpg")):
            if kept is not None and p.stem not in kept:
                continue
            k = int(p.stem.rsplit("_f", 1)[1])
            a = next((a for a, b in T.get(i, []) if a - 1 <= k <= b + 1), -1)
            if (i, a) not in seen:
                seg.setdefault((i, a), []).append(p)
    if kept is not None:
        with ThreadPoolExecutor(16) as ex_:
            # grigi: least colored first; verdi: greenest first (minus sign). On the sheet the first and the median
            f, th = (_gray, gray_thr) if name == "grigi" else (lambda p: -_green(p), -green_thr)
            col = {k: list(ex_.map(f, ps)) for k, ps in seg.items()}
        seg = {k: [p for _, p in sorted(zip(col[k], ps))] for k, ps in seg.items() if min(col[k]) <= th}
    out = raw / f"fogli_{name}{len(list(raw.glob(f'fogli_{name}*'))) + 1 if seen else ''}"
    out.mkdir(exist_ok=True)
    rows, tiles = [], []
    for n, ((i, a), ps) in enumerate(sorted(seg.items()), 1):
        t = np.hstack([cv2.resize(cv2.imread(str(p)), (240, 192)) for p in ((ps[0], ps[len(ps) // 2]) if kept is not None else (ps[len(ps) // 4], ps[3 * len(ps) // 4]))])
        cv2.rectangle(t, (0, 0), (52, 24), (0, 0, 0), -1)
        cv2.putText(t, str(n), (3, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        tiles.append(cv2.copyMakeBorder(t, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=(255, 255, 255)))
        rows.append((n, i, a, len(ps)))
    for k in range(0, len(tiles), per):
        g = tiles[k:k + per] + [np.zeros_like(tiles[0])] * (-len(tiles[k:k + per]) % 4)
        cv2.imwrite(str(out / f"{k // per + 1:02d}.jpg"), np.vstack([np.hstack(g[j:j + 4]) for j in range(0, len(g), 4)]))
    with open(out / "indice.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["n", "id", "inizio_k", "ritagli"])
        w.writerows(rows)
    print(f"dvids: {len(rows)} tratti in {-(-len(rows) // per)} fogli -> {out}")


def exclude_smoke(src: Path, sheets: str, rejects: Path, name: str = "fumo") -> None:
    """Operator rows `n F` on `0_raw/<sheets>/` -> rows `id inizio_k n` in
    `0_raw/<name>.txt` (fumo or grigi), read by tools/preselection.py."""
    d = src / "0_raw" / sheets
    idx = {r["n"]: r for r in csv.DictReader(open(d / "indice.csv"))}
    rej = [ln.split()[:2] for ln in open(rejects) if ln.strip()]
    with open(src / "0_raw" / f"{name}.txt", "a") as f:
        f.write(f"# {sheets}\n")
        f.writelines(f"{idx[n]['id']} {idx[n]['inizio_k']} {n} {c}\n" for n, c in rej)
    print(f"dvids: {len(rej)} tratti {name} su {len(idx)} di {sheets}")


def video_frames(src: Path, ids: list[str], seconds: float = 0, size: float = 0, cls: str = "",
                 conf: float = 0.25, workers: int = 8, min_size: float = 0) -> None:
    """The segments of `tratti.csv` (with at least one preview on the sky) at `FPS` frames
    per second -> `1_frame/DVIDS_<id>/images/DVIDS_<id>_f<k>.jpg`, k = round(t*FPS)
    (stable: no re-downloads), and YOLO labels from the proposals detector on the whole
    frame (proposals, to be reviewed in CVAT)."""
    import shutil
    import tempfile

    import cv2
    model = detector()
    info = {r["id"]: r for r in csv.DictReader(open(src / "0_raw" / "video.csv"))}
    by = {}
    for r in _segments(src):
        if ((not ids or r["id"] in ids) and (not size or float(r["grandezza"]) < size) and float(r["grandezza"]) >= min_size
                and (not cls or info[r["id"]]["classe"] == cls)):
            by.setdefault(r["id"], []).append(r)
    # in rounds over the videos (1st segment of each video, then the 2nd...): stopping early gives variety
    segs = [s for k in range(max(map(len, by.values()), default=0)) for v in by.values() if k < len(v) for s in v[k:k + 1]]
    spent, todo = {}, []
    for r in segs:
        v = f"DVIDS_{r['id']}"
        k0 = round(float(r["inizio"]) * FPS)
        if (src / "1_frame" / v / "labels" / f"{v}_f{k0:05d}.txt").exists():
            continue
        c = info[r["id"]]["classe"]
        if seconds and spent.get(c, 0) >= seconds:
            continue
        spent[c] = spent.get(c, 0) + float(r["fine"]) - float(r["inizio"])
        todo.append(r)

    def fetch(r):
        a, b = float(r["inizio"]), float(r["fine"])
        tmp = tempfile.mkdtemp(dir=src)
        for _ in range(3):
            try:
                if subprocess.run(["ffmpeg", "-v", "error", "-y", "-rw_timeout", "30000000", "-ss", str(a), "-i",
                                   info[r["id"]]["mp4"], "-t", str(b - a), "-vf", f"fps={FPS}", "-q:v", "3",
                                   f"{tmp}/%05d.jpg"], timeout=900).returncode == 0:
                    break
            except subprocess.TimeoutExpired:
                pass
        return r, tmp

    with ThreadPoolExecutor(workers) as ex:  # parallel download, YOLO here
        for r, tmp in ex.map(fetch, todo):
            v = f"DVIDS_{r['id']}"
            dst = src / "1_frame" / v
            (dst / "images").mkdir(parents=True, exist_ok=True)
            (dst / "labels").mkdir(exist_ok=True)
            k0 = round(float(r["inizio"]) * FPS)
            n, frames = 0, sorted(Path(tmp).glob("*.jpg"))
            for j, p in enumerate(frames):
                stem = f"{v}_f{k0 + j:05d}"
                im = cv2.imread(str(p))
                if im is None:
                    continue
                H, W = im.shape[:2]
                res = model(im, conf=conf, imgsz=min(max(H, W), 1920) // 32 * 32, verbose=False)[0]
                lines = [f"{_cls(x.cls, info[r['id']]['classe'])} " + " ".join(f"{u:.6f}" for u in x.xywhn[0].tolist())
                         + "\n" for x in res.boxes]
                shutil.move(str(p), dst / "images" / f"{stem}.jpg")
                (dst / "labels" / f"{stem}.txt").write_text("".join(lines))
                n += len(lines)
            shutil.rmtree(tmp, ignore_errors=True)
            print(f"dvids: {v} {r['inizio']}-{r['fine']} s, {len(frames)} frame, {n} box", flush=True)
    print(f"dvids: secondi per classe in questo giro {spent}")


# --- comune ------------------------------------------------------------------------


def run_pipeline(src: Path, ss: list[str]) -> None:
    """annotati, dedup, crop of pipeline.py; the crop labels become the
    proposals (`5_box/proposte/`) for count and CVAT."""
    import shutil

    # segments with a failed download: folder without frames
    ss = [s for s in ss if any((src / "1_frame" / f"{pipeline.PREFIX}_{s}" / "images").glob("*.jpg"))]
    pipeline.annotated(src, ss)
    pipeline.dedup(src, ss)
    pipeline.crop(src, ss)
    for s in ss:
        v = f"{pipeline.PREFIX}_{s}"
        shutil.copytree(src / "4_ritagli" / v / "labels", src / "5_box" / "proposte" / v, dirs_exist_ok=True)


def sessions(src: Path, thr: float = 0.9) -> None:
    """Sessions -> `6_sessioni/sessioni.csv`. DVIDS: one per video (same shot).
    FBD: average-linkage groups (threshold `thr`) of the DINOv2 embedding of the mid-clip
    frame, like the automatic FRED sessions (fixed cameras: same place)."""
    ss = seqs(src)
    P = pipeline.PREFIX
    if P == "DVIDS":
        sess = {s: f"{P}_{s}" for s in ss}
    elif P == "AUAV":  # <date>_<hour>_<camera>_<piece>: the pieces of the same recording together
        sess = {s: f"{P}_{'_'.join(s.split('_')[:2])}" for s in ss}
    else:
        import cv2
        import torch

        from tools.common import propose_sessions

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        m = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", trust_repo=True).to(dev).eval()
        mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
        emb = {}
        with torch.no_grad():
            for s in ss:
                imgs = sorted((src / "1_frame" / f"{P}_{s}" / "images").glob("*.jpg"))
                im = cv2.imread(str(imgs[len(imgs) // 2]))
                x = (cv2.resize(im, (448, 252))[..., ::-1] / 255 - mean) / std
                x = torch.tensor(x.transpose(2, 0, 1)[None], dtype=torch.float32, device=dev)
                emb[s] = torch.nn.functional.normalize(m(x), dim=1).cpu().numpy()
        vids, _, groups = propose_sessions(emb, thr)
        sess = {s: f"{P}_S{g:03d}" for s, g in zip(vids, groups)}
    out = src / "6_sessioni"
    out.mkdir(exist_ok=True)
    with open(out / "sessioni.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["clip", "sessione"])
        w.writerows((f"{P}_{s}/s0", sess[s]) for s in ss)
    print(f"{P}: {len(ss)} clip in {len(set(sess.values()))} sessioni")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fonte", choices=list(SOURCES), required=True)
    p.add_argument("--step", required=True, choices=["frame", "frame_extra", "escludi", "fogli_fumo", "fumo", "fogli_grigi", "grigi", "fogli_verdi", "verdi", "cerca", "info", "anteprima", "doppioni", "scegli", "fogli", "catena", "sessioni",
                                                     "conta", "cvat", "cvat_import"])
    p.add_argument("--seq", nargs="+", default=[], help="sequenze (vuoto: tutte quelle in 1_frame)")
    p.add_argument("--secondi", type=float, default=0, help="DVIDS frame: secondi nuovi per classe in questo giro (0: tutti)")
    p.add_argument("--fogli", default="", help="DVIDS escludi: cartella dei fogli rivisti (fogli_tratti<N>)")
    p.add_argument("--scarti", type=Path, help="DVIDS escludi: file con le righe `n codice` della revisione")
    p.add_argument("--classe", default="", help="DVIDS fogli e frame: solo i video di questa classe")
    p.add_argument("--grandezza", type=float, default=0, help="DVIDS fogli e frame: solo i tratti con un velivolo di grandezza sqrt(w*h) sotto questa (px)")
    p.add_argument("--minimo", type=float, default=0, help="DVIDS fogli e frame: solo i tratti con il velivolo piu' piccolo almeno di questa grandezza (px)")
    a = p.parse_args(argv)
    pipeline.PREFIX, src = SOURCES[a.fonte]
    if a.step == "frame":
        if a.fonte == "fbd":
            xml_video_frames(src)
        elif a.fonte == "antiuav":
            download_image_frames(src)
        else:
            video_frames(src, a.seq, a.secondi, a.grandezza, a.classe, min_size=a.minimo)
    if a.step == "frame_extra":  # Anti-UAV: more drones in the sky, between the frames already used (26/10/2026)
        antiuav_extra()
    if a.step == "cerca":
        search_videos(src, cls=a.classe)
    if a.step == "escludi":
        exclude_segments(src, a.fogli, a.scarti)
    if a.step == "fogli_fumo":
        smoke_sheets(src)
    if a.step == "fumo":
        exclude_smoke(src, a.fogli, a.scarti)
    if a.step == "fogli_grigi":
        smoke_sheets(src, name="grigi")
    if a.step == "grigi":
        exclude_smoke(src, a.fogli, a.scarti, name="grigi")
    if a.step == "fogli_verdi":
        smoke_sheets(src, name="verdi")
    if a.step == "verdi":
        exclude_smoke(src, a.fogli, a.scarti, name="verdi")
    if a.step == "doppioni":
        video_duplicates(src)
    if a.step == "info":
        video_info(src, a.classe)
    if a.step == "anteprima":
        video_previews(src, a.classe)
    if a.step == "scegli":
        choose_segments(src)
    if a.step == "fogli":
        segment_sheets(src, cls=a.classe, size=a.grandezza, min_size=a.minimo)
    if a.step == "catena":
        run_pipeline(src, a.seq or seqs(src))
    if a.step == "sessioni":
        sessions(src)
    if a.step == "conta":
        pipeline.count(src)
    if a.step == "cvat":
        pipeline.cvat_export(src, per_task=1000)
    if a.step == "cvat_import":
        pipeline.cvat_import(src)
    return 0


if __name__ == "__main__":
    sys.exit(main())
