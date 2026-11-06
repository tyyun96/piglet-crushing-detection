"""
Preprocessing step 12: the detector training recipe of the previous study [9], with and without leakage
(sensitivity analysis for reviewer comment R1-3; the main detector chosen in §3.2 is NOT changed).

[9] recipe (archives audited by prep_11): every labelled original + 7 offline copies (torchvision ColorJitter
brightness=3 / contrast=2 / saturation=2 / hue=0.3; three random-noise patches inside the top-left 400x400 px), each
of the 8 rotated by 90/180/270 degrees and flipped left-right / top-bottom (x48 per original); YOLOv5s
(models/yolov5s.yaml + COCO yolov5s.pt), --img 416 --batch 16 --epochs 50, default hyperparameters (hyp.scratch-low),
seed 0. The copies are regenerated here with the archive's operations (seeded -> reproducible); the random pixel
values therefore differ from [9], the composition of the dataset does not.

Runs (all with the [9] recipe)
  r9_split : all 204 originals, [9]'s own YOLOv5 split lists (random over the 9792 augmented images, so copies of the
             same frame are on both sides)                    -> reproduces [9]'s validation numbers
  r9_ctrl  : all 204 originals, split by ORIGINAL (prep_05 'ctrl' originals: 173 train / 31 val, x48 each)
  r9_holdA : camera-A originals removed (prep_05 'holdA': 102 / 18) -> leakage-free for camera A (litter A-R)
  r9_holdB : camera-B originals removed (prep_05 'holdB': 159 / 28) -> leakage-free for camera B (litters B-L, B-R)
The train/val originals of r9_ctrl / r9_holdA / r9_holdB are those of the existing fold detectors (prep_05/06), so the
comparison with them changes only the training recipe.
Detections are cached with tag _r9lolo (camera-A videos <- r9_holdA, camera-B videos <- r9_holdB) and _r9split
(r9_split on the 11 manuscript clips: closest reconstruction of the lost manuscript YOLOv5s, reproduction in §6).
"""
import json
import multiprocessing as mp
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from video_registry import VIDEOS  # noqa: E402

PY = str(ROOT / ".venv/bin/python")
V5 = ROOT / "third_party/yolov5"
SRC = ROOT / "data/new data origin"
FOLD_DS = ROOT / "cache/detector_training"         # prep_05 datasets (which originals are train / val)
WORK = ROOT / "cache/detector_training_r9"
AUG = WORK / "aug"                                  # images/ and labels/ side by side (YOLOv5 label lookup)
TRANSFORMS = ("", "_90", "_180", "_270", "_LR", "_TB")
RUNS = ["r9_holdA", "r9_holdB", "r9_split", "r9_ctrl"]
SEED = 1000                                         # augmentation seed (+ original index)


def aug_name(i, copy, t=""):
    return f"o{i:03d}_c{copy}{t}"


def transformed(lb, t):
    """boxes [cls, x, y, w, h] after the PIL transpose (same as label_rotation.py of [9])"""
    c, x, y, w, h = lb.T
    if t == "_90":
        x, y, w, h = y, 1 - x, h, w
    elif t == "_180":
        x, y = 1 - x, 1 - y
    elif t == "_270":
        x, y, w, h = 1 - y, x, h, w
    elif t == "_LR":
        x = 1 - x
    elif t == "_TB":
        y = 1 - y
    return np.stack([c, x, y, w, h], 1)


def augment_original(i):
    """the 48 images of original i, with the operations of image_augmentation.py / image_rotation.py"""
    import torch
    from PIL import Image
    from torchvision import transforms
    torch.set_num_threads(1)
    if (AUG / "labels" / f"{aug_name(i, 7, '_TB')}.txt").exists():
        return i
    src = SRC / f"train ({i}).jpg"
    lab = SRC / f"train ({i}).txt"
    lb = np.loadtxt(lab, ndmin=2).reshape(-1, 5) if lab.stat().st_size else np.zeros((0, 5))
    torch.manual_seed(SEED + i)
    rnd = random.Random(SEED + i)
    nrs = np.random.RandomState(SEED + i)
    img = Image.open(src).convert("RGB")
    copies = {0: None}                                       # copy 0 = the original file itself
    for k, jitter in enumerate([dict(brightness=3), dict(contrast=2), dict(saturation=2), dict(hue=0.3)], start=1):
        copies[k] = transforms.ColorJitter(**jitter)(img)
    for k in range(5, 8):                                    # random-noise patch, rows/cols inside 0..399
        a = np.array(img)
        r = rnd.sample(range(400), 2)
        c = rnd.sample(range(400), 2)
        n = (max(r) - min(r)) * (max(c) - min(c)) * 3
        a[min(r):max(r), min(c):max(c), :] = nrs.choice(256, n, replace=True).reshape(max(r) - min(r), max(c) - min(c), 3)
        copies[k] = Image.fromarray(a)
    ops = {"_90": Image.ROTATE_90, "_180": Image.ROTATE_180, "_270": Image.ROTATE_270,
           "_LR": Image.FLIP_LEFT_RIGHT, "_TB": Image.FLIP_TOP_BOTTOM}
    for k, im in copies.items():
        dst = AUG / "images" / f"{aug_name(i, k)}.jpg"
        if im is None:
            if not dst.exists():
                os.symlink(src, dst)
            im = img
        else:
            im.save(dst, "JPEG")                             # PIL default quality, as in [9]
            im = Image.open(dst).convert("RGB")              # [9] rotated the saved JPEG files
        for t, op in ops.items():
            im.transpose(op).save(AUG / "images" / f"{aug_name(i, k, t)}.jpg", "JPEG")
        for t in TRANSFORMS:
            np.savetxt(AUG / "labels" / f"{aug_name(i, k, t)}.txt", transformed(lb, t), fmt=["%d"] + ["%.6f"] * 4)
    return i


def fold_originals(fold):
    """train / val originals of a prep_05 fold (file names there: train_<i>.jpg)"""
    out = {}
    for s in ("train", "val"):
        out[s] = sorted(int(re.fullmatch(r"train_(\d+)", p.stem).group(1)) for p in (FOLD_DS / fold / "images" / s).iterdir())
    return out


def build_lists():
    lists = {}
    for run, fold in (("r9_ctrl", "ctrl"), ("r9_holdA", "holdA"), ("r9_holdB", "holdB")):
        o = fold_originals(fold)
        lists[run] = {s: [aug_name(i, k, t) for i in o[s] for k in range(8) for t in TRANSFORMS] for s in ("train", "val")}
    im = pd.read_csv(ROOT / "meta/previous_study_images.csv")                      # prep_11
    im = im[im.split_list == "yolov5"]
    tr = {"none": ""} | {t.strip("_"): t for t in TRANSFORMS[1:]}
    lists["r9_split"] = {s: [aug_name(int(re.search(r"\d+", r.original).group()), r.copy, tr[r.transform])
                             for r in im[im.split == s].itertuples()] for s in ("train", "val")}
    for run, d in lists.items():
        root = WORK / "data" / run
        root.mkdir(parents=True, exist_ok=True)
        for s, names in d.items():
            (root / f"{s}.txt").write_text("".join(f"{AUG / 'images' / n}.jpg\n" for n in names))
        (root / "data.yaml").write_text(f"train: {root / 'train.txt'}\nval: {root / 'val.txt'}\n"
                                        "nc: 3\nnames:\n  0: piglet\n  1: crushing\n  2: crushed\n")
    return {run: {s: len(v) for s, v in d.items()} for run, d in lists.items()}


def cmd(run):
    return [PY, "train.py", "--img", "416", "--batch-size", "16", "--epochs", "50", "--data", str(WORK / "data" / run / "data.yaml"),
            "--cfg", "models/yolov5s.yaml", "--weights", "yolov5s.pt", "--seed", "0", "--workers", "4", "--cache", "ram",
            "--project", str(WORK / "runs"), "--name", run, "--exist-ok"]


def done(run):
    return (WORK / "runs" / run / "results.png").exists()          # written by YOLOv5 when training has finished


def train_all(max_parallel=2):
    pending = [r for r in RUNS if not done(r)]
    running = {}
    while pending or running:
        for r, p in list(running.items()):
            if p.poll() is not None:
                running.pop(r)
                print(time.strftime("%H:%M:%S"), "finished", r, "exit", p.returncode, flush=True)
        while pending and len(running) < max_parallel:
            r = pending.pop(0)
            log = open(WORK / f"train_{r}.log", "w")
            running[r] = subprocess.Popen(cmd(r), cwd=V5, stdout=log, stderr=subprocess.STDOUT,
                                          env=dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2"))
            print(time.strftime("%H:%M:%S"), "started", r, flush=True)
        time.sleep(30)
    for r in RUNS:
        assert done(r), f"run {r} did not finish (see {WORK / f'train_{r}.log'})"


def best_val(run):
    r = pd.read_csv(WORK / "runs" / run / "results.csv")
    r.columns = [c.strip() for c in r.columns]
    i = r["metrics/mAP_0.5:0.95"].idxmax()
    last = r.iloc[-1]
    return dict(val_mAP50=float(r.loc[i, "metrics/mAP_0.5"]), val_mAP50_95=float(r.loc[i, "metrics/mAP_0.5:0.95"]),
                best_epoch=int(r.index[i]), epochs_run=len(r), last_val_mAP50=float(last["metrics/mAP_0.5"]),
                last_val_mAP50_95=float(last["metrics/mAP_0.5:0.95"]))


def main():
    (AUG / "images").mkdir(parents=True, exist_ok=True)
    (AUG / "labels").mkdir(parents=True, exist_ok=True)
    with mp.get_context("spawn").Pool(4) as pool:
        for k, _ in enumerate(pool.imap_unordered(augment_original, range(1, 205)), start=1):
            if k % 50 == 0:
                print(time.strftime("%H:%M:%S"), "augmented originals:", k, flush=True)
    sizes = build_lists()
    print("datasets:", sizes, flush=True)
    train_all()
    rows = [dict(run=r, n_train=sizes[r]["train"], n_val=sizes[r]["val"], weights=str(WORK / "runs" / r / "weights/best.pt"),
                 **best_val(r)) for r in RUNS]
    pd.DataFrame(rows).to_csv(WORK / "val_summary.csv", index=False)
    print(pd.DataFrame(rows).drop(columns="weights").round(3).to_string(), flush=True)

    det = str(Path(__file__).resolve().parent / "prep_02_detect.py")
    for run, cam in (("r9_holdA", "A"), ("r9_holdB", "B")):
        vids = [v for v in VIDEOS if VIDEOS[v]["camera"] == cam]
        subprocess.run([PY, det, "--workers", "3", "--tag", "_r9lolo", "--weights", str(WORK / "runs" / run / "weights/best.pt"),
                        "--backend", "yolov5", "--imgsz", "416", "--only", *vids], check=True)
    clips = [v for v in VIDEOS if VIDEOS[v]["kind"] == "clip"]
    subprocess.run([PY, det, "--workers", "3", "--tag", "_r9split", "--weights", str(WORK / "runs/r9_split/weights/best.pt"),
                    "--backend", "yolov5", "--imgsz", "416", "--only", *clips], check=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
