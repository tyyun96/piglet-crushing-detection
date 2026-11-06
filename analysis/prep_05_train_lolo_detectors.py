"""
Preprocessing step 5: leakage-free detectors (reviewer comment 3).

The detector shipped with the manuscript (train-20230728T031826Z-001.zip, YOLOv8s) was trained on
frames that lie inside the evaluation clips (see meta/train_image_matches.csv).  Its training originals
are the 204 labelled images in 'new data origin' (the complete set; [9] trained on 48 offline-augmented
copies of each, see prep_11 / prep_12).  We retrain YOLOv8s with the same base settings (args.yaml of the
original run: yolov8s.pt, imgsz 416, batch 16, seed 0; here up to 100 epochs, originals only) with a
camera-level hold-out:
  fold 'holdA' : all images except those from camera-A recordings  -> used to test litter A-R
  fold 'holdB' : all images except those from camera-B recordings  -> used to test litters B-L/B-R
  'ctrl'       : all 204 images (same recipe, includes the leaking frames) -> control
(camera B images show both B-L and B-R, so a litter-level split is impossible for camera B.)
Afterwards the detections of every relevant video are cached with tag _lolo (fold detector of the
camera) and _ctrl.
"""
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from video_registry import VIDEOS  # noqa: E402

SRC = ROOT / "data/new data origin"
WORK = ROOT / "cache/detector_training"
PY = str(ROOT / ".venv/bin/python")


def image_groups():
    m = pd.read_csv(ROOT / "meta/train_image_matches.csv")
    grp = {}
    for r in m.itertuples():
        if r.mad < 5 and r.best_video in VIDEOS:
            grp[r.image] = VIDEOS[r.best_video]["camera"]       # A, B or N
        else:
            grp[r.image] = "other_farm"
    return grp


def build_dataset(name, images, seed=0):
    root = WORK / name
    if root.exists():
        shutil.rmtree(root)
    rnd = random.Random(seed)
    imgs = sorted(images)
    rnd.shuffle(imgs)
    n_val = max(1, int(round(0.15 * len(imgs))))
    split = {"val": imgs[:n_val], "train": imgs[n_val:]}
    for s, lst in split.items():
        (root / "images" / s).mkdir(parents=True)
        (root / "labels" / s).mkdir(parents=True)
        for im in lst:
            stem = im[:-4]
            safe = stem.replace(" ", "_").replace("(", "").replace(")", "")
            os.symlink(SRC / im, root / "images" / s / f"{safe}.jpg")
            lab = SRC / f"{stem}.txt"
            if lab.exists():
                shutil.copy(lab, root / "labels" / s / f"{safe}.txt")
            else:
                (root / "labels" / s / f"{safe}.txt").write_text("")
    (root / "data.yaml").write_text(
        f"path: {root}\ntrain: images/train\nval: images/val\nnames:\n  0: piglet\n  1: crushing\n  2: crushed\n")
    return root / "data.yaml", {k: len(v) for k, v in split.items()}


def train(name, data_yaml):
    from ultralytics import YOLO
    out = WORK / "runs" / name
    if (out / "weights/best.pt").exists():
        return out / "weights/best.pt"
    model = YOLO("yolov8s.pt")
    model.train(data=str(data_yaml), epochs=100, patience=20, imgsz=416, batch=16, seed=0,
                deterministic=True, workers=4, project=str(WORK / "runs"), name=name, exist_ok=True,
                verbose=False, plots=False)
    return out / "weights/best.pt"


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    os.chdir(WORK)          # yolov8s.pt (COCO-pretrained) is downloaded here
    grp = image_groups()
    folds = {
        "holdA": [i for i, g in grp.items() if g != "A"],
        "holdB": [i for i, g in grp.items() if g != "B"],
        "ctrl": list(grp),
    }
    info = {}
    for name, imgs in folds.items():
        y, sizes = build_dataset(name, imgs)
        w = train(name, y)
        info[name] = dict(n_images=len(imgs), split=sizes, weights=str(w))
        print(name, info[name], flush=True)
    json.dump(dict(groups=grp, folds=info), open(WORK / "folds.json", "w"), indent=1)
    # cache detections: fold detector per camera (_lolo) and control (_ctrl)
    vidsA = [v for v in VIDEOS if VIDEOS[v]["camera"] == "A"]
    vidsB = [v for v in VIDEOS if VIDEOS[v]["camera"] == "B"]
    det = str(Path(__file__).resolve().parent / "prep_02_detect.py")
    for tag, wA, wB in (("_lolo", info["holdA"]["weights"], info["holdB"]["weights"]),
                        ("_ctrl", info["ctrl"]["weights"], info["ctrl"]["weights"])):
        for vids, w in ((vidsA, wA), (vidsB, wB)):
            subprocess.run([PY, det, "--workers", "3", "--tag", tag, "--weights", w, "--only", *vids], check=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
