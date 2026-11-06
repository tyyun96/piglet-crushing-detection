"""
Preprocessing step 6: re-train YOLOv5s (the detector named in the manuscript; its weights were lost)
with the original anchor-based YOLOv5 code base (third_party/yolov5; YOLOv5 default schedule: 300 epochs,
patience 50 -- with 100 epochs the YOLOv5s validation mAP was still rising), and compare it with YOLOv8s
under the same data splits as prep_05 (ctrl = all 204 labelled images; holdA / holdB = camera-level
hold-out, i.e. leakage-free for the held-out camera) and two input sizes (416, 640).

Pre-specified detector-selection rule (does NOT look at the evaluation events):
  for each hold-out fold, choose the (architecture, imgsz) with the highest mAP50-95 on that fold's
  OWN validation split (images from the training cameras only); ties -> smaller imgsz.
The selected fold detectors are cached with tag _best (camera A videos <- holdA model, camera B <- holdB).
Additionally cached: _v5ctrl (YOLOv5s trained on all images, clips only: manuscript reproduction)
and _v5lolo (YOLOv5s fold detectors, for the detector comparison).
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from video_registry import VIDEOS  # noqa: E402

PY = str(ROOT / ".venv/bin/python")
V5 = ROOT / "third_party/yolov5"
SRC_DS = ROOT / "cache/detector_training"          # datasets built by prep_05 (ultralytics caches inside)
WORK = ROOT / "cache/detector_training_v5"
FOLDS = ["ctrl", "holdA", "holdB"]


def v5_dataset(fold):
    """separate copy (symlinks) so that YOLOv5 and Ultralytics label caches do not collide"""
    dst = WORK / "data" / fold
    if (dst / "data.yaml").exists():
        return dst / "data.yaml"
    for split in ("train", "val"):
        (dst / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / split).mkdir(parents=True, exist_ok=True)
        for im in (SRC_DS / fold / "images" / split).iterdir():
            os.symlink(os.path.realpath(im), dst / "images" / split / im.name)
            shutil.copy(SRC_DS / fold / "labels" / split / (im.stem + ".txt"), dst / "labels" / split / (im.stem + ".txt"))
    (dst / "data.yaml").write_text(
        f"path: {dst}\ntrain: images/train\nval: images/val\nnames:\n  0: piglet\n  1: crushing\n  2: crushed\n")
    return dst / "data.yaml"


def v5_cmd(fold, imgsz):
    name = f"v5s_{fold}_{imgsz}"
    return [PY, "train.py", "--data", str(v5_dataset(fold)), "--weights", "yolov5s.pt", "--img", str(imgsz),
            "--batch-size", "16", "--epochs", "300", "--patience", "50", "--seed", "0", "--workers", "3",
            "--project", str(WORK / "runs"), "--name", name, "--exist-ok"]


def v5_done(fold, imgsz):
    # results.png is written by YOLOv5 only when training has finished
    return (WORK / "runs" / f"v5s_{fold}_{imgsz}" / "results.png").exists()


def train_v5_parallel(jobs, max_parallel=3):
    """train all missing YOLOv5s runs, at most max_parallel at a time (GPU is mostly idle with these
    tiny datasets; CPU dataloaders are the bottleneck).  Runs already in progress (best.pt but no
    results.png, e.g. started by an earlier invocation) are waited for, not restarted."""
    import time
    pending = [j for j in jobs if not v5_done(*j)]
    running = {}
    for j in list(pending):
        run = WORK / "runs" / f"v5s_{j[0]}_{j[1]}"
        if (run / "weights/last.pt").exists():
            running[j] = None                      # started elsewhere -> just wait
            pending.remove(j)
    while pending or running:
        for j, pr in list(running.items()):
            if v5_done(*j) or (pr is not None and pr.poll() is not None):
                running.pop(j)
        while pending and sum(1 for p in running.values() if p is not None) < max_parallel:
            j = pending.pop(0)
            log = open(WORK / f"train_v5s_{j[0]}_{j[1]}.log", "w")
            running[j] = subprocess.Popen(v5_cmd(*j), cwd=V5, stdout=log, stderr=subprocess.STDOUT)
            print("started", j, flush=True)
        time.sleep(20)
    for j in jobs:
        assert v5_done(*j), f"YOLOv5 run {j} did not finish"


def train_v8(fold, imgsz):
    from ultralytics import YOLO
    name = f"v8s_{fold}_{imgsz}"
    out = WORK / "runs" / name
    if imgsz == 416:                                   # trained by prep_05 with the same recipe
        return SRC_DS / "runs" / fold
    if (out / "weights/best.pt").exists():
        return out
    os.chdir(SRC_DS)
    YOLO("yolov8s.pt").train(data=str(SRC_DS / fold / "data.yaml"), epochs=100, patience=20, imgsz=imgsz, batch=16,
                             seed=0, deterministic=True, workers=4, project=str(WORK / "runs"), name=name,
                             exist_ok=True, verbose=False, plots=False)
    return out


def best_val(run, arch):
    r = pd.read_csv(run / "results.csv")
    r.columns = [c.strip() for c in r.columns]
    c50, c95 = (("metrics/mAP_0.5", "metrics/mAP_0.5:0.95") if arch == "v5s" else ("metrics/mAP50(B)", "metrics/mAP50-95(B)"))
    i = r[c95].idxmax()
    return dict(val_mAP50=float(r.loc[i, c50]), val_mAP50_95=float(r.loc[i, c95]), best_epoch=int(r.index[i]), epochs_run=len(r))


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    import threading
    v5_jobs = [(f, sz) for f in FOLDS for sz in (416, 640)]
    for f in FOLDS:
        v5_dataset(f)
    th = threading.Thread(target=train_v5_parallel, args=(v5_jobs, 2)); th.start()
    for fold in FOLDS:                       # YOLOv8s runs in this process meanwhile
        for imgsz in (416, 640):
            train_v8(fold, imgsz)
    th.join()
    rows = []
    for fold in FOLDS:
        for imgsz in (416, 640):
            for arch in ("v5s", "v8s"):
                run = (WORK / "runs" / f"v5s_{fold}_{imgsz}") if arch == "v5s" else train_v8(fold, imgsz)
                rows.append(dict(fold=fold, arch=arch, imgsz=imgsz, run=str(run), weights=str(run / "weights/best.pt"),
                                 **best_val(run, arch)))
                print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(WORK / "val_summary.csv", index=False)
    sel = {}
    for fold in ("holdA", "holdB"):
        d = df[df.fold == fold].sort_values(["val_mAP50_95", "imgsz"], ascending=[False, True])
        sel[fold] = d.iloc[0].to_dict()
    json.dump(sel, open(WORK / "selected.json", "w"), indent=1, default=str)
    print("selected:", {k: (v["arch"], v["imgsz"]) for k, v in sel.items()}, flush=True)

    det = str(Path(__file__).resolve().parent / "prep_02_detect.py")
    vidsA = [v for v in VIDEOS if VIDEOS[v]["camera"] == "A"]
    vidsB = [v for v in VIDEOS if VIDEOS[v]["camera"] == "B"]
    clips = [v for v in VIDEOS if VIDEOS[v]["kind"] == "clip"]

    def cache(tag, weights, arch, imgsz, vids):
        backend = "yolov5" if arch == "v5s" else "ultralytics"
        subprocess.run([PY, det, "--workers", "3", "--tag", tag, "--weights", weights, "--backend", backend,
                        "--imgsz", str(imgsz), "--only", *vids], check=True)

    g = lambda f, a, s: df[(df.fold == f) & (df.arch == a) & (df.imgsz == s)].iloc[0].weights
    # (1) YOLOv5s on all images (manuscript-like detector) -> clips (reproduction)
    cache("_v5ctrl", g("ctrl", "v5s", 416), "v5s", 416, clips)
    # (2) selected leakage-free detectors
    for fold, vids in (("holdA", vidsA), ("holdB", vidsB)):
        s = sel[fold]
        cache("_best", s["weights"], s["arch"], int(s["imgsz"]), vids)
    # (3) YOLOv5s leakage-free (416) for the detector comparison
    for fold, vids in (("holdA", vidsA), ("holdB", vidsB)):
        cache("_v5lolo", g(fold, "v5s", 416), "v5s", 416, vids)
    print("done", flush=True)


if __name__ == "__main__":
    main()
