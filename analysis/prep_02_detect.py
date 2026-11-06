"""
Preprocessing step 2 (not part of the paper analysis itself):
run the piglet detector (YOLOv8s, weights from train-20230728T031826Z-001.zip) once on
every frame of every video and cache the raw detections, so that all downstream analyses
(HPA / CDA / TPA, threshold search, sensitivity analysis) run on cached detections only.

Outputs (per video id):
  cache/det/<vid>.parquet   frame, cls, conf, x1, y1, x2, y2   (conf >= CONF_STORE)
  cache/thumb/<vid>.npy     uint8 [n_frames, 36, 64] grayscale thumbnails (for alignment)
  cache/meta/<vid>.json     fps, size, n_frames, timing

Usage:  python prep_02_detect.py [--workers 3] [--only VID ...] [--weights path]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from video_registry import VIDEOS  # noqa: E402

CONF_STORE = 0.05     # store everything above this; thresholds are applied later
NMS_IOU = 0.7         # ultralytics default for YOLOv8
IMGSZ = 416           # training image size of the provided weights (args.yaml)
BATCH = 64
THUMB = (64, 36)


def run_one(vid, weights, out_root, tag, backend="ultralytics", imgsz=IMGSZ, nms_iou=NMS_IOU):
    import cv2
    import numpy as np
    import pandas as pd
    import torch
    from ultralytics import YOLO

    cv2.setNumThreads(3)
    torch.set_num_threads(1)
    det_dir = out_root / f"det{tag}"
    det_dir.mkdir(parents=True, exist_ok=True)
    (out_root / "thumb").mkdir(parents=True, exist_ok=True)
    (out_root / f"meta{tag}").mkdir(parents=True, exist_ok=True)
    out_det = det_dir / f"{vid}.parquet"
    if out_det.exists():
        return vid, "skip"
    path = ROOT / VIDEOS[vid]["path"]
    if backend == "yolov5":      # original (anchor-based) YOLOv5 code base, third_party/yolov5
        model = torch.hub.load(str(ROOT / "third_party/yolov5"), "custom", path=str(weights), source="local",
                               verbose=False)
        model.conf, model.iou, model.max_det = CONF_STORE, nms_iou, 300
        model = model.to(0)
    else:
        model = YOLO(str(weights))
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rows, thumbs = [], []
    t_dec = t_inf = 0.0
    fidx = 0
    done = False
    while not done:
        frames = []
        t0 = time.perf_counter()
        while len(frames) < BATCH:
            ok, fr = cap.read()
            if not ok:
                done = True
                break
            frames.append(fr)
            thumbs.append(cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), THUMB, interpolation=cv2.INTER_AREA))
        t_dec += time.perf_counter() - t0
        if not frames:
            break
        t0 = time.perf_counter()
        if backend == "yolov5":
            with torch.no_grad():
                res = model([f[..., ::-1] for f in frames], size=imgsz)      # AutoShape expects RGB
            t_inf += time.perf_counter() - t0
            for k, det in enumerate(res.xyxy):
                det = det.cpu().numpy()
                for x1, y1, x2, y2, c, k_cls in det:
                    rows.append((fidx + k, int(k_cls), c, x1, y1, x2, y2))
        else:
            res = model.predict(frames, imgsz=imgsz, conf=CONF_STORE, iou=nms_iou, half=True,
                                verbose=False, device=0)
            t_inf += time.perf_counter() - t0
            for k, r in enumerate(res):
                b = r.boxes
                if len(b) == 0:
                    continue
                xyxy = b.xyxy.cpu().numpy()
                cls = b.cls.cpu().numpy().astype(np.int8)
                conf = b.conf.cpu().numpy()
                for j in range(len(b)):
                    rows.append((fidx + k, cls[j], conf[j], *xyxy[j]))
        fidx += len(frames)
    cap.release()
    df = pd.DataFrame(rows, columns=["frame", "cls", "conf", "x1", "y1", "x2", "y2"])
    df = df.astype({"frame": "int32", "cls": "int8", "conf": "float32", "x1": "float32",
                    "y1": "float32", "x2": "float32", "y2": "float32"})
    df.to_parquet(out_det, index=False)
    tp = out_root / "thumb" / f"{vid}.npy"
    if not tp.exists():
        np.save(tp, np.stack(thumbs))
    meta = dict(vid=vid, path=str(path), fps=fps, width=w, height=h, n_frames=fidx,
                n_frames_meta=n_meta, decode_s=t_dec, infer_s=t_inf, weights=str(weights),
                imgsz=imgsz, conf_store=CONF_STORE, nms_iou=nms_iou, batch=BATCH, backend=backend)
    json.dump(meta, open(out_root / f"meta{tag}" / f"{vid}.json", "w"), indent=1)
    return vid, f"{fidx} frames, dec {t_dec:.0f}s inf {t_inf:.0f}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--weights", default=str(ROOT / "extracted/train/train/weights/best.pt"))
    ap.add_argument("--tag", default="", help="suffix for det/meta dirs (e.g. _clean)")
    ap.add_argument("--backend", default="ultralytics", choices=["ultralytics", "yolov5"])
    ap.add_argument("--imgsz", type=int, default=IMGSZ)
    ap.add_argument("--nms-iou", type=float, default=None, help="default 0.7 (ultralytics) / 0.45 (yolov5)")
    args = ap.parse_args()
    vids = args.only or list(VIDEOS)
    vids = sorted(vids, key=lambda v: -os.path.getsize(ROOT / VIDEOS[v]["path"]))
    out_root = ROOT / "cache"
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        nms = args.nms_iou if args.nms_iou is not None else (0.45 if args.backend == "yolov5" else NMS_IOU)
        jobs = [pool.apply_async(run_one, (v, Path(args.weights), out_root, args.tag, args.backend, args.imgsz, nms))
                for v in vids]
        for j in jobs:
            vid, msg = j.get()
            print(time.strftime("%H:%M:%S"), vid, msg, flush=True)


if __name__ == "__main__":
    main()
