"""
Preprocessing step 4:
 (a) locate every 1-min evaluation clip inside its continuous source recording
     (thumbnail matching -> exact frame offset), so that event times can be expressed on the
     timeline of the uninterrupted recordings;
 (b) leakage audit: locate every available detector-training image ('new data origin', 204
     labelled images) inside the recordings, to show which training frames overlap the
     evaluation clips / events.

Outputs: meta/clip_alignment.csv, meta/train_image_matches.csv
"""
import glob
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from video_registry import VIDEOS  # noqa: E402

CLIP_SOURCE = {"T1": "A_f269", "C1": "A_f269", "T2": "A_kt1958", "T3": "A_kt2044", "C2": "A_kt2044",
               "T4": "B_f413", "C3": "B_f413", "T5": "B_f442", "C4": "B_f442", "C5": "B_f449",
               "T6": "B_f456"}


def load_thumbs(vid):
    return np.load(ROOT / "cache/thumb" / f"{vid}.npy").astype(np.float32)


def meta(vid):
    return json.load(open(ROOT / "cache/meta" / f"{vid}.json"))


def align_clip(clip, src):
    a, b = load_thumbs(clip), load_thumbs(src)
    fa, fb = meta(clip)["fps"], meta(src)["fps"]
    # compare at 2 samples / s over the clip, for every candidate start frame in the source
    ts = np.arange(1.0, len(a) / fa - 1.0, 0.5)
    ia = np.round(ts * fa).astype(int)
    A = a[ia] - a[ia].mean(axis=(1, 2), keepdims=True)
    best = (1e9, None)
    ib_rel = np.round(ts * fb).astype(int)
    for off in range(0, len(b) - ib_rel.max() - 1):
        B = b[off + ib_rel]
        B = B - B.mean(axis=(1, 2), keepdims=True)
        err = np.abs(A - B).mean()
        if err < best[0]:
            best = (err, off)
    err, off = best
    return dict(clip=clip, source=src, offset_frame=off, offset_s=off / fb, mad=err,
                fps_clip=fa, fps_source=fb, clip_len_s=len(a) / fa)


def audit_training_images():
    """nearest video frame (L1 distance between mean-centred 64x36 thumbnails, computed on GPU)"""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    vids = [v for v in VIDEOS if VIDEOS[v]["kind"] == "raw" or v[0] in "TC"]
    files = sorted(glob.glob(str(ROOT / "data/new data origin/*.jpg")))
    q, shapes = [], []
    for f in files:
        im = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        h, w = im.shape
        shapes.append((w, h))
        if h == 1088:          # padded 1080p export
            im = im[:1080]
        t = cv2.resize(im, (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
        q.append(t - t.mean())
    Q = torch.tensor(np.stack(q), device=dev)
    best_d = torch.full((len(files),), 1e9, device=dev)
    best_v = [None] * len(files); best_f = [None] * len(files)
    for v in vids:
        T = torch.tensor(load_thumbs(v).reshape(-1, 64 * 36), device=dev)
        T = T - T.mean(1, keepdim=True)
        for s0 in range(0, len(T), 8192):
            d = torch.cdist(Q, T[s0:s0 + 8192], p=1) / (64 * 36)
            m, k = d.min(1)
            upd = m < best_d
            for i in torch.nonzero(upd).flatten().tolist():
                best_v[i] = v; best_f[i] = s0 + int(k[i])
            best_d = torch.where(upd, m, best_d)
    rows = []
    for i, f in enumerate(files):
        rows.append(dict(image=os.path.basename(f), width=shapes[i][0], height=shapes[i][1],
                         best_video=best_v[i], best_frame=best_f[i], mad=float(best_d[i])))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    (ROOT / "meta").mkdir(exist_ok=True)
    rows = []
    todo = {} if (ROOT / "meta/clip_alignment.csv").exists() else CLIP_SOURCE
    for c, s in todo.items():
        r = align_clip(c, s)
        print(r, flush=True)
        rows.append(r)
    if rows:
        pd.DataFrame(rows).to_csv(ROOT / "meta/clip_alignment.csv", index=False)
    df = audit_training_images()
    df.to_csv(ROOT / "meta/train_image_matches.csv", index=False)
    print(df.groupby("best_video").size())
