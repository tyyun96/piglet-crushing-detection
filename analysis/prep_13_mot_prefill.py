"""
Preprocessing step 13: pre-filled boxes and identities for the manual MOT ground truth (reviewer comment R1-10).

The observer corrects these in outputs/mot_annotation/annotate_mot.ipynb (source analysis/mot_src.py); the corrected file
annotations/mot_gt/<clip>_<pen>.csv is what notebook §13 evaluates the trackers against.

Pre-fill: detections of the provided YOLOv8s (cache/det, conf >= 0.25; this detector saw frames of the evaluation clips during
training, so its boxes are close to the truth - it is used here only as a labelling aid, never as the evaluated detector),
centre inside the pen ROI, linked at the full frame rate by a plain IoU linker (Hungarian assignment, IoU >= 0.3, a track
survives 1 s without a match; no motion model, i.e. none of the evaluated trackers; a tracklet starting <= 1 s after another
ended, within half a box diagonal of its last box, continues it) and sampled at 2 frames per second
(the frames of outputs/mot_annotation_template/<clip>). Detections with 0.05 <= conf < 0.25 that do not overlap a pre-filled
box (IoU < 0.3) are kept as candidates the observer can accept.
usage: python prep_13_mot_prefill.py [--clip T6 --pen L]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from video_registry import VIDEOS  # noqa: E402

PEN_ROI = {"A": {"L": (0, 990), "R": (990, 1920)}, "B": {"L": (0, 850), "R": (850, 1920)}}   # = core.PEN_ROI
CONF, CAND_CONF, LINK_IOU, MAX_GAP_S = 0.25, 0.05, 0.3, 1.0
TEMPLATE = ROOT / "outputs" / "mot_annotation_template"
OUT = ROOT / "outputs" / "mot_annotation"


def iou(a, b):
    """pairwise IoU of boxes a (n x 4) and b (m x 4), x1 y1 x2 y2"""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda c: (c[:, 2] - c[:, 0]) * (c[:, 3] - c[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def link(det, n_frames, max_gap):
    """plain IoU linker over all frames -> det with a 'tid' column"""
    det = det.sort_values(["frame", "conf"], ascending=[True, False]).reset_index(drop=True)
    tid = np.full(len(det), -1)
    last_box, last_frame, nxt = {}, {}, 0
    by_frame = det.groupby("frame").indices
    for f in range(n_frames):
        idx = by_frame.get(f, np.array([], int))
        act = [t for t in last_box if f - last_frame[t] <= max_gap]
        if len(idx) and act:
            m = iou(np.array([last_box[t] for t in act]), det.loc[idx, ["x1", "y1", "x2", "y2"]].to_numpy())
            r, c = linear_sum_assignment(-m)
            for i, j in zip(r, c):
                if m[i, j] >= LINK_IOU:
                    tid[idx[j]] = act[i]
        for j in idx:
            if tid[j] < 0:
                tid[j] = nxt; nxt += 1
            last_box[tid[j]] = det.loc[j, ["x1", "y1", "x2", "y2"]].to_numpy(float); last_frame[tid[j]] = f
    return det.assign(tid=tid)


def stitch(trk, fps, gap_s=1.0, k=0.5):
    """join a tracklet to one that starts <= gap_s after it ended, within k x box diagonal of its last box
    (closest first; each tracklet gets at most one successor and one predecessor)"""
    t = trk.sort_values("frame")
    first, last = t.groupby("tid").first(), t.groupby("tid").last()
    centre = lambda b: np.array([(b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2])
    diag = lambda b: np.hypot(b.x2 - b.x1, b.y2 - b.y1)
    cand = []
    for a in last.index:
        for b in first.index:
            g = first.frame[b] - last.frame[a]
            if 0 < g <= gap_s * fps:
                dist = np.linalg.norm(centre(last.loc[a]) - centre(first.loc[b]))
                dg = (diag(last.loc[a]) + diag(first.loc[b])) / 2
                if dist <= k * dg:
                    cand.append((dist / dg, a, b))
    succ, pred, new = set(), set(), {}
    for _, a, b in sorted(cand):
        if a not in succ and b not in pred:
            succ.add(a); pred.add(b); new[b] = a
    root = lambda x: root(new[x]) if x in new else x
    return trk.assign(tid=trk.tid.map(lambda x: root(x))), len(new)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="T6")
    ap.add_argument("--pen", default="L")
    args = ap.parse_args()
    clip, pen = args.clip, args.pen
    meta = json.load(open(ROOT / "cache" / "meta" / f"{clip}.json"))
    fps, n = meta["fps"], meta["n_frames"]
    frames = sorted(int(p.stem.split("_")[1]) for p in (TEMPLATE / clip).glob("frame_*.jpg"))
    d = pd.read_parquet(ROOT / "cache" / "det" / f"{clip}.parquet")
    s = 1920.0 / meta["width"]
    d[["x1", "y1", "x2", "y2"]] = d[["x1", "y1", "x2", "y2"]] * s
    x0, x1 = PEN_ROI[VIDEOS[clip]["camera"]][pen]
    cx = (d.x1 + d.x2) / 2
    d = d[(cx >= x0) & (cx < x1)]
    trk = link(d[d.conf >= CONF], n, int(round(MAX_GAP_S * fps)))
    trk, n_stitched = stitch(trk, fps)
    pre = trk[trk.frame.isin(frames)].copy()
    order = pre.groupby("tid").frame.min().sort_values(kind="stable").index          # ids 1, 2, ... by first appearance
    pre["id"] = pre.tid.map({t: i + 1 for i, t in enumerate(order)})
    pre["source"] = "prefill"
    cand = d[(d.conf >= CAND_CONF) & (d.conf < CONF) & d.frame.isin(frames)].copy()
    keep = []
    for f, g in cand.groupby("frame"):
        p = pre[pre.frame == f][["x1", "y1", "x2", "y2"]].to_numpy()
        b = g[["x1", "y1", "x2", "y2"]].to_numpy()
        ok = np.ones(len(g), bool) if not len(p) else iou(b, p).max(1) < LINK_IOU
        # candidates overlapping each other: keep the most confident
        g = g[ok].sort_values("conf", ascending=False)
        chosen = []
        for r in g.itertuples():
            bb = np.array([[r.x1, r.y1, r.x2, r.y2]])
            if not chosen or iou(bb, np.array(chosen)).max() < 0.5:
                chosen.append(bb[0]); keep.append(r.Index)
    cand = cand.loc[keep]
    out = OUT / f"{clip}_{pen}"
    out.mkdir(parents=True, exist_ok=True)
    cols = ["frame", "id", "x1", "y1", "x2", "y2", "conf", "cls", "source"]
    pre.sort_values(["frame", "id"])[cols].round(1).to_csv(out / "prefill.csv", index=False)
    cand.sort_values(["frame", "conf"], ascending=[True, False])[["frame", "x1", "y1", "x2", "y2", "conf", "cls"]].round(
        {"x1": 1, "y1": 1, "x2": 1, "y2": 1, "conf": 3}).to_csv(out / "candidates.csv", index=False)
    json.dump(dict(clip=clip, pen=pen, camera=VIDEOS[clip]["camera"], fps=fps, n_frames=n, frames=frames, roi=[x0, x1],
                   detector=meta["weights"], conf=CONF, candidate_conf=CAND_CONF, link_iou=LINK_IOU, max_gap_s=MAX_GAP_S,
                   stitched_tracklets=n_stitched),
              open(out / "meta.json", "w"), indent=1)
    per = pre.groupby("frame").size()
    print(f"{clip} pen {pen}: {len(frames)} frames, {len(pre)} pre-filled boxes ({per.median():.0f} per frame, "
          f"{per.min()}-{per.max()}), {pre.id.nunique()} identities, {len(cand)} candidates -> {out}")


if __name__ == "__main__":
    main()
