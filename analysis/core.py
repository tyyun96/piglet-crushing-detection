# =====================================================================================
# core.py -- algorithm implementations used by crushing_revision_analysis.ipynb
# (the notebook inlines this file cell-by-cell; markers '# %% [section]' delimit cells)
# =====================================================================================

# %% [imports]
import json
import math
import os
from collections import Counter, deque
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

def _project_root():
    """project folder (the one containing analysis/): $CRUSHING_ROOT, else searched upwards from this file / the working directory"""
    if os.environ.get("CRUSHING_ROOT"):
        return Path(os.environ["CRUSHING_ROOT"])
    for base in ([Path(__file__).resolve().parent] if "__file__" in globals() else []) + [Path.cwd()]:
        for q in [base, *base.parents]:
            if (q / "analysis" / "core.py").exists():
                return q
    raise RuntimeError("set CRUSHING_ROOT to the project folder (the one containing analysis/)")


ROOT = _project_root()
CACHE = ROOT / "cache"
CLS_PIGLET, CLS_TRAP, CLS_CRUSHED = 0, 1, 2   # YOLOv8s classes: piglet / crushing(=visible trapping) / crushed


# %% [data-access]
def video_meta(vid, tag=""):
    return json.load(open(CACHE / f"meta{tag}" / f"{vid}.json"))


_DET_CACHE = {}


def load_dets(vid, tag=""):
    """raw cached detections (conf >= 0.05) in pixel coords of a 1920x1080 frame"""
    key = (vid, tag)
    if key not in _DET_CACHE:
        d = pd.read_parquet(CACHE / f"det{tag}" / f"{vid}.parquet")
        m = video_meta(vid, tag)
        s = 1920.0 / m["width"]
        for c in ("x1", "y1", "x2", "y2"):
            d[c] = d[c] * s
        _DET_CACHE[key] = d
    return _DET_CACHE[key]


# pen regions of interest (1920x1080 coordinates); boundary = central partition pole
PEN_ROI = {
    "A": {"L": (0, 990), "R": (990, 1920)},     # camera A: left sow not farrowed -> no litter
    "B": {"L": (0, 850), "R": (850, 1920)},     # camera B: both pens with litters
    "N": {"ALL": (0, 1920)},
    "X": {"ALL": (0, 1920)},                    # independent farm: one crate per camera
}


def frames_dets(vid, conf=0.25, classes=(0, 1, 2), pen=None, camera=None, tag=""):
    """list (len = n_frames) of arrays [x1,y1,x2,y2,conf,cls] after confidence / class / ROI filtering"""
    d = load_dets(vid, tag)
    d = d[(d.conf >= conf) & d.cls.isin(classes)]
    if pen is not None:
        x0, x1 = PEN_ROI[camera][pen]
        cx = (d.x1 + d.x2) / 2
        d = d[(cx >= x0) & (cx < x1)]
    n = video_meta(vid, tag)["n_frames"]
    arr = d[["x1", "y1", "x2", "y2", "conf", "cls"]].to_numpy(np.float64)
    fr = d.frame.to_numpy()
    out = [np.zeros((0, 6))] * n
    if len(fr):
        bounds = np.searchsorted(fr, np.arange(n + 1))
        out = [arr[bounds[i]:bounds[i + 1]] for i in range(n)]
    return out


def count_series(vid, conf=0.25, classes=(0, 1, 2), pen=None, camera=None, tag=""):
    """per-frame number of detections (vectorised, no tracking needed)"""
    d = load_dets(vid, tag)
    d = d[(d.conf >= conf) & d.cls.isin(classes)]
    if pen is not None:
        x0, x1 = PEN_ROI[camera][pen]
        cx = (d.x1 + d.x2) / 2
        d = d[(cx >= x0) & (cx < x1)]
    n = video_meta(vid, tag)["n_frames"]
    return np.bincount(d.frame.to_numpy(), minlength=n)[:n]


# %% [helpers]
def rolling_mode_int(x, w):
    """causal rolling mode of a non-negative int series over the last w samples (ties -> larger)"""
    x = np.asarray(x, int)
    K = int(x.max()) + 1 if len(x) else 1
    onehot_cum = np.zeros((len(x) + 1, K), np.int32)
    onehot_cum[1:] = np.cumsum(np.eye(K, dtype=np.int32)[x], axis=0)
    idx = np.arange(1, len(x) + 1)
    lo = np.maximum(0, idx - w)
    counts = onehot_cum[idx] - onehot_cum[lo]
    return (K - 1 - np.argmax(counts[:, ::-1], axis=1)).astype(int)


def rolling_median_int(x, w):
    s = pd.Series(np.asarray(x, float))
    return np.round(s.rolling(w, min_periods=1).median().to_numpy()).astype(int)



# %% [sow-motion]
def pen_motion(rec, pen, camera, lag_s=1.0, thr=20):
    """Sow-activity proxy: fraction of pen pixels (64x36 grey thumbnails, ~30x30 px cells) whose
    intensity changed by > thr grey levels over lag_s seconds.  Large values = sow standing up,
    lying down or rolling (the postural changes during which trapping/crushing happens)."""
    th = np.load(CACHE / "thumb" / f"{rec}.npy").astype(np.int16)
    fps = video_meta(rec)["fps"]
    x0, x1 = PEN_ROI[camera][pen]
    c0, c1 = int(x0 / 30), int(math.ceil(x1 / 30))
    sub = th[:, :, c0:c1]
    lag = max(1, int(round(lag_s * fps)))
    d = np.zeros(len(th))
    d[lag:] = (np.abs(sub[lag:] - sub[:-lag]) > thr).mean(axis=(1, 2))
    return d


def gate_alerts(evs, motion, fps, thr=0.08, pre_s=20.0):
    """keep only alerts preceded by a sow-activity burst: max(motion) >= thr within
    [onset - pre_s, alert]"""
    if len(evs) == 0:
        return evs
    keep = []
    for r in evs.itertuples():
        a = max(0, int(r.onset_frame - pre_s * fps)); b = int(r.alert_frame) + 1
        keep.append(motion[a:b].max() >= thr if b > a else False)
    return evs[np.array(keep, bool)]


# %% [sort]
def iou_matrix(a, b):
    """IoU between boxes a [N,4] and b [M,4] (x1,y1,x2,y2)"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    xx1 = np.maximum(a[:, None, 0], b[None, :, 0]); yy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    xx2 = np.minimum(a[:, None, 2], b[None, :, 2]); yy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-9)


def _z(b):      # box -> measurement [u, v, s, r]
    w = b[:, 2] - b[:, 0]; h = b[:, 3] - b[:, 1]
    return np.stack([b[:, 0] + w / 2, b[:, 1] + h / 2, w * h, w / np.maximum(h, 1e-6)], 1)


def _box(x):    # state -> box
    s = np.clip(x[:, 2], 1e-6, None); r = np.clip(x[:, 3], 1e-6, None)
    w = np.sqrt(s * r); h = s / np.maximum(w, 1e-6)
    return np.stack([x[:, 0] - w / 2, x[:, 1] - h / 2, x[:, 0] + w / 2, x[:, 1] + h / 2], 1)


class SortTracker:
    """SORT (Bewley et al. 2016): constant-velocity Kalman filter on [u,v,s,r] + Hungarian IoU
    association.  Vectorised over tracks.  Parameters follow the manuscript:
    iou_thr = 0.3 (Algorithm 1, line 9), max_age = t_expire (frames a track may stay unmatched)."""
    F = np.eye(7); F[0, 4] = F[1, 5] = F[2, 6] = 1
    H = np.eye(4, 7)
    Q = np.eye(7); Q[-1, -1] *= 0.01; Q[4:, 4:] *= 0.01
    R = np.eye(4); R[2:, 2:] *= 10.0

    def __init__(self, iou_thr=0.3, max_age=1):
        self.iou_thr, self.max_age = iou_thr, max_age
        self.X = np.zeros((0, 7)); self.P = np.zeros((0, 7, 7))
        self.ids = np.zeros(0, int); self.age = np.zeros(0, int); self.miss = np.zeros(0, int)
        self.next_id = 1

    def step(self, dets):
        """dets [N,>=4]; returns (matches [(det_idx, track_id)], new_ids [(det_idx, track_id)],
        removed_ids, boxes_of_live_tracks dict id->box)"""
        # --- predict
        if len(self.X):
            neg = (self.X[:, 6] + self.X[:, 2]) <= 0
            self.X[neg, 6] = 0
            self.X = self.X @ self.F.T
            self.P = np.einsum("ij,njk,lk->nil", self.F, self.P, self.F) + self.Q
        pred = _box(self.X) if len(self.X) else np.zeros((0, 4))
        # --- associate
        iou = iou_matrix(dets[:, :4], pred) if len(dets) else np.zeros((0, len(pred)))
        matches = []
        if iou.size:
            r, c = linear_sum_assignment(-iou)
            matches = [(i, j) for i, j in zip(r, c) if iou[i, j] >= self.iou_thr]
        md = {i for i, _ in matches}; mt = {j for _, j in matches}
        # --- update matched
        if matches:
            di = np.array([i for i, _ in matches]); tj = np.array([j for _, j in matches])
            z = _z(dets[di, :4])
            Pm = self.P[tj]; Xm = self.X[tj]
            S = np.einsum("ij,njk,lk->nil", self.H, Pm, self.H) + self.R
            K = np.einsum("nij,kj,nkl->nil", Pm, self.H, np.linalg.inv(S))
            y = z - Xm @ self.H.T
            self.X[tj] = Xm + np.einsum("nij,nj->ni", K, y)
            self.P[tj] = np.einsum("nij,njk->nik", np.eye(7)[None] - K @ self.H, Pm)
        self.miss += 1
        self.age += 1
        if matches:
            self.miss[tj] = 0
        # --- delete expired tracks (unmatched for more than max_age frames)
        keep = self.miss <= self.max_age
        removed = self.ids[~keep].tolist()
        out_matches = [(i, int(self.ids[j])) for i, j in matches]
        self.X, self.P, self.ids = self.X[keep], self.P[keep], self.ids[keep]
        self.age, self.miss = self.age[keep], self.miss[keep]
        # --- create tracks for unmatched detections
        new = [i for i in range(len(dets)) if i not in md]
        new_ids = []
        if new:
            z = _z(dets[new, :4])
            Xn = np.zeros((len(new), 7)); Xn[:, :4] = z
            Pn = np.tile(np.eye(7) * 10.0, (len(new), 1, 1)); Pn[:, 4:, 4:] *= 1000.0
            ids = np.arange(self.next_id, self.next_id + len(new)); self.next_id += len(new)
            self.X = np.vstack([self.X, Xn]); self.P = np.concatenate([self.P, Pn])
            self.ids = np.concatenate([self.ids, ids]); self.age = np.concatenate([self.age, np.zeros(len(new), int)])
            self.miss = np.concatenate([self.miss, np.zeros(len(new), int)])
            new_ids = list(zip(new, ids.tolist()))
        return out_matches, new_ids, removed


def run_tracker(frames, iou_thr=0.3, max_age=1, tracker="sort", fps=30.0):
    """runs a tracker over a list of per-frame detection arrays.
    Returns a long DataFrame: frame, tid, cx, cy, w, h, cls, conf, is_new"""
    rows = []
    if tracker == "sort":
        tr = SortTracker(iou_thr, max_age)
        for k, d in enumerate(frames):
            matches, new_ids, _ = tr.step(d)
            for i, tid in matches:
                b = d[i]; rows.append((k, tid, (b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b[2] - b[0], b[3] - b[1], b[5], b[4], 0))
            for i, tid in new_ids:
                b = d[i]; rows.append((k, tid, (b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b[2] - b[0], b[3] - b[1], b[5], b[4], 1))
    elif tracker == "bytetrack":
        rows = _run_bytetrack(frames, fps=fps)
    return pd.DataFrame(rows, columns=["frame", "tid", "cx", "cy", "w", "h", "cls", "conf", "is_new"])


def _run_bytetrack(frames, fps=30, track_buffer_s=1.0):
    """ByteTrack (Zhang et al. 2022) via the Ultralytics implementation, fed with cached detections.
    Ultralytics defaults except the lost-track buffer, which is set in seconds (track_buffer_s)."""
    from ultralytics.engine.results import Boxes
    from ultralytics.trackers.byte_tracker import BYTETracker
    from ultralytics.utils import IterableSimpleNamespace
    args = IterableSimpleNamespace(tracker_type="bytetrack", track_high_thresh=0.25, track_low_thresh=0.1,
                                   new_track_thresh=0.25, track_buffer=int(round(track_buffer_s * fps)),
                                   match_thresh=0.8, fuse_score=True)
    bt = BYTETracker(args)
    seen = set()
    rows = []
    for k, d in enumerate(frames):
        d = d if len(d) else np.zeros((0, 6))
        out = bt.update(Boxes(d.astype(np.float32), (1080, 1920)).cpu().numpy())
        for o in out:   # x1,y1,x2,y2,id,score,cls,idx
            tid = int(o[4])
            rows.append((k, tid, (o[0] + o[2]) / 2, (o[1] + o[3]) / 2, o[2] - o[0], o[3] - o[1], o[6], o[5],
                         int(tid not in seen)))
            seen.add(tid)
    return rows


# %% [hpa-original]
def hpa_original(n_det, t=300, d=150):
    """Algorithm 2 of the submitted manuscript, implemented verbatim (frame-based).
    n_det: per-frame number of detected objects (whole frame or one pen).
    Returns DataFrame with N_Tot, N_Det, M_cur, P_HT, N_HT per frame.
    Notes reproduced from the original: N_Tot is the running maximum of N_Det; M_cur (mode of
    N_Tot-N_Det over the last t frames) is refreshed every t/2 frames; N_HT changes by +/-1 only."""
    N_Tot = 0; D = 0; M_cur = 0; M_prev = 0; C_prev = 0; N_HT = 0
    d_list = []
    half = t // 2
    out = np.zeros((len(n_det), 5), dtype=np.int32)
    for k, nd in enumerate(n_det):
        nd = int(nd)
        d_list.append(N_Tot - nd)
        if N_Tot - nd > 0:
            D += 1
        else:
            D = 0
            N_Tot = nd
        P_HT = 1 if D > d else 0
        L = len(d_list)
        if L > half and L % half == 0:
            M_prev = M_cur
            win = d_list[-t:]
            M_cur = Counter(win).most_common(1)[0][0]     # 'max(count(d_list[-t:]))' = mode
            C_prev = 1
        if M_cur > M_prev:
            if P_HT == 1 and C_prev == 1:
                N_HT += 1; C_prev = 0
        else:
            if P_HT == 0 and C_prev == 1:
                N_HT -= 1; C_prev = 0
        if N_HT < 0:
            N_HT = 0
        out[k] = (N_Tot, nd, M_cur, P_HT, N_HT)
    return pd.DataFrame(out, columns=["N_Tot", "N_Det", "M_cur", "P_HT", "N_HT"])


# %% [cda-original]
def cda_original(tracks, f=300, m=200.0, movement="displacement", start_frame=0):
    """Algorithm 1 (CDA) of the submitted manuscript: every track created from an unmatched
    detection (T_new) is followed for f frames; if it moved <= m pixels, N_CR += 1.
    Tracks that disappear before f frames are discarded.  'start_frame': tracks born before this
    frame are ignored (0 = original behaviour, which also evaluates the tracks created on frame 0).
    Returns DataFrame of crushing decisions: tid, birth_frame, decision_frame, movement."""
    if len(tracks) == 0:
        return pd.DataFrame(columns=["tid", "birth", "decision", "move", "cx", "cy", "diag"])
    g = tracks.sort_values(["tid", "frame"]).groupby("tid", sort=False)
    rows = []
    for tid, tr in g:
        b = int(tr.frame.iloc[0])
        if b < start_frame:
            continue
        sub = tr[tr.frame <= b + f]
        if int(sub.frame.iloc[-1]) < b + f:      # track lost before the decision time
            continue
        x, y = sub.cx.to_numpy(), sub.cy.to_numpy()
        if movement == "displacement":
            mv = float(np.hypot(x[-1] - x[0], y[-1] - y[0]))
        elif movement == "max_displacement":
            mv = float(np.max(np.hypot(x - x[0], y - y[0])))
        else:   # path length
            mv = float(np.sum(np.hypot(np.diff(x), np.diff(y))))
        diag = float(np.median(np.hypot(sub.w, sub.h)))
        rows.append((tid, b, b + f, mv, x[-1], y[-1], diag))
    df = pd.DataFrame(rows, columns=["tid", "birth", "decision", "move", "cx", "cy", "diag"])
    df["crushed"] = df.move <= m
    return df


def cda_series(dec, n_frames):
    """cumulative N_CR per frame from CDA decisions"""
    s = np.zeros(n_frames, int)
    for k in dec.loc[dec.crushed, "decision"]:
        if k < n_frames:
            s[int(k)] += 1
    return np.cumsum(s)


# %% [hpa-v2]
def hpa_v2(n_det, fps, d_s=5.0, t_s=5.0, step_min=1, smooth_s=1.0, tot_window_s=600.0,
           warmup_s=30.0, known_total=None, max_event_s=120.0, max_total=None):
    """Revised hidden-trapping detector for ONE pen (all thresholds in seconds, frame-rate free).
    It keeps the logic of the original HPA (a sustained rise of the stabilised missing count is a
    hidden-trapping event, a sustained fall means hidden piglets re-appeared) but
      * evaluates every frame instead of every t/2 frames,
      * counts piglets by the MAGNITUDE of the step (original: +/-1 per update),
      * uses one variable per concept (reviewer comment 6):
          n_s   stabilised visible count          = causal median of N_Det over smooth_s
          N_Tot expected litter size              = causal max of n_s over tot_window_s (or known_total;
                                                    max_total caps the rolling max, e.g. at the counted litter size)
          M     current number of missing detections = max(0, N_Tot - n_s)
          M_st  stabilised missing level          = causal mode of M over t_s
          L     last confirmed missing level (reference)
          H     current estimated number of hidden(-trapped) piglets
          E     cumulative number of hidden-trapping events (= alerts)
    Rule: if M_st >= L + step_min for d_s seconds -> alert, dH = mode(M_st over those d_s) - L,
          H += dH, E += 1, L <- new level;  if M_st <= L - 1 for d_s seconds -> re-appearance,
          H <- max(0, H - (L - new level)), L <- new level (no alert)."""
    n = len(n_det)
    n_s = rolling_median_int(n_det, max(1, int(round(smooth_s * fps))))
    if known_total is None:
        N_Tot = pd.Series(n_s).rolling(max(1, int(tot_window_s * fps)), min_periods=1).max().to_numpy().astype(int)
        if max_total is not None:
            N_Tot = np.minimum(N_Tot, int(max_total))
    else:
        N_Tot = np.full(n, int(known_total))
    M = np.clip(N_Tot - n_s, 0, None)
    M_st = rolling_mode_int(M, max(1, int(round(t_s * fps))))
    d_f = max(1, int(round(d_s * fps))); warm = int(warmup_s * fps)
    H = np.zeros(n, int); E = np.zeros(n, int); Lr = np.zeros(n, int)
    L = int(M_st[min(warm, n - 1)]) if n else 0
    up = down = 0; h = 0; cnt = 0
    alerts, reapp = [], []
    for k in range(n):
        if k >= warm:
            m = int(M_st[k])
            up = up + 1 if m >= L + step_min else 0
            down = down + 1 if m <= L - 1 else 0
            if up >= d_f:
                seg = M_st[k - d_f + 1:k + 1]
                new = int(np.bincount(seg).argmax())
                dh = max(step_min, new - L)
                h += dh; cnt += 1
                alerts.append(dict(onset_frame=k - d_f + 1, alert_frame=k, level_before=L, level_after=new,
                                   dH=dh, H_after=h, N_Tot=int(N_Tot[k])))
                L = new; up = down = 0
            elif down >= d_f:
                seg = M_st[k - d_f + 1:k + 1]
                new = int(np.bincount(seg).argmax())          # most frequent missing level over the last d seconds
                dr = L - new
                if h > 0:
                    reapp.append(dict(frame=k, dR=min(dr, h), H_after=max(0, h - dr)))
                h = max(0, h - dr); L = new; up = down = 0
        else:
            L = int(M_st[k])
        H[k] = h; E[k] = cnt; Lr[k] = L
    ts = pd.DataFrame(dict(n_det=np.asarray(n_det, int), n_s=n_s, N_Tot=N_Tot, M=M, M_st=M_st, L=Lr, H=H, E=E))
    evs = pd.DataFrame(alerts, columns=["onset_frame", "alert_frame", "level_before", "level_after", "dH",
                                        "H_after", "N_Tot"])
    # end of an event = first re-appearance step after the alert, at most max_event_s after the alert
    ends = []
    rf = np.array([r["frame"] for r in reapp]) if reapp else np.zeros(0, int)
    cap = int(round(max_event_s * fps))
    for r in evs.itertuples():
        nxt = rf[rf > r.alert_frame]
        ends.append(int(min(nxt[0] if len(nxt) else n - 1, r.alert_frame + cap, n - 1)))
    evs["end_frame"] = ends
    for c in ("onset_frame", "alert_frame", "end_frame"):
        evs[c.replace("_frame", "_s")] = evs[c] / fps
    return ts, evs


def visible_trap_v2(n_vt, fps, smooth_s=1.0, d_s=2.0):
    """image-based visible-trapping channel: stabilised count of 'crushing'-class boxes and
    sustained-presence alerts (>= d_s seconds)"""
    vt = rolling_median_int(n_vt, max(1, int(round(smooth_s * fps))))
    on = vt >= 1
    events = []
    k = 0; n = len(vt); d_f = int(round(d_s * fps))
    while k < n:
        if on[k]:
            j = k
            while j < n and on[j]:
                j += 1
            if j - k >= d_f:
                events.append(dict(onset_frame=k, alert_frame=k + d_f - 1, end_frame=j - 1,
                                   n_max=int(vt[k:j].max())))
            k = j
        else:
            k += 1
    evs = pd.DataFrame(events, columns=["onset_frame", "alert_frame", "end_frame", "n_max"])
    for c in ("onset_frame", "alert_frame", "end_frame"):
        evs[c.replace("_frame", "_s")] = evs[c] / fps
    return vt, evs


def tpa_v2(ts_hpa, vt):
    """TPA fusion (per pen, per frame):  N_trap(k) = N_VT(k) + H(k).
    Partially visible trapped piglets are detected boxes (counted in N_Det, hence never 'missing'),
    so they can contribute to N_VT only; fully hidden piglets contribute to H only -> no double count."""
    return np.asarray(vt, int) + ts_hpa["H"].to_numpy()


# %% [cda-v2]
def cda_v2(tracks, fps, hpa_events=None, f_s=10.0, m_rel=0.5, gate_after_s=30.0, warmup_s=30.0,
           crushed_boxes=None, require_crushed=False):
    """Revised suspected-crushing (immobility) alert.
    Candidate = track newly created after warm-up; if hpa_events is given, only tracks created during
    a hidden-trapping event or within gate_after_s after it ends (i.e. piglets re-appearing from
    under the sow) are considered (HPA-gated CDA).  Immobile = maximum displacement of the box centre
    during the first f_s seconds <= m_rel x median box diagonal (size-normalised threshold).
    Optionally require a 'crushed'-class detection on the track (image-based confirmation)."""
    if len(tracks) == 0:
        return pd.DataFrame(columns=["tid", "birth", "decision", "move_rel", "crushed"])
    f = int(round(f_s * fps)); warm = int(warmup_s * fps)
    gates = None
    if hpa_events is not None:
        gates = [(int(r.onset_frame), int(r.end_frame + gate_after_s * fps)) for r in hpa_events.itertuples()]
    rows = []
    for tid, tr in tracks.groupby("tid", sort=False):
        b = int(tr.frame.iloc[0])
        if b < warm:
            continue
        if gates is not None and not any(g0 <= b <= g1 for g0, g1 in gates):
            continue
        sub = tr[tr.frame <= b + f]
        if int(sub.frame.iloc[-1]) < b + f - int(0.2 * f):    # must be observed until (almost) f
            continue
        x, y = sub.cx.to_numpy(), sub.cy.to_numpy()
        diag = float(np.median(np.hypot(sub.w, sub.h)))
        mv = float(np.max(np.hypot(x - x[0], y - y[0]))) / max(diag, 1.0)
        n_crushed_cls = int((sub.cls == CLS_CRUSHED).sum())
        ok = mv <= m_rel and (n_crushed_cls > 0 if require_crushed else True)
        rows.append((tid, b, b + f, mv, n_crushed_cls, ok))
    return pd.DataFrame(rows, columns=["tid", "birth", "decision", "move_rel", "n_crushed_cls", "crushed"])


def track_summary(tracks, fps, f_list_s=(1, 3, 5, 10, 20, 30), f_list_frames=(30, 90, 150, 300, 600)):
    """Per-track features used by both CDA versions, so that CDA thresholds can be varied without
    re-tracking.  For each window length f (seconds for CDA-v2, frames for the original CDA):
      full_*  : track still alive (observed) at birth + f (original: exactly; v2: >= 80% of f)
      mrel_*  : max displacement of the box centre within f, divided by the median box diagonal (v2)
      path_*  : path length (px) of the centre within f (original CDA movement)
      ncr_*   : number of frames with a 'crushed'-class box on the track within f"""
    if len(tracks) == 0:
        return pd.DataFrame()
    tr = tracks.sort_values(["tid", "frame"])
    fr = tr.frame.to_numpy(); tid = tr.tid.to_numpy(); cx = tr.cx.to_numpy(); cy = tr.cy.to_numpy()
    dg = np.hypot(tr.w.to_numpy(), tr.h.to_numpy()); cl = tr.cls.to_numpy()
    starts = np.r_[0, np.where(np.diff(tid) != 0)[0] + 1]; ends = np.r_[starts[1:], len(tid)]
    rows = []
    for a, b in zip(starts, ends):
        f0 = fr[a]; x = cx[a:b]; y = cy[a:b]; ff = fr[a:b] - f0
        rec = dict(tid=int(tid[a]), birth=int(f0), last=int(fr[b - 1]), n_obs=int(b - a), cx0=float(x[0]), cy0=float(y[0]),
                   diag=float(np.median(dg[a:b])))
        disp = np.hypot(x - x[0], y - y[0]); step = np.r_[0, np.hypot(np.diff(x), np.diff(y))]; cum = np.cumsum(step)
        for fs in f_list_s:
            n = int(round(fs * fps)); j = np.searchsorted(ff, n, side="right")
            rec[f"full_s{fs}"] = bool(ff[-1] >= 0.8 * n)
            rec[f"mrel_s{fs}"] = float(disp[:j].max() / max(rec["diag"], 1.0))
            rec[f"ncr_s{fs}"] = int((cl[a:a + j] == CLS_CRUSHED).sum())
        for fn in f_list_frames:
            j = np.searchsorted(ff, fn, side="right")
            rec[f"full_f{fn}"] = bool(ff[-1] >= fn)
            rec[f"path_f{fn}"] = float(cum[j - 1])
            rec[f"disp_f{fn}"] = float(disp[j - 1])
        rows.append(rec)
    return pd.DataFrame(rows)


def cda_orig_from_summary(ts_, f=300, m=200.0, movement="path", start_frame=0):
    """original CDA decisions from a track summary (frames / pixels)"""
    if len(ts_) == 0:
        return ts_
    d = ts_[(ts_.birth >= start_frame) & ts_[f"full_f{f}"]].copy()
    d["decision"] = d.birth + f
    d["crushed"] = d[f"{'path' if movement == 'path' else 'disp'}_f{f}"] <= m
    return d[d.crushed]


def cda_v2_from_summary(ts_, fps, f_s=10, m_rel=0.5, gates=None, warmup_s=30.0, require_crushed=False):
    """revised CDA (suspected crushing / immobility alert) from a track summary.
    gates: list of (start_frame, end_frame) windows in which candidate tracks must be born (HPA gating)"""
    if len(ts_) == 0:
        return ts_
    d = ts_[(ts_.birth >= warmup_s * fps) & ts_[f"full_s{f_s}"] & (ts_[f"mrel_s{f_s}"] <= m_rel)].copy()
    if require_crushed:
        d = d[d[f"ncr_s{f_s}"] > 0]
    if gates is not None:
        b = d.birth.to_numpy()
        ok = np.zeros(len(d), bool)
        for g0, g1 in gates:
            ok |= (b >= g0) & (b <= g1)
        d = d[ok]
    d["decision"] = d.birth + int(round(f_s * fps))
    return d


def mot_proxies(tracks, fps):
    """tracking-quality proxies that need no ground truth"""
    if len(tracks) == 0:
        return {}
    g = tracks.groupby("tid").frame.agg(["min", "max", "size"])
    dur = (g["max"] - g["min"] + 1) / fps
    minutes = (tracks.frame.max() + 1) / fps / 60
    det_per_frame = len(tracks) / (tracks.frame.max() + 1)
    long_ids = g.index[dur >= 10]
    return dict(n_tracks=len(g), births_per_min=len(g) / minutes,
                births_per_piglet_per_min=len(g) / minutes / max(det_per_frame, 1e-9),
                mean_track_s=float(dur.mean()), median_track_s=float(dur.median()),
                frac_dets_in_tracks_ge10s=float(tracks.tid.isin(long_ids).mean()))


# %% [evaluation]
def r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    sst = ((y - y.mean()) ** 2).sum()
    return float(1 - ((y - p) ** 2).sum() / sst) if sst > 0 else float("nan")


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def match_alerts(alerts, gt, pre_s=10.0, post_s=60.0, exclude=None):
    """Event-level matching with a PRE-SPECIFIED temporal window.
    An alert at time a (s) matches GT event g of the same recording and pen if
        onset_g - pre_g <= a <= onset_g + post_g
    (pre_g/post_g from gt columns 'pre_s'/'post_s' if present, else the defaults).  Alerts with
    pen == 'ALL' (whole-frame methods) may match an event in any pen of that recording.
    Alerts are processed in time order: the first alert matched to an event is a TP, further ones
    are duplicates (DUP; an alert prefers a not-yet-detected event if several are possible);
    unmatched alerts are FP unless inside an 'exclude' window (unannotated candidate events ->
    EXCLUDED); events without alert are FN.
    alerts: DataFrame[rec, pen, alert_s]; gt: DataFrame[event_id, rec, pen, onset_s]"""
    alerts = alerts.sort_values("alert_s", kind="stable").reset_index(drop=True)
    gt = gt.copy().reset_index(drop=True)
    if "pre_s" not in gt:
        gt["pre_s"] = pre_s
    if "post_s" not in gt:
        gt["post_s"] = post_s
    n = len(alerts)
    status = np.array(["FP"] * n, dtype=object); match = np.array([None] * n, dtype=object)
    first = np.full(len(gt), np.nan); n_al = np.zeros(len(gt), int)
    if n and len(gt):
        ar, ap, at = alerts.rec.to_numpy(), alerts.pen.to_numpy(), alerts.alert_s.to_numpy()
        cand = ((ar[:, None] == gt.rec.to_numpy()[None]) &
                ((ap[:, None] == gt.pen.to_numpy()[None]) | (ap[:, None] == "ALL")) &
                (at[:, None] >= (gt.onset_s - gt.pre_s).to_numpy()[None]) &
                (at[:, None] <= (gt.onset_s + gt.post_s).to_numpy()[None]))
        on = gt.onset_s.to_numpy()
        for i in np.where(cand.any(1))[0]:
            js = np.where(cand[i])[0]
            free = js[n_al[js] == 0]
            pool = free if len(free) else js
            j = pool[np.argmin(np.abs(on[pool] - at[i]))]
            match[i] = gt.event_id.iat[j]
            if n_al[j] == 0:
                status[i] = "TP"; first[j] = at[i]
            else:
                status[i] = "DUP"
            n_al[j] += 1
    if exclude is not None and n and len(exclude):
        ar, ap, at = alerts.rec.to_numpy(), alerts.pen.to_numpy(), alerts.alert_s.to_numpy()
        ex = ((ar[:, None] == exclude.rec.to_numpy()[None]) &
              ((ap[:, None] == exclude.pen.to_numpy()[None]) | (ap[:, None] == "ALL")) &
              (at[:, None] >= exclude.start_s.to_numpy()[None]) & (at[:, None] <= exclude.end_s.to_numpy()[None])).any(1)
        status[(status == "FP") & ex] = "EXCLUDED"
    alerts["match"] = match; alerts["status"] = status
    gt["first_alert_s"] = first; gt["n_alerts"] = n_al
    gt["detected"] = gt.n_alerts > 0
    gt["latency_s"] = gt.first_alert_s - gt.onset_s
    return alerts, gt


def window_confusion(alerts, gt, recs_pens, fps_of, nfr_of, win_s=60.0, exclude=None):
    """time-window confusion matrix (gives specificity / NPV): every pen-recording is cut into
    consecutive win_s windows; a window is GT-positive if it contains an annotated onset, predicted
    positive if it contains >= 1 alert; windows overlapping an exclude window are dropped."""
    tp = fp = fn = tn = 0
    for rec, pen in recs_pens:
        dur = nfr_of[rec] / fps_of[rec]
        edges = np.arange(0, dur + win_s, win_s)
        g = gt[(gt.rec == rec) & (gt.pen == pen)].onset_s.to_numpy()
        a = alerts[(alerts.rec == rec) & ((alerts.pen == pen) | (alerts.pen == "ALL"))].alert_s.to_numpy()
        for w0, w1 in zip(edges[:-1], edges[1:]):
            if exclude is not None and len(exclude[(exclude.rec == rec) & (exclude.pen == pen) &
                                                   (exclude.start_s < w1) & (exclude.end_s > w0)]):
                continue
            pos = bool(((g >= w0) & (g < w1)).any()); pred = bool(((a >= w0) & (a < w1)).any())
            tp += pos and pred; fp += (not pos) and pred; fn += pos and (not pred); tn += (not pos) and (not pred)
    return dict(TP=tp, FP=fp, FN=fn, TN=tn, sensitivity=tp / (tp + fn) if tp + fn else np.nan,
                specificity=tn / (tn + fp) if tn + fp else np.nan, PPV=tp / (tp + fp) if tp + fp else np.nan,
                NPV=tn / (tn + fn) if tn + fn else np.nan)


def event_metrics(alerts, gt, hours):
    tp = int((alerts.status == "TP").sum()); fp = int((alerts.status == "FP").sum())
    dup = int((alerts.status == "DUP").sum()); fn = int((~gt.detected).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) and not np.isnan(prec + rec) else float("nan")
    return dict(TP=tp, FP=fp, FN=fn, DUP=dup, precision=prec, recall=rec, F1=f1,
                FA_per_pen_h=fp / hours if hours else float("nan"),
                dup_rate=dup / (tp + dup) if tp + dup else float("nan"),
                latency_median_s=float(gt.latency_s.median()) if gt.detected.any() else float("nan"),
                latency_max_s=float(gt.latency_s.max()) if gt.detected.any() else float("nan"))


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


