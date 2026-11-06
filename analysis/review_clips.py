"""Short review videos of alerts (used by outputs/alert_review/label_alerts.ipynb and
prep_07_render_review_clips.py).

Each clip shows ONE pen, from alert-half to alert+half seconds, with
  - the offset to the alert time (e.g. '-3.2 s') and the camera clock time,
  - a red border while |offset| < 0.5 s (the alert moment),
  - a time bar at the bottom: red tick = alert time, white marker = current frame.
Encoded as H.264 (yuv420p) so that it plays inside JupyterLab.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "threads;2")   # limit decoder threads (shared server)
import cv2
import imageio_ffmpeg

cv2.setNumThreads(2)
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("CRUSHING_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "analysis"))
from video_registry import VIDEOS  # noqa: E402

PEN_ROI = {"A": {"L": (0, 990), "R": (990, 1920)}, "B": {"L": (0, 850), "R": (850, 1920)},
           "N": {"ALL": (0, 1920)}, "X": {"ALL": (0, 1920)}}
CLIPS = ROOT / "outputs" / "alert_review" / "_clips"


def clip_path(rec, pen, t, half):
    return CLIPS / f"{rec}_{pen}_{float(t):07.1f}s_pm{int(half)}s.mp4"


def make_clip(rec, pen, t, half=15, fps_out=12, height=540, force=False):
    t = float(t)
    out = clip_path(rec, pen, t, half)
    if out.exists() and not force:
        return out
    CLIPS.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(ROOT / VIDEOS[rec]["path"]))
    f = cap.get(cv2.CAP_PROP_FPS)
    t0 = max(0.0, t - half)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t0 * f)))
    x0, x1 = PEN_ROI[rec[0]][pen]
    w = int(round(height * (x1 - x0) / 1080 / 16)) * 16
    start = pd.Timestamp(VIDEOS[rec]["start"])
    tmp = out.with_name(out.stem + f".part{os.getpid()}.mp4")      # unique per process -> no clashes
    wr = imageio_ffmpeg.write_frames(str(tmp), (w, height), fps=fps_out, codec="libx264", pix_fmt_out="yuv420p",
                                     quality=6, macro_block_size=16,
                                     output_params=["-movflags", "+faststart", "-threads", "2"])
    wr.send(None)
    step = f / fps_out
    n_src = int(round((t + half - t0) * f))
    nxt = 0.0
    for i in range(n_src):
        ok, fr = cap.read()
        if not ok:
            break
        if i + 1e-6 < nxt:
            continue
        nxt += step
        tt = t0 + i / f
        off = tt - t
        c = cv2.resize(cv2.resize(fr, (1920, 1080))[:, x0:x1], (w, height))
        if abs(off) < 0.5:
            cv2.rectangle(c, (0, 0), (w - 1, height - 1), (0, 0, 255), 8)
            cv2.putText(c, "ALERT", (w - 120, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
        cv2.rectangle(c, (0, 48), (150, 90), (0, 0, 0), -1)
        cv2.putText(c, f"{off:+5.1f} s", (8, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        clock = (start + pd.to_timedelta(tt, unit="s")).strftime("%m-%d %H:%M:%S")
        cv2.putText(c, clock, (10, height - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        # time bar
        y = height - 10
        cv2.rectangle(c, (0, y), (w - 1, height - 1), (60, 60, 60), -1)
        span = (t + half) - t0
        xa = int((t - t0) / span * (w - 1)); xc = int((tt - t0) / span * (w - 1))
        cv2.rectangle(c, (xa - 2, y - 4), (xa + 2, height - 1), (0, 0, 255), -1)
        cv2.rectangle(c, (xc - 1, y), (xc + 1, height - 1), (255, 255, 255), -1)
        wr.send(cv2.cvtColor(c, cv2.COLOR_BGR2RGB).tobytes())
    wr.close()
    cap.release()
    tmp.rename(out)
    return out


# ---------------------------------------------------------------------------------------------
# Continuous (blind) annotation: the whole recording of one pen cut into consecutive segments.
# No alert information is drawn. Overlay: segment time (m:ss, large), camera clock, time bar
# with a tick every 10 s.
SEGMENTS = ROOT / "outputs" / "continuous_annotation" / "_segments"


def segment_path(rec, pen, start_s, seg_s):
    return SEGMENTS / f"{rec}_{pen}_{int(start_s):05d}s_{int(seg_s)}s.mp4"


def segment_starts(rec, seg_s=120):
    cap = cv2.VideoCapture(str(ROOT / VIDEOS[rec]["path"]))
    dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return [(s, min(s + seg_s, dur)) for s in np.arange(0, dur, seg_s) if min(s + seg_s, dur) - s >= 1.0], dur


def render_segments(rec, pen, seg_s=120, fps_out=12, height=540):
    """decode the recording once and write every missing segment of this pen"""
    segs, _ = segment_starts(rec, seg_s)
    todo = [(s, e) for s, e in segs if not segment_path(rec, pen, s, seg_s).exists()]
    if not todo:
        return 0
    SEGMENTS.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(ROOT / VIDEOS[rec]["path"]))
    f = cap.get(cv2.CAP_PROP_FPS)
    x0, x1 = PEN_ROI[rec[0]][pen]
    w = int(round(height * (x1 - x0) / 1080 / 16)) * 16
    start = pd.Timestamp(VIDEOS[rec]["start"])
    step = f / fps_out
    i = int(round(todo[0][0] * f))
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    for s, e in todo:
        out = segment_path(rec, pen, s, seg_s)
        i0, i1 = int(round(s * f)), int(round(e * f))
        while i < i0:                       # skip frames of already rendered segments
            cap.grab(); i += 1
        tmp = out.with_name(out.stem + f".part{os.getpid()}.mp4")
        wr = imageio_ffmpeg.write_frames(str(tmp), (w, height), fps=fps_out, codec="libx264", pix_fmt_out="yuv420p",
                                         quality=6, macro_block_size=16,
                                         output_params=["-movflags", "+faststart", "-threads", "2"])
        wr.send(None)
        nxt = float(i0)
        while i < i1:
            ok, fr = cap.read()
            i += 1
            if not ok:
                break
            if i - 1 + 1e-6 < nxt:
                continue
            nxt += step
            tt = (i - 1) / f
            off = max(0.0, tt - s)              # first frame can sit a few ms before s (frame rounding)
            ds = int(off * 10)                  # tenths of a second, truncated like a stopwatch
            c = cv2.resize(cv2.resize(fr, (1920, 1080))[:, x0:x1], (w, height))
            cv2.rectangle(c, (0, 0), (190, 58), (0, 0, 0), -1)
            cv2.putText(c, f"{ds // 600}:{ds % 600 / 10:04.1f}", (8, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 3)
            clock = (start + pd.to_timedelta(tt, unit="s")).strftime("%m-%d %H:%M:%S")
            cv2.putText(c, clock, (10, height - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y = height - 10
            cv2.rectangle(c, (0, y), (w - 1, height - 1), (60, 60, 60), -1)
            span = e - s
            for k in range(10, int(span), 10):
                xk = int(k / span * (w - 1))
                cv2.rectangle(c, (xk, y - (6 if k % 60 == 0 else 2)), (xk, height - 1), (150, 150, 150), -1)
            xc = int(off / span * (w - 1))
            cv2.rectangle(c, (xc - 1, y), (xc + 1, height - 1), (255, 255, 255), -1)
            wr.send(cv2.cvtColor(c, cv2.COLOR_BGR2RGB).tobytes())
        wr.close()
        tmp.rename(out)
    cap.release()
    return len(todo)


# ---------------------------------------------------------------------------------------------
# Adjudication of discrepancies between the alert labels and the continuous annotation: one pen,
# a time range of the recording, the period to judge highlighted in yellow on the time bar.
# Overlay: RECORDING time (m:ss, large; used to type an onset) and camera clock.
ADJ = ROOT / "outputs" / "continuous_annotation" / "_adjudication"


def range_clip_path(rec, pen, t0, t1):
    return ADJ / f"{rec}_{pen}_{int(t0):05d}-{int(t1):05d}s.mp4"


def make_range_clip(rec, pen, t0, t1, hi0, hi1, fps_out=12, height=544):
    out = range_clip_path(rec, pen, t0, t1)
    if out.exists():
        return out
    ADJ.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(ROOT / VIDEOS[rec]["path"]))
    f = cap.get(cv2.CAP_PROP_FPS)
    t1 = min(t1, cap.get(cv2.CAP_PROP_FRAME_COUNT) / f)
    x0, x1 = PEN_ROI[rec[0]][pen]
    w = int(round(height * (x1 - x0) / 1080 / 16)) * 16
    start = pd.Timestamp(VIDEOS[rec]["start"])
    i0, i1 = int(round(t0 * f)), int(round(t1 * f))
    cap.set(cv2.CAP_PROP_POS_FRAMES, i0)
    tmp = out.with_name(out.stem + f".part{os.getpid()}.mp4")
    wr = imageio_ffmpeg.write_frames(str(tmp), (w, height), fps=fps_out, codec="libx264", pix_fmt_out="yuv420p",
                                     quality=6, macro_block_size=16, output_params=["-movflags", "+faststart", "-threads", "2"])
    wr.send(None)
    step, nxt, span = f / fps_out, float(i0), t1 - t0
    for i in range(i0, i1):
        ok, fr = cap.read()
        if not ok:
            break
        if i + 1e-6 < nxt:
            continue
        nxt += step
        tt = i / f
        ds = int(tt * 10)
        c = cv2.resize(cv2.resize(fr, (1920, 1080))[:, x0:x1], (w, height))
        if hi0 <= tt <= hi1:
            cv2.rectangle(c, (0, 0), (w - 1, height - 1), (0, 220, 255), 6)
        cv2.rectangle(c, (0, 0), (190, 58), (0, 0, 0), -1)
        cv2.putText(c, f"{ds // 600}:{ds % 600 / 10:04.1f}", (8, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 3)
        cv2.putText(c, (start + pd.to_timedelta(tt, unit="s")).strftime("%m-%d %H:%M:%S"), (10, height - 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y = height - 10
        cv2.rectangle(c, (0, y), (w - 1, height - 1), (60, 60, 60), -1)
        xa, xb = int((hi0 - t0) / span * (w - 1)), int((hi1 - t0) / span * (w - 1))
        cv2.rectangle(c, (max(0, xa), y), (max(xa + 2, xb), height - 1), (0, 220, 255), -1)
        xc = int((tt - t0) / span * (w - 1))
        cv2.rectangle(c, (xc - 1, y - 4), (xc + 1, height - 1), (255, 255, 255), -1)
        wr.send(cv2.cvtColor(c, cv2.COLOR_BGR2RGB).tobytes())
    wr.close(); cap.release()
    tmp.rename(out)
    return out
