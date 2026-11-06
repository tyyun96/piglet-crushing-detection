"""
Preprocessing step 3: read back the per-frame outputs of the ORIGINAL (submitted-manuscript)
implementation from the red text overlay burned into the 'crushing result' videos.

Overlay (red text, OpenCV Hershey font):
  left  : 'Hidden Trapping Predict: <N_HT>', 'Crushing Predict: <N_CR>'
  right : 'Objects Count: <N_Tot>', 'Detected Objects: <N_Det>', 'Disappeared Objects: <M_cur>'
Method: red-pixel mask -> split into text lines -> connected components -> glyphs to the
right of the last colon -> each glyph classified against rendered digit templates.

Output: cache/ocr/<C#>.csv with columns frame, N_HT, N_CR, N_Tot, N_Det, M_cur
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FONT = cv2.FONT_HERSHEY_SIMPLEX
RESULT_VIDEOS = {
    "C1": "data/crushing result/crushing1(2m34s) (2).mp4",
    "C2": "data/crushing result/crushing2(8m21s).mp4",
    "C3": "data/crushing result/crushing3(12m30s).mp4",
    "C4": "data/crushing result/crushing4(1m24s).mp4",
    "C5": "data/crushing result/crushing5(3m59s).mp4",
}
CANVAS = 24  # glyphs are scaled to height 22 (aspect kept) and centred on a 24x24 canvas
# frames whose overlay values were verified by eye -> used to harvest digit glyph templates
KNOWN = [
    ("C1", 900, dict(N_Tot=14, N_Det=10, M_cur=3)),
    ("C1", -3, dict(N_HT=0, N_CR=3, N_Tot=17, N_Det=11, M_cur=2)),
    ("C3", -3, dict(N_HT=2, N_CR=0, N_Tot=18, N_Det=14, M_cur=4)),
    ("C4", -3, dict(N_HT=1, N_CR=1, N_Tot=24, N_Det=21, M_cur=5)),
    ("C5", -3, dict(N_HT=5, N_CR=3, N_Tot=26, N_Det=17, M_cur=11)),
]
def red_mask(img):
    b, g, r = img[..., 0].astype(int), img[..., 1].astype(int), img[..., 2].astype(int)
    return (r > 150) & (g < 110) & (b < 110) & (r - g > 80)


def _norm(glyph):
    h, w = glyph.shape
    nw = max(1, int(round(w * 22 / h)))
    g = cv2.resize(glyph.astype(np.uint8) * 255, (min(nw, CANVAS), 22), interpolation=cv2.INTER_AREA)
    c = np.zeros((CANVAS, CANVAS), np.float32)
    x0 = (CANVAS - g.shape[1]) // 2
    c[1:23, x0:x0 + g.shape[1]] = g / 255.0
    return c


TEMPL = {}   # digit -> list of normalised glyphs (filled by build_templates())


def classify(glyph):
    g = _norm(glyph)
    best, bd = None, 1e9
    for d, lst in TEMPL.items():
        for t in lst:
            dist = np.abs(g - t).mean()
            if dist < bd:
                bd, best = dist, d
    return best, bd


def split_lines(mask, min_gap=2):
    prof = mask.sum(1) > 0
    lines, start, gap = [], None, 0
    for y, v in enumerate(prof):
        if v:
            if start is None:
                start = y
            gap = 0
        elif start is not None:
            gap += 1
            if gap >= min_gap:
                lines.append((start, y - gap + 1))
                start, gap = None, 0
    if start is not None:
        lines.append((start, len(prof)))
    return [l for l in lines if l[1] - l[0] >= 12]


def line_glyphs(line_mask):
    """glyph bitmaps to the right of the last colon of a text line"""
    ncc, lab, st, _ = cv2.connectedComponentsWithStats(line_mask.astype(np.uint8), connectivity=8)
    comps = [tuple(st[i, :4]) for i in range(1, ncc) if st[i, 4] >= 2]
    if not comps:
        return None
    comps.sort(key=lambda c: c[0])
    glyphs = []
    for x, y, w, h in comps:
        if glyphs and x <= glyphs[-1][1]:
            g = glyphs[-1]
            g[1] = max(g[1], x + w); g[2] = min(g[2], y); g[3] = max(g[3], y + h); g[4] += 1
            g[5].append(h)
        else:
            glyphs.append([x, x + w, y, y + h, 1, [h]])
    colon_idx = None
    for i, (x0, x1, y0, y1, k, hs) in enumerate(glyphs):
        if k >= 2 and (x1 - x0) <= 8 and max(hs) <= 8:
            colon_idx = i
    if colon_idx is None:
        return None
    return [line_mask[y0:y1, x0:x1] for x0, x1, y0, y1, k, hs in glyphs[colon_idx + 1:]]


def read_line_number(line_mask):
    gl = line_glyphs(line_mask)
    if not gl:
        return None, 1.0
    ds = [classify(g) for g in gl]
    return int("".join(d for d, _ in ds)), max(e for _, e in ds)


def line_masks(fr):
    W = fr.shape[1]
    rm = red_mask(fr[:140])
    out = {}
    for side, keys in (("L", ["N_HT", "N_CR"]), ("R", ["N_Tot", "N_Det", "M_cur"])):
        m = rm[:, :W // 2] if side == "L" else rm[:, W // 2:]
        for key, b in zip(keys, BASELINES[side]):
            out[key] = m[b - 24:b + 2]
    return out


def build_templates():
    TEMPL.clear()
    for vid, f, vals in KNOWN:
        cap = cv2.VideoCapture(str(ROOT / RESULT_VIDEOS[vid]))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, f if f >= 0 else n + f)
        ok, fr = cap.read()
        lm = line_masks(fr)
        for key, v in vals.items():
            gl = line_glyphs(lm[key])
            if gl is None or len(gl) != len(str(v)):
                print("template harvest mismatch", vid, f, key, v, None if gl is None else len(gl))
                continue
            for d, g in zip(str(v), gl):
                TEMPL.setdefault(d, []).append(_norm(g))
    if "9" not in TEMPL and "6" in TEMPL:   # 9 = 6 rotated by 180 degrees in this font
        TEMPL["9"] = [np.rot90(t, 2).copy() for t in TEMPL["6"]]
    return {d: len(v) for d, v in sorted(TEMPL.items())}


BASELINES = {"L": [61, 91], "R": [31, 61, 91]}   # text baselines (px) in all result videos


def read_frame(fr):
    out, err = {}, 0.0
    for key, m in line_masks(fr).items():
        v, e = read_line_number(m)
        out[key] = v
        err = max(err, e)
    out["ocr_err"] = round(err, 4)
    return out


def ocr_video(vid, rel, step=1):
    cap = cv2.VideoCapture(str(ROOT / rel))
    rows, f = [], 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if f % step == 0:
            rec = {"frame": f}
            rec.update(read_frame(fr))
            rows.append(rec)
        f += 1
    out = ROOT / "cache/ocr"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / f"{vid}.csv", index=False)
    return vid, f


if __name__ == "__main__":
    print("templates:", build_templates())
    vids = sys.argv[1:] or list(RESULT_VIDEOS)
    for v in vids:
        print(ocr_video(v, RESULT_VIDEOS[v]), flush=True)
