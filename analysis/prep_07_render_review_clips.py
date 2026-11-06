"""
Preprocessing step 7: pre-render the review video (alert +/- 15 s, one pen) of every alert in the
review sheets, so that label_alerts.ipynb shows each video instantly.
usage: python prep_07_render_review_clips.py [--workers 2] [--half 15]
"""
import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import review_clips as rc  # noqa: E402


def job(a):
    rec, pen, t, half = a
    try:
        rc.make_clip(rec, pen, t, half=half)
        return None
    except Exception as e:      # keep going; the labeler renders missing clips on demand
        return f"{rec} {pen} {t}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--half", type=float, default=15)
    args = ap.parse_args()
    jobs = []
    sheets = sorted((rc.ROOT / "outputs/alert_review").glob("*/review_sheet.csv"),
                    key=lambda p: (not p.parent.name.startswith("TPA"), p.parent.name))     # trapping alerts first
    for sheet in sheets:
        d = pd.read_csv(sheet, dtype=str)
        jobs += [(r.rec, r.pen, float(r.alert_s), args.half) for r in d.itertuples()]
    jobs = [j for j in dict.fromkeys(jobs) if not rc.clip_path(j[0], j[1], j[2], j[3]).exists()]
    print(len(jobs), "clips to render", flush=True)
    with Pool(args.workers) as pool:
        for i, err in enumerate(pool.imap_unordered(job, jobs), 1):
            if err:
                print("ERROR", err, flush=True)
            if i % 25 == 0:
                print(i, "done", flush=True)
    print("finished", flush=True)


if __name__ == "__main__":
    main()
