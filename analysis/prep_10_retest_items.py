"""
Preprocessing step 10: segments for the test-retest re-annotation of the continuous annotation (same observer).
Stratified sample: every 2-min segment in which the first pass (observer1) placed a trapping / crushing / uncertain /
found-at-release mark, plus the same number of randomly drawn segments without any mark (seed 0); shown in random order.
Each clip shows the segment plus 30 s of context before it (the segment itself is highlighted in yellow).
Writes outputs/continuous_annotation/retest_items.csv and renders the clips.
usage: python prep_10_retest_items.py [--workers 2]
"""
import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import review_clips as rc  # noqa: E402

DIR = rc.ROOT / "outputs" / "continuous_annotation"
CONTEXT_S = 30.0
SEED = 0


def build_items():
    seg = pd.read_csv(DIR / "segments.csv").fillna({"reviewed_at": ""})
    seg = seg[(seg.annotator == "observer1") & (seg.reviewed_at.astype(str) != "")].reset_index(drop=True)
    ann = pd.read_csv(DIR / "annotations.csv")
    ann = ann[(ann.annotator == "observer1") & ann.type.isin(["trapping", "crushing", "uncertain", "found_at_release"])]
    has = [bool(((ann.rec == s.rec) & (ann.pen == s.pen) & (ann.onset_s >= s.seg_start_s) & (ann.onset_s < s.seg_end_s)).any())
           for s in seg.itertuples()]
    seg["stratum"] = ["event segment" if h else "no-event segment" for h in has]
    ev = seg[seg.stratum == "event segment"]
    ne = seg[seg.stratum == "no-event segment"].sample(n=min(len(ev), int((~pd.Series(has)).sum())), random_state=SEED)
    items = pd.concat([ev, ne]).sample(frac=1.0, random_state=SEED).reset_index(drop=True)     # shuffled order
    items.insert(0, "item", range(1, len(items) + 1))
    items["clip_start_s"] = (items.seg_start_s - CONTEXT_S).clip(lower=0.0)
    return items[["item", "stratum", "litter", "rec", "pen", "seg_start_s", "seg_end_s", "clip_start_s"]]


def job(a):
    rc.make_range_clip(*a)
    return a[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    items = build_items()
    out = DIR / "retest_items.csv"
    if out.exists():                                   # never reshuffle once the re-annotation may have started
        items = pd.read_csv(out)
    else:
        items.to_csv(out, index=False)
    print(len(items), "items:", items.stratum.value_counts().to_dict(), flush=True)
    jobs = [(r.rec, r.pen, r.clip_start_s, r.seg_end_s, r.seg_start_s, r.seg_end_s) for r in items.itertuples()]
    with Pool(args.workers) as pool:
        for i, _ in enumerate(pool.imap_unordered(job, jobs), 1):
            if i % 5 == 0 or i == len(jobs):
                print(i, "/", len(jobs), "clips", flush=True)
    print("finished", flush=True)


if __name__ == "__main__":
    main()
