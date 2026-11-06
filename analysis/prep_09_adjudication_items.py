"""
Preprocessing step 9: items for the adjudication of discrepancies between the two human references
(observer labels of the TDA-G alerts vs the blind continuous annotation), and their video clips.

An item is an observer-confirmed alert episode (outputs/tables/T24_observer_confirmed_episodes_TPA_v2_gate.csv,
target litters) with no trapping/crushing/uncertain/found-at-release mark of the continuous annotation within 60 s,
plus marks whose type needs a check (listed in CHECK_MARKS). Writes
outputs/continuous_annotation/adjudication_items.csv and renders the clips (episode +/- 30 s).
usage: python prep_09_adjudication_items.py [--workers 2]
"""
import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import review_clips as rc  # noqa: E402

ROOT = rc.ROOT
DIR = ROOT / "outputs" / "continuous_annotation"
TARGET = ["A-R", "B-L"]
NEAR_S = 60.0
PAD_S = 30.0
# marks whose type the annotator wanted to check (B_f413: several piglets trapped at once, then rescued by a person)
CHECK_MARKS = [dict(rec="B_f413", pen="L", onset_s=257.0, question="type: crushing (death suspected) or trapping (rescued alive)?")]


def build_items():
    E = pd.read_csv(ROOT / "outputs" / "tables" / "T24_observer_confirmed_episodes_TPA_v2_gate.csv")
    E = E[E.litter.isin(TARGET)]
    A = pd.read_csv(DIR / "annotations.csv")
    A = A[(A.annotator == "observer1") & (A.type != "end")]
    rows = []
    for r in E.itertuples():
        near = A[(A.rec == r.rec) & (A.pen == r.pen) & (A.onset_s >= r.start - NEAR_S) & (A.onset_s <= r.end + NEAR_S)]
        if len(near):
            continue
        rows.append(dict(kind="alert_episode", rec=r.rec, pen=r.pen, litter=r.litter, hi_start_s=r.start, hi_end_s=r.end,
                         clip_start_s=max(0.0, r.start - PAD_S), clip_end_s=r.end + PAD_S,
                         question="continuation of an annotated event / new trapping missed / not a trapping / unclear?"))
    for c in CHECK_MARKS:
        m = A[(A.rec == c["rec"]) & (A.pen == c["pen"]) & (A.onset_s == c["onset_s"])]
        if not len(m):
            continue
        ends = pd.read_csv(DIR / "annotations.csv").query("rec == @c['rec'] and pen == @c['pen'] and type == 'end' and onset_s > @c['onset_s']")
        end = float(ends.onset_s.min()) if len(ends) else c["onset_s"] + 60.0
        rows.append(dict(kind="check_type", rec=c["rec"], pen=c["pen"], litter=m.litter.iloc[0], hi_start_s=c["onset_s"], hi_end_s=end,
                         clip_start_s=max(0.0, c["onset_s"] - 15.0), clip_end_s=end + 15.0, question=c["question"]))
    items = pd.DataFrame(rows).sort_values(["litter", "rec", "hi_start_s"]).reset_index(drop=True)
    items.insert(0, "item", range(1, len(items) + 1))
    return items


def job(a):
    rc.make_range_clip(*a)
    return a[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    items = build_items()
    items.to_csv(DIR / "adjudication_items.csv", index=False)
    print(len(items), "items:", items.kind.value_counts().to_dict(), flush=True)
    jobs = [(r.rec, r.pen, r.clip_start_s, r.clip_end_s, r.hi_start_s, r.hi_end_s) for r in items.itertuples()]
    with Pool(args.workers) as pool:
        for i, _ in enumerate(pool.imap_unordered(job, jobs), 1):
            print(i, "/", len(jobs), "clips", flush=True)
    print("finished", flush=True)


if __name__ == "__main__":
    main()
