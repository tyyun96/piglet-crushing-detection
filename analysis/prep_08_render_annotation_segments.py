"""
Preprocessing step 8: cut the continuous recordings of each litter's pen into consecutive
segments (default 120 s, one pen, 12 fps, no alert information) for the blind continuous
annotation notebook outputs/continuous_annotation/annotate_continuous.ipynb.
usage: python prep_08_render_annotation_segments.py [--litters A-R B-L] [--seg 120] [--workers 3]
"""
import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import review_clips as rc  # noqa: E402
from video_registry import VIDEOS  # noqa: E402

LITTER_PEN = {"A-R": ("A", "R"), "B-L": ("B", "L"), "B-R": ("B", "R")}


def litter_jobs(litters, seg_s):
    jobs = []
    for lit in litters:
        cam, pen = LITTER_PEN[lit]
        recs = sorted((v for v in VIDEOS if VIDEOS[v]["kind"] == "raw" and VIDEOS[v]["camera"] == cam),
                      key=lambda v: VIDEOS[v]["start"])
        jobs += [(r, pen, seg_s) for r in recs]
    return jobs


def job(a):
    rec, pen, seg_s = a
    try:
        return f"{rec} {pen}: {rc.render_segments(rec, pen, seg_s=seg_s)} segments written"
    except Exception as e:      # keep going; report at the end
        return f"ERROR {rec} {pen}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--litters", nargs="+", default=["A-R", "B-L"])
    ap.add_argument("--seg", type=float, default=120)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    jobs = litter_jobs(args.litters, args.seg)
    print(len(jobs), "recordings", flush=True)
    with Pool(args.workers) as pool:
        for msg in pool.imap_unordered(job, jobs):
            print(msg, flush=True)
    print("finished", flush=True)


if __name__ == "__main__":
    main()
