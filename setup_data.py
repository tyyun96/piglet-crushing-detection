"""
Puts the Zenodo dataset where the code expects it.

usage:  python setup_data.py <dataset folder> [--copy]

<dataset folder> is the folder into which the dataset archives were unzipped (it contains annotations/, detections/, ...).
By default large files are linked (no extra disk space) and small text files (CSV, JSON, YAML) are copied, because the
notebook rewrites some of them; --copy copies everything. Existing files are never overwritten. Run it from the repository
folder (the one containing analysis/), then run analysis.ipynb.
"""
import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "analysis"))
from video_registry import VIDEOS  # noqa: E402

# dataset folder (relative to the dataset root)  ->  project folder (relative to the repository root)
FOLDERS = [
    ("annotations/continuous_annotation", "outputs/continuous_annotation"),
    ("annotations/alert_labels/TPA_v2_gate", "outputs/alert_review/TPA_v2_gate"),
    ("annotations/alert_labels/CDA_v2_HPA_activity_gated", "outputs/alert_review/CDA_v2_HPA_activity_gated"),
    ("annotations/mot_ground_truth", "annotations/mot_gt"),
    ("annotations/mot_prefill", "outputs/mot_annotation"),
    ("sow_activity_thumbnails", "cache/thumb"),
    ("overlay_ocr", "cache/ocr"),
    ("metadata", "meta"),
    ("precomputed", "cache/precomputed"),
    ("figure_frames", "cache/frames"),
    ("detectors/provided_yolov8s", "extracted/train/train"),
    ("detectors/detector_training", "cache/detector_training"),
    ("detectors/detector_training_v5", "cache/detector_training_v5"),
    ("detectors/detector_training_r9", "cache/detector_training_r9"),
    ("reference_results/tables", "reference_results/tables"),
    ("reference_results/figures", "reference_results/figures"),
    # optional parts (only if they are part of the dataset)
    ("detector_training_images", "data/new data origin"),
    ("mot_frames", "outputs/mot_annotation_template"),
]


SMALL = {".csv", ".json", ".yaml", ".txt"}     # always copied: the notebook rewrites some of them (e.g. alert review sheets)


def place(src, dst, copy):
    """link / copy every file below src to the same relative place below dst"""
    n = 0
    for f in sorted(p for p in src.rglob("*") if p.is_file()):
        t = dst / f.relative_to(src)
        if t.exists() or t.is_symlink():
            continue
        t.parent.mkdir(parents=True, exist_ok=True)
        if copy or f.suffix in SMALL:
            shutil.copy2(f, t)
        else:
            t.symlink_to(f.resolve())
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--copy", action="store_true")
    a = ap.parse_args()
    ds = Path(a.dataset).resolve()
    if not (ds / "annotations").exists():
        sys.exit(f"{ds} does not look like the unzipped dataset (no annotations/ folder)")
    total = 0
    for s, d in FOLDERS:
        if (ds / s).exists():
            n = place(ds / s, ROOT / d, a.copy)
            total += n
            print(f"{s:55s} -> {d}  ({n} files)")
    det = ds / "detections"                       # det<tag>/ and meta<tag>/ -> cache/
    for sub in sorted(p for p in det.iterdir() if p.is_dir()) if det.exists() else []:
        n = place(sub, ROOT / "cache" / sub.name, a.copy)
        total += n
        print(f"detections/{sub.name:44s} -> cache/{sub.name}  ({n} files)")
    vids = ds / "videos"                          # optional: <video id>.mp4 -> path in video_registry.py
    if vids.exists():
        n = 0
        for v, info in VIDEOS.items():
            f = vids / f"{v}.mp4"
            t = ROOT / info["path"]
            if f.exists() and not (t.exists() or t.is_symlink()):
                t.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, t) if a.copy else t.symlink_to(f.resolve())
                n += 1
        total += n
        print(f"{'videos':55s} -> data/ (paths of video_registry.py)  ({n} files)")
    print(f"done: {total} files {'copied' if a.copy else 'linked'}. Now run analysis.ipynb (see README.md).")


if __name__ == "__main__":
    main()
