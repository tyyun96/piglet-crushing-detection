"""
Preprocessing step 11: audit of the detector dataset of the previous study [9] (reviewer comment R1-3).

Inputs (project folder)
  image_augmentation-main.zip, yolov4_piglet_crushing-main.zip, yolov5_piglet_crushing-main.zip,
  yolov8_piglet_crushing-main.zip   GitHub archives of [9] (the zip comment is the commit hash)
  data/new data origin/             the 204 labelled originals 'train (i).jpg/.txt'
  train-20230728T031826Z-001.zip    run folder of the provided YOLOv8s (args.yaml, results.csv)
  meta/train_image_matches.csv, meta/clip_alignment.csv, cache/thumb/   (prep_04 / prep_02)

How [9] built its dataset (scripts of image_augmentation-main):
  every original 'train (i)' (i = 1..204) + 7 copies (ColorJitter brightness / contrast / saturation / hue, three
  random-noise patches), renamed 'train (N)' with N = 8(i-1)+1 .. 8i (1632 images); each of these rotated by
  90/180/270 degrees and flipped left-right / top-bottom (suffix _90/_180/_270/_LR/_TB) -> 9792 images; split 70/30
  at random over the 9792 images (sklearn train_test_split, random_state=1000) -> dataset/train.txt, val.txt.

Steps
  1. map every image of the [9] split lists to its original (i = (N-1)//8 + 1) and verify the mapping with the labels
     stored in the YOLOv5 label caches of the archive (the 8 copies carry the boxes of the original, the rotated /
     flipped images the transformed boxes);
  2. count the copies of every original in the training and validation lists (YOLOv5 and YOLOv8 splits of [9]);
  3. locate the demo video of [9] (yolov8 archive, dataset/video/test_video1.mp4) in the recordings (thumbnail matching);
  4. record the training settings written in the archives and in the run folder of the provided YOLOv8s.
Outputs: meta/previous_study_images.csv (one row per image and split list), meta/previous_study_originals.csv
(one row per original), meta/previous_study_summary.json
"""
import hashlib
import io
import json
import pickle
import re
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
from numpy.lib import format as npf
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from video_registry import VIDEOS  # noqa: E402

ARCHIVES = {"augmentation": "image_augmentation-main.zip", "yolov4": "yolov4_piglet_crushing-main.zip",
            "yolov5": "yolov5_piglet_crushing-main.zip", "yolov8": "yolov8_piglet_crushing-main.zip"}
PROVIDED = "train-20230728T031826Z-001.zip"
ORIG = ROOT / "data/new data origin"
N_ORIG, N_COPIES = 204, 8                      # 8 = original + 7 colour / noise copies
NAME = re.compile(r"train \((\d+)\)(_90|_180|_270|_LR|_TB)?")


class SafeUnpickler(pickle.Unpickler):
    """the label caches are dicts pickled by np.save; only numpy arrays and plain containers may be rebuilt"""
    ALLOWED = {("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
               ("numpy", "ndarray"), ("numpy", "dtype"), ("numpy.core.multiarray", "scalar"),
               ("numpy._core.multiarray", "scalar"), ("builtins", "set"), ("builtins", "frozenset"),
               ("collections", "OrderedDict")}

    def find_class(self, module, name):
        if (module, name) in self.ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked global {module}.{name}")


def read_label_cache(data):
    f = io.BytesIO(data)
    version = npf.read_magic(f)
    _, _, dtype = npf.read_array_header_1_0(f) if version == (1, 0) else npf.read_array_header_2_0(f)
    assert dtype.hasobject, dtype
    obj = SafeUnpickler(f).load()
    return obj.item() if getattr(obj, "shape", None) == () else obj


def transformed(lb, t):
    """boxes [cls, x, y, w, h] (normalised) after the PIL transpose of image_rotation.py"""
    c, x, y, w, h = lb.T
    if t == "_90":                      # ROTATE_90 = counter-clockwise
        x, y, w, h = y, 1 - x, h, w
    elif t == "_180":
        x, y = 1 - x, 1 - y
    elif t == "_270":
        x, y, w, h = 1 - y, x, h, w
    elif t == "_LR":
        x = 1 - x
    elif t == "_TB":
        y = 1 - y
    return np.stack([c, x, y, w, h], 1)


def same_boxes(a, b, tol=1e-3):
    if len(a) != len(b):
        return False
    if len(a) == 0:
        return True
    d = np.abs(a[:, None, 1:] - b[None, :, 1:]).sum(-1) + 1e3 * (a[:, None, 0] != b[None, :, 0])
    r, c = linear_sum_assignment(d)
    return bool((a[r, 0] == b[c, 0]).all() and np.abs(a[r] - b[c]).max() < tol)


def nb_cells(zf, member):
    nb = json.loads(zf.read(member))
    out = []
    for c in nb["cells"]:
        src = "".join(c.get("source", []))
        txt = "".join("".join(o.get("text", [])) for o in c.get("outputs", []) if o.get("output_type") == "stream")
        out.append((src, txt))
    return out


def metric_lines(text):
    """final 'Class Images Instances P R mAP50 mAP50-95' rows printed by YOLOv5 / YOLOv8"""
    rows = re.findall(r"^\s*(all|piglet|crushing|crushed)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$",
                      re.sub(r"\x1b\[[0-9;]*m", "", text), flags=re.M)
    last = {}
    for r in rows:                                      # the last block is the final validation
        last[r[0]] = dict(images=int(r[1]), instances=int(r[2]), P=float(r[3]), R=float(r[4]), mAP50=float(r[5]),
                          mAP50_95=float(r[6]))
    return last


def locate_video(data):
    """thumbnail matching (64x36 grey, mean-centred mean absolute difference) against all continuous recordings"""
    import cv2
    cv2.setNumThreads(4)
    with tempfile.TemporaryDirectory(dir=ROOT / "cache") as td:
        p = Path(td) / "query.mp4"
        p.write_bytes(data)
        cap = cv2.VideoCapture(str(p))
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        th = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            th.append(cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), (64, 36), interpolation=cv2.INTER_AREA))
        cap.release()
    q = np.stack(th).astype(np.float32)
    q -= q.mean(axis=(1, 2), keepdims=True)
    probes = np.linspace(int(fps), len(q) - int(fps), 12).astype(int)
    res = []
    for vid, v in VIDEOS.items():
        tp = ROOT / "cache/thumb" / f"{vid}.npy"
        if v["kind"] != "raw" or not tp.exists():
            continue
        T = np.load(tp, mmap_mode="r")
        f2 = json.load(open(ROOT / "cache/meta" / f"{vid}.json"))["fps"]
        best = []
        for k in probes:
            bm = (np.inf, -1)
            for s in range(0, len(T), 20000):
                B = T[s:s + 20000].astype(np.float32)
                B -= B.mean(axis=(1, 2), keepdims=True)
                d = np.abs(B - q[k]).mean(axis=(1, 2))
                j = int(d.argmin())
                if d[j] < bm[0]:
                    bm = (float(d[j]), s + j)
            best.append((k / fps, bm[1] / f2, bm[0]))
        b = np.array(best)
        res.append(dict(video=vid, mad=float(np.median(b[:, 2])), offset_s=float(np.median(b[:, 1] - b[:, 0])),
                        offset_sd_s=float(np.std(b[:, 1] - b[:, 0]))))
    res = sorted(res, key=lambda r: r["mad"])
    return dict(fps=fps, width=size[0], height=size[1], n_frames=len(q), duration_s=len(q) / fps, best=res[0],
                runner_up=res[1])


def main():
    zs = {k: zipfile.ZipFile(ROOT / v) for k, v in ARCHIVES.items()}
    summary = {"archives": {k: dict(file=ARCHIVES[k], commit=zs[k].comment.decode(),
                                    date=max("%04d-%02d-%02d" % i.date_time[:3] for i in zs[k].infolist()),
                                    files=[i.filename for i in zs[k].infolist() if not i.is_dir()]) for k in zs}}

    # ---- 1. labels of all 9792 images (YOLOv5 label caches) and of the 204 originals -------------------------
    v5 = "yolov5_piglet_crushing-main/dataset/"
    labels = {}
    for split in ("train", "val"):
        for k, v in read_label_cache(zs["yolov5"].read(v5 + f"{split}.cache")).items():
            if k.endswith(".jpg"):
                labels[PurePosixPath(k).stem] = np.asarray(v[0], dtype=float).reshape(-1, 5)
    orig = {}
    for t in ORIG.glob("train (*).txt"):
        i = int(re.fullmatch(r"train \((\d+)\)", t.stem).group(1))
        a = np.loadtxt(t, ndmin=2).reshape(-1, 5) if t.stat().st_size else np.zeros((0, 5))
        orig[i] = np.unique(a, axis=0)                   # YOLOv5 drops duplicate rows when it builds the cache
    assert sorted(orig) == list(range(1, N_ORIG + 1)), "data/new data origin must hold train (1) .. train (204)"
    check = []
    for stem, lb in labels.items():
        m = NAME.fullmatch(stem)
        n, t = int(m.group(1)), m.group(2) or ""
        check.append(same_boxes(lb, transformed(orig[(n - 1) // N_COPIES + 1], t)))
    summary["label_check"] = dict(images_in_label_caches=len(labels), labels_match_original=int(sum(check)),
                                  max_N=max(int(NAME.fullmatch(s).group(1)) for s in labels))
    print("label check:", summary["label_check"], flush=True)

    # ---- 2. split lists -> originals --------------------------------------------------------------------------
    rows = []
    for repo in ("yolov5", "yolov8"):
        for split in ("train", "val"):
            lines = zs[repo].read(f"{repo}_piglet_crushing-main/dataset/{split}.txt").decode().splitlines()
            for line in filter(None, map(str.strip, lines)):
                stem = PurePosixPath(line).stem
                m = NAME.fullmatch(stem)
                n, t = int(m.group(1)), m.group(2) or ""
                i = (n - 1) // N_COPIES + 1
                rows.append(dict(split_list=repo, split=split, image=stem, N=n, original=f"train ({i})", i=i,
                                 copy=(n - 1) % N_COPIES, transform=t.strip("_") or "none",
                                 labels_match=stem in labels and same_boxes(labels[stem], transformed(orig[i], t))))
    img = pd.DataFrame(rows)
    img.drop(columns="i").to_csv(ROOT / "meta/previous_study_images.csv", index=False)
    cnt = img.groupby(["i", "split_list", "split"]).size().unstack(["split_list", "split"], fill_value=0)
    cnt.columns = [f"{a}_{b}" for a, b in cnt.columns]
    per = pd.DataFrame({"i": range(1, N_ORIG + 1)})
    per["original"] = [f"train ({i})" for i in per.i]
    per["n_boxes"] = [len(orig[i]) for i in per.i]
    per = per.merge(cnt.reset_index(), on="i", how="left").fillna(0)
    per[[c for c in per.columns if c.startswith("yolov")]] = per[[c for c in per.columns if c.startswith("yolov")]].astype(int)
    per.drop(columns="i").to_csv(ROOT / "meta/previous_study_originals.csv", index=False)
    for repo in ("yolov5", "yolov8"):
        tr = per.set_index("i")[f"{repo}_train"]
        va = img[(img.split_list == repo) & (img.split == "val")]
        c = tr.loc[va.i].to_numpy()
        summary[f"split_{repo}"] = dict(n_train=int((img[(img.split_list == repo)].split == "train").sum()), n_val=len(va),
                                        originals_with_train_copies=int((tr > 0).sum()),
                                        originals_with_val_copies=int((per[f"{repo}_val"] > 0).sum()),
                                        originals_only_in_val=int(((tr == 0) & (per.set_index("i")[f"{repo}_val"] > 0)).sum()),
                                        val_images_with_same_frame_in_train=int((c > 0).sum()),
                                        train_copies_per_val_image=dict(min=int(c.min()), median=float(np.median(c)), max=int(c.max())))
        print(repo, summary[f"split_{repo}"], flush=True)
    s5 = {k: set(img[(img.split_list == "yolov5") & (img.split == k)].image) for k in ("train", "val")}
    s8 = {k: set(img[(img.split_list == "yolov8") & (img.split == k)].image) for k in ("train", "val")}
    summary["yolov5_vs_yolov8_split"] = dict(same_train_list=s5["train"] == s8["train"],
                                             train_overlap=len(s5["train"] & s8["train"]))
    md5 = lambda z, m: hashlib.md5(z.read(m)).hexdigest()
    summary["yolov8_label_caches_are_yolov5_copies"] = all(
        md5(zs["yolov5"], v5 + f"{s}.cache") == md5(zs["yolov8"], f"yolov8_piglet_crushing-main/dataset/{s}.cache")
        for s in ("train", "val"))
    u8 = read_label_cache(zs["yolov8"].read("yolov8_piglet_crushing-main/dataset/export/labels.cache"))
    u8_imgs = {PurePosixPath(d["im_file"]).stem for d in u8["labels"]}
    summary["ultralytics_label_cache"] = dict(n_images=len(u8_imgs), equals_yolov8_val_list=u8_imgs == s8["val"],
                                              version=str(u8.get("version")))

    # ---- 3. demo video of [9] ------------------------------------------------------------------------------
    loc = locate_video(zs["yolov8"].read("yolov8_piglet_crushing-main/dataset/video/test_video1.mp4"))
    b = loc["best"]
    start, end = b["offset_s"], b["offset_s"] + loc["duration_s"]
    al = pd.read_csv(ROOT / "meta/clip_alignment.csv")
    loc["overlapping_clips"] = [r.clip for r in al.itertuples()
                                if r.source == b["video"] and r.offset_s < end and r.offset_s + r.clip_len_s > start]
    tm = pd.read_csv(ROOT / "meta/train_image_matches.csv")
    fps_src = json.load(open(ROOT / "cache/meta" / f"{b['video']}.json"))["fps"]
    tm = tm[(tm.best_video == b["video"]) & (tm.mad < 5)]
    t_img = tm.best_frame / fps_src
    loc["training_originals_inside"] = sorted(tm[(t_img >= start) & (t_img <= end)].image.str[:-4],
                                              key=lambda s: int(re.search(r"\d+", s).group()))
    loc["window_s"] = [round(start, 1), round(end, 1)]
    t0 = pd.Timestamp(VIDEOS[b["video"]]["start"])
    loc["window_clock"] = [str(t0 + pd.Timedelta(seconds=start))[:19], str(t0 + pd.Timedelta(seconds=end))[:19]]
    summary["demo_video_test_video1"] = loc
    print("test_video1:", b["video"], loc["window_s"], loc["overlapping_clips"], flush=True)

    # ---- 4. training settings written in the archives ------------------------------------------------------
    aug = zs["augmentation"].read("image_augmentation-main/image_augmentation.py").decode()
    rot = zs["augmentation"].read("image_augmentation-main/image_rotation.py").decode()
    summary["augmentation"] = dict(
        color_jitter=re.findall(r"ColorJitter\((\w+=[\d.]+)\)", aug),
        noise_patches=dict(n=len(range(*map(int, re.search(r"for j in range\((\d+),\s*(\d+)\)", aug).groups()))),
                           region_px=int(re.search(r"random\.sample\(range\((\d+)\)", aug).group(1))),
        geometric=sorted(set(re.findall(r"Image\.(ROTATE_\d+|FLIP_\w+)", rot))),
        jpeg_save="PIL default (quality 75)")
    cmds = {}
    for repo, member in (("yolov5", "yolov5_piglet_crushing-main/yolov5_piglet_crushing.ipynb"),
                         ("yolov8", "yolov8_piglet_crushing-main/yolov8_piglet_cushing.ipynb")):
        cells = nb_cells(zs[repo], member)
        cmds[repo] = dict(
            split=[l.strip() for s, _ in cells for l in s.splitlines() if "train_test_split(" in l and "=" in l],
            train=[l.strip() for s, _ in cells for l in s.splitlines() if "train.py" in l or "model.train(" in l],
            val=[l.strip() for s, _ in cells for l in s.splitlines() if "val.py" in l],
            final_validation=[dict(command=[l.strip() for l in s.splitlines() if "train.py" in l or "val.py" in l
                                            or "model.train(" in l], metrics=metric_lines(o))
                              for s, o in cells if metric_lines(o)])
    summary["notebooks"] = cmds
    cfg = zs["yolov4"].read("yolov4_piglet_crushing-main/yolov4-custom.cfg").decode()
    proc = zs["yolov4"].read("yolov4_piglet_crushing-main/process.py").decode()
    every = round(100 / int(re.search(r"percentage_test = (\d+)", proc).group(1)))
    summary["yolov4"] = dict(
        cfg={k: re.search(rf"^{k}\s*=\s*([^\n#]+)", cfg, flags=re.M).group(1).strip()
             for k in ("batch", "subdivisions", "width", "height", "max_batches", "steps", "classes")},
        split_rule=f"process.py: 1 of every {every} images (file-system order) of the augmented image folder -> test.txt")
    pz = zipfile.ZipFile(ROOT / PROVIDED)
    args = dict(re.findall(r"^(\w+): (.*)$", pz.read("train/args.yaml").decode(), flags=re.M))
    res = pd.read_csv(io.BytesIO(pz.read("train/results.csv")))
    res.columns = [c.strip() for c in res.columns]
    d8 = zs["yolov8"].read("yolov8_piglet_crushing-main/dataset/data.yaml").decode()
    summary["provided_yolov8s"] = dict(
        args={k: args[k] for k in ("model", "data", "epochs", "patience", "batch", "imgsz", "seed")},
        uses_yolov8_archive_dataset=args["data"].strip() == "/mydrive/yolov8_piglet_crushing/dataset/data.yaml",
        yolov8_data_yaml=dict(re.findall(r"^(train|val): (.*)$", d8, flags=re.M)),
        epochs_run=len(res), final_val=dict(mAP50=float(res["metrics/mAP50(B)"].iloc[-1]),
                                            mAP50_95=float(res["metrics/mAP50-95(B)"].iloc[-1])))
    json.dump(summary, open(ROOT / "meta/previous_study_summary.json", "w"), indent=1, ensure_ascii=False, default=str)
    print("written meta/previous_study_images.csv, meta/previous_study_originals.csv, meta/previous_study_summary.json")


if __name__ == "__main__":
    main()
