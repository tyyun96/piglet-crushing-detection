# Hidden-trapping and suspected-crushing detection for piglets — analysis code

Code for the manuscript **"Detecting hidden piglet trapping from temporal changes in visible piglet counts in farrowing-pen video: An offline proof-of-concept study"**
(PLOS ONE, revised version of PONE-D-26-22779).
Dataset: Zenodo, DOI [10.5281/zenodo.23999721](https://doi.org/10.5281/zenodo.23999721).
Archived code: Zenodo, DOI [10.5281/zenodo.24076186](https://doi.org/10.5281/zenodo.24076186) (v1.0.0, the version used in the
paper; all versions: [10.5281/zenodo.24076185](https://doi.org/10.5281/zenodo.24076185)).

The pipeline detects piglets in overhead video of farrowing pens with a YOLO detector, counts them per pen and

- raises a **hidden-trapping alert (HDA, hidden-trapping detection algorithm)** when piglets stay missing after the count drops, optionally only after a burst of sow
  activity (HDA-G, sow-activity gate);
- raises a **trapping alert (TDA / TDA-G, trapping detection algorithm)** for a hidden piglet or a piglet visibly trapped under the sow (detector class
  `crushing`);
- raises a **suspected-crushing alert (SCA)** for a piglet track that re-appears during a hidden-trapping episode and stays
  motionless (post-hoc immobility detection, not a confirmation of death).

`analysis.ipynb` contains every analysis of the paper: reproduction of the submitted manuscript, detector-training leakage and
leakage-free detectors, evaluation on the continuous (unedited) recordings against the manuscript events, a blind annotation of
the whole footage (primary reference) and an adjudicated reference, leave-one-litter-out cross-validation (LOLO-CV), sensitivity
analyses, tracking quality (proxy measures; no MOT ground truth was made — the notebook computes MOTA, IDF1, HOTA and ID switches if one is added), runtime, an
independent farm, and annotation reliability.
Every table is written to `outputs/tables/*.csv` and every figure to `outputs/figures/*.pdf` (and `.png`).

## Repository

| path | content |
|---|---|
| `analysis.ipynb` | all analyses, executed (English) |
| `analysis/core.py` | algorithms: detection-cache access, pen regions, sow-activity proxy, SORT, the original HPA/CDA of the submitted manuscript (HDA-F / SCA-F in the paper), the revised HDA/TDA/SCA, event matching and metrics |
| `analysis/nb_src.py`, `analysis/build_notebook.py` | source of the notebook (jupytext, Korean and English text) and its builder: `python analysis/build_notebook.py --lang en` |
| `analysis/video_registry.py` | the videos: ID, camera, start time (camera clock) |
| `analysis/prep_*.py`, `analysis/v5_val.py` | preprocessing that needs the raw videos, the detector-training images or a GPU; their outputs are part of the dataset |
| `analysis/*_src.py`, `annotation_tools/` | the Jupyter tools the observer used for labelling alerts, annotating the continuous footage, adjudication and the test–retest re-annotation, and an unused tool for a MOT ground truth (user interface in Korean) |
| `setup_data.py` | puts the Zenodo dataset where the code expects it |
| `CITATION.cff` | citation metadata (title, authors, license) for GitHub's "Cite this repository" and the Zenodo archive |

### Names

The code keeps the internal names of the analysis (HPA = HDA, TPA = TDA, CDA = SCA; `orig` = the frame-based versions of the
submitted manuscript, `v2` = the refined versions). The result tables give the paper's names in the column `paper_name`
(`PAPER_NAME` in the notebook), except that the frame-based versions are still called `HDA-original` and `SCA-original`:

| paper | `method` in the code and tables | `paper_name` in the tables | Table 5 (`T9_same_clip_ablation_metrics`) |
|---|---|---|---|
| HDA-F, whole image | `HPA-orig, whole frame (manuscript)` | HDA-original (whole frame) | |
| HDA-F, per pen | `HPA-orig, per pen` | HDA-original (per pen) | `hpa_orig_pen` |
| HDA | `HPA-v2` | HDA | `hpa_v2` |
| HDA-G | `HPA-v2 + sow-activity gate` | HDA-G | `hpa_v2_gate` |
| image-based model | `Visible trapping (image-only)` | Visible trapping (image only) | `image_only` |
| TDA-F, per pen | | | `tpa_orig_pen` |
| TDA | `TPA-v2 (HPA-v2 OR visible)` | TDA | `tpa_v2` |
| TDA-G | `TPA-v2 + gate` | TDA-G | `tpa_v2_gate` |
| SCA-F, per pen | `CDA-orig (sort, t_expire=1 frame, f=300 fr, m=200 px)` | SCA-original | `cda_orig_pen` |
| SCA, no HDA gating | `CDA-v2, no HPA (tracker only)` | SCA (tracker only) | `cda_v2_noHPA` |
| SCA, HDA gating | `CDA-v2, HPA-gated` | SCA (HDA-gated) | `cda_v2_gated` |
| SCA, HDA-G gating | `CDA-v2, HPA+gate-gated` | SCA-G (HDA-G-gated) | `cda_v2_gated_gate` |
| SCA, crushed class required | `CDA-v2, HPA-gated + crushed class` | SCA (HDA-gated + crushed class) | `cda_v2_gated_crushedcls` |
| SCA, ByteTrack | `CDA-v2, HPA-gated, ByteTrack` | SCA (HDA-gated, ByteTrack) | `cda_v2_gated_bytetrack` |
| crushed-class detector; running-maximum difference; counting model (Mcur only) | | | `image_only_crushed`; `raw_runmax_diff`; `counting_orig` |

The previous study is Yun et al. (2025), *J Anim Sci Technol* 67:666, https://doi.org/10.5187/jast.2024.e106 — reference [14]
of the revised paper. The notebook was written for the submitted manuscript and calls it [9] (also in the tags `_r9…`).

## Reproduce the results

1. **Environment** — Python 3.11, `pip install -r analysis/requirements.txt` (exact versions of the analysis; the CUDA packages are
   needed only for detection and detector training).
2. **Data** — download the [dataset](https://doi.org/10.5281/zenodo.23999721) from Zenodo (11 files, about 1.3 GB) and unzip all
   archives into one folder, e.g. `../crushing_dataset/`.
3. **Place the data** — `python setup_data.py ../crushing_dataset` (symbolic links; `--copy` to copy).
4. **Run** — `jupyter nbconvert --to notebook --execute analysis.ipynb --output analysis_run.ipynb`, or open the notebook in
   Jupyter. The project folder is found from the working directory (or set `CRUSHING_ROOT`).
   Without the raw videos everything is recomputed from the per-frame detections of the dataset (about 25 min with 8 worker
   processes on the analysis server); four results that need the raw videos, the training images or a GPU (held-out detector
   evaluation, runtime benchmark) are read from `cache/precomputed/`, and the video frames shown in figures from `cache/frames/`.
   The tables can be compared with `reference_results/tables/` of the dataset.

The analysis is deterministic (same inputs → identical tables). Heavy intermediate results are cached in `cache/results/`
(set `RECOMPUTE = True` in the first code cell to recompute them).

### From the raw videos (only if the videos are available)

| step | script | output |
|---|---|---|
| detection on every frame (YOLOv8s / YOLOv5s, conf ≥ 0.05 stored) | `prep_02_detect.py` | `cache/det*/`, `cache/meta*/`, `cache/thumb/` |
| read the overlay numbers of the manuscript's result videos | `prep_03_ocr_overlay.py` | `cache/ocr/` |
| align the 1-min evaluation clips with the continuous recordings; trace the training images back to video frames | `prep_04_align.py` | `meta/clip_alignment.csv`, `meta/train_image_matches.csv` |
| leakage-free YOLOv8s (camera hold-out) | `prep_05_train_lolo_detectors.py` | `cache/detector_training/` |
| YOLOv5s retraining, detector selection by a pre-specified rule | `prep_06_train_yolov5.py` (+ `v5_val.py`) | `cache/detector_training_v5/` |
| audit of the previous study's dataset (its four GitHub archives) | `prep_11_previous_study_dataset.py` | `meta/previous_study_*` |
| the previous study's training recipe with and without leakage | `prep_12_train_previous_recipe.py` | `cache/detector_training_r9/` |
| review clips, annotation segments, adjudication and retest items, MOT pre-fill | `prep_07`–`prep_10`, `prep_13` | `outputs/…` |

YOLOv5 runs use [ultralytics/yolov5](https://github.com/ultralytics/yolov5) at commit `f9578796` cloned into `third_party/yolov5`.
The analysis server had an NVIDIA H100 NVL GPU; the notebook itself needs no GPU.

## Annotation tools

The human references were made with the notebooks in `annotation_tools/` (built from `analysis/*_src.py`); they read and write
the annotation files at the paths used by the analysis (`outputs/continuous_annotation/`, `outputs/alert_review/`,
`annotations/mot_gt/`). The observer is a Korean speaker, so the instructions and prompts are in Korean.

## License

Code: MIT (see `LICENSE`). The detectors are trained and run with Ultralytics YOLOv8 and YOLOv5, which are licensed under AGPL-3.0.
Dataset: CC BY 4.0 (see the Zenodo record).

## Citation

Article: [to be added after publication]

Dataset: Yun T, Yun J, Lee W. Hidden-trapping and suspected-crushing detection for piglets in farrowing pens: dataset [Data set].
Version 1.0. Zenodo; 2026. https://doi.org/10.5281/zenodo.23999721

Code: Yun T, Yun J, Lee W. Hidden-trapping and suspected-crushing detection for piglets: analysis code [Software].
Version 1.0.0. Zenodo; 2026. https://doi.org/10.5281/zenodo.24076186
