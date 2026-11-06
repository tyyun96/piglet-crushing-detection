"""Evaluate a YOLOv5 (original code base) model on a YOLO-format dataset; prints one JSON line.
usage: python v5_val.py <weights> <data.yaml> <imgsz>"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/yolov5"))
import val  # noqa: E402  (third_party/yolov5/val.py)

_ap = {}
_ap_per_class = val.ap_per_class


def _keep_ap(*a, **k):
    """val.run returns per-class mAP50-95 only; keep the per-class AP50 of its ap_per_class call"""
    out = _ap_per_class(*a, **k)
    _ap.update(ap50=out[5][:, 0], cls=out[6], names=k.get("names", {}))
    return out


val.ap_per_class = _keep_ap
w, data, imgsz = sys.argv[1], sys.argv[2], int(sys.argv[3])
(mp, mr, map50, map_, *_), maps, _ = val.run(data=data, weights=w, imgsz=imgsz, batch_size=16, task="val", plots=False,
                                              project=str(ROOT / "cache/detector_training_v5/val_tmp"), name="v",
                                              exist_ok=True, verbose=False)
res = dict(P=float(mp), R=float(mr), mAP50=float(map50), mAP50_95=float(map_))
for c, a in zip(_ap.get("cls", []), _ap.get("ap50", [])):
    res[f"AP50_{_ap['names'][int(c)]}"] = float(a)
print("RESULT " + json.dumps(res))
