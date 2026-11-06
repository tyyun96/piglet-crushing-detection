# ---
# jupyter:
#   kernelspec:
#     display_name: Python (crushing .venv)
#     language: python
#     name: crushing-venv
# ---

# %% [markdown]
# # 추적 정답 만들기 (MOT ground truth) — T6 클립 · B-L 펜
#
# 리뷰어(R1-10)가 요구한 **추적 성능 지표(IDF1 · MOTA · ID switch · HOTA)** 를 계산하기 위한 정답이다.
# 클립 T6(60초)를 **0.5초마다 한 장씩, 121장** 본다. 검출기가 박스를, 간단한 연결기가 번호를 **미리 채워 두었으므로
# 프레임마다 확인하고 틀린 것만 고친다**. 고칠 때마다 바로 저장되고, 언제든 `q` 로 멈췄다가 다시 실행하면 이어진다.
# 121장을 다 확인하면 정답 파일(`annotations/mot_gt/T6_L.csv`)이 만들어지고, 분석 노트북 §13 이 추적기를 이 정답으로 평가한다.
#
# ## 기준
# - **왼쪽 펜(B-L)의 자돈만**. 몸이 조금이라도 보이면 박스가 있어야 한다. 모돈 밑에 완전히 가려진 자돈은 박스 없음.
# - **같은 자돈 = 같은 번호** (클립 끝까지). 가려졌다가 다시 나온 자돈이 누구인지 **확실하면** 원래 번호로 바꾸고,
#   **확실하지 않으면 새 번호를 그대로 둔다**(추측하지 않는다).
# - 박스는 보이는 부분 기준. 미리 채운 박스가 대체로 맞으면 그대로 둔다(크게 틀린 것만 지우고 다시 넣는다).
# - 모돈, 사람 손, 오른쪽 펜의 자돈은 박스 없음.
#
# ## 화면
# - **왼쪽 큰 그림 = 지금 프레임, 오른쪽 작은 그림 = 바로 앞 프레임(0.5초 전)**. 같은 자돈에 같은 번호가 붙어 있는지 비교한다.
#   두 그림은 자돈이 있는 부분만 확대해 보여 주고, 오른쪽 아래 작은 그림이 펜 전체(노란 틀 = 확대한 부분)다. 번호는 박스 가운데에 있다.
# - 색 박스와 숫자 = 자돈 번호. **노란 테두리 = 이 프레임에서 처음 나온 번호**. 회색 점선과 글자(a, b, …) = 검출기가 확신이 낮아 빼 둔 후보.
# - 그림 아래에 처음 나온 번호마다 **"아마 이 번호" 후보**(조금 전에 사라진 번호 중 가까운 것)가 나온다. 맞으면 `36=30` 처럼 잇는다.
# - 헷갈리면 `v` 로 앞뒤 2초 영상을 본다.
#
# ## 입력 (아무것도 안 쓰고 Enter = 이 프레임 확인 끝 → 다음)
# | 입력 | 뜻 |
# |---|---|
# | `36=30` | 번호 36을 30으로 바꾼다 (이 프레임부터 끝까지) — 끊긴 번호 잇기 |
# | `36=30!` | 이 프레임에서만 바꾼다 |
# | `5<>12` | 5와 12를 맞바꾼다 (이 프레임부터 끝까지) — 두 자돈의 번호가 서로 바뀐 경우 |
# | `-7` | 이 프레임에서 7번 박스를 지운다 (자돈이 아님 / 한 자돈에 박스 두 개) |
# | `-7*` | 이 프레임부터 끝까지 7번을 지운다 |
# | `+a` / `+a 12` | 후보 a 를 새 번호로 / 12번으로 추가 |
# | `+12` | 보이는데 박스가 없는 12번을 앞뒤 프레임의 12번 박스로 보간해 추가 |
# | `+box 300 400 420 470 12` | 좌표(x1 y1 x2 y2)로 직접 12번 추가 — `grid` 로 좌표 눈금을 켜고 끈다 |
# | `v` | 앞뒤 2초 영상 보기 |
# | `full` | 확대(자돈이 있는 부분만) ↔ 펜 전체 보기 |
# | `m 메모` | 이 프레임 메모 |
# | `u` | 방금 바꾼 것 되돌리기 |
# | `b` 이전 · `g 37` 37번째 장으로 · `q` 종료 | |

# %%
CLIP, PEN = "T6", "L"
ANNOTATOR = "observer1"
CUR_W, PREV_W = 640, 420          # 그림 폭(px). 화면이 좁으면 줄인다

# %%
import base64
import json
import os
import sys
import warnings
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import pandas as pd
from IPython.display import HTML, clear_output, display

warnings.filterwarnings("ignore", category=FutureWarning)
ROOT = Path(os.environ.get("CRUSHING_ROOT") or next(q for q in [Path.cwd(), *Path.cwd().parents] if (q / "analysis" / "core.py").exists()))   # project folder (contains analysis/)
sys.path.insert(0, str(ROOT / "analysis"))
from video_registry import VIDEOS  # noqa: E402

WORK = ROOT / "outputs" / "mot_annotation" / f"{CLIP}_{PEN}"          # prep_13_mot_prefill.py
FRAMES = ROOT / "outputs" / "mot_annotation_template" / CLIP
GT_DIR = ROOT / "annotations" / "mot_gt"
META = json.load(open(WORK / "meta.json"))
X0, X1 = META["roi"]
FPS, FR = META["fps"], META["frames"]
N = len(FR)
COLS = ["frame", "id", "x1", "y1", "x2", "y2", "source"]
FONT = cv2.FONT_HERSHEY_SIMPLEX
TAB20 = ["1f77b4", "ff7f0e", "2ca02c", "d62728", "9467bd", "8c564b", "e377c2", "7f7f7f", "bcbd22", "17becf",
         "aec7e8", "ffbb78", "98df8a", "ff9896", "c5b0d5", "c49c94", "f7b6d2", "c7c7c7", "dbdb8d", "9edae5"]
PAL = [tuple(int(h[i:i + 2], 16) for i in (4, 2, 0)) for h in TAB20]       # BGR
CAND = pd.read_csv(WORK / "candidates.csv")


def iou(a, b):
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda c: (c[:, 2] - c[:, 0]) * (c[:, 3] - c[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def load_state():
    b = pd.read_csv(WORK / "work.csv") if (WORK / "work.csv").exists() else pd.read_csv(WORK / "prefill.csv")[COLS]
    st = (json.load(open(WORK / "state.json")) if (WORK / "state.json").exists()
          else dict(annotator=ANNOTATOR, started=pd.Timestamp.now().isoformat(timespec="seconds"), reviewed={}, memo={}, log=[]))
    return b, st


def save_state(b, st):
    b.sort_values(["frame", "id"]).to_csv(WORK / "work.csv", index=False)
    json.dump(st, open(WORK / "state.json", "w"), ensure_ascii=False, indent=1)


def candidates(f, b):
    c = CAND[CAND.frame == f]
    bb = b[b.frame == f][["x1", "y1", "x2", "y2"]].to_numpy(float)
    if len(c) and len(bb):
        c = c[iou(c[["x1", "y1", "x2", "y2"]].to_numpy(float), bb).max(1) < 0.3]
    c = c.reset_index(drop=True)
    return c.assign(letter=[chr(97 + i) for i in range(len(c))])


def dashed(img, p1, p2, col, step=8):
    (x1, y1), (x2, y2) = p1, p2
    for x in range(x1, x2, 2 * step):
        cv2.line(img, (x, y1), (min(x + step, x2), y1), col, 1); cv2.line(img, (x, y2), (min(x + step, x2), y2), col, 1)
    for y in range(y1, y2, 2 * step):
        cv2.line(img, (x1, y), (x1, min(y + step, y2)), col, 1); cv2.line(img, (x2, y), (x2, min(y + step, y2)), col, 1)


def region(boxes, margin=70, min_size=380):
    """part of the pen to show: all boxes of the current and previous frame (and candidates) plus a margin"""
    if not len(boxes):
        return X0, 0, X1, 1080
    x1, y1 = boxes.x1.min() - margin, boxes.y1.min() - margin
    x2, y2 = boxes.x2.max() + margin, boxes.y2.max() + margin
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = max(x2 - x1, min_size), max(y2 - y1, min_size)
    x1, x2 = cx - w / 2, cx + w / 2
    y1, y2 = cy - h / 2, cy + h / 2
    sx = max(0, X0 - x1) - max(0, x2 - X1); sy = max(0, -y1) - max(0, y2 - 1080)      # shift back inside the pen
    x1, x2, y1, y2 = max(X0, x1 + sx), min(X1, x2 + sx), max(0, y1 + sy), min(1080, y2 + sy)
    return int(x1), int(y1), int(x2), int(y2)


def render(f, boxes, width, new=(), dup=(), cand=None, grid=False, reg=None, max_scale=2.5):
    img = cv2.imread(str(FRAMES / f"frame_{f:05d}.jpg"))
    if img.shape[1] != 1920:
        img = cv2.resize(img, (1920, 1080))
    rx1, ry1, rx2, ry2 = reg or (X0, 0, X1, 1080)
    img = img[ry1:ry2, rx1:rx2].copy()
    s = min(width / img.shape[1], max_scale)
    img = cv2.resize(img, (int(round(img.shape[1] * s)), int(round(img.shape[0] * s))), interpolation=cv2.INTER_AREA)
    fs = 0.6 if width >= 500 else 0.5
    P = lambda x, y: (int(round((x - rx1) * s)), int(round((y - ry1) * s)))
    if grid:
        step = 50 if rx2 - rx1 < 600 else 100
        for gx in range((rx1 // step + 1) * step, rx2, step):
            cv2.line(img, P(gx, ry1), P(gx, ry2), (255, 255, 255), 1)
            cv2.putText(img, str(gx), (P(gx, 0)[0] + 2, 14), FONT, 0.4, (255, 255, 255), 1)
        for gy in range((ry1 // step + 1) * step, ry2, step):
            cv2.line(img, P(rx1, gy), P(rx2, gy), (255, 255, 255), 1)
            cv2.putText(img, str(gy), (2, P(rx1, gy)[1] - 2), FONT, 0.4, (255, 255, 255), 1)
    if cand is not None:
        for r in cand.itertuples():
            p1, p2 = P(r.x1, r.y1), P(r.x2, r.y2)
            dashed(img, p1, p2, (210, 210, 210))
            c = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
            cv2.putText(img, r.letter, c, FONT, fs + 0.1, (0, 0, 0), 4)
            cv2.putText(img, r.letter, c, FONT, fs + 0.1, (240, 240, 240), 2)
    for r in boxes.itertuples():                       # boxes first, labels on top
        p1, p2 = P(r.x1, r.y1), P(r.x2, r.y2)
        if r.id in new:
            cv2.rectangle(img, (p1[0] - 3, p1[1] - 3), (p2[0] + 3, p2[1] + 3), (0, 255, 255), 3)
        if r.id in dup:
            cv2.rectangle(img, (p1[0] - 3, p1[1] - 3), (p2[0] + 3, p2[1] + 3), (0, 0, 255), 3)
        cv2.rectangle(img, p1, p2, PAL[int(r.id) % len(PAL)], 2)
    placed = []                                        # id label near the box centre, moved until it overlaps no other label
    for r in boxes.itertuples():
        p1, p2 = P(r.x1, r.y1), P(r.x2, r.y2)
        col = PAL[int(r.id) % len(PAL)]
        lab = str(int(r.id))
        (tw, th), _ = cv2.getTextSize(lab, FONT, fs, 2)
        cx, cy = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
        for dy in (0, th + 10, -(th + 10), 2 * (th + 10), -2 * (th + 10), 3 * (th + 10)):
            L = (cx - tw // 2 - 3, cy + dy - th // 2 - 4, cx + tw // 2 + 3, cy + dy + th // 2 + 4)
            if not any(L[0] < q[2] and q[0] < L[2] and L[1] < q[3] and q[1] < L[3] for q in placed):
                break
        placed.append(L)
        cv2.rectangle(img, L[:2], L[2:], col, -1)
        cv2.rectangle(img, L[:2], L[2:], (0, 0, 0), 1)
        cv2.putText(img, lab, (L[0] + 3, L[3] - 4), FONT, fs, (0, 0, 0), 3)
        cv2.putText(img, lab, (L[0] + 3, L[3] - 4), FONT, fs, (255, 255, 255), 1)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode()


def overview(f, reg, width=200):
    """whole pen, small, with the shown part outlined"""
    img = cv2.imread(str(FRAMES / f"frame_{f:05d}.jpg"))
    if img.shape[1] != 1920:
        img = cv2.resize(img, (1920, 1080))
    img = img[:, X0:X1].copy()
    cv2.rectangle(img, (reg[0] - X0, reg[1]), (reg[2] - X0, reg[3]), (0, 255, 255), 8)
    s = width / img.shape[1]
    img = cv2.resize(img, (width, int(round(img.shape[0] * s))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf.tobytes()).decode()


def frame_info(k, b):
    """ids first seen in this frame (with likely earlier ids), ids back after a gap, ids of the previous frame missing now"""
    f = FR[k]
    cur = b[b.frame == f]
    first = b.groupby("id").frame.min()
    new = [i for i in sorted(cur.id.unique()) if first[i] == f and k > 0]
    prev_ids = set(b[b.frame == FR[k - 1]].id) if k > 0 else set()
    back = [i for i in sorted(cur.id.unique()) if i not in prev_ids and first[i] < f]
    gone = sorted(prev_ids - set(cur.id))
    recent = b[(b.frame < f) & (b.frame >= FR[max(0, k - 6)])]                  # last 3 s
    lost = recent[~recent.id.isin(cur.id)].sort_values("frame").groupby("id").last()
    sugg = {}
    for i in new:
        bx = cur[cur.id == i].iloc[0]
        c = np.array([(bx.x1 + bx.x2) / 2, (bx.y1 + bx.y2) / 2])
        d = [(float(np.hypot(*(c - np.array([(r.x1 + r.x2) / 2, (r.y1 + r.y2) / 2])))), int(lid), (f - r.frame) / FPS)
             for lid, r in lost.iterrows()]
        sugg[i] = sorted(d)[:3]
    dup = sorted(cur.id[cur.id.duplicated()].unique())
    return new, sugg, back, gone, dup


def video(f, half=2.0, height=544, fps_out=15):
    out = WORK / "_video" / f"v_{f:05d}.mp4"
    if not out.exists():
        out.parent.mkdir(exist_ok=True)
        cap = cv2.VideoCapture(str(ROOT / VIDEOS[CLIP]["path"]))
        fps = cap.get(cv2.CAP_PROP_FPS)
        a, z = max(0, int(f - half * fps)), int(f + half * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, a)
        w = int(round(height * (X1 - X0) / 1080 / 16)) * 16
        tmp = out.with_name(out.stem + ".part.mp4")
        wr = imageio_ffmpeg.write_frames(str(tmp), (w, height), fps=fps_out, codec="libx264", pix_fmt_out="yuv420p", quality=6,
                                         macro_block_size=16, output_params=["-movflags", "+faststart", "-threads", "2"])
        wr.send(None)
        step, nxt = fps / fps_out, float(a)
        for i in range(a, z + 1):
            ok, fr = cap.read()
            if not ok:
                break
            if i + 1e-6 < nxt:
                continue
            nxt += step
            c = cv2.resize(cv2.resize(fr, (1920, 1080))[:, X0:X1], (w, height))
            if abs(i - f) <= fps / fps_out:
                cv2.rectangle(c, (0, 0), (w - 1, height - 1), (0, 255, 255), 6)
            cv2.putText(c, f"{(i - f) / fps:+.1f} s", (8, 30), FONT, 0.8, (0, 255, 255), 2)
            wr.send(cv2.cvtColor(c, cv2.COLOR_BGR2RGB).tobytes())
        wr.close(); cap.release(); tmp.rename(out)
    b64 = base64.b64encode(out.read_bytes()).decode()
    return (f'<video width="{CUR_W}" controls autoplay loop muted playsinline><source src="data:video/mp4;base64,{b64}" '
            'type="video/mp4"></video><div>노란 테두리 = 지금 프레임. 영상을 닫으려면 Enter</div>')


def export(b, st):
    """write the ground truth once every frame has been checked"""
    dup = b.groupby(["frame", "id"]).size()
    if (dup > 1).any():
        print("같은 번호가 한 프레임에 두 번 있는 곳:", [f"{FR.index(f) + 1}번째 장(번호 {i})" for f, i in dup[dup > 1].index][:10])
        return False
    GT_DIR.mkdir(parents=True, exist_ok=True)
    b.sort_values(["frame", "id"])[["frame", "id", "x1", "y1", "x2", "y2"]].to_csv(GT_DIR / f"{CLIP}_{PEN}.csv", index=False)
    pre = pd.read_csv(WORK / "prefill.csv")
    kept = pre.merge(b, on=["frame", "x1", "y1", "x2", "y2"], how="inner")
    ops = pd.Series([e["cmd"].split()[0][0] if e["cmd"][0] in "+-" else ("swap" if "<>" in e["cmd"] else "relabel")
                     for e in st["log"] if e["cmd"][0] in "+-" or "=" in e["cmd"] or "<>" in e["cmd"]], dtype=object)
    json.dump(dict(clip=CLIP, pen=PEN, roi=[X0, X1], fps=FPS, frames=FR, annotator=st["annotator"], started=st["started"],
                   finished=max(st["reviewed"].values()), n_boxes=len(b), n_ids=int(b.id.nunique()),
                   prefill_boxes=len(pre), prefill_kept=len(kept), prefill_deleted=len(pre) - len(kept),
                   added=b.source.value_counts().drop("prefill", errors="ignore").to_dict(),
                   edit_commands=ops.value_counts().to_dict(), memo=st["memo"]),
              open(GT_DIR / f"{CLIP}_{PEN}.json", "w"), ensure_ascii=False, indent=1)
    return True


def annotate():
    b, st = load_state()
    hist, msg, grid, full, show_video = [], "", False, False, None
    todo = [k for k in range(N) if str(FR[k]) not in st["reviewed"]]
    print(f"{N}장 중 {N - len(todo)}장 확인됨")
    input("시작하려면 Enter > ")
    k = todo[0] if todo else 0
    while 0 <= k < N:
        f = FR[k]
        cur = b[b.frame == f]
        new, sugg, back, gone, dup = frame_info(k, b)
        cand = candidates(f, b)
        clear_output(wait=True)
        done = len(st["reviewed"])
        status = "확인함" if str(f) in st["reviewed"] else '<span style="color:#d62728">미확인</span>'
        head = f"<b>[{k + 1}/{N}] 프레임 {f} ({f / FPS:.1f} s)</b> · 박스 {len(cur)}개 · {status} · 전체 {done}/{N}장 확인"
        prev = b[b.frame == FR[k - 1]] if k > 0 else cur.iloc[:0]
        xy = ["x1", "y1", "x2", "y2"]
        reg = None if full else region(pd.concat([cur[xy], prev[xy], cand[xy]]))
        img_cur = render(f, cur, CUR_W, new=set(new), dup=set(dup), cand=cand, grid=grid, reg=reg)
        prev_html = '<div style="margin-left:10px">'
        if k > 0:
            pf = FR[k - 1]
            prev_html += (f'<div>앞 프레임 ({pf / FPS:.1f} s)</div>'
                          f'<img src="data:image/jpeg;base64,{render(pf, prev, PREV_W, reg=reg)}">')
        if reg:
            prev_html += f'<div>펜 전체 (노란 틀 = 보이는 부분)</div><img src="data:image/jpeg;base64,{overview(f, reg)}">'
        prev_html += "</div>"
        lines = []
        for i in new:
            s = ", ".join(f"<b>{lid}</b> ({dt:.1f}초 전, {dist:.0f}px)" for dist, lid, dt in sugg[i]) or "없음"
            lines.append(f"처음 나온 번호 <b>{i}</b> → 아마: {s}")
        if back:
            lines.append("다시 나타난 번호: " + ", ".join(map(str, back)))
        if gone:
            lines.append("앞 프레임에 있었는데 없는 번호: " + ", ".join(map(str, gone)) + " (보이는데 박스가 없으면 <code>+번호</code>)")
        if len(cand):
            lines.append("후보: " + ", ".join(f"{r.letter} ({r.conf:.2f})" for r in cand.itertuples()))
        if dup:
            lines.append('<span style="color:#d62728">같은 번호가 두 번: ' + ", ".join(map(str, dup)) + "</span>")
        if st["memo"].get(str(f)):
            lines.append("메모: " + st["memo"][str(f)])
        if msg:
            lines.append(f'<span style="color:#1f77b4">{msg}</span>')
        display(HTML(f'<div>{head}</div><div style="display:flex;align-items:flex-start"><div><div>지금 프레임</div>'
                     f'<img src="data:image/jpeg;base64,{img_cur}"></div>{prev_html}</div>'
                     f'<div style="margin-top:6px;line-height:1.6">{"<br>".join(lines)}</div>'
                     + (show_video or "")))
        show_video, msg = None, ""
        cmd = input("Enter=확인·다음 | 36=30 · 36=30! · 5<>12 · -7 · -7* · +a · +a 12 · +12 · +box x1 y1 x2 y2 id · v · grid · full · m 메모 · u · b · g 37 · q > ").strip()
        before = b.copy()
        try:
            if cmd in ("", "n"):
                st["reviewed"][str(f)] = pd.Timestamp.now().isoformat(timespec="seconds")
                save_state(b, st)
                k += 1
                continue
            if cmd == "q":
                break
            if cmd == "b":
                k = max(0, k - 1); continue
            if cmd.startswith("g "):
                k = min(N - 1, max(0, int(cmd.split()[1]) - 1)); continue
            if cmd == "grid":
                grid = not grid; continue
            if cmd == "full":
                full = not full; continue
            if cmd == "v":
                show_video = video(f); continue
            if cmd.startswith("m "):
                st["memo"][str(f)] = cmd[2:].strip(); save_state(b, st); msg = "메모 저장"; continue
            if cmd == "u":
                if hist:
                    b = hist.pop(); st["log"].pop(); save_state(b, st); msg = "되돌림"
                else:
                    msg = "되돌릴 것이 없음"
                continue
            if "<>" in cmd:
                x, y = (int(v) for v in cmd.split("<>"))
                ma, my = (b.id == x) & (b.frame >= f), (b.id == y) & (b.frame >= f)
                b.loc[ma, "id"], b.loc[my, "id"] = y, x
                msg = f"{x} ↔ {y} 맞바꿈 (이 프레임부터)"
            elif "=" in cmd:
                only = cmd.endswith("!")
                x, y = (int(v) for v in cmd.rstrip("!").split("="))
                m = (b.id == x) & ((b.frame == f) if only else (b.frame >= f))
                if not m.any():
                    msg = f"{x}번이 {'이 프레임에' if only else '이 프레임부터'} 없음"; continue
                clash = sorted(set(b[m].frame) & set(b[b.id == y].frame))
                if clash:
                    msg = (f"{y}번이 이미 있는 장: " + ", ".join(str(FR.index(c) + 1) for c in clash[:8]) +
                           f" → 그 장의 {y}번을 먼저 정리하거나 <code>{x}<>{y}</code> 로 맞바꾸세요"); continue
                b.loc[m, "id"] = y
                msg = f"{x} → {y} ({m.sum()}개 박스{', 이 프레임만' if only else ''})"
            elif cmd.startswith("-"):
                allf = cmd.endswith("*")
                x = int(cmd[1:].rstrip("*"))
                m = (b.id == x) & ((b.frame >= f) if allf else (b.frame == f))
                if not m.any():
                    msg = f"{x}번이 없음"; continue
                b = b[~m].reset_index(drop=True)
                msg = f"{x}번 박스 {m.sum()}개 지움"
            elif cmd.startswith("+box"):
                x1, y1, x2, y2, i = (float(v) for v in cmd.split()[1:6])
                i = int(i)
                if not (X0 <= x1 < x2 <= X1 and 0 <= y1 < y2 <= 1080):
                    msg = f"좌표 범위: x {X0}–{X1}, y 0–1080, x1<x2, y1<y2"; continue
                if (cur.id == i).any():
                    msg = f"이 프레임에 {i}번이 이미 있음"; continue
                b = pd.concat([b, pd.DataFrame([dict(frame=f, id=i, x1=x1, y1=y1, x2=x2, y2=y2, source="manual")])], ignore_index=True)
                msg = f"{i}번 추가(좌표)"
            elif cmd.startswith("+"):
                parts = cmd[1:].split()
                if parts[0].isalpha():
                    r = cand[cand.letter == parts[0]]
                    if not len(r):
                        msg = f"후보 {parts[0]} 없음"; continue
                    r = r.iloc[0]
                    i = int(parts[1]) if len(parts) > 1 else int(b.id.max()) + 1
                    if (cur.id == i).any():
                        msg = f"이 프레임에 {i}번이 이미 있음"; continue
                    b = pd.concat([b, pd.DataFrame([dict(frame=f, id=i, x1=r.x1, y1=r.y1, x2=r.x2, y2=r.y2, source="candidate")])],
                                  ignore_index=True)
                    msg = f"후보 {parts[0]} → {i}번"
                else:
                    i = int(parts[0])
                    if (cur.id == i).any():
                        msg = f"이 프레임에 {i}번이 이미 있음"; continue
                    p = b[(b.id == i) & (b.frame < f)].sort_values("frame").tail(1)
                    q = b[(b.id == i) & (b.frame > f)].sort_values("frame").head(1)
                    if not len(p) and not len(q):
                        msg = f"{i}번이 다른 프레임에 없어 보간할 수 없음 (<code>+box</code> 사용)"; continue
                    c = ["x1", "y1", "x2", "y2"]
                    if len(p) and len(q):
                        w = (f - p.frame.iloc[0]) / (q.frame.iloc[0] - p.frame.iloc[0])
                        box = p[c].to_numpy(float)[0] * (1 - w) + q[c].to_numpy(float)[0] * w
                    else:
                        box = (p if len(p) else q)[c].to_numpy(float)[0]
                    b = pd.concat([b, pd.DataFrame([dict(frame=f, id=i, **dict(zip(c, box.round(1))), source="interp")])],
                                  ignore_index=True)
                    msg = f"{i}번 보간해 추가"
            else:
                msg = "입력을 확인하세요"; continue
        except (ValueError, IndexError):
            msg = "형식을 확인하세요 (예: 36=30, 5<>12, -7, +a, +12)"; continue
        hist.append(before); hist[:] = hist[-30:]
        st["log"].append(dict(frame=int(f), cmd=cmd, at=pd.Timestamp.now().isoformat(timespec="seconds")))
        save_state(b, st)
    clear_output(wait=True)
    done = len(st["reviewed"])
    print(f"저장됨: {WORK / 'work.csv'} — {done}/{N}장 확인")
    if done == N and export(b, st):
        print(f"정답 파일: {GT_DIR / f'{CLIP}_{PEN}.csv'} (박스 {len(b)}개, 번호 {b.id.nunique()}개)")


# %%
annotate()
