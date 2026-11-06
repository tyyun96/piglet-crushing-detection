# ---
# jupyter:
#   kernelspec:
#     display_name: Python (crushing .venv)
#     language: python
#     name: crushing-venv
# ---

# %% [markdown]
# # 연속 주석 재주석 (test–retest, 같은 관찰자)
#
# 연속 영상 주석(첫 주석, 2026-10-21)의 신뢰도를 보기 위해, **같은 사람이 시간이 지난 뒤 일부 구간을 다시 주석**한다.
# **첫 주석 후 2주 이상 지나서(2026-11-04 이후)** 하는 것이 좋다. 판정 시각이 기록되므로 실제 간격이 자동으로 계산된다.
#
# - 대상: 2분 구간 26개(영상 52분). 순서는 섞여 있다. 첫 주석의 표시는 보여 주지 않는다.
# - 시작 전에 첫 주석 파일(`annotations.csv`), 재확인 결과, 분석 노트북을 다시 보지 않는다.
# - 표시 기준은 첫 주석과 같다(아래 표). 언제든 `q` 로 멈추고, 다시 실행하면 이어진다.
#
# ## 화면
# - 각 영상은 **구간 앞 30초(맥락) + 2분 구간**이다. **노란 테두리·노란 막대 = 주석할 2분 구간**. 앞 30초는 상황 파악용이다.
# - 왼쪽 위 노란 숫자 = **녹화 시각**. 시각은 이 숫자 그대로 입력한다(예: `t 3:25`).
#
# ## 키 (첫 주석과 같음)
# | 입력 | 뜻 |
# |---|---|
# | `t 3:25` / `t 3:25 2` | 깔림 시작(마리 수 생략 = 1) |
# | `c 3:25` | 깔린 새끼가 죽은 것으로 보임(깔리기 시작한 시각) |
# | `u 3:25` | 깔림일 수도 있지만 확신이 없음 |
# | `r 3:25` | 시작은 못 봤고, 모돈이 일어날 때 밑에서 드러남(드러난 시각) |
# | `e 3:50` | 깔림 끝(새끼가 다시 나옴) |
# | `x` 마지막 표시 지우기 · `m` 메모 · `n` **이 구간 다 봤음 → 다음** · `b` 이전 · `q` 종료 | |
#
# 한 사건은 시작에 한 번만 표시한다. 노란 구간이 시작될 때 이미 깔려 있는 새끼는 그 시작이 앞 30초 안에 보이면 그 시각으로 `t`, 안 보이면 `u` 또는 `r`.

# %%
ANNOTATOR = "observer1_retest"
VIDEO_WIDTH = 600
DEFAULT_SPEED = 2

# %%
import base64
import os
import sys
import uuid
import warnings
from pathlib import Path

import pandas as pd
from IPython.display import HTML, clear_output, display

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(os.environ.get("CRUSHING_ROOT") or next(q for q in [Path.cwd(), *Path.cwd().parents] if (q / "analysis" / "core.py").exists()))   # project folder (contains analysis/)
sys.path.insert(0, str(ROOT / "analysis"))
import review_clips as rc  # noqa: E402
from video_registry import VIDEOS  # noqa: E402

DIR = ROOT / "outputs" / "continuous_annotation"
ITEMS, ANN, SEG = DIR / "retest_items.csv", DIR / "annotations.csv", DIR / "segments.csv"
TYPES = {"t": "trapping", "c": "crushing", "u": "uncertain", "e": "end", "r": "found_at_release"}
ANN_COLS = ["annotator", "litter", "rec", "pen", "seg_start_s", "type", "t_in_seg_s", "onset_s", "n_piglets", "clock_time", "marked_at"]
SEG_COLS = ["annotator", "litter", "rec", "pen", "seg_start_s", "seg_end_s", "reviewed_at", "memo"]


def mmss(t):
    ds = int(round(t * 10))
    return f"{ds // 600}:{ds % 600 / 10:04.1f}"


def parse_time(s):
    if ":" in s:
        m, sec = s.split(":", 1)
        return int(m) * 60 + float(sec)
    return float(s)


def clock(rec, t):
    return (pd.Timestamp(VIDEOS[rec]["start"]) + pd.to_timedelta(t, unit="s")).strftime("%m-%d %H:%M:%S")


def video_html(path, width=VIDEO_WIDTH, speed=DEFAULT_SPEED):
    vid = "v" + uuid.uuid4().hex[:8]
    b64 = base64.b64encode(Path(path).read_bytes()).decode()
    btn = "".join(f'<button style="margin-right:6px" onclick="document.getElementById(\'{vid}\').playbackRate={r}">{r}×</button>'
                  for r in (1, 2, 3, 4))
    return (f'<video id="{vid}" width="{width}" controls autoplay muted playsinline '
            f'onloadedmetadata="this.defaultPlaybackRate={speed}; this.playbackRate={speed}">'
            f'<source src="data:video/mp4;base64,{b64}" type="video/mp4"></video>'
            f'<div style="margin:4px 0 8px 0">재생 속도: {btn}</div>')


def load(path, cols):
    return pd.read_csv(path).fillna({"memo": "", "reviewed_at": ""}) if path.exists() else pd.DataFrame(columns=cols)


def retest(annotator=ANNOTATOR):
    items = pd.read_csv(ITEMS)
    ann, st = load(ANN, ANN_COLS), load(SEG, SEG_COLS)
    key = lambda r, p, s: (r, p, round(float(s), 1))
    def done_keys():
        m = st[(st.annotator == annotator) & (st.reviewed_at.astype(str) != "")]
        return {key(r, p, s) for r, p, s in zip(m.rec, m.pen, m.seg_start_s)}
    todo = [i for i in range(len(items)) if key(items.rec[i], items.pen[i], items.seg_start_s[i]) not in done_keys()]
    print(f"재주석 {len(items)}개 구간 중 {len(items) - len(todo)}개 완료")
    input("시작하려면 Enter > ")
    pos = todo[0] if todo else 0
    while 0 <= pos < len(items):
        it = items.iloc[pos]
        rec, pen, s, e, c0 = it.rec, it.pen, float(it.seg_start_s), float(it.seg_end_s), float(it.clip_start_s)
        clear_output(wait=True)
        state = "완료(다시 보는 중)" if key(rec, pen, s) in done_keys() else "미완료"
        print(f"[{pos + 1}/{len(items)}] {rec} 펜 {pen} · 주석할 구간 {mmss(s)}–{mmss(e)} (앞 30초는 맥락) · {state}")
        display(HTML(video_html(rc.make_range_clip(rec, pen, c0, e, s, e))))
        mine = lambda: ann[(ann.annotator == annotator) & (ann.rec == rec) & (ann.pen == pen) & (ann.seg_start_s.astype(float).round(1) == round(s, 1))]
        def show_marks():
            m = mine()
            print("이 구간 표시: " + (", ".join(f"{r.type} @ {mmss(r.onset_s)}" + ("" if r.type == "end" else f" ×{int(r.n_piglets)}")
                                            for r in m.itertuples()) or "없음"))
        show_marks()
        step = None
        while step is None:
            cmd = input("t/c/u/r 시각 [마리] | e 시각(깔림 끝) | x 지우기 | m 메모 | n 다 봤음→다음 | b 이전 | q 종료 > ").strip().lower()
            parts = cmd.split()
            if not parts:
                continue
            if parts[0] in TYPES and len(parts) >= 2:
                try:
                    t = parse_time(parts[1]); n = int(parts[2]) if len(parts) > 2 else 1
                except ValueError:
                    print("형식: t 3:25  /  t 3:25 2"); continue
                if not (s - 30 <= t <= e + 1):
                    print(f"시각은 {mmss(s - 30)} ~ {mmss(e)} 사이여야 함"); continue
                if parts[0] == "e":
                    a_ = ann[(ann.annotator == annotator) & (ann.rec == rec) & (ann.pen == pen) & (ann.onset_s.astype(float) < t)].sort_values("onset_s")
                    op = 0
                    for ty in a_.type:
                        op = op + 1 if ty in ("trapping", "crushing") else (max(0, op - 1) if ty == "end" else op)
                    if op == 0:
                        print("이 시각 앞에 끝나지 않은 t/c 표시가 없음"); continue
                row = dict(annotator=annotator, litter=it.litter, rec=rec, pen=pen, seg_start_s=s, type=TYPES[parts[0]], t_in_seg_s=t - s,
                           onset_s=round(t, 1), n_piglets=n, clock_time=clock(rec, t), marked_at=pd.Timestamp.now().isoformat(timespec="seconds"))
                ann = pd.concat([ann, pd.DataFrame([row])], ignore_index=True); ann.to_csv(ANN, index=False)
                print(f"저장: {row['type']} @ {mmss(t)}"); show_marks()
            elif parts[0] == "x":
                m = mine()
                if len(m):
                    ann = ann.drop(m.index[-1]); ann.to_csv(ANN, index=False); print("마지막 표시를 지웠음"); show_marks()
            elif parts[0] in ("m", "n"):
                old = st[(st.annotator == annotator) & (st.rec == rec) & (st.pen == pen) & (st.seg_start_s.astype(float).round(1) == round(s, 1))]
                memo = str(old.memo.iloc[-1]) if len(old) else ""
                rv = str(old.reviewed_at.iloc[-1]) if len(old) else ""
                if parts[0] == "m":
                    memo = input("메모 > ")
                else:
                    rv = pd.Timestamp.now().isoformat(timespec="seconds")
                st = pd.concat([st.drop(old.index), pd.DataFrame([dict(annotator=annotator, litter=it.litter, rec=rec, pen=pen, seg_start_s=s,
                                                                      seg_end_s=e, reviewed_at=rv, memo=memo)])], ignore_index=True)
                st.to_csv(SEG, index=False)
                if parts[0] == "n":
                    step = "n"
                else:
                    print("메모 저장")
            elif parts[0] in ("b", "q"):
                step = parts[0]
            else:
                print("키를 확인하세요")
        if step == "q":
            break
        pos = max(0, pos - 1) if step == "b" else pos + 1
    clear_output(wait=True)
    m = st[(st.annotator == annotator) & (st.reviewed_at.astype(str) != "")]
    print(f"저장됨: {ANN}\n{annotator}: {len(m)}/{len(items)} 구간 완료")

# %%
retest()
