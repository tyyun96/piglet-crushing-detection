# ---
# jupyter:
#   kernelspec:
#     display_name: Python (crushing .venv)
#     language: python
#     name: crushing-venv
# ---

# %% [markdown]
# # 연속 영상 전체 주석 (블라인드) — 진짜 재현율을 위한 정답 만들기
#
# 지금 재현율은 원고의 사건 11개만 정답으로 쓴다. 그런데 경보 판정에서 확인된 에피소드 30개 중 원고에 주석된 것은 9개뿐이었다.
# 그래서 **연속 녹화 전체를 처음부터 끝까지 보면서** 깔림이 시작된 시각을 모두 표시한다. 이것이 새 정답이 된다.
#
# - 대상: 목표 집단 펜 **A-R(1.66시간), B-L(1.10시간)** = 2.76시간. 2분 구간 약 85개를 시간 순서대로 본다.
# - **블라인드**: 화면에는 알고리즘 경보가 전혀 나오지 않는다. 주석하는 동안 경보 검토 영상·판정 시트·분석 노트북을 보지 않는다.
# - 표시할 때마다 `annotations.csv` 에 바로 저장된다. 언제든 `q` 로 멈추고, 다시 실행하면 안 본 구간부터 이어진다.
#
# ## 화면
# - 펜 하나만 잘라낸 2분 영상(초당 12프레임)이 재생된다. 왼쪽 위 **노란 숫자 = 구간 안의 시각**(예: `0:37.5`). 왼쪽 아래는 카메라 시각.
# - 아래 막대: 흰 표시 = 현재 위치, 회색 눈금 = 10초마다(1분마다 긴 눈금).
# - 영상 아래 버튼으로 속도(1× / 2× / 3× / 4×)를 바꾼다. 영상을 클릭하면 멈춤, 막대를 끌면 되감기.
# - 깔림이 보이면 **영상을 멈추고, 노란 숫자를 읽어** 아래 입력칸에 입력한다.
#
# ## 무엇을 표시하나 (경보 판정 때와 같은 기준)
# | 입력 | 뜻 | 기준 |
# |---|---|---|
# | `t 37` | **trapping** 이 0:37 에 시작 | 새끼 몸의 일부 또는 전부가 **모돈 몸 아래에 깔리거나 끼이기 시작한** 시각. 대개 모돈이 눕거나 자세를 바꿀 때. 모돈이 누운 자리에 있던 새끼가 사라지고 다시 안 나오면 trapping |
# | `c 37` | **crushing** (사망 의심) | 깔린 새끼가 그 뒤 **움직이지 않거나 죽은 것으로 보인다**. `t` 대신 `c` 로, **깔리기 시작한 시각**에 표시. 깔린 순간이 안 보였으면 처음 알아챈 시각 |
# | `u 37` | **uncertain** | 깔림일 수도 있지만 판단할 근거가 부족하다(평가에서 따로 다룸) |
# | `r 85` | **일어날 때 발견** | 깔리는 순간은 못 봤고(시작 모름), **모돈이 일어나거나 몸을 돌릴 때 그 밑에서 깔려 있던 새끼가 드러났다**. 드러난(풀려난) 시각에 표시. 죽은 것으로 보이면 메모(`m`). `e` 는 붙이지 않는다 |
# | `e 85` | **깔림 끝** | 깔렸던 새끼가 **다시 나온(풀려난) 시각**. 이 펜에서 가장 최근에 표시한, 아직 안 끝난 `t`/`c` 를 닫는다. 앞 구간에서 시작된 깔림도 된다. 두 마리 이상이면 마지막 새끼가 나온 시각. 죽어서 안 나오거나 끝을 못 봤으면 생략 |
# | `t 37 2` | 두 마리가 동시에 | 마지막 숫자 = 마리 수(생략하면 1) |
# | `t 1:05`, `t -3` | 시각 쓰는 법 | `분:초` 도 된다. 앞 구간 끝에서 시작된 걸 다음 구간에서 알아챘으면 음수(최대 -30초) |
# | `x` | 이 구간의 마지막 표시 지우기 | |
# | `m` | 이 구간 메모 | |
# | `n` | **이 구간 다 봤음 → 다음** | 사건이 없어도 반드시 `n` 으로 넘긴다(본 구간이 기록되어야 평가에 들어감) |
# | `b` / `q` | 이전 구간 / 종료 | |
#
# **깔림이 아닌 것**: 새끼가 스스로 모돈 배·다리 사이로 파고들어 젖을 먹거나 자는 것(눌리지 않음), 모돈 뒤로 가려졌다가 멀쩡히 다시 나오는 것,
# 새끼끼리 겹치는 것.
#
# **한 사건에는 표시 하나**: 깔린 새끼가 죽은 것으로 보이면 `t` 와 `c` 를 둘 다 넣지 말고 `c` 하나만 넣는다(같은 시각의 `t`·`c` 는 분석에서 한 사건으로 합쳐진다).
# **시작을 못 본 깔림**(모돈이 일어날 때 밑에서 드러난 새끼)은 시작 시각을 추측하지 말고 `r` 로 드러난 시각에 표시한다.
#
# **한 사건은 시작에 한 번만 표시한다.** 앞 구간에서 표시한 깔림이 다음 구간에도 이어지면 다시 `t` 를 넣지 않고, 새끼가 다시 나오는 것을 보면
# 그 구간에서 `e 시각` 으로 끝을 표시한다(끝 시각이 있으면, 깔림이 계속되는 동안 울린 경보를 오경보가 아니라 같은 사건의 중복 경보로 셀 수 있다).
# 깔림이 이어지는 중에 **다른 새끼가 새로 깔리면** 새 `t`. 같은 새끼가 풀려났다가 다시 깔리면 새 `t`.
# 애매하면 `b` 로 앞 구간을 다시 보고, 그래도 모르겠으면 `u`.

# %%
ANNOTATOR = "observer1"          # 주석자 이름(두 번째 주석자는 "observer2")
LITTERS = ["A-R", "B-L"]         # 목표 집단. 자돈이 큰 B-R 은 넣지 않음
SEG_S = 120                      # 구간 길이(초). prep_08 로 미리 만든 영상과 같아야 함
SUBSET = None                    # None = 모든 구간. 0.35 처럼 비율을 주면 복마다 무작위로 그 비율의 구간만(시간 순서로) 보여 준다
SEED = 0                         # SUBSET 의 무작위 선택 고정(두 번째 관찰자끼리 같은 구간)
DEFAULT_SPEED = 2                # 처음 재생 속도
VIDEO_WIDTH = 600                # 영상 표시 폭(px)

# %%
import base64
import os
import sys
import uuid
import warnings
from pathlib import Path

import pandas as pd
from IPython.display import HTML, clear_output, display

warnings.filterwarnings("ignore", category=FutureWarning)      # pandas concat with an empty frame

ROOT = Path(os.environ.get("CRUSHING_ROOT") or next(q for q in [Path.cwd(), *Path.cwd().parents] if (q / "analysis" / "core.py").exists()))   # project folder (contains analysis/)
sys.path.insert(0, str(ROOT / "analysis"))
import review_clips as rc  # noqa: E402
from video_registry import VIDEOS  # noqa: E402

LITTER_PEN = {"A-R": ("A", "R"), "B-L": ("B", "L"), "B-R": ("B", "R")}
OUT = ROOT / "outputs" / "continuous_annotation"
ANN = OUT / "annotations.csv"
SEG = OUT / "segments.csv"
TYPES = {"t": "trapping", "c": "crushing", "u": "uncertain", "e": "end", "r": "found_at_release"}
ANN_COLS = ["annotator", "litter", "rec", "pen", "seg_start_s", "type", "t_in_seg_s", "onset_s", "n_piglets",
            "clock_time", "marked_at"]
SEG_COLS = ["annotator", "litter", "rec", "pen", "seg_start_s", "seg_end_s", "reviewed_at", "memo"]


def plan(litters, seg_s):
    rows = []
    for lit in litters:
        cam, pen = LITTER_PEN[lit]
        recs = sorted((v for v in VIDEOS if VIDEOS[v]["kind"] == "raw" and VIDEOS[v]["camera"] == cam),
                      key=lambda v: VIDEOS[v]["start"])
        for r in recs:
            segs, _ = rc.segment_starts(r, seg_s)
            rows += [dict(litter=lit, rec=r, pen=pen, seg_start_s=float(s), seg_end_s=float(e)) for s, e in segs]
    return pd.DataFrame(rows)


def load(path, cols):
    return pd.read_csv(path).fillna({"memo": "", "reviewed_at": ""}) if path.exists() else pd.DataFrame(columns=cols)


def parse_time(s):
    s = s.strip()
    if ":" in s:
        m, sec = s.split(":", 1)
        sign = -1 if m.startswith("-") else 1
        return sign * (abs(int(m)) * 60 + float(sec))
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


def annotate(annotator=ANNOTATOR, litters=LITTERS, seg_s=SEG_S, subset=SUBSET, seed=SEED):
    OUT.mkdir(parents=True, exist_ok=True)
    P = plan(litters, seg_s)
    if subset:                                      # random share of the segments of each litter, shown in time order
        P = P.groupby("litter", group_keys=False).sample(frac=subset, random_state=seed).sort_index().reset_index(drop=True)
    ann, st = load(ANN, ANN_COLS), load(SEG, SEG_COLS)
    key = lambda r, p, s: (r, p, round(float(s), 1))
    def reviewed():
        m = st[(st.annotator == annotator) & (st.reviewed_at.astype(str) != "")]
        return {key(r, p, s) for r, p, s in zip(m.rec, m.pen, m.seg_start_s)}
    done = reviewed()
    todo = [i for i in range(len(P)) if key(P.rec[i], P.pen[i], P.seg_start_s[i]) not in done]
    total_min = (P.seg_end_s - P.seg_start_s).sum() / 60
    print(f"주석자 {annotator} | 복 {litters} | 구간 {len(P)}개 ({total_min:.0f}분) 중 {len(P) - len(todo)}개 완료, {len(todo)}개 남음")
    input("시작하려면 Enter > ")
    pos = todo[0] if todo else 0
    while 0 <= pos < len(P):
        g = P.iloc[pos]
        rec, pen, s, e = g.rec, g.pen, float(g.seg_start_s), float(g.seg_end_s)
        clear_output(wait=True)
        done = reviewed()
        done_min = sum((P.seg_end_s[i] - P.seg_start_s[i]) for i in range(len(P)) if key(P.rec[i], P.pen[i], P.seg_start_s[i]) in done) / 60
        state = "완료(다시 보는 중)" if key(rec, pen, s) in done else "미완료"
        print(f"[구간 {pos + 1}/{len(P)}] {g.litter} · {rec} · {int(s // 60)}:{int(s % 60):02d}–{int(e // 60)}:{int(e % 60):02d}"
              f" ({clock(rec, s)} – {clock(rec, e)[-8:]}) · {state} · 전체 {done_min:.0f}/{total_min:.0f}분 완료")
        path = rc.segment_path(rec, pen, s, seg_s)
        if not path.exists():
            print("영상 준비 중(이 녹화는 처음 한 번 1분 정도 걸림)...")
            rc.render_segments(rec, pen, seg_s=seg_s)
        display(HTML(video_html(path)))
        mine = lambda: ann[(ann.annotator == annotator) & (ann.rec == rec) & (ann.pen == pen) & (ann.seg_start_s.astype(float).round(1) == round(s, 1))]
        def show_marks():
            m = mine()
            print("이 구간 표시: " + (", ".join(f"{r.type} @ {r.t_in_seg_s:.1f}s" + ("" if r.type == "end" else f" ×{int(r.n_piglets)}")
                                            for r in m.itertuples()) or "없음"))
        show_marks()
        while True:
            cmd = input("t/c/u/r 시각 [마리] | e 시각(깔림 끝) | x 지우기 | m 메모 | n 다 봤음→다음 | b 이전 | q 종료 > ").strip().lower()
            if cmd[:1] in TYPES and len(cmd) > 1:
                parts = cmd[1:].split()
                try:
                    t_in = parse_time(parts[0]); n = int(parts[1]) if len(parts) > 1 else 1
                except (ValueError, IndexError):
                    print("형식: t 37  /  t 1:05  /  t 37 2"); continue
                if not (-30 <= t_in <= e - s + 1):
                    print(f"시각은 -30 ~ {e - s:.0f} 초 사이여야 함"); continue
                if cmd[0] == "e":                 # an end needs an earlier, still open trapping/crushing mark in this pen
                    a_ = ann[(ann.annotator == annotator) & (ann.rec == rec) & (ann.pen == pen) & (ann.onset_s.astype(float) < s + t_in)].sort_values("onset_s")
                    open_ = 0
                    for ty in a_.type:
                        open_ = open_ + 1 if ty in ("trapping", "crushing") else (max(0, open_ - 1) if ty == "end" else open_)
                    if open_ == 0:
                        print("이 시각 앞에 끝나지 않은 t/c 표시가 없음"); continue
                row = dict(annotator=annotator, litter=g.litter, rec=rec, pen=pen, seg_start_s=s, type=TYPES[cmd[0]],
                           t_in_seg_s=t_in, onset_s=round(s + t_in, 1), n_piglets=n, clock_time=clock(rec, s + t_in),
                           marked_at=pd.Timestamp.now().isoformat(timespec="seconds"))
                ann = pd.concat([ann, pd.DataFrame([row])], ignore_index=True); ann.to_csv(ANN, index=False)
                print(f"저장: {row['type']} @ {t_in:.1f}s ({row['clock_time']})" + ("" if row["type"] == "end" else f" ×{n}")); show_marks()
            elif cmd == "x":
                m = mine()
                if len(m):
                    ann = ann.drop(m.index[-1]); ann.to_csv(ANN, index=False); print("마지막 표시를 지웠음"); show_marks()
            elif cmd == "m":
                memo = input("메모 > ")
                old = st[(st.annotator == annotator) & (st.rec == rec) & (st.pen == pen) & (st.seg_start_s.astype(float).round(1) == round(s, 1))]
                rv = old.reviewed_at.iloc[-1] if len(old) else ""           # keep the 'reviewed' state
                st = pd.concat([st.drop(old.index), pd.DataFrame([dict(annotator=annotator, litter=g.litter, rec=rec, pen=pen, seg_start_s=s,
                                                                      seg_end_s=e, reviewed_at=rv, memo=memo)])], ignore_index=True)
                st.to_csv(SEG, index=False); print("메모 저장")
            elif cmd == "n":
                old = st[(st.annotator == annotator) & (st.rec == rec) & (st.pen == pen) & (st.seg_start_s.astype(float).round(1) == round(s, 1))]
                memo = str(old.memo.iloc[-1]) if len(old) else ""
                st = st.drop(old.index)
                st = pd.concat([st, pd.DataFrame([dict(annotator=annotator, litter=g.litter, rec=rec, pen=pen, seg_start_s=s, seg_end_s=e,
                                                       reviewed_at=pd.Timestamp.now().isoformat(timespec="seconds"), memo=memo)])],
                               ignore_index=True)
                st.to_csv(SEG, index=False)
                pos += 1; break
            elif cmd == "b":
                pos = max(0, pos - 1); break
            elif cmd == "q":
                pos = -1; break
    clear_output(wait=True)
    m = st[(st.annotator == annotator) & (st.reviewed_at.fillna("") != "")]
    print(f"저장됨: {ANN}\n{annotator}: 구간 {len(m)}/{len(P)}개 완료 ({(m.seg_end_s - m.seg_start_s).sum() / 60:.0f}/{total_min:.0f}분)")
    a = ann[ann.annotator == annotator]
    print(a.groupby(["litter", "type"]).n_piglets.agg(["size", "sum"]).rename(columns={"size": "표시 수", "sum": "마리 수"}).to_string()
          if len(a) else "표시 없음")

# %%
annotate()
