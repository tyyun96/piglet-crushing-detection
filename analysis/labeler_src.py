# ---
# jupyter:
#   kernelspec:
#     display_name: Python (crushing .venv)
#     language: python
#     name: crushing-venv
# ---

# %% [markdown]
# # 경보 판정 노트북 — 판정 신뢰도 확인용 (영상 기본)
#
# 첫 판정(`observer1`)이 끝난 경보 중 **목표 집단(신생 자돈 복: A-R, B-L)의 경보만** 무작위로 골라 **두 번째 판정**을 한다.
# 자돈이 큰 복(B-R, 고위험기를 지난 복 — 분석 노트북 §2.6)의 경보는 나오지 않는다. 판정은 키를 누를 때마다 `review_sheet.csv` 에 바로 저장된다.
#
# **누가, 무엇으로**
# - 공동저자 1명이 독립적으로 판정 → `COL = "observer2"` (관찰자 간 일치도, 권장)
# - 첫 판정자 본인이 **시간이 지난 뒤**(첫 판정 2026-10-09) 다시 판정 → `COL = "observer1_repeat"` (관찰자 내 일치도 — 논문에 그렇게 명시)
# - 둘 다 같은 `seed` 를 쓰므로 **같은 경보 묶음**(목표 집단 120개 중 무작위 30% = 36개)을 보게 된다.
# - 판정할 때마다 시각이 `{COL}_time` 열에 기록된다 → 분석 노트북 §16 이 첫 판정과의 **실제 간격(일)** 을 계산한다(논문에는 그 값을 쓴다).
#
# **블라인드**: 두 번째 판정자는 `review_sheet.csv`(첫 판정이 들어 있음)와 `review_key.csv`(원고 GT 와의 대조)를 열어 보지 않는다. 화면에는 지금 판정 열의 값만 나온다.
#
# ## 화면
# - 경보마다 해당 **펜만 잘라낸 영상**(경보 시점 ±15초, 초당 12프레임)이 **자동 재생·반복**된다.
# - 왼쪽 위 `-3.2 s` = 경보 시점 기준 시간. 경보 순간(±0.5초)에는 **빨간 테두리 + ALERT** 표시.
# - 아래 시간 막대: **빨간 눈금 = 경보 시점**, 흰 표시 = 지금 재생 위치. 왼쪽 아래는 카메라 시각.
# - 영상 아래 버튼으로 재생 속도(0.5× / 1× / 2×)를 바꾼다. 영상의 기본 컨트롤로 멈춤·되감기도 된다.
#
# ## 키 (아래 입력칸에 입력 후 Enter)
# | 키 | 판정 | 기준 |
# |---|---|---|
# | `1` | **trapping** | 새끼 몸의 일부 또는 전부가 **모돈 몸 아래에 깔리거나 끼어 있다**. 몸이 완전히 안 보여도, 모돈이 눕기 직전 그 자리에 있던 새끼가 사라지고 영상 끝까지(필요하면 `w` 로 ±30초까지) 다시 나오지 않으면 trapping |
# | `2` | **crushing** | 깔린 뒤 다시 나타난 새끼가 **움직이지 않거나 죽은 것으로 보인다**(사망 의심) |
# | `3` | **no_event** | **깔림이 없다고 판단할 수 있다**. 이어서 이유 한 글자: `a` 모돈 뒤·반대편에 가려졌을 뿐(다시 멀쩡히 나옴) / `b` 새끼끼리 겹침·무리·젖먹이 / `c` 보온등·조명·칸막이 봉 때문에 검출만 빠짐 / `d` 사람·관리 작업 / `e` 기타 |
# | `4` | **uncertain** | 영상을 보고도 깔렸는지 아닌지 **판단할 근거가 없다**(정밀도 계산에서 제외하고 개수만 보고) |
# | `w` | 넓게 보기 | ±30초 영상(처음 누를 때 몇 초 걸림) |
# | `p` | 사진 | −10, −5, 0, +5, +10초 사진 한 줄 |
# | `m` | 메모 | 자유 메모 |
# | `b` / `n` | 이전 / 다음(건너뛰기) | |
# | `q` | 종료 | 지금까지 판정은 이미 저장됨 |
#
# 애매하면 먼저 `w` 로 넓게 본다. 근거가 조금이라도 있으면 `1` 또는 `3` 으로 판정하고, 정말 판단할 수 없을 때만 `4` 를 쓴다. 이유는 `m` 으로 메모해 둔다.
#
# **블라인드**: 화면에는 원고 GT 와의 일치 여부를 보여주지 않는다. 그 대조 결과는 판정 시트가 아니라 `review_key.csv` 에 따로 있다(판정을 마치기 전에는 열지 않는다).
#
# ## 혼자 판정할 때 신뢰도 확보 (리뷰어 R1-4)
# - 혼자 전부 판정해도 정밀도·시간당 오경보 계산에는 충분하다(`COL = "observer1"`).
# - 신뢰도를 보고하려면: 공동저자가 무작위 30% 를 독립 판정(`COL = "observer2"`)하거나,
#   본인이 시간이 지난 뒤 무작위 30% 를 순서를 섞어 다시 판정(`COL = "observer1_repeat"`) → 관찰자 내 일치도. 간격은 기록된 실제 날짜로 보고한다.
# - 판정 후 분석 노트북을 다시 실행하면 §8.5(판정 기반 지표)·§16(일치도)이 자동으로 갱신된다.

# %%
SHEET = "TPA_v2_gate"            # 트래핑 경보 시트
COL = "observer1_repeat"         # 본인 재판정 = "observer1_repeat" / 공동저자 독립 판정 = "observer2"
LITTERS = ["A-R", "B-L"]         # 목표 집단(신생 자돈 복)만. 자돈이 큰 B-R 제외. None = 모든 복
SUBSET = 0.3                     # 목표 집단 경보 중 무작위 30% (약 36개; κ 신뢰구간을 위해 20% 대신 30%)
SHUFFLE = True                   # 순서 섞기(첫 판정과 같은 순서로 보지 않도록)
VIDEO_WIDTH = 720                # 영상 표시 폭(px). 화면이 작으면 줄인다

# %%
import base64
import os
import random
import sys
import uuid
from pathlib import Path

import pandas as pd
from IPython.display import HTML, Image, clear_output, display

ROOT = Path(os.environ.get("CRUSHING_ROOT") or next(q for q in [Path.cwd(), *Path.cwd().parents] if (q / "analysis" / "core.py").exists()))   # project folder (contains analysis/)
sys.path.insert(0, str(ROOT / "analysis"))
import review_clips as rc  # noqa: E402

LABELS = {"1": "trapping", "2": "crushing", "3": "no_event", "4": "uncertain"}
REASONS = {"a": "hidden_behind_or_beside_sow", "b": "piglets_overlapping_huddling_nursing",
           "c": "detector_miss_light_or_bars", "d": "person_or_management", "e": "other"}


def video_html(path, width=VIDEO_WIDTH):
    vid = "v" + uuid.uuid4().hex[:8]
    b64 = base64.b64encode(Path(path).read_bytes()).decode()
    btn = "".join(f'<button style="margin-right:6px" onclick="document.getElementById(\'{vid}\').playbackRate={r}">{r}×</button>'
                  for r in (0.5, 1, 2))
    return (f'<video id="{vid}" width="{width}" controls autoplay loop muted playsinline>'
            f'<source src="data:video/mp4;base64,{b64}" type="video/mp4"></video>'
            f'<div style="margin:4px 0 8px 0">재생 속도: {btn}</div>')


def litter_of(rec, pen):
    return f"{rec[0]}-{pen}"          # A-R, B-L, B-R


def labeler(sheet=SHEET, col=COL, subset=SUBSET, shuffle=SHUFFLE, litters=LITTERS, seed=0):
    d = ROOT / "outputs" / "alert_review" / sheet
    path = d / "review_sheet.csv"
    df = pd.read_csv(path, dtype=str).fillna("")
    for c in (col, f"{col}_reason", f"{col}_notes", f"{col}_time"):
        if c not in df:
            df[c] = ""
    pool = [i for i in df.index if not litters or litter_of(df.at[i, "rec"], df.at[i, "pen"]) in litters]
    rnd = random.Random(seed)
    order = sorted(rnd.sample(pool, max(1, int(round(subset * len(pool)))))) if subset else list(pool)
    print(f"판정 대상: {len(order)}개 (복 {litters or '전부'}의 경보 {len(pool)}개 중"
          f"{' 무작위 ' + str(int(subset * 100)) + '%' if subset else ' 전부'}; 제외된 경보 {len(df) - len(pool)}개)")
    input("시작하려면 Enter > ")
    if shuffle:
        rnd.shuffle(order)
    todo = [i for i in order if df.at[i, col] == ""]
    pos = order.index(todo[0]) if todo else 0
    half = 15
    while 0 <= pos < len(order):
        i = order[pos]; r = df.loc[i]
        clear_output(wait=True)
        done = int((df.loc[order, col] != "").sum())
        print(f"[{done}/{len(order)} 판정됨]  #{r['id']}  {r['rec']} 펜 {r['pen']}  경보 t={r['alert_s']} s  ({r['clock_time']})"
              f"   현재 판정: {r[col] or '-'} {r[f'{col}_reason']}  메모: {r[f'{col}_notes']}"
              + ("   [±30초 영상]" if half == 30 else ""))
        clip = rc.make_clip(r["rec"], r["pen"], float(r["alert_s"]), half=half)
        display(HTML(video_html(clip)))
        cmd = input("1 trapping | 2 crushing | 3 no_event | 4 uncertain | w ±30초 | p 사진 | m 메모 | b 이전 | n 다음 | q 종료 > ").strip().lower()
        if cmd in LABELS:
            df.at[i, col] = LABELS[cmd]
            if cmd == "3":
                rs = input("이유: a 모돈에 가려짐 | b 새끼 겹침·무리·젖먹이 | c 조명·봉 때문에 검출 누락 | d 사람·관리 | e 기타 > ").strip().lower()
                df.at[i, f"{col}_reason"] = REASONS.get(rs, rs)
            else:
                df.at[i, f"{col}_reason"] = ""
            df.at[i, f"{col}_time"] = pd.Timestamp.now().isoformat(timespec="seconds")   # 실제 재판정 간격 보고용
            df.to_csv(path, index=False)
            pos += 1; half = 15
        elif cmd == "w":
            half = 30 if half == 15 else 15
        elif cmd == "p":
            display(Image(filename=str(d / r["image"]), width=1100))
            input("사진을 본 뒤 Enter > ")
        elif cmd == "m":
            df.at[i, f"{col}_notes"] = input("메모 > ")
            df.to_csv(path, index=False)
        elif cmd == "b":
            pos = max(0, pos - 1); half = 15
        elif cmd == "n":
            pos += 1; half = 15
        elif cmd == "q":
            break
    clear_output(wait=True)
    n_done = int((df.loc[order, col] != "").sum())
    print(f"저장됨: {path}\n{col}: {n_done}/{len(order)} 판정 완료")
    print(df.loc[order, col].replace("", "(미판정)").value_counts().to_string())

# %%
labeler()
