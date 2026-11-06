# ---
# jupyter:
#   kernelspec:
#     display_name: Python (crushing .venv)
#     language: python
#     name: crushing-venv
# ---

# %% [markdown]
# # 불일치 재확인 (adjudication) — 경보 판정 vs 연속 영상 주석
#
# 경보 판정에서 trapping/crushing 으로 확인된 에피소드 중 **연속 주석에는 표시가 없는 19개**와, 깔린 새끼 수·사망 여부가 불확실한
# **B_f413 장면 1개**를 다시 본다. 논문에는 '두 기준이 어긋난 경우를 같은 관찰자가 재검토했다'고 쓰고, 재검토 전후 결과를 둘 다 보고한다.
# 연속 주석(`annotations.csv`)과 경보 판정은 고치지 않고, 결정은 `adjudication.csv` 에 따로 저장된다.
#
# ## 화면
# - 확인할 구간 앞뒤 30초가 포함된 영상. **노란 테두리 + 아래 노란 막대 = 확인할 구간**(경보가 사건으로 판정됐던 때).
# - 왼쪽 위 노란 숫자 = **녹화 시각**(구간 시각이 아님). 새 깔림의 시작을 입력할 때 이 숫자를 쓴다.
# - 영상 위에 이 녹화·펜에서 연속 주석으로 표시한 사건 목록이 번호와 함께 나온다.
#
# ## 키 (경보 에피소드)
# | 입력 | 뜻 |
# |---|---|
# | `1 2` | 노란 구간의 깔림/죽은 새끼는 **앞에서 표시한 사건 2번이 계속되는 중**이다(새 사건 아님) |
# | `2 t 3:25` | 연속 주석에서 **놓친 새 깔림**이다. 녹화 시각 3:25 에 시작(죽은 것으로 보이면 `2 c 3:25`, 시작을 못 보면 `2 r 3:25` = 드러난 시각) |
# | `3` | **깔림이 아니다**(경보 판정이 틀렸음 — 새끼가 모돈 옆에 붙어 자거나 젖을 먹는 것, 가려졌을 뿐 등) |
# | `4` | 판단할 수 없다 |
# | `m` 메모 · `b` 이전 · `n` 건너뛰기 · `q` 종료 | |
#
# ## 키 (B_f413 장면)
# `c` = 깔린 새끼 중 죽은 것으로 보이는 새끼가 있다(crushing 유지) · `t` = 사람이 구출해 살아 있는 것으로 보인다(trapping 으로 바꿈) · `4` = 모르겠다.
# 몇 마리였는지 짐작되면 `m` 메모로 남긴다.

# %%
ANNOTATOR = "observer1"
VIDEO_WIDTH = 600

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

DIR = ROOT / "outputs" / "continuous_annotation"
ITEMS = DIR / "adjudication_items.csv"
OUTF = DIR / "adjudication.csv"
COLS = ["item", "kind", "rec", "pen", "hi_start_s", "hi_end_s", "decision", "ref_type", "ref_onset_s", "new_type", "new_onset_s",
        "memo", "annotator", "decided_at"]
NEW_TYPES = {"t": "trapping", "c": "crushing", "r": "found_at_release"}


def mmss(t):
    ds = int(round(t * 10))
    return f"{ds // 600}:{ds % 600 / 10:04.1f}"


def parse_time(s):
    if ":" in s:
        m, sec = s.split(":", 1)
        return int(m) * 60 + float(sec)
    return float(s)


def video_html(path, width=VIDEO_WIDTH):
    vid = "v" + uuid.uuid4().hex[:8]
    b64 = base64.b64encode(Path(path).read_bytes()).decode()
    btn = "".join(f'<button style="margin-right:6px" onclick="document.getElementById(\'{vid}\').playbackRate={r}">{r}×</button>'
                  for r in (0.5, 1, 2, 3))
    return (f'<video id="{vid}" width="{width}" controls autoplay muted playsinline>'
            f'<source src="data:video/mp4;base64,{b64}" type="video/mp4"></video>'
            f'<div style="margin:4px 0 8px 0">재생 속도: {btn}</div>')


def marks_of(rec, pen):
    a = pd.read_csv(DIR / "annotations.csv")
    a = a[(a.annotator == ANNOTATOR) & (a.rec == rec) & (a.pen == pen)].sort_values("onset_s")
    rows, cur = [], None             # trapping/crushing marks <= 5 s apart = one event (crushing wins), as in the analysis
    for x in a[a.type != "end"].itertuples():
        if (cur is not None and x.type in ("trapping", "crushing") and cur["type"] in ("trapping", "crushing")
                and x.onset_s - cur["last_s"] <= 5.0):
            cur["type"] = "crushing" if "crushing" in (cur["type"], x.type) else "trapping"
            cur["n_piglets"] = max(cur["n_piglets"], int(x.n_piglets)); cur["last_s"] = x.onset_s
            continue
        if cur is not None:
            rows.append(cur)
        cur = dict(type=x.type, onset_s=float(x.onset_s), last_s=float(x.onset_s), n_piglets=int(x.n_piglets))
    if cur is not None:
        rows.append(cur)
    ev = pd.DataFrame(rows, columns=["type", "onset_s", "last_s", "n_piglets"])
    ends = sorted(a[a.type == "end"].onset_s)
    ev["end_s"] = float("nan")
    for t in ends:                                   # an end closes the latest open trapping/crushing before it
        op = ev[(ev.type.isin(["trapping", "crushing"])) & (ev.onset_s < t) & ev.end_s.isna()]
        if len(op):
            ev.loc[op.onset_s.idxmax(), "end_s"] = t
    return ev


def adjudicate():
    items = pd.read_csv(ITEMS)
    res = pd.read_csv(OUTF) if OUTF.exists() else pd.DataFrame(columns=COLS)
    done = set(res[res.decision.fillna("") != ""].item)
    print(f"재확인 {len(items)}개 중 {len(done)}개 완료")
    input("시작하려면 Enter > ")
    todo = [i for i in range(len(items)) if items.item[i] not in done]
    pos = todo[0] if todo else 0
    while 0 <= pos < len(items):
        it = items.iloc[pos]
        clear_output(wait=True)
        prev = res[res.item == it["item"]]
        print(f"[{pos + 1}/{len(items)}] {it.rec} 펜 {it.pen} ({it.litter}) · 확인할 구간 {mmss(it.hi_start_s)}–{mmss(it.hi_end_s)}"
              + (f" · 현재 결정: {prev.decision.iloc[-1]}" if len(prev) else ""))
        ev = marks_of(it.rec, it.pen)
        print("이 녹화·펜의 연속 주석 표시: " + ("없음" if not len(ev) else ""))
        for k, e in ev.iterrows():
            end = "" if e.type not in ("trapping", "crushing") else (f", 끝 {mmss(e.end_s)}" if e.end_s == e.end_s else ", 끝 표시 없음")
            print(f"   {k + 1}) {e.type} @ {mmss(e.onset_s)} ×{int(e.n_piglets)}{end}")
        display(HTML(video_html(rc.make_range_clip(it.rec, it.pen, it.clip_start_s, it.clip_end_s, it.hi_start_s, it.hi_end_s))))
        if it.kind == "check_type":
            prompt = "c 사망 의심(crushing 유지) | t 구출되어 살아 있음(trapping) | 4 모르겠음 | m 메모 | b 이전 | n 건너뛰기 | q 종료 > "
        else:
            prompt = "1 번호 계속되는 중 | 2 t/c/r 시각 놓친 새 깔림 | 3 깔림 아님 | 4 판단 불가 | m 메모 | b 이전 | n 건너뛰기 | q 종료 > "
        base = dict(item=it["item"], kind=it.kind, rec=it.rec, pen=it.pen, hi_start_s=it.hi_start_s, hi_end_s=it.hi_end_s,
                    annotator=ANNOTATOR)
        step = None
        while step is None:                       # commands for this item; the video keeps playing
            parts = input(prompt).strip().lower().split()
            if not parts:
                continue
            dec = None
            if it.kind == "check_type" and parts[0] in ("c", "t", "4"):
                dec = dict(decision={"c": "keep_crushing", "t": "change_to_trapping", "4": "unclear"}[parts[0]])
            elif it.kind != "check_type" and parts[0] == "1":
                try:
                    e = ev.iloc[int(parts[1]) - 1]
                except (IndexError, ValueError):
                    print("형식: 1 2  (목록의 번호)"); continue
                if e.onset_s > it.hi_end_s:
                    print("그 사건은 노란 구간보다 뒤에 시작함 — 앞의 사건 번호를 고르거나 2/3/4"); continue
                dec = dict(decision="continuation", ref_type=e.type, ref_onset_s=e.onset_s)
            elif it.kind != "check_type" and parts[0] == "2":
                try:
                    ty = NEW_TYPES[parts[1]]; t = parse_time(parts[2])
                except (IndexError, KeyError, ValueError):
                    print("형식: 2 t 3:25  /  2 c 3:25  /  2 r 3:25"); continue
                if not (it.clip_start_s - 1 <= t <= it.clip_end_s + 1):
                    print(f"시각은 이 영상 범위({mmss(it.clip_start_s)}–{mmss(it.clip_end_s)}) 안이어야 함"); continue
                dec = dict(decision="new_event", new_type=ty, new_onset_s=t)
            elif it.kind != "check_type" and parts[0] in ("3", "4"):
                dec = dict(decision={"3": "not_trapping", "4": "unclear"}[parts[0]])
            elif parts[0] == "m":
                cur = res[res.item == it["item"]]
                row = cur.iloc[-1].to_dict() if len(cur) else dict(base, decision="", decided_at="")
                row["memo"] = input("메모 > ")
                res = pd.concat([res[res.item != it["item"]], pd.DataFrame([row])], ignore_index=True)
                res.to_csv(OUTF, index=False); print("메모 저장"); continue
            elif parts[0] in ("b", "n", "q"):
                step = parts[0]; continue
            else:
                print("키를 확인하세요"); continue
            cur = res[res.item == it["item"]]
            memo = cur.memo.iloc[-1] if len(cur) and cur.memo.iloc[-1] == cur.memo.iloc[-1] else ""
            row = dict(base, ref_type="", ref_onset_s=float("nan"), new_type="", new_onset_s=float("nan"), memo=memo)
            row.update(dec, decided_at=pd.Timestamp.now().isoformat(timespec="seconds"))
            res = pd.concat([res[res.item != it["item"]], pd.DataFrame([row])], ignore_index=True)
            res.to_csv(OUTF, index=False)
            print("저장:", row["decision"]); step = "n"
        if step == "q":
            break
        pos = max(0, pos - 1) if step == "b" else pos + 1
    clear_output(wait=True)
    res = pd.read_csv(OUTF) if OUTF.exists() else res
    print(f"저장됨: {OUTF}\n완료 {int((res.decision.fillna('') != '').sum())}/{len(items)}")
    if len(res):
        print(res.decision.value_counts().to_string())

# %%
adjudicate()
