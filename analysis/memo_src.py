# ---
# jupyter:
#   kernelspec:
#     display_name: Python (crushing .venv)
#     language: python
#     name: crushing-venv
# ---

# %% [markdown]
# # 재확인 항목에 메모 남기기
# 셀을 실행하면 재확인 20개 항목의 현재 결정과 메모가 표로 나온다. 항목 번호를 넣고 메모를 쓰면 `adjudication.csv` 의 메모 뒤에 덧붙여 저장된다
# (기존 메모는 지워지지 않음). 결정은 바뀌지 않는다. 여러 항목에 같은 메모를 달려면 `15 16` 처럼 번호를 띄어 쓴다. 끝내려면 `q`.

# %%
import os
from pathlib import Path

import pandas as pd
from IPython.display import display

ROOT = Path(os.environ.get("CRUSHING_ROOT") or next(q for q in [Path.cwd(), *Path.cwd().parents] if (q / "analysis" / "core.py").exists()))   # project folder (contains analysis/)
OUTF = ROOT / "outputs" / "continuous_annotation" / "adjudication.csv"


def table():
    a = pd.read_csv(OUTF)
    show = a[["item", "rec", "pen", "hi_start_s", "decision", "memo"]].copy()
    show["구간"] = [f"{int(t // 60)}:{int(t % 60):02d}" for t in show.hi_start_s]
    with pd.option_context("display.max_colwidth", 120, "display.width", 250):
        display(show[["item", "rec", "pen", "구간", "decision", "memo"]].fillna(""))


def add_memo():
    table()
    while True:
        ids = input("항목 번호 (예: 15 또는 15 16, 끝내려면 q) > ").strip()
        if ids.lower() == "q":
            break
        try:
            items = [int(x) for x in ids.split()]
        except ValueError:
            print("번호를 숫자로 넣으세요"); continue
        a = pd.read_csv(OUTF)
        if not items or not set(items) <= set(a["item"]):
            print("없는 번호가 있음"); continue
        text = input("메모 > ").strip()
        if not text:
            continue
        a["memo"] = a.memo.astype(object)
        for it in items:
            k = a.index[a["item"] == it][0]
            old = a.at[k, "memo"] if a.at[k, "memo"] == a.at[k, "memo"] else ""
            a.at[k, "memo"] = f"{old} | {text}" if old else text
        a.to_csv(OUTF, index=False)
        print("저장:", items)
    table()

# %%
add_memo()
