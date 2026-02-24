"""
결과 분석 + 시각화 스크립트
Usage:
  python scripts/04_analyze.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager

from src.db.client import get_connection

# -------------------------------------------------------------------
# 폰트 설정 (macOS 한글)
# -------------------------------------------------------------------
def _set_korean_font():
    candidates = ["AppleGothic", "Apple SD Gothic Neo", "NanumGothic", "Malgun Gothic"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            matplotlib.rcParams["font.family"] = name
            break
    matplotlib.rcParams["axes.unicode_minus"] = False

_set_korean_font()

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 정렬 순서
ORDER_GPT = [
    "Vanilla", "Oracle",
    "Retrieved/ko-sroberta", "Retrieved/bge-m3", "Retrieved/openai",
    "Retrieved/hyde-bge-m3", "Retrieved/hybrid-bge-m3",
]
ORDER_O4 = ["Vanilla", "Oracle", "Retrieved/bge-m3", "Retrieved/hybrid-bge-m3"]

# 색상
COLORS = {
    "Vanilla": "#9E9E9E",
    "Oracle": "#4CAF50",
    "Retrieved/ko-sroberta": "#2196F3",
    "Retrieved/bge-m3": "#F44336",
    "Retrieved/openai": "#FF9800",
    "Retrieved/hyde-bge-m3": "#9C27B0",
    "Retrieved/hybrid-bge-m3": "#00BCD4",
}

# x축 표시 이름 (짧게)
DISPLAY = {
    "Vanilla": "Vanilla",
    "Oracle": "Oracle",
    "Retrieved/ko-sroberta": "ko-sroberta",
    "Retrieved/bge-m3": "bge-m3",
    "Retrieved/openai": "openai",
    "Retrieved/hyde-bge-m3": "HyDE\n(bge-m3)",
    "Retrieved/hybrid-bge-m3": "Hybrid\n(bge-m3)",
}

# Recall@k 그래프용 모델 표시 이름
MODEL_DISPLAY = {
    "ko-sroberta": "ko-sroberta",
    "bge-m3": "bge-m3",
    "openai": "openai",
    "hyde-bge-m3": "HyDE",
    "hybrid-bge-m3": "Hybrid",
}


# -------------------------------------------------------------------
# 데이터 로드
# -------------------------------------------------------------------
def load_results() -> pd.DataFrame:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT setting, model_name, llm_model, subject,
                       correct, recall_at_1, recall_at_3, recall_at_5
                FROM experiment_results
                ORDER BY setting, model_name, llm_model, subject
                """
            )
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description]
    return pd.DataFrame(rows, columns=cols)


def _label(setting, model):
    if setting == "vanilla":
        return "Vanilla"
    if setting == "oracle":
        return "Oracle"
    return f"Retrieved/{model}"


def summarize(df: pd.DataFrame, llm_model: str, order: list) -> pd.DataFrame:
    """llm_model별 세팅×모델 Accuracy + Recall@k 집계"""
    sub = df[df["llm_model"] == llm_model]
    rows = []
    for (setting, model), g in sub.groupby(["setting", "model_name"], dropna=False):
        model = model if pd.notna(model) else None
        acc = g["correct"].mean() * 100
        r1 = g["recall_at_1"].mean() * 100 if g["recall_at_1"].notna().any() else None
        r3 = g["recall_at_3"].mean() * 100 if g["recall_at_3"].notna().any() else None
        r5 = g["recall_at_5"].mean() * 100 if g["recall_at_5"].notna().any() else None
        rows.append({
            "label": _label(setting, model),
            "setting": setting,
            "model": model or "-",
            "accuracy": round(acc, 1),
            "recall@1": round(r1, 1) if r1 is not None else "-",
            "recall@3": round(r3, 1) if r3 is not None else "-",
            "recall@5": round(r5, 1) if r5 is not None else "-",
        })
    result = pd.DataFrame(rows)
    result["_sort"] = result["label"].apply(lambda x: order.index(x) if x in order else 99)
    return result.sort_values("_sort").drop(columns="_sort").reset_index(drop=True)


def subject_accuracy(df: pd.DataFrame, llm_model: str, order: list) -> pd.DataFrame:
    """과목별 Accuracy"""
    sub = df[df["llm_model"] == llm_model]
    subjects = ["민사법", "형사법", "공법"]
    rows = []
    for (setting, model), g in sub.groupby(["setting", "model_name"], dropna=False):
        model = model if pd.notna(model) else None
        row = {"label": _label(setting, model)}
        for subj in subjects:
            sg = g[g["subject"] == subj]
            row[subj] = round(sg["correct"].mean() * 100, 1) if len(sg) > 0 else None
        rows.append(row)
    result = pd.DataFrame(rows)
    result["_sort"] = result["label"].apply(lambda x: order.index(x) if x in order else 99)
    return result.sort_values("_sort").drop(columns="_sort").reset_index(drop=True)


# -------------------------------------------------------------------
# 출력
# -------------------------------------------------------------------
def print_summary(summary: pd.DataFrame, title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)
    print(summary[["label", "accuracy", "recall@1", "recall@3", "recall@5"]].to_string(index=False))
    print()


def print_subject(subj_df: pd.DataFrame, title: str):
    print(f"{'=' * 60}")
    print(f"  {title} — 과목별 Accuracy (%)")
    print("=" * 60)
    print(subj_df.to_string(index=False))
    print()


# -------------------------------------------------------------------
# 시각화
# -------------------------------------------------------------------
def plot_accuracy_bar(summary: pd.DataFrame, filename: str, title: str):
    """세팅별 Accuracy 막대 그래프"""
    labels = summary["label"].tolist()
    accs = summary["accuracy"].tolist()
    colors = [COLORS.get(l, "#607D8B") for l in labels]
    disp_labels = [DISPLAY.get(l, l) for l in labels]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(disp_labels, accs, color=colors, width=0.6, edgecolor="white", linewidth=1.2)

    ax.axhline(y=20, color="#9E9E9E", linestyle="--", linewidth=1, label="랜덤 기댓값 (20%)")
    vanilla_acc = summary.loc[summary["label"] == "Vanilla", "accuracy"].values[0]
    ax.axhline(y=vanilla_acc, color="#607D8B", linestyle=":", linewidth=1.2, label=f"Vanilla ({vanilla_acc}%)")

    for bar, val in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val}%",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    y_max = max(accs) * 1.18
    ax.set_ylim(0, max(y_max, 55))
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.tick_params(axis="x", labelsize=9)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    plt.xticks(rotation=10, ha="right")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  저장: {out}")


def plot_recall_accuracy(summary: pd.DataFrame, filename: str, title: str):
    """Recall@5 vs Accuracy 산점도 (Retrieved 세팅만)"""
    retrieved = summary[summary["setting"] == "retrieved"].copy()
    retrieved = retrieved[retrieved["recall@5"] != "-"]
    retrieved["recall@5"] = retrieved["recall@5"].astype(float)

    fig, ax = plt.subplots(figsize=(7, 5))
    for _, row in retrieved.iterrows():
        color = COLORS.get(row["label"], "#607D8B")
        ax.scatter(row["recall@5"], row["accuracy"], color=color, s=120, zorder=5)
        disp = MODEL_DISPLAY.get(row["model"], row["model"])
        ax.annotate(disp, xy=(row["recall@5"], row["accuracy"]), xytext=(5, 5),
                    textcoords="offset points", fontsize=10)

    oracle_acc = summary.loc[summary["label"] == "Oracle", "accuracy"].values[0]
    vanilla_acc = summary.loc[summary["label"] == "Vanilla", "accuracy"].values[0]
    ax.axhline(y=oracle_acc, color="#4CAF50", linestyle="--", linewidth=1.2, label=f"Oracle ({oracle_acc}%)")
    ax.axhline(y=vanilla_acc, color="#9E9E9E", linestyle=":", linewidth=1.2, label=f"Vanilla ({vanilla_acc}%)")

    ax.set_xlabel("Recall@5 (%)", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  저장: {out}")


def plot_recall_at_k(summary: pd.DataFrame, filename: str, title: str):
    """Retrieved 세팅별 Recall@1/3/5 그룹 막대"""
    retrieved = summary[summary["setting"] == "retrieved"].copy()
    models = [MODEL_DISPLAY.get(m, m) for m in retrieved["model"].tolist()]
    k_labels = ["recall@1", "recall@3", "recall@5"]
    k_names = ["Recall@1", "Recall@3", "Recall@5"]
    k_colors = ["#BBDEFB", "#64B5F6", "#1565C0"]

    x = range(len(models))
    width = 0.25

    all_vals = [float(v) for col in k_labels for v in retrieved[col] if v != "-"]
    y_max = max(all_vals) * 1.15 if all_vals else 80

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (k, name, color) in enumerate(zip(k_labels, k_names, k_colors)):
        vals = [float(v) if v != "-" else 0 for v in retrieved[k]]
        offset = (i - 1) * width
        bars = ax.bar([xi + offset for xi in x], vals, width=width, label=name, color=color, edgecolor="white")
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{val:.0f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(list(x))
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylim(0, y_max)
    ax.set_ylabel("Recall (%)", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  저장: {out}")


def plot_subject_accuracy(subj_df: pd.DataFrame, filename: str, title: str):
    """과목별 Accuracy 히트맵"""
    subjects = ["민사법", "형사법", "공법"]
    labels = [DISPLAY.get(l, l) for l in subj_df["label"].tolist()]
    data = subj_df[subjects].values.astype(float)

    vmax = max(data.max(), 50)
    fig, ax = plt.subplots(figsize=(8, max(3, len(labels) * 0.6)))
    im = ax.imshow(data, cmap="RdYlGn", vmin=20, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(subjects)))
    ax.set_xticklabels(subjects, fontsize=11)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    for i in range(len(labels)):
        for j in range(len(subjects)):
            val = data[i, j]
            text_color = "white" if val < 28 or val > vmax * 0.88 else "black"
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center", fontsize=10, color=text_color)

    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04, label="Accuracy (%)")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  저장: {out}")


def plot_llm_comparison(summary_gpt: pd.DataFrame, summary_o4: pd.DataFrame, filename: str):
    """그림 5: gpt-4o-mini vs o4-mini 세팅별 Accuracy 비교"""
    shared_labels = ["Vanilla", "Oracle", "Retrieved/bge-m3", "Retrieved/hybrid-bge-m3"]
    gpt_row = summary_gpt[summary_gpt["label"].isin(shared_labels)].set_index("label")
    o4_row = summary_o4[summary_o4["label"].isin(shared_labels)].set_index("label")

    disp_labels = [DISPLAY.get(l, l) for l in shared_labels]
    gpt_vals = [gpt_row.loc[l, "accuracy"] if l in gpt_row.index else 0 for l in shared_labels]
    o4_vals = [o4_row.loc[l, "accuracy"] if l in o4_row.index else 0 for l in shared_labels]

    x = range(len(shared_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars_gpt = ax.bar([xi - width / 2 for xi in x], gpt_vals, width=width,
                      label="GPT-4o-mini", color="#F44336", edgecolor="white")
    bars_o4 = ax.bar([xi + width / 2 for xi in x], o4_vals, width=width,
                     label="o4-mini", color="#3F51B5", edgecolor="white")

    for bars, vals in [(bars_gpt, gpt_vals), (bars_o4, o4_vals)]:
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.axhline(y=20, color="#9E9E9E", linestyle="--", linewidth=1, label="랜덤 기댓값 (20%)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(disp_labels, fontsize=11)
    ax.set_ylim(0, max(max(gpt_vals), max(o4_vals)) * 1.2)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("GPT-4o-mini vs o4-mini 세팅별 Accuracy 비교", fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  저장: {out}")


# -------------------------------------------------------------------
# 에세이 Judge 데이터 로드
# -------------------------------------------------------------------
def load_essay_results() -> pd.DataFrame:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT setting, model_name,
                       AVG(judge_legal_accuracy)    AS legal_acc,
                       AVG(judge_reasoning_quality) AS reasoning_q,
                       AVG(judge_citation_fidelity) AS citation_f,
                       AVG(judge_total)             AS avg_total,
                       COUNT(*)                     AS n
                FROM essay_results
                GROUP BY setting, model_name
                ORDER BY CASE setting WHEN 'vanilla' THEN 1 WHEN 'oracle' THEN 2 ELSE 3 END
            """)
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description]
    return pd.DataFrame(rows, columns=cols)


def load_error_analysis() -> pd.DataFrame:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT error_type, COUNT(*) AS cnt
                FROM error_analysis
                GROUP BY error_type
                ORDER BY cnt DESC
            """)
            rows = cur.fetchall()
    return pd.DataFrame(rows, columns=["error_type", "cnt"])


# -------------------------------------------------------------------
# 에세이 Judge 시각화
# -------------------------------------------------------------------
def plot_essay_judge(essay_df: pd.DataFrame, filename: str):
    """fig6: 세팅별 에세이 Judge 점수 (3가지 기준 그룹 막대)"""
    metrics = ["legal_acc", "reasoning_q", "citation_f"]
    metric_names = ["Legal Accuracy", "Reasoning Quality", "Citation Fidelity"]

    # 세팅 레이블
    def _essay_label(row):
        if row["setting"] == "vanilla":
            return "Vanilla"
        if row["setting"] == "oracle":
            return "Oracle"
        return f"Retrieved/{row['model_name']}"

    essay_df = essay_df.copy()
    essay_df["label"] = essay_df.apply(_essay_label, axis=1)

    setting_colors = {
        "Vanilla": "#9E9E9E",
        "Oracle": "#4CAF50",
    }
    for label in essay_df["label"]:
        if label not in setting_colors:
            setting_colors[label] = "#F44336"

    labels = essay_df["label"].tolist()
    n_settings = len(labels)
    n_metrics = len(metrics)
    x = range(n_metrics)
    width = 0.22
    offsets = [(i - (n_settings - 1) / 2) * width for i in range(n_settings)]

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (_, row) in enumerate(essay_df.iterrows()):
        vals = [float(row[m]) for m in metrics]
        color = setting_colors.get(row["label"], "#607D8B")
        bars = ax.bar(
            [xi + offsets[i] for xi in x],
            vals,
            width=width,
            label=row["label"],
            color=color,
            edgecolor="white",
        )
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.04,
                f"{val:.2f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
            )

    ax.set_xticks(list(x))
    ax.set_xticklabels(metric_names, fontsize=11)
    ax.set_ylim(1, 5.5)
    ax.set_ylabel("Judge 점수 (1~5점)", fontsize=12)
    ax.set_title("KCL-Essay LLM-as-a-Judge 세팅별 점수 비교", fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  저장: {out}")


def plot_error_analysis(error_df: pd.DataFrame, filename: str):
    """fig7: 오답 유형 분류 (수평 막대 + 비율 표시)"""
    total = error_df["cnt"].sum()
    error_df = error_df.copy()
    error_df["ratio"] = error_df["cnt"] / total * 100

    # 색상 매핑
    color_map = {
        "Misinterpretation": "#EF5350",
        "Distraction": "#FF9800",
        "Noise Dominance": "#42A5F5",
        "Irrelevant": "#66BB6A",
    }
    colors = [color_map.get(t, "#90A4AE") for t in error_df["error_type"]]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(error_df["error_type"], error_df["cnt"], color=colors, edgecolor="white", height=0.5)

    for bar, (_, row) in zip(bars, error_df.iterrows()):
        ax.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{int(row['cnt'])}개  ({row['ratio']:.1f}%)",
            va="center", fontsize=11, fontweight="bold",
        )

    ax.set_xlim(0, error_df["cnt"].max() * 1.45)
    ax.set_xlabel("케이스 수", fontsize=12)
    ax.set_title(
        f"o4-mini+bge-m3 오답 유형 분류  (Recall@5=True, correct=False, n={total})",
        fontsize=12, fontweight="bold", pad=12,
    )
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  저장: {out}")


# -------------------------------------------------------------------
# 메인
# -------------------------------------------------------------------
def main():
    print("결과 데이터 로드 중...")
    df = load_results()
    print(f"  총 {len(df)}개 행 로드 완료")
    print(f"  LLM 모델 목록: {df['llm_model'].unique().tolist()}")

    # --- gpt-4o-mini 분석 ---
    GPT = "gpt-4o-mini"
    summary_gpt = summarize(df, GPT, ORDER_GPT)
    subj_gpt = subject_accuracy(df, GPT, ORDER_GPT)

    print_summary(summary_gpt, f"전체 결과 요약 [{GPT}]")
    print_subject(subj_gpt, GPT)

    vanilla_acc = summary_gpt.loc[summary_gpt["label"] == "Vanilla", "accuracy"].values[0]
    oracle_acc = summary_gpt.loc[summary_gpt["label"] == "Oracle", "accuracy"].values[0]
    retrieved_only = summary_gpt[summary_gpt["setting"] == "retrieved"]
    best_retrieved = retrieved_only["accuracy"].max()
    best_model = retrieved_only.loc[retrieved_only["accuracy"] == best_retrieved, "model"].values[0]

    print("=" * 60)
    print(f"  핵심 수치 [{GPT}]")
    print("=" * 60)
    print(f"  Oracle - Vanilla 갭         : +{oracle_acc - vanilla_acc:.1f}%p")
    print(f"  Oracle - Best Retrieved 갭  : {oracle_acc - best_retrieved:.1f}%p  (best: {best_model})")
    print(f"  Vanilla 초과 모델           : {best_model}  ({best_retrieved}% > {vanilla_acc}%)")
    print()

    # --- o4-mini 분석 ---
    O4 = "o4-mini-2025-04-16"
    summary_o4 = summarize(df, O4, ORDER_O4)
    print_summary(summary_o4, f"전체 결과 요약 [{O4}]")

    # --- 그래프 생성 ---
    print("그래프 생성 중...")
    plot_accuracy_bar(summary_gpt, "fig1_accuracy_bar_gpt4omini.png",
                      f"세팅별 Accuracy 비교 [{GPT}]")
    plot_recall_accuracy(summary_gpt, "fig2_recall_vs_accuracy_gpt4omini.png",
                         f"Recall@5 vs Accuracy [{GPT}]")
    plot_recall_at_k(summary_gpt, "fig3_recall_at_k_gpt4omini.png",
                     f"검색 전략별 Recall@k [{GPT}]")
    plot_subject_accuracy(subj_gpt, "fig4_subject_accuracy_gpt4omini.png",
                          f"과목별 Accuracy [{GPT}]")
    plot_llm_comparison(summary_gpt, summary_o4, "fig5_llm_comparison.png")

    # --- 에세이 Judge + 오답 분류 그래프 ---
    essay_df = load_essay_results()
    error_df = load_error_analysis()

    if not essay_df.empty:
        plot_essay_judge(essay_df, "fig6_essay_judge_scores.png")
    if not error_df.empty:
        plot_error_analysis(error_df, "fig7_error_type_analysis.png")

    print(f"\n완료. 그래프 저장 위치: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
