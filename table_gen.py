#!/usr/bin/env python3
"""
Generate Table 2 (AUROC) and Table 3 (RMSE) as PNG images.
Scale cal-housing RMSE values by 100,000 for better readability.
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # non-interactive backend

BASE = "/Users/kwokwailok/Desktop/comp7404"
EPSILON_ORDER = ["0.5", "1.0", "2.0", "4.0", "8.0", "Non-Private"]

CLS_ORDER = ["adult-income", "credit-fraud", "healthcare", "telco-churn"]
REG_ORDER = ["cal-housing", "elevators", "pol", "wine-quality"]

# ══════════════════════════════════════════════════════════════════
#  1. LOAD & UNIFY ALL CSVs
# ══════════════════════════════════════════════════════════════════

def load_unified():
    rows = []

    def _eps(row):
        e = str(row["epsilon"]).strip()
        m = str(row.get("model", ""))
        if "Non" in e or "NonPrivate" in m:
            return "Non-Private"
        return e

    def _scale_val(dataset, metric, value):
        """Scale cal-housing RMSE by 100,000 for readability."""
        if dataset == "cal-housing" and metric == "RMSE":
            return value * 100000
        return value

    # ── EBM results ───────────────────────────────────────────────
    p = os.path.join(BASE, "experiment_results4.csv")
    if os.path.exists(p):
        df = pd.read_csv(p)
        df["epsilon"] = df["epsilon"].astype(str).str.strip()
        for _, r in df.iterrows():
            m = r["model"]
            val = _scale_val(r["dataset"], r["metric_name"], r["metric_value"])
            if m == "EBM-NonPrivate":
                for tm in ["DPEBM-Classic", "DPEBM-GDP"]:
                    rows.append((r["dataset"], tm, "Non-Private", r["metric_name"], val))
            elif m in ("DPEBM-Classic", "DPEBM-GDP"):
                rows.append((r["dataset"], m, r["epsilon"], r["metric_name"], val))

    # ── LR Classification ─────────────────────────────────────────
    p = os.path.join(BASE, "lr_classification_results.csv")
    if os.path.exists(p):
        df = pd.read_csv(p)
        for _, r in df.iterrows():
            rows.append((r["dataset"], "Logistic Regression", _eps(r), "AUROC", r["AUROC"]))

    # ── LR Regression ─────────────────────────────────────────────
    p = os.path.join(BASE, "lr_regression_results.csv")
    if os.path.exists(p):
        df = pd.read_csv(p)
        for _, r in df.iterrows():
            val = _scale_val(r["dataset"], "RMSE", r["RMSE"])
            rows.append((r["dataset"], "Linear Regression", _eps(r), "RMSE", val))

    # ── DPBoost Classification ────────────────────────────────────
    p = os.path.join(BASE, "dpboost_classification_results.csv")
    if os.path.exists(p):
        df = pd.read_csv(p)
        for _, r in df.iterrows():
            rows.append((r["dataset"], "DPBoost", _eps(r), "AUROC", r["AUROC"]))

    # ── DPBoost Regression ────────────────────────────────────────
    p = os.path.join(BASE, "dpboost_regression_results.csv")
    if os.path.exists(p):
        df = pd.read_csv(p)
        for _, r in df.iterrows():
            val = _scale_val(r["dataset"], "RMSE", r["RMSE"])
            rows.append((r["dataset"], "DPBoost", _eps(r), "RMSE", val))

    return pd.DataFrame(rows, columns=["dataset", "model", "epsilon", "metric", "value"])


# ══════════════════════════════════════════════════════════════════
#  2. COMPUTE MEAN ± STD LOOKUP
# ══════════════════════════════════════════════════════════════════

def build_lookup(df):
    g = (df.groupby(["dataset", "model", "epsilon", "metric"])["value"]
           .agg(["mean", "std", "count"]).reset_index())
    g["std"] = g["std"].fillna(0)
    lk = {}
    for _, r in g.iterrows():
        lk[(r["dataset"], r["model"], r["epsilon"], r["metric"])] = (
            r["mean"], r["std"], int(r["count"]))
    return lk


# ══════════════════════════════════════════════════════════════════
#  3. FORMAT HELPER
# ══════════════════════════════════════════════════════════════════

def _fmt(mean, std, metric):
    if metric == "AUROC":
        return f"{mean:.3f} ± {std:.3f}"
    else:
        # For scaled RMSE (like cal-housing) or naturally large values
        if mean >= 1000:
            return f"{mean:,.0f} ± {std:,.0f}"  # Added commas for thousands
        else:
            return f"{mean:.3f} ± {std:.3f}"


def _ordered_datasets(lk, metric, preferred):
    found = list(dict.fromkeys(ds for (ds, m, e, mt) in lk if mt == metric))
    ordered = [d for d in preferred if d in found]
    for d in found:
        if d not in ordered:
            ordered.append(d)
    return ordered


# ══════════════════════════════════════════════════════════════════
#  4. RENDER TABLE AS PNG IMAGE
# ══════════════════════════════════════════════════════════════════

def render_table_image(lk, metric, models, preferred_ds, title, higher_better, output_path):
    datasets = _ordered_datasets(lk, metric, preferred_ds)
    if not datasets:
        print(f"  No data for metric={metric}")
        return

    col_headers = ["DATASET", "ε"] + models
    cell_text = []
    bold_mask = []

    for ds in datasets:
        eps_present = [e for e in EPSILON_ORDER if any((ds, m, e, metric) in lk for m in models)]
        if not eps_present: continue

        for ei, eps in enumerate(eps_present):
            vals = {}
            for m in models:
                key = (ds, m, eps, metric)
                if key in lk:
                    vals[m] = lk[key][0]

            best = None
            if vals:
                best = (max if higher_better else min)(vals.values())

            ds_label = ds.upper() if ei == 0 else ""
            eps_label = eps
            row_text = [ds_label, eps_label]
            row_bold = [False, eps == "Non-Private"]

            for m in models:
                key = (ds, m, eps, metric)
                if key in lk:
                    mean, std, _ = lk[key]
                    s = _fmt(mean, std, metric)
                    is_best = (best is not None and abs(mean - best) < 1e-8)
                    row_text.append(s)
                    row_bold.append(is_best)
                else:
                    row_text.append("—")
                    row_bold.append(False)

            cell_text.append(row_text)
            bold_mask.append(row_bold)

    n_rows, n_cols = len(cell_text), len(col_headers)
    col_widths = [0.14, 0.09] + [0.19] * len(models)
    fig_width = 16
    fig_height = max(3, 0.42 * n_rows + 1.8)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=25, loc="left", fontfamily="serif")

    table = ax.table(cellText=cell_text, colLabels=col_headers, colWidths=col_widths, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)

    # Styling
    for j in range(n_cols):
        cell = table[0, j]
        cell.set_text_props(fontweight="bold", fontfamily="serif", fontsize=10)
        cell.set_facecolor("#d5d5d5")

    for i in range(n_rows):
        for j in range(n_cols):
            cell = table[i + 1, j]
            cell.set_facecolor("#f9f9f9" if i % 2 == 0 else "#ffffff")
            if bold_mask[i][j]:
                cell.set_text_props(fontweight="bold", fontfamily="serif")
            else:
                cell.set_text_props(fontfamily="serif")
            if j == 0 and cell_text[i][0] != "":
                cell.set_text_props(fontweight="bold", ha="left")

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved → {output_path}")


# ══════════════════════════════════════════════════════════════════
#  5. MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    print("Loading and scaling data …")
    df = load_unified()
    lk = build_lookup(df)

    cls_models = ["DPBoost", "Logistic Regression", "DPEBM-Classic", "DPEBM-GDP"]
    reg_models = ["DPBoost", "Linear Regression", "DPEBM-Classic", "DPEBM-GDP"]

    render_table_image(
        lk, "AUROC", cls_models, CLS_ORDER,
        title="Table 2. Area Under the ROC Curve (AUROC) algorithm comparison. Higher is better.",
        higher_better=True,
        output_path=os.path.join(BASE, "table2_auroc.png"),
    )

    render_table_image(
        lk, "RMSE", reg_models, REG_ORDER,
        title="Table 3. Root Mean Squared Error (RMSE) algorithm comparison. Lower is better. (cal-housing scaled ×100k)",
        higher_better=False,
        output_path=os.path.join(BASE, "table3_rmse.png"),
    )

    print("\nDone!")

if __name__ == "__main__":
    main()