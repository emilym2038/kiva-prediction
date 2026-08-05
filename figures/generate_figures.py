"""Poster figures for the Kiva moral-distance project. Regenerate any time
the underlying result files change: `python figures/generate_figures.py`.

Reads only already-computed result files under model_output/ -- no model
fitting happens here. Figure 4's source data (GBT SHAP dependence + RF
partial dependence for log_dist, as raw numbers rather than only the
existing rendered PNGs) is produced by
model_output/robustness/export_saturation_curve_data.py; run that first if
model_output/robustness/{gbt_shap_dependence_log_dist_data,rf_pdp_log_dist_data}.csv
don't exist yet.
"""
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# =============================================================================
# shared style -- define once, every figure below reuses this
# =============================================================================
plt.style.use("seaborn-v0_8-whitegrid")

ACCENT = "#d8a983"       # dark teal -- the headline finding (geographic distance / best model)
GREY_DARK = "#4A2F00"    # primary neutral -- everything else
GREY_LIGHT = "#f3cd9f"   # secondary neutral -- only where a 3rd distinct-but-muted tone is needed
GREY_LINE = "#f7e3c8"    # reference lines / gridline-adjacent chrome
INK = "#0b0b0b"
SURFACE = "#fcfcfb"

FONT_TITLE, FONT_LABEL, FONT_TICK, FONT_ANNOT = 20, 16, 14, 13
FIG_WIDTH = 11  # inches -- consistent across every figure so they sit together on the poster

plt.rcParams.update({
    "font.size": FONT_TICK,
    "axes.titlesize": FONT_TITLE,
    "axes.titleweight": "bold",
    "axes.labelsize": FONT_LABEL,
    "xtick.labelsize": FONT_TICK,
    "ytick.labelsize": FONT_TICK,
    "legend.fontsize": FONT_ANNOT,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": GREY_LINE,
    "grid.color": "#e1e0d9",
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": GREY_DARK,
    "ytick.color": GREY_DARK,
})

# plain-English labels -- every figure maps through this, never a raw column name
LABELS = {
    "log_dist": "Geographic distance",
    "comlang_off": "Shared official language",
    "comlang_ethno": "Shared spoken language",
    "contig": "Shared border",
    "col_dep_ever": "Colonial ties",
    "comcol": "Shared colonizer",
    "comrelig": "Religious similarity",
    "diplo_disagreement": "Diplomatic distance",
    "cult_dist": "Cultural distance",
    "asjp_ldnd_dominant": "Linguistic distance",
    "relig_dist": "Religious distance",
    "guillen_admin_dist": "Administrative distance",
    "guillen_demo_dist": "Demographic distance",
    "guillen_econ_dist": "Economic distance",
    "guillen_know_dist": "Knowledge distance",
    "log_amnt": "Loan amount",
    "brwr_pic": "Borrower has photo",
    "desc_en_len": "Description length",
    "female_share": "Female borrower share",
    "has_desc_en": "Has description",
    "chance baseline": "Chance",
    "conditional logit": "Conditional logit",
    "GBT": "Gradient-boosted trees",
    "random forest": "Random forest",
}
DIST_FEATS = ["log_dist", "contig", "comlang_off", "comlang_ethno",
              "col_dep_ever", "comcol", "comrelig", "diplo_disagreement"]
# top 8 of all 15 distance-type features (original + round-1 cultural/
# linguistic + round-2 religious/Guillen) by RF permutation importance --
# supersedes the original-8-only view now that the later rounds exist
FIG1_TOP8 = ["log_dist", "cult_dist", "guillen_know_dist", "comrelig", "guillen_econ_dist",
             "guillen_admin_dist", "asjp_ldnd_dominant", "guillen_demo_dist"]

MODEL_OUT = "model_output"
ROBUST = f"{MODEL_OUT}/robustness"
CONDLOGIT = f"{MODEL_OUT}/conditional_logit"
OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)


def save_all(fig, name):
    for ext in ["svg", "pdf"]:
        fig.savefig(f"{OUT_DIR}/{name}.{ext}", bbox_inches="tight")
    fig.savefig(f"{OUT_DIR}/{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_DIR}/{name}.{{svg,pdf,png}}")


# =============================================================================
# Figure 1 -- distance coefficient plot (conditional logit)
# =============================================================================
def fig1_distance_coefficients():
    # extended spec (headline 8 + cult_dist/asjp_ldnd_dominant/relig_dist/4x
    # Guillen), not the original condlogit_100k_coefficients.csv -- that file
    # only ever fit the original 8, so it can't cover FIG1_TOP8's new features
    df = pd.read_csv(f"{ROBUST}/condlogit_extended_distance_coefficients.csv")
    df = df[df.feature.isin(FIG1_TOP8)].copy()
    df["abs_coef"] = df["coef"].abs()
    df["significant"] = (df["ci_2_5_country_pair"] > 0) | (df["ci_97_5_country_pair"] < 0)
    df = df.sort_values("abs_coef", ascending=True).reset_index(drop=True)  # ascending -> largest ends up at top

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 6.5))
    y = np.arange(len(df))
    for i, row in df.iterrows():
        color = ACCENT if row.feature == "log_dist" else GREY_DARK
        ax.plot([row.ci_2_5_country_pair, row.ci_97_5_country_pair], [i, i], color=color, lw=2.5, solid_capstyle="round")
        ax.plot(row.coef, i, marker="o", markersize=13,
                 markerfacecolor=color if row.significant else "white",
                 markeredgecolor=color, markeredgewidth=2.2, zorder=5)

    ax.axvline(0, color=GREY_DARK, linestyle="--", linewidth=1.5, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([LABELS[f] for f in df.feature])
    for i, row in df.iterrows():
        if row.feature == "log_dist":
            ax.get_yticklabels()[i].set_color(ACCENT)
            ax.get_yticklabels()[i].set_fontweight("bold")

    ax.set_xlabel("Feature coefficient")
    ax.grid(axis="y", visible=False)

    xmin, xmax = ax.get_xlim()
    span = xmax - xmin
    ax.set_xlim(xmin - 0.03 * span, xmax + 0.03 * span)

    legend_elems = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=GREY_DARK, markeredgecolor=GREY_DARK,
               markersize=11, label="95% CI excludes zero"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor=GREY_DARK,
               markersize=11, markeredgewidth=2, label="95% CI includes zero"),
    ]
    ax.legend(handles=legend_elems, loc="lower right", frameon=False, fontsize=FONT_ANNOT)
    fig.tight_layout()
    save_all(fig, "fig1_distance_coefficients")


# =============================================================================
# Figure 2 -- pooled vs. conditional (focused version; see report at bottom)
# =============================================================================
FIG2_FOCUS = ["log_dist", "comrelig", "diplo_disagreement", "comlang_off"]
# chosen for comparably-scaled pooled-model CIs (contig/comcol's pooled
# bootstrap CIs span ~1.0+ log-odds, wide enough on their own to flatten the
# other bars against the axis) -- this set stays legible while still showing
# two reversals (log_dist emerges, comrelig disappears), one stable null
# (diplo_disagreement), and one robust-in-both (comlang_off)


def fig2_pooled_vs_conditional():
    # dot-and-whisker, matched pairs -- same visual language as fig1 (a CI
    # line + a dot, filled/hollow by significance), just two dots per feature
    # row (pooled vs. conditional) instead of one
    pooled = pd.read_csv(f"{MODEL_OUT}/distance_coefficients.csv")
    pooled = pooled[pooled.spec == "main"].set_index("feature")
    cond = pd.read_csv(f"{CONDLOGIT}/condlogit_100k_coefficients.csv").set_index("feature")

    feats = FIG2_FOCUS
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 5.5))
    y = np.arange(len(feats))
    offset = 0.16

    for i, f in enumerate(feats):
        prow, crow = pooled.loc[f], cond.loc[f]
        for yi, row, color in [(i + offset, prow, GREY_DARK), (i - offset, crow, ACCENT)]:
            sig = (row.ci_2_5_country_pair > 0) or (row.ci_97_5_country_pair < 0)
            ax.plot([row.ci_2_5_country_pair, row.ci_97_5_country_pair], [yi, yi],
                    color=color, lw=2.5, solid_capstyle="round")
            ax.plot(row.coef, yi, marker="o", markersize=13,
                    markerfacecolor=color if sig else "white",
                    markeredgecolor=color, markeredgewidth=2.2, zorder=5)

    ax.axvline(0, color=GREY_DARK, linestyle="--", linewidth=1.5, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([LABELS[f] for f in feats])
    ax.set_xlabel("Feature coefficient")
    ax.grid(axis="y", visible=False)

    legend_elems = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=GREY_DARK, markeredgecolor=GREY_DARK,
               markersize=11, label="Pooled model"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=ACCENT, markeredgecolor=ACCENT,
               markersize=11, label="Lender-level model"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor=GREY_DARK,
               markersize=11, markeredgewidth=2, label="95% CI includes zero"),
    ]
    ax.legend(handles=legend_elems, loc="lower right", frameon=False, fontsize=FONT_ANNOT)
    fig.tight_layout()
    save_all(fig, "fig2_pooled_vs_conditional")


# =============================================================================
# Figure 3 -- category composition (stacked bar, RF permutation importance)
# =============================================================================
CATEGORY_MAP = {
    "original_loan": "Loan characteristics",
    "original_sector": "Loan characteristics",
    "original_distance": "Distance",
    "round1_distance": "Distance",
    "round2_distance": "Distance",
    "new_nlp": "Narrative / text (NLP)",
}
CATEGORY_COLORS = {
    "Loan characteristics": GREY_DARK,
    "Distance": ACCENT,
    "Narrative / text (NLP)": GREY_LIGHT,
}
CATEGORY_ORDER = ["Loan characteristics", "Distance", "Narrative / text (NLP)"]


def fig3_category_composition():
    df = pd.read_csv(f"{ROBUST}/rf_feature_importance_round2.csv")
    df["category"] = df["feature_group"].map(CATEGORY_MAP)
    assert df["category"].notna().all(), "unmapped feature_group values"
    totals = df.groupby("category")["permutation_importance_mean"].sum()
    pct = (totals / totals.sum() * 100).reindex(CATEGORY_ORDER)

    # Percentage-only inside each segment (short enough to fit even the
    # narrowest, ~6%-wide slice) -- category names live in the legend below
    # instead of as in-segment text, which overflowed adjacent segments when
    # a full two-line "name + percentage" label was wider than its slice.
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 3.4))
    left = 0
    for cat in CATEGORY_ORDER:
        v = pct[cat]
        ax.barh(0, v, left=left, height=0.6, color=CATEGORY_COLORS[cat],
                edgecolor=SURFACE, linewidth=2, label=f"{cat} ({v:.0f}%)")
        text_color = "white" if cat != "Narrative / text (NLP)" else INK
        ax.text(left + v / 2, 0, f"{v:.0f}%", ha="center", va="center",
                fontsize=FONT_LABEL, color=text_color, fontweight="bold")
        left += v

    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, 0.6)
    ax.set_yticks([])
    ax.set_xlabel("Share of model importance (%)")
    ax.grid(axis="x", visible=True)
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=3, frameon=False, fontsize=FONT_ANNOT)
    fig.tight_layout()
    save_all(fig, "fig3_category_composition")
    return pct


# =============================================================================
# Figure 4 -- distance saturation curve (GBT SHAP + RF PDP)
# =============================================================================
def fig4_saturation():
    gbt_path = f"{ROBUST}/gbt_shap_dependence_log_dist_data.csv"
    rf_path = f"{ROBUST}/rf_pdp_log_dist_data.csv"
    if not (os.path.exists(gbt_path) and os.path.exists(rf_path)):
        print("SKIPPED fig4_saturation -- run export_saturation_curve_data.py first "
              f"(looked for {gbt_path}, {rf_path})")
        return None

    gbt = pd.read_csv(gbt_path)
    rf = pd.read_csv(rf_path)

    shared_min = max(gbt["log_dist"].min(), rf["log_dist"].min())
    shared_max = min(gbt["log_dist"].max(), rf["log_dist"].max())

    # GBT: bin raw SHAP points into a smoothed mean curve (it's a scatter of
    # per-row values, not already a curve, unlike PDP's grid)
    gbt_in = gbt[(gbt.log_dist >= shared_min) & (gbt.log_dist <= shared_max)].sort_values("log_dist")
    window = max(len(gbt_in) // 60, 50)
    gbt_smooth_x = gbt_in["log_dist"].rolling(window, center=True, min_periods=1).mean()
    gbt_smooth_y = gbt_in["shap_value"].rolling(window, center=True, min_periods=1).mean()

    rf_in = rf[(rf.log_dist >= shared_min) & (rf.log_dist <= shared_max)].sort_values("log_dist")
    rf_effect = rf_in["pred_prob"] - rf_in["pred_prob"].mean()

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 6))
    ax.plot(gbt_smooth_x, gbt_smooth_y, color=ACCENT, linewidth=3, label="Gradient-boosted trees")
    ax.plot(rf_in["log_dist"], rf_effect, color=GREY_DARK, linewidth=3, linestyle="--",
            label="Random forest")
    ax.axhline(0, color=GREY_LINE, linewidth=1)

    ax.set_xlabel("Geographic distance (log km)")
    ax.set_ylabel("Effect on funding probability")
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    save_all(fig, "fig4_saturation")
    return shared_min, shared_max


# =============================================================================
# Figure 5 -- model accuracy comparison
# =============================================================================
def fig5_accuracy():
    df = pd.read_csv(f"{ROBUST}/model_comparison.csv")
    order = ["chance baseline", "conditional logit", "GBT", "random forest"]
    df = df.set_index("model").loc[order].reset_index()
    colors = [GREY_LIGHT, "#cfb078", ACCENT, GREY_DARK]
    chance = df.loc[df.model == "chance baseline", "top1_accuracy"].iloc[0]

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 6.5))
    x = np.arange(len(df))
    bars = ax.bar(x, df["top1_accuracy"], color=colors, width=0.6, zorder=3)
    ax.axhline(chance, color=INK, linestyle="--", linewidth=1.8, zorder=4)
    ax.annotate(f"chance = {chance:.3f} (1/6)", xy=(len(df) - 0.5, chance), xytext=(0, 8),
                textcoords="offset points", ha="right", fontsize=FONT_ANNOT, color=INK)

    for xi, row in zip(x, df.itertuples()):
        ax.annotate(f"{row.top1_accuracy:.3f}", xy=(xi, row.top1_accuracy), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=FONT_ANNOT, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[m] for m in df.model], fontsize=FONT_TICK)
    ax.set_ylabel("Top-1 accuracy")
    ax.set_ylim(0, max(df["top1_accuracy"]) * 1.25)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save_all(fig, "fig5_accuracy")


if __name__ == "__main__":
    fig1_distance_coefficients()
    fig2_pooled_vs_conditional()
    fig3_pct = fig3_category_composition()
    fig4_range = fig4_saturation()
    fig5_accuracy()

    report = {
        "fig2_version": "focused",
        "fig2_features_shown": FIG2_FOCUS,
        "fig3_includes_nlp": True,
        "fig3_category_pct": fig3_pct.to_dict() if fig3_pct is not None else None,
        "fig4_shared_x_range_log_dist": list(fig4_range) if fig4_range else "SKIPPED - source data missing",
    }
    print("\n=== REPORT ===")
    print(json.dumps(report, indent=2, default=float))
    with open(f"{OUT_DIR}/generation_report.json", "w") as f:
        json.dump(report, f, indent=2, default=float)
