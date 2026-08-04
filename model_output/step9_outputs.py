"""Step 9: assemble final outputs -- coefficients CSV, metrics JSON, ranking figure."""
import json
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "model_output"

main_coef = pd.read_csv(f"{OUT}/distance_coefficients_main_only.csv", index_col=0)
fe_compare = pd.read_csv(f"{OUT}/partner_fe_comparison.csv", index_col=0)
ci_pair = pd.read_csv(f"{OUT}/bootstrap_ci_country_pair.csv", index_col=0)
ci_partner = pd.read_csv(f"{OUT}/bootstrap_ci_partner.csv", index_col=0)

# ---- combined coefficient table ----
main_out = pd.DataFrame({
    "feature": main_coef.index,
    "spec": "main",
    "coef": main_coef["coef"].values,
    "sign": main_coef["sign"].values,
    "ci_2_5_country_pair": ci_pair.loc[main_coef.index, "p2_5"].values,
    "ci_97_5_country_pair": ci_pair.loc[main_coef.index, "p97_5"].values,
    "ci_2_5_partner": ci_partner.loc[main_coef.index, "p2_5"].values,
    "ci_97_5_partner": ci_partner.loc[main_coef.index, "p97_5"].values,
})
fe_out = pd.DataFrame({
    "feature": fe_compare.index,
    "spec": "partner_fe",
    "coef": fe_compare["coef_partner_fe"].values,
    "sign": np.where(fe_compare["coef_partner_fe"].values < 0, "negative", "positive"),
    "ci_2_5_country_pair": np.nan,
    "ci_97_5_country_pair": np.nan,
    "ci_2_5_partner": np.nan,
    "ci_97_5_partner": np.nan,
})
combined = pd.concat([main_out, fe_out], ignore_index=True)
combined.to_csv(f"{OUT}/distance_coefficients.csv", index=False)
print("Wrote distance_coefficients.csv")
print(combined)

# ---- update model_metrics.json ----
with open(f"{OUT}/model_metrics_partial.json") as f:
    metrics = json.load(f)
metrics["partner_fe"] = {
    "n_partner_dummies": 243,
}
with open(f"{OUT}/model_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("Wrote model_metrics.json")

# ---- ranking figure ----
ranked = main_coef.sort_values("coef", key=lambda s: s.abs(), ascending=True)
features = ranked.index.tolist()
coefs = ranked["coef"].values
lo = ci_pair.loc[features, "p2_5"].values
hi = ci_pair.loc[features, "p97_5"].values
err_lo = coefs - lo
err_hi = hi - coefs

POS_COLOR = "#2166AC"  # blue, positive assoc. with funding
NEG_COLOR = "#B2182B"  # red, negative assoc. with funding
colors = [POS_COLOR if c >= 0 else NEG_COLOR for c in coefs]

fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
y_pos = np.arange(len(features))
ax.barh(y_pos, coefs, color=colors, height=0.6, zorder=3)
ax.errorbar(coefs, y_pos, xerr=[err_lo, err_hi], fmt="none", ecolor="#333333",
            elinewidth=1.2, capsize=3, zorder=4)
ax.axvline(0, color="#999999", linewidth=1, zorder=2)
ax.set_yticks(y_pos)
ax.set_yticklabels(features)
ax.set_xlabel("Standardized coefficient (log-odds of funding)")
ax.set_title("Distance-type ranking: standardized elastic-net coefficients\n(bootstrap 95% CI, clustered by country_pair)")
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
ax.grid(axis="x", color="#e5e5e5", linewidth=0.8, zorder=1)
plt.tight_layout()
plt.savefig(f"{OUT}/coefficient_ranking.png")
print("Wrote coefficient_ranking.png")
