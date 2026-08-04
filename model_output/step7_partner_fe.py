"""Step 7: partner fixed-effects robustness spec."""
import json
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from common import load_clean, build_matrix, DIST_FEATS

OUT = "model_output"

with open(f"{OUT}/model_metrics_partial.json") as f:
    metrics = json.load(f)
C = metrics["selected_C"]
l1 = metrics["selected_l1_ratio"]

df = load_clean()
train_mask = df["split_group"] == "train"

partner_dummies = pd.get_dummies(df["partner"].astype(int).astype(str), prefix="partner", drop_first=True)
partner_cols = partner_dummies.columns.tolist()
df = pd.concat([df, partner_dummies], axis=1)

X_train, COLS, scaler = build_matrix(df[train_mask], fit=True, extra_bin=partner_cols)
y_train = df.loc[train_mask, "funded"].to_numpy()

print(f"Partner-FE train matrix: {X_train.shape} ({len(partner_cols)} partner dummies)")

clf = LogisticRegression(penalty="elasticnet", solver="saga", C=C, l1_ratio=l1,
                          max_iter=2000, tol=1e-4)
clf.fit(X_train, y_train)

coef_series = pd.Series(clf.coef_[0], index=COLS)
dist_coef_fe = coef_series.loc[DIST_FEATS]

main_coef = pd.read_csv(f"{OUT}/distance_coefficients_main_only.csv", index_col=0)["coef"]

compare = pd.DataFrame({
    "coef_main": main_coef,
    "coef_partner_fe": dist_coef_fe,
})
compare["abs_main_rank"] = compare["coef_main"].abs().rank(ascending=False)
compare["abs_fe_rank"] = compare["coef_partner_fe"].abs().rank(ascending=False)
compare = compare.sort_values("abs_main_rank")
print("REPORT: distance coefficients, main spec vs partner-FE spec:")
print(compare)

compare.to_csv(f"{OUT}/partner_fe_comparison.csv")
joblib.dump(clf, f"{OUT}/model_partner_fe.joblib")
joblib.dump(scaler, f"{OUT}/scaler_partner_fe.joblib")
joblib.dump(COLS, f"{OUT}/feature_cols_partner_fe.joblib")

print("=== STEP 7 COMPLETE ===")
