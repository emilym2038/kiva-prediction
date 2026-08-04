"""Elastic-net logistic regression distance-ranking model for the moral-distance study."""
import json
import time
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

RNG = np.random.default_rng(42)
OUT = "model_output"

DIST_FEATS = ["log_dist", "contig", "comlang_off", "comlang_ethno",
              "col_dep_ever", "comcol", "comrelig", "diplo_disagreement"]
BORROWER_CONT = ["log_amnt", "female_share", "brwr_pic", "desc_en_len"]
BORROWER_FLAG = ["has_desc_en"]
SECTOR_COLS = [f"sector_{i}" for i in range(1, 16)]
LENDER_CONT = ["lender_loan_cnt", "tenure_days"]
LENDER_FLAG = ["tenure_backdated"]

SPARSE_COLS = ["comrelig", "diplo_disagreement"]  # ~1.5% null, median-impute + indicator

print("=== Loading data ===")
df = pd.read_parquet("cleaned_csv/pairs_dataset_fixed.parquet")
print(df.shape)

# ---- Step 2: missingness handling ----
core_dist_cols = ["log_dist", "contig", "comlang_off", "comlang_ethno", "col_dep_ever", "comcol"]
null_before = df[core_dist_cols].isnull().any(axis=1).sum()
print(f"REPORT: residual null rows in core distance cols (log_dist/contig/comlang/col_dep/comcol): {null_before}")
df = df.dropna(subset=core_dist_cols).reset_index(drop=True)
print(f"Rows after dropping residual-null core-distance rows: {len(df)}")

for col in SPARSE_COLS:
    n_missing = df[col].isnull().sum()
    print(f"REPORT: sparse col '{col}' missing rows: {n_missing} ({n_missing/len(df):.4%})")

FEATURE_COLS = DIST_FEATS + BORROWER_CONT + BORROWER_FLAG + SECTOR_COLS + LENDER_CONT + LENDER_FLAG
CONTINUOUS = ["log_dist"] + BORROWER_CONT + LENDER_CONT  # standardize these (+ comrelig/diplo continuous too)
# comrelig & diplo_disagreement are continuous distance features -> standardize as well
CONTINUOUS += ["comrelig", "diplo_disagreement"]
BINARY = ["contig", "comlang_off", "comlang_ethno", "col_dep_ever", "comcol"] + BORROWER_FLAG + SECTOR_COLS + LENDER_FLAG

train_mask = df["split_group"] == "train"
test_mask = df["split_group"] == "test"

# median-impute sparse cols using TRAIN medians only, add missingness indicators
impute_info = {}
for col in SPARSE_COLS:
    med = df.loc[train_mask, col].median()
    ind_col = f"{col}_missing"
    df[ind_col] = df[col].isnull().astype(int)
    df[col] = df[col].fillna(med)
    FEATURE_COLS.append(ind_col)
    BINARY.append(ind_col)
    impute_info[col] = {"train_median": float(med), "n_imputed_total": int(df[ind_col].sum())}
    print(f"Imputed {col} with train median {med:.4f}; touched {df[ind_col].sum()} rows total")

print("=== Preprocessing: standardize continuous, one-hot already present for sector ===")

def build_matrix(frame, scaler=None, fit=False):
    X_cont = frame[CONTINUOUS].to_numpy(dtype=float)
    if fit:
        scaler = StandardScaler().fit(X_cont)
    X_cont_s = scaler.transform(X_cont)
    X_bin = frame[BINARY].to_numpy(dtype=float)
    X = np.hstack([X_cont_s, X_bin])
    cols = CONTINUOUS + BINARY
    return X, cols, scaler

X_train, COLS, scaler = build_matrix(df[train_mask], fit=True)
y_train = df.loc[train_mask, "funded"].to_numpy()
X_test, _, _ = build_matrix(df[test_mask], scaler=scaler)
y_test = df.loc[test_mask, "funded"].to_numpy()
groups_train = df.loc[train_mask, "lender_id"].to_numpy()

print(f"Train matrix: {X_train.shape}, Test matrix: {X_test.shape}")
print(f"Class balance train: {y_train.mean():.4f}, test: {y_test.mean():.4f}")

# ---- Step 4: grouped CV hyperparameter search ----
print("=== Step 4: GroupKFold CV for C / l1_ratio ===")
C_grid = [0.03, 0.1, 0.3, 1.0]
l1_grid = [0.1, 0.5, 0.9]
gkf = GroupKFold(n_splits=3)
splits = list(gkf.split(X_train, y_train, groups=groups_train))

# To keep runtime feasible: coarse search on subsample-informed full folds but saga on full X is expensive.
# Use max_iter reasonable, n_jobs for parallel fits per fold via looping.
cv_results = []
best = None
for C in C_grid:
    for l1 in l1_grid:
        aucs = []
        for tr_idx, val_idx in splits:
            clf = LogisticRegression(penalty="elasticnet", solver="saga", C=C, l1_ratio=l1,
                                      max_iter=200, tol=1e-3)
            clf.fit(X_train[tr_idx], y_train[tr_idx])
            p = clf.predict_proba(X_train[val_idx])[:, 1]
            aucs.append(roc_auc_score(y_train[val_idx], p))
        mean_auc = float(np.mean(aucs))
        cv_results.append({"C": C, "l1_ratio": l1, "cv_auc_mean": mean_auc, "cv_auc_folds": aucs})
        print(f"C={C}, l1_ratio={l1}: CV AUC={mean_auc:.5f} folds={[round(a,4) for a in aucs]}")
        if best is None or mean_auc > best["cv_auc_mean"]:
            best = cv_results[-1]

print(f"REPORT: selected C={best['C']}, l1_ratio={best['l1_ratio']}, CV AUC mean={best['cv_auc_mean']:.5f}")
with open(f"{OUT}/cv_results.json", "w") as f:
    json.dump(cv_results, f, indent=2)

# ---- Step 5: fit final model ----
print("=== Step 5: fit final model on full training set ===")
final_clf = LogisticRegression(penalty="elasticnet", solver="saga", C=best["C"], l1_ratio=best["l1_ratio"],
                                max_iter=2000, tol=1e-4)
final_clf.fit(X_train, y_train)

coef = final_clf.coef_[0]
coef_series = pd.Series(coef, index=COLS)
dist_coef = coef_series[DIST_FEATS + [c for c in SPARSE_COLS]].copy()
# Note comrelig/diplo already in DIST_FEATS list, avoid dup
dist_coef = coef_series.loc[DIST_FEATS]
dist_ranked = dist_coef.reindex(dist_coef.abs().sort_values(ascending=False).index)
print("REPORT: standardized distance coefficients (main spec), ranked by |coef|:")
print(dist_ranked)

n_zero = int((coef_series.loc[DIST_FEATS].abs() < 1e-8).sum())
print(f"REPORT: distance coefficients regularized to exactly zero: {n_zero} / {len(DIST_FEATS)}")

# ---- Step 6: evaluate on lender-grouped test + temporal test ----
print("=== Step 6: evaluation ===")
p_test = final_clf.predict_proba(X_test)[:, 1]
auc_lender = roc_auc_score(y_test, p_test)
pr_lender = average_precision_score(y_test, p_test)
print(f"Lender-grouped test: AUC={auc_lender:.4f}, PR-AUC={pr_lender:.4f} (base rate {y_test.mean():.4f})")

temporal_train_mask = df["split_temporal"] == "train"
temporal_test_mask = df["split_temporal"] == "test"
X_ttrain, _, temporal_scaler = build_matrix(df[temporal_train_mask], fit=True)
y_ttrain = df.loc[temporal_train_mask, "funded"].to_numpy()
X_ttest, _, _ = build_matrix(df[temporal_test_mask], scaler=temporal_scaler)
y_ttest = df.loc[temporal_test_mask, "funded"].to_numpy()

temporal_clf = LogisticRegression(penalty="elasticnet", solver="saga", C=best["C"], l1_ratio=best["l1_ratio"],
                                   max_iter=2000, tol=1e-4)
temporal_clf.fit(X_ttrain, y_ttrain)
p_ttest = temporal_clf.predict_proba(X_ttest)[:, 1]
auc_temporal = roc_auc_score(y_ttest, p_ttest)
pr_temporal = average_precision_score(y_ttest, p_ttest)
print(f"Temporal test: AUC={auc_temporal:.4f}, PR-AUC={pr_temporal:.4f} (base rate {y_ttest.mean():.4f})")

# save primary artifacts so far (bootstrap / partner-FE will append)
joblib.dump(final_clf, f"{OUT}/model_main.joblib")
joblib.dump(scaler, f"{OUT}/scaler_main.joblib")
joblib.dump(COLS, f"{OUT}/feature_cols_main.joblib")

metrics = {
    "selected_C": best["C"],
    "selected_l1_ratio": best["l1_ratio"],
    "cv_auc_mean": best["cv_auc_mean"],
    "cv_auc_folds": best["cv_auc_folds"],
    "n_distance_coefs_zeroed": n_zero,
    "test_lender_grouped": {"auc": auc_lender, "pr_auc": pr_lender, "base_rate": float(y_test.mean()), "n_rows": int(len(y_test))},
    "test_temporal": {"auc": auc_temporal, "pr_auc": pr_temporal, "base_rate": float(y_ttest.mean()), "n_rows": int(len(y_ttest))},
    "n_rows_total_after_cleaning": int(len(df)),
    "n_rows_train_lender_split": int(train_mask.sum()),
    "n_rows_test_lender_split": int(test_mask.sum()),
    "sparse_impute_info": impute_info,
    "residual_null_rows_dropped": int(null_before),
}
with open(f"{OUT}/model_metrics_partial.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("=== STAGE 1 COMPLETE (through Step 6) ===")
