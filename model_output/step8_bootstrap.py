"""Step 8: cluster bootstrap (country_pair primary, partner secondary) for CIs on distance coefs."""
import json
import time
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from common import load_clean, build_matrix, DIST_FEATS

OUT = "model_output"
N_BOOT = 300

with open(f"{OUT}/model_metrics_partial.json") as f:
    metrics = json.load(f)
C = metrics["selected_C"]
l1 = metrics["selected_l1_ratio"]

df = load_clean()
train_df = df[df["split_group"] == "train"].reset_index(drop=True)


def one_boot(seed, cluster_col):
    rng = np.random.default_rng(seed)
    clusters = train_df[cluster_col].unique()
    sampled = rng.choice(clusters, size=len(clusters), replace=True)
    # build resampled frame by concatenating rows for each sampled cluster
    idx_by_cluster = train_df.groupby(cluster_col).indices
    idxs = np.concatenate([idx_by_cluster[c] for c in sampled])
    boot_df = train_df.iloc[idxs]
    y = boot_df["funded"].to_numpy()
    if y.min() == y.max():
        return None
    X, cols, _ = build_matrix(boot_df, fit=True)
    clf = LogisticRegression(penalty="elasticnet", solver="saga", C=C, l1_ratio=l1,
                              max_iter=150, tol=1e-3)
    clf.fit(X, y)
    coef = pd.Series(clf.coef_[0], index=cols)
    return coef.loc[DIST_FEATS].to_dict()


def run_bootstrap(cluster_col, n_boot=N_BOOT, n_jobs=-1):
    t0 = time.time()
    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(one_boot)(seed, cluster_col) for seed in range(n_boot)
    )
    results = [r for r in results if r is not None]
    print(f"{cluster_col} bootstrap: {len(results)}/{n_boot} valid reps in {time.time()-t0:.1f}s")
    boot_df = pd.DataFrame(results)
    ci = pd.DataFrame({
        "p2_5": boot_df.quantile(0.025),
        "p97_5": boot_df.quantile(0.975),
        "boot_mean": boot_df.mean(),
        "boot_std": boot_df.std(),
        "n_valid_reps": len(results),
    })
    return ci, boot_df


if __name__ == "__main__":
    ci_pair, boot_pair = run_bootstrap("country_pair", N_BOOT)
    print("REPORT: country_pair-clustered bootstrap CIs:")
    print(ci_pair)
    ci_pair.to_csv(f"{OUT}/bootstrap_ci_country_pair.csv")
    boot_pair.to_csv(f"{OUT}/bootstrap_draws_country_pair.csv", index=False)

    ci_partner, boot_partner = run_bootstrap("partner", N_BOOT)
    print("REPORT: partner-clustered bootstrap CIs:")
    print(ci_partner)
    ci_partner.to_csv(f"{OUT}/bootstrap_ci_partner.csv")
    boot_partner.to_csv(f"{OUT}/bootstrap_draws_partner.csv", index=False)

    print("=== STEP 8 COMPLETE ===")
