"""Shared preprocessing for the elastic-net distance-ranking model."""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

DIST_FEATS = ["log_dist", "contig", "comlang_off", "comlang_ethno",
              "col_dep_ever", "comcol", "comrelig", "diplo_disagreement"]
BORROWER_CONT = ["log_amnt", "female_share", "brwr_pic", "desc_en_len"]
BORROWER_FLAG = ["has_desc_en"]
SECTOR_COLS = [f"sector_{i}" for i in range(1, 16)]
LENDER_CONT = ["lender_loan_cnt", "tenure_days"]
LENDER_FLAG = ["tenure_backdated"]
SPARSE_COLS = ["comrelig", "diplo_disagreement"]

CONTINUOUS = ["log_dist"] + BORROWER_CONT + LENDER_CONT + ["comrelig", "diplo_disagreement"]
BINARY = ["contig", "comlang_off", "comlang_ethno", "col_dep_ever", "comcol"] + BORROWER_FLAG + SECTOR_COLS + LENDER_FLAG + ["comrelig_missing", "diplo_disagreement_missing"]

CORE_DIST_COLS = ["log_dist", "contig", "comlang_off", "comlang_ethno", "col_dep_ever", "comcol"]


def load_clean(path="cleaned_csv/pairs_dataset_fixed.parquet", train_mask_col="split_group"):
    df = pd.read_parquet(path)
    df = df.dropna(subset=CORE_DIST_COLS).reset_index(drop=True)
    train_mask = df[train_mask_col] == "train"
    for col in SPARSE_COLS:
        med = df.loc[train_mask, col].median()
        ind_col = f"{col}_missing"
        df[ind_col] = df[col].isnull().astype(int)
        df[col] = df[col].fillna(med)
    return df


def build_matrix(frame, scaler=None, fit=False, extra_cont=None, extra_bin=None):
    cont_cols = CONTINUOUS + (extra_cont or [])
    bin_cols = BINARY + (extra_bin or [])
    X_cont = frame[cont_cols].to_numpy(dtype=float)
    if fit:
        scaler = StandardScaler().fit(X_cont)
    X_cont_s = scaler.transform(X_cont)
    X_bin = frame[bin_cols].to_numpy(dtype=float)
    X = np.hstack([X_cont_s, X_bin])
    cols = cont_cols + bin_cols
    return X, cols, scaler
