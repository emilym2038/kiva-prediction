"""
Build the lender-loan choice dataset for the moral-distance study.

See the project spec (conversation record) for full rationale. This script
implements Steps 0-8:

  0. Load + type-check (MATLAB datenum -> real datetime)
  1. Clean the loan table (required-not-null spine; was_funded is an outcome,
     not missing data)
  2. Clean the lender table (resolve nation_code -> ISO3 via the full Kiva
     country table, which covers lender-only countries too)
  3. Build the (lender_nation, borrower_nation) distance lookup once
  4. Subsample lenders (not pairs) with loan_cnt >= 5
  5. Risk-set negative sampling via a sorted sweep (not pairwise)
  6. Assemble the long feature matrix
  7. Lender-group and temporal split columns
  8. Write parquet outputs + build_summary.json + validation checks

Run `python pairs_dataset_scripts/build_pairs_dataset.py --help` for params.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from matlab_dates import matlab_serial_to_datetime

REPO_ROOT = Path(__file__).resolve().parent.parent


class ShrinkageLog:
    """Accumulates an audit trail of row counts / stats at every pipeline step."""

    def __init__(self):
        self.entries = []

    def log(self, step, note, n=None, **extra):
        entry = {"step": step, "note": note}
        if n is not None:
            entry["n_rows"] = int(n)
        entry.update({k: (int(v) if isinstance(v, (np.integer,)) else
                           float(v) if isinstance(v, (np.floating,)) else v)
                      for k, v in extra.items()})
        self.entries.append(entry)
        suffix = f" | {extra}" if extra else ""
        n_str = f": {n:,}" if n is not None else ""
        print(f"[{step}] {note}{n_str}{suffix}")

    def to_list(self):
        return self.entries


# ---------------------------------------------------------------------------
# Step 0: load + type-check
# ---------------------------------------------------------------------------
def load_and_typecheck(borrower_path, lender_path, log: ShrinkageLog):
    borrowers = pd.read_csv(borrower_path)
    lenders = pd.read_csv(lender_path)
    log.log("step0", "raw borrower_info.csv rows", len(borrowers))
    log.log("step0", "raw lender_info.csv rows", len(lenders))

    for col in ("time_posted", "time_funded"):
        borrowers[col] = matlab_serial_to_datetime(borrowers[col])
        if not pd.api.types.is_datetime64_any_dtype(borrowers[col]):
            raise TypeError(f"borrower_info.{col} failed to convert to datetime64")
    lenders["memb_since"] = matlab_serial_to_datetime(lenders["memb_since"])
    if not pd.api.types.is_datetime64_any_dtype(lenders["memb_since"]):
        raise TypeError("lender_info.memb_since failed to convert to datetime64")

    log.log("step0", "time_posted/time_funded/memb_since confirmed datetime64", None)
    return borrowers, lenders


# ---------------------------------------------------------------------------
# Step 1: clean the loan table
# ---------------------------------------------------------------------------
REQUIRED_LOAN_COLS = [
    "nation_code", "amnt", "amnt_funded", "sector", "partner", "status",
    "brwr_female", "brwr_male", "brwr_pic", "activity", "use", "time_posted",
]


def clean_loans(borrowers: pd.DataFrame, log: ShrinkageLog) -> pd.DataFrame:
    n0 = len(borrowers)
    null_mask = borrowers[REQUIRED_LOAN_COLS].isnull().any(axis=1)
    borrowers = borrowers[~null_mask].copy()
    log.log("step1", "rows dropped for null required-not-null spine column",
            n0 - len(borrowers), pct_of_input=round((n0 - len(borrowers)) / n0 * 100, 3))

    borrowers["was_funded"] = borrowers["time_funded"].notna()
    never_funded = ~borrowers["was_funded"]
    log.log("step1", "never-funded loans kept (time_funded null = outcome, not missing)",
            int(never_funded.sum()), pct=round(never_funded.mean() * 100, 3))

    # Cross-check: never-funded rows should show partial/zero funding, not full.
    ratio = (borrowers.loc[never_funded, "amnt_funded"] / borrowers.loc[never_funded, "amnt"])
    fully_funded_but_never_marked = (ratio >= 0.999).sum()
    log.log("step1", "never-funded rows with amnt_funded>=amnt (inconsistency, kept anyway)",
            int(fully_funded_but_never_marked))
    log.log("step1", "never-funded status code distribution",
            None, status_counts=borrowers.loc[never_funded, "status"].value_counts().to_dict())

    ties = int((borrowers["time_funded"] == borrowers["time_posted"]).sum())
    log.log("step1", "exact-tie rows (time_funded == time_posted, using '<' not '<=')", ties)

    borrowers["has_desc_en"] = borrowers["desc_en"].notna()
    log.log("step1", "desc_en null rate", None,
            pct_null=round((~borrowers["has_desc_en"]).mean() * 100, 3))

    borrowers["was_prefunded"] = borrowers["time_funded"] < borrowers["time_posted"]
    n_prefunded = int(borrowers["was_prefunded"].sum())
    log.log("step1", "was_prefunded rows remaining (should be 0; upstream clean_borrower.py "
                      "already drops these)", n_prefunded)
    if n_prefunded:
        borrowers = borrowers[~borrowers["was_prefunded"]].copy()
        log.log("step1", "dropped residual prefunded rows", n_prefunded)

    return borrowers


# ---------------------------------------------------------------------------
# Step 2: clean the lender table + resolve nation_code -> gravity ISO3
# ---------------------------------------------------------------------------
def clean_lenders(lenders: pd.DataFrame, mapping: pd.DataFrame, log: ShrinkageLog) -> pd.DataFrame:
    n0 = len(lenders)
    required = ["nation_code", "memb_since", "loan_cnt"]
    null_mask = lenders[required].isnull().any(axis=1)
    lenders = lenders[~null_mask].copy()
    log.log("step2", "lender rows dropped for null nation_code/memb_since/loan_cnt",
            n0 - len(lenders))

    iso3_lookup = mapping.set_index("nation_code")["gravity_iso3"]
    lenders["nation_iso3"] = lenders["nation_code"].map(iso3_lookup)
    unresolved = lenders["nation_iso3"].isna()
    log.log("step2", "lender rows with nation_code unresolved to a gravity ISO3 (dropped)",
            int(unresolved.sum()),
            unresolved_nation_codes=sorted(lenders.loc[unresolved, "nation_code"].unique().tolist()))
    lenders = lenders[~unresolved].copy()

    return lenders


# ---------------------------------------------------------------------------
# Step 3: country-pair distance lookup (built once)
# ---------------------------------------------------------------------------
GRAVITY_BINARY_COLS = ["contig", "comlang_off", "comlang_ethno", "col_dep_ever", "comcol", "comrelig"]
GRAVITY_EXTRA_COLS = ["diplo_disagreement"]  # foreign-relations proxy, added per user confirmation


def build_country_pair_lookup(gravity_path, log: ShrinkageLog) -> pd.DataFrame:
    g = pd.read_csv(gravity_path, index_col=0)
    log.log("step3", "raw gravity_filtered_more rows (all years)", len(g))

    check_cols = ["dist"] + GRAVITY_BINARY_COLS
    # Vectorized time-invariance check: groupby + nunique-after-dropna per
    # column, not a per-pair boolean mask scan (that was O(pairs x rows) and
    # took ~5 minutes on its own -- the exact pairwise anti-pattern this
    # pipeline is supposed to avoid, just misplaced in a diagnostic check).
    nunique_per_pair = g.groupby(["iso3_o", "iso3_d"])[check_cols].agg(lambda s: s.dropna().nunique())
    violations = int((nunique_per_pair > 1).to_numpy().sum())
    log.log("step3", "dyadic columns non-constant across years for a pair (dropna'd; "
                      "0 expected for time-invariant CEPII vars)", violations)

    # Collapse to one row per pair. Some dissolved/merged entities (e.g. the
    # Netherlands Antilles, ANT) have NaN dyadic values in later years -- sort
    # so a non-null `dist` row wins the groupby().first() when one exists.
    g = g.sort_values("dist", na_position="last")
    keep_cols = ["iso3_o", "iso3_d", "dist"] + GRAVITY_BINARY_COLS + GRAVITY_EXTRA_COLS
    pairs = g[keep_cols].groupby(["iso3_o", "iso3_d"], as_index=False).first()

    # A handful of CEPII rows (Vatican self-pair, a Serbia/Serbia&Montenegro
    # transition-year artifact) record dist==0, which log() turns into -inf.
    # Treat as missing rather than let a degenerate distance corrupt log_dist.
    zero_dist = int((pairs["dist"] == 0).sum())
    log.log("step3", "pair rows with dist==0 (degenerate; nulled before log, not -inf)", zero_dist)
    pairs.loc[pairs["dist"] <= 0, "dist"] = np.nan
    pairs["log_dist"] = np.log(pairs["dist"])
    pairs = pairs.rename(columns={"iso3_o": "lender_nation", "iso3_d": "borrower_nation"})
    pairs = pairs.drop(columns=["dist"])

    log.log("step3", "collapsed country-pair rows", len(pairs))
    null_rates = pairs[["log_dist"] + GRAVITY_BINARY_COLS + GRAVITY_EXTRA_COLS].isna().mean() * 100
    log.log("step3", "distance-feature null rates (%) in the pair lookup", None,
            null_rates=null_rates.round(3).to_dict())

    return pairs


# ---------------------------------------------------------------------------
# Step 4: subsample lenders
# ---------------------------------------------------------------------------
def subsample_lenders(lenders, borrowers, funding_path, min_loan_cnt, n_lenders, seed, log: ShrinkageLog):
    eligible = lenders[lenders["loan_cnt"] >= min_loan_cnt].copy()
    log.log("step4", f"lenders with loan_cnt >= {min_loan_cnt}", len(eligible), of=len(lenders))

    clean_loan_ids = set(borrowers["loan_id"])
    eligible_lender_ids = set(eligible["lender"])

    funding = pd.read_csv(funding_path, usecols=["lender", "loan_id"])
    log.log("step4", "raw lender_loan_funding.csv edges", len(funding))
    funding = funding[funding["lender"].isin(eligible_lender_ids) & funding["loan_id"].isin(clean_loan_ids)]
    log.log("step4", "edges after restricting to eligible lenders + clean loans", len(funding))

    per_lender_n = funding.groupby("lender").size()
    usable_lender_ids = per_lender_n.index.to_numpy()
    log.log("step4", "eligible lenders with >=1 usable positive edge", len(usable_lender_ids),
            of=len(eligible))

    rng = np.random.default_rng(seed)
    n_lenders = min(n_lenders, len(usable_lender_ids))
    sampled_ids = rng.choice(usable_lender_ids, size=n_lenders, replace=False)
    log.log("step4", "lenders sampled (random sample of LENDERS, not pairs)", n_lenders, seed=seed)

    positives = funding[funding["lender"].isin(set(sampled_ids))].copy()
    log.log("step4", "positive pairs from sampled lenders' full choice history", len(positives))

    sampled_lenders_df = lenders[lenders["lender"].isin(set(sampled_ids))].copy()
    return sampled_lenders_df, positives


# ---------------------------------------------------------------------------
# Step 5: risk-set negative sampling (sorted sweep, not pairwise)
# ---------------------------------------------------------------------------
def make_overlap_bounds(overlap_mode: str, min_overlap_days: float = 0.0):
    """
    Returns a function overlap_ok(q_start, q_end, cand_start, cand_end) -> bool array,
    applied to a candidate set that has ALREADY passed the exact any-overlap
    bounding-box filter (cand_start <= q_end and cand_end >= q_start). This is
    the swappable predicate hook -- 'any' accepts everything that reaches it;
    a stricter variant (min-overlap-duration / midpoint-window) can be dropped
    in here later without touching the sweep/bucket machinery.
    """
    if overlap_mode == "any":
        def predicate(q_start, q_end, cand_start, cand_end):
            return np.ones(len(cand_start), dtype=bool)
        return predicate
    elif overlap_mode == "min_duration":
        min_td = pd.Timedelta(days=min_overlap_days)

        def predicate(q_start, q_end, cand_start, cand_end):
            overlap_start = np.maximum(cand_start.astype("datetime64[ns]"), np.datetime64(q_start))
            overlap_end = np.minimum(cand_end.astype("datetime64[ns]"), np.datetime64(q_end))
            return (overlap_end - overlap_start) >= np.timedelta64(min_td)
        return predicate
    else:
        raise ValueError(f"Unknown overlap_mode: {overlap_mode}")


def build_loan_universe(borrowers: pd.DataFrame, snapshot_end: pd.Timestamp):
    loans = borrowers[["loan_id", "time_posted", "time_funded", "was_funded"]].copy()
    loans["window_end"] = loans["time_funded"].fillna(snapshot_end)
    return loans


def sample_negatives(positives, borrowers, sampled_lenders_df, k, overlap_mode, min_overlap_days,
                      seed, log: ShrinkageLog):
    snapshot_end = max(borrowers["time_posted"].max(), borrowers["time_funded"].max())
    loans = build_loan_universe(borrowers, snapshot_end)
    log.log("step5", "snapshot_end (right-censoring point for never-funded loans)", None,
            snapshot_end=str(snapshot_end))

    funded_loans = loans[loans["was_funded"]].sort_values("time_posted").reset_index(drop=True)
    neverfunded_loans = loans[~loans["was_funded"]].sort_values("time_posted").reset_index(drop=True)

    max_duration_days = (funded_loans["window_end"] - funded_loans["time_posted"]).dt.total_seconds().max() / 86400
    log.log("step5", "max funded-loan duration (days); bounds the sweep lookback window",
            None, max_duration_days=round(max_duration_days, 2))

    f_posted = funded_loans["time_posted"].values.astype("datetime64[ns]")
    f_end = funded_loans["window_end"].values.astype("datetime64[ns]")
    f_ids = funded_loans["loan_id"].to_numpy()

    nf_posted = neverfunded_loans["time_posted"].values.astype("datetime64[ns]")
    nf_ids = neverfunded_loans["loan_id"].to_numpy()

    borrower_by_loan = borrowers.set_index("loan_id")[["time_posted", "time_funded", "was_funded"]]

    lender_memb_since = sampled_lenders_df.set_index("lender")["memb_since"]
    lender_funded_ids = positives.groupby("lender")["loan_id"].apply(lambda s: np.sort(s.to_numpy()))

    overlap_predicate = make_overlap_bounds(overlap_mode, min_overlap_days)
    max_lookback = np.timedelta64(int(np.ceil(max_duration_days)) + 1, "D")

    rng = np.random.default_rng(seed)
    neg_rows = []
    shortfall_count = 0
    t0 = time.time()

    for i, row in enumerate(positives.itertuples(index=False)):
        lender_id, loan_id = row.lender, row.loan_id
        binfo = borrower_by_loan.loc[loan_id]
        q_start = np.datetime64(binfo["time_posted"])
        q_end = np.datetime64(binfo["time_funded"]) if binfo["was_funded"] else np.datetime64(snapshot_end)

        memb_since = np.datetime64(lender_memb_since.get(lender_id))
        lo_bound = max(q_start - max_lookback, memb_since)

        # --- funded candidates: bounding-box any-overlap filter via searchsorted + mask ---
        lo_f = np.searchsorted(f_posted, lo_bound, side="left")
        hi_f = np.searchsorted(f_posted, q_end, side="right")
        cand_f_ids = f_ids[lo_f:hi_f]
        cand_f_posted = f_posted[lo_f:hi_f]
        cand_f_end = f_end[lo_f:hi_f]
        keep_f = cand_f_end >= q_start
        cand_f_ids = cand_f_ids[keep_f]
        cand_f_posted = cand_f_posted[keep_f]
        cand_f_end = cand_f_end[keep_f]

        # --- never-funded candidates: window = [posted, snapshot_end], so any-overlap
        #     reduces to "posted in [lo_bound, q_end]" (their end is always >= q_start
        #     since snapshot_end >= q_start for any in-sample query) ---
        lo_nf = np.searchsorted(nf_posted, lo_bound, side="left")
        hi_nf = np.searchsorted(nf_posted, q_end, side="right")
        cand_nf_ids = nf_ids[lo_nf:hi_nf]
        cand_nf_posted = nf_posted[lo_nf:hi_nf]
        cand_nf_end = np.full(len(cand_nf_ids), np.datetime64(snapshot_end))

        cand_ids = np.concatenate([cand_f_ids, cand_nf_ids])
        cand_posted = np.concatenate([cand_f_posted, cand_nf_posted])
        cand_end = np.concatenate([cand_f_end, cand_nf_end])

        # swappable overlap predicate (identity for 'any', already-exact bounding box above)
        pred_keep = overlap_predicate(q_start, q_end, cand_posted, cand_end)
        cand_ids = cand_ids[pred_keep]

        # exclude the positive loan itself + any other loan this lender actually funded
        own_funded = lender_funded_ids.get(lender_id, np.array([], dtype=cand_ids.dtype))
        cand_ids = cand_ids[~np.isin(cand_ids, own_funded)]

        n_avail = len(cand_ids)
        if n_avail < k:
            shortfall_count += 1
        n_draw = min(k, n_avail)
        if n_draw > 0:
            drawn = rng.choice(cand_ids, size=n_draw, replace=False)
            for nid in drawn:
                neg_rows.append((lender_id, nid, loan_id))

        if (i + 1) % 50000 == 0:
            elapsed = time.time() - t0
            print(f"  ... sampled negatives for {i+1:,}/{len(positives):,} positives "
                  f"({elapsed:.1f}s elapsed)")

    log.log("step5", "positives with fewer than k available candidates (took all available)",
            shortfall_count)
    # `neg_occasions` keeps (lender, negative_loan, positive_loan) provenance --
    # a lender can be offered the same negative loan across two different
    # choice occasions (their candidate windows can overlap), so the same
    # (lender, loan_id) negative pair may be drawn more than once. That
    # duplication carries no extra feature information (none of Step 6's
    # features vary by occasion), so the output table dedupes on
    # (lender, loan_id); `neg_occasions` is kept around only so validation can
    # check each draw against the specific positive window it was sampled for.
    neg_occasions = pd.DataFrame(neg_rows, columns=["lender", "loan_id", "pos_loan_id"])
    neg_df = neg_occasions[["lender", "loan_id"]].drop_duplicates().reset_index(drop=True)
    log.log("step5", "negative draws before dedup", len(neg_occasions), k=k, overlap_mode=overlap_mode)
    log.log("step5", "unique negative (lender, loan) pairs after dedup",
            len(neg_df), n_collisions=len(neg_occasions) - len(neg_df))
    return neg_df, neg_occasions


# ---------------------------------------------------------------------------
# Step 6: assemble the feature matrix
# ---------------------------------------------------------------------------
def zscore(s: pd.Series):
    mu, sd = s.mean(), s.std()
    return (s - mu) / sd if sd > 0 else s * 0.0


def assemble_features(positives, negatives, borrowers, sampled_lenders_df, mapping, pair_lookup, log: ShrinkageLog):
    pos = positives[["lender", "loan_id"]].copy()
    pos["funded"] = 1
    neg = negatives[["lender", "loan_id"]].copy()
    neg["funded"] = 0
    pairs = pd.concat([pos, neg], ignore_index=True)
    log.log("step6", "assembled rows before feature join (positives + negatives)", len(pairs),
            n_pos=len(pos), n_neg=len(neg))

    loan_cols = ["loan_id", "amnt", "sector", "brwr_female", "brwr_male", "brwr_pic",
                 "has_desc_en", "desc_en", "partner", "nation_code", "time_posted", "was_prefunded"]
    pairs = pairs.merge(borrowers[loan_cols], on="loan_id", how="left")

    nation_lookup = mapping.set_index("nation_code")["gravity_iso3"]
    pairs["borrower_nation"] = pairs["nation_code"].map(nation_lookup)

    lender_cols = ["lender", "nation_iso3", "loan_cnt", "memb_since"]
    lc = sampled_lenders_df[lender_cols].rename(columns={"nation_iso3": "lender_nation",
                                                          "loan_cnt": "lender_loan_cnt"})
    pairs = pairs.merge(lc, on="lender", how="left")

    before = len(pairs)
    pairs = pairs.merge(pair_lookup, on=["lender_nation", "borrower_nation"], how="left")
    dist_cols = ["log_dist"] + GRAVITY_BINARY_COLS + GRAVITY_EXTRA_COLS
    dist_null = pairs[dist_cols].isna().any(axis=1)
    log.log("step6", "rows with unresolved country-pair distance join (kept, flagged null)",
            int(dist_null.sum()))
    missing_pairs = (pairs.loc[dist_null, ["lender_nation", "borrower_nation"]]
                     .drop_duplicates().to_dict("records"))
    if missing_pairs:
        log.log("step6", "distinct country pairs that failed the distance join", len(missing_pairs),
                sample=missing_pairs[:20])
    assert len(pairs) == before, "distance join changed row count -- pair_lookup must be deduplicated"

    # borrower controls
    pairs["log_amnt"] = np.log(pairs["amnt"])
    pairs["female_share"] = np.where(
        (pairs["brwr_female"] + pairs["brwr_male"]) > 0,
        pairs["brwr_female"] / (pairs["brwr_female"] + pairs["brwr_male"]),
        np.nan,
    )
    pairs["desc_en_len"] = pairs["desc_en"].fillna("").str.len()  # hook: swap for an embedding later
    sector_dummies = pd.get_dummies(pairs["sector"].astype("Int64"), prefix="sector")
    pairs = pd.concat([pairs, sector_dummies], axis=1)

    # lender controls
    pairs["tenure_days"] = (pairs["time_posted"] - pairs["memb_since"]).dt.total_seconds() / 86400
    neg_tenure = (pairs["tenure_days"] < 0).sum()
    log.log("step6", "rows with negative tenure_days (loan posted before lender joined; "
                      "should not occur given memb_since floor in Step 5, sanity check)", int(neg_tenure))

    # standardized (z-scored) versions of skewed continuous features -- computed
    # over the full assembled sample (splits are logical columns, not physical
    # partitions yet, so there is no single "train" to fit a scaler on here;
    # re-standardize per chosen split at modeling time if leakage sensitivity matters).
    pairs["log_dist_z"] = zscore(pairs["log_dist"])
    pairs["log_amnt_z"] = zscore(pairs["log_amnt"])
    pairs["tenure_days_z"] = zscore(pairs["tenure_days"])
    pairs["lender_loan_cnt_z"] = zscore(np.log1p(pairs["lender_loan_cnt"]))
    pairs["brwr_pic_z"] = zscore(np.log1p(pairs["brwr_pic"]))

    # bookkeeping
    pairs["country_pair"] = pairs["lender_nation"].astype(str) + "_" + pairs["borrower_nation"].astype(str)
    pairs["nation_imputed"] = False  # no nation_code imputation happens in this pipeline
    pairs = pairs.rename(columns={"lender": "lender_id"})

    keep_cols = (
        ["lender_id", "loan_id", "funded"] +
        dist_cols +
        ["log_dist_z", "log_amnt", "log_amnt_z", "female_share", "brwr_pic", "brwr_pic_z",
         "has_desc_en", "desc_en_len", "partner", "lender_loan_cnt", "lender_loan_cnt_z",
         "tenure_days", "tenure_days_z"] +
        list(sector_dummies.columns) +
        ["country_pair", "time_posted", "was_prefunded", "nation_imputed"]
    )
    pairs = pairs[keep_cols]
    log.log("step6", "final assembled feature matrix rows/cols", len(pairs), n_cols=len(pairs.columns))
    return pairs


# ---------------------------------------------------------------------------
# Step 7: splits (columns, not physical partitions)
# ---------------------------------------------------------------------------
def add_splits(pairs: pd.DataFrame, test_frac: float, seed: int, log: ShrinkageLog):
    rng = np.random.default_rng(seed)
    # pandas 3.0 returns an Arrow-backed StringArray from .unique() on string
    # columns; np.random.Generator.shuffle silently mishandles non-ndarray
    # sequences (can duplicate/drop entries), so force a plain object ndarray.
    unique_lenders = pairs["lender_id"].unique().to_numpy(dtype=object)
    rng.shuffle(unique_lenders)
    n_test = int(len(unique_lenders) * test_frac)
    test_lenders = set(unique_lenders[:n_test])
    pairs["split_group"] = np.where(pairs["lender_id"].isin(test_lenders), "test", "train")
    log.log("step7", "GroupShuffleSplit by lender_id", None,
            n_test_lenders=n_test, n_train_lenders=len(unique_lenders) - n_test,
            test_frac_rows=round((pairs["split_group"] == "test").mean(), 4))

    threshold = pairs["time_posted"].quantile(1 - test_frac)
    pairs["split_temporal"] = np.where(pairs["time_posted"] >= threshold, "test", "train")
    log.log("step7", "temporal split threshold (train = earlier, test = later time_posted)",
            None, threshold=str(threshold),
            test_frac_rows=round((pairs["split_temporal"] == "test").mean(), 4))
    return pairs


# ---------------------------------------------------------------------------
# Step 8: validation checks
# ---------------------------------------------------------------------------
def run_validations(pairs, positives, borrowers, sampled_lenders_df, neg_occasions, k, log: ShrinkageLog):
    checks = {}

    # 1. funded==1 rows reproduce exactly the original positives for sampled lenders
    recon = set(zip(pairs.loc[pairs["funded"] == 1, "lender_id"], pairs.loc[pairs["funded"] == 1, "loan_id"]))
    orig = set(zip(positives["lender"], positives["loan_id"]))
    checks["check1_positive_reconstruction_exact"] = (recon == orig)
    checks["check1_n_positive_rows"] = len(recon)
    checks["check1_n_original_positives"] = len(orig)

    # 2. every negative draw's loan window overlaps the specific positive window
    #    it was sampled against, and was posted after that lender's memb_since.
    #    Checked against `neg_occasions` (pre-dedup, so provenance is intact),
    #    not the deduped output table.
    borrower_by_loan = borrowers.set_index("loan_id")[["time_posted", "time_funded", "was_funded"]]
    memb_since = sampled_lenders_df.set_index("lender")["memb_since"]
    snapshot_end = max(borrowers["time_posted"].max(), borrowers["time_funded"].max())

    def window(loan_id):
        b = borrower_by_loan.loc[loan_id]
        return b["time_posted"], (b["time_funded"] if b["was_funded"] else snapshot_end)

    bad_overlap = 0
    bad_memb = 0
    sample_check = neg_occasions.sample(min(20000, len(neg_occasions)), random_state=0)
    for row in sample_check.itertuples(index=False):
        n_start, n_end = window(row.loan_id)
        p_start, p_end = window(row.pos_loan_id)
        if not (n_start <= p_end and p_start <= n_end):
            bad_overlap += 1
        ms = memb_since.get(row.lender)
        if pd.notna(ms) and n_start < ms:
            bad_memb += 1
    checks["check2_negatives_window_does_not_overlap_its_positive_in_sample"] = bad_overlap
    checks["check2_negatives_posted_before_memb_since_in_sample"] = bad_memb
    checks["check2_sample_size"] = len(sample_check)

    # 3. no was_prefunded rows
    checks["check3_was_prefunded_rows"] = int(pairs["was_prefunded"].sum())

    # 4. distance-feature null rates
    dist_cols = ["log_dist"] + GRAVITY_BINARY_COLS + GRAVITY_EXTRA_COLS
    checks["check4_distance_null_rates_pct"] = (pairs[dist_cols].isna().mean() * 100).round(3).to_dict()

    # 5. class balance
    actual_ratio = pairs["funded"].mean()
    intended_ratio = 1 / (k + 1)
    checks["check5_class_balance_actual"] = round(float(actual_ratio), 4)
    checks["check5_class_balance_intended"] = round(intended_ratio, 4)
    checks["check5_within_10pct_relative"] = abs(actual_ratio - intended_ratio) / intended_ratio < 0.10

    # 6. dtypes
    checks["check6_time_posted_is_datetime"] = str(pairs["time_posted"].dtype).startswith("datetime64")

    for k_, v_ in checks.items():
        log.log("step8_validation", k_, None, value=v_)
    return checks


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--borrower-csv", default=str(REPO_ROOT / "cleaned_csv" / "borrower_info.csv"))
    ap.add_argument("--lender-csv", default=str(REPO_ROOT / "cleaned_csv" / "lender_info.csv"))
    ap.add_argument("--funding-csv", default=str(REPO_ROOT / "cleaned_csv" / "lender_loan_funding.csv"))
    ap.add_argument("--gravity-csv", default=str(REPO_ROOT / "distance_data" / "gravity_filtered_more.csv"))
    ap.add_argument("--mapping-csv", default=str(REPO_ROOT / "distance_data" / "kiva_country_mapping.csv"))
    ap.add_argument("--output-dir", default=str(REPO_ROOT / "cleaned_csv"))
    ap.add_argument("--min-loan-cnt", type=int, default=5)
    ap.add_argument("--n-lenders", type=int, default=10000)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--overlap-mode", choices=["any", "min_duration"], default="any")
    ap.add_argument("--min-overlap-days", type=float, default=0.0)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    log = ShrinkageLog()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    borrowers, lenders = load_and_typecheck(args.borrower_csv, args.lender_csv, log)
    borrowers = clean_loans(borrowers, log)

    mapping = pd.read_csv(args.mapping_csv, keep_default_na=False, na_values=[""])
    lenders = clean_lenders(lenders, mapping, log)

    pair_lookup = build_country_pair_lookup(args.gravity_csv, log)
    pair_lookup.to_parquet(out_dir / "country_pair_distances.parquet", index=False)
    log.log("step3", f"wrote {out_dir / 'country_pair_distances.parquet'}", None)

    sampled_lenders_df, positives = subsample_lenders(
        lenders, borrowers, args.funding_csv, args.min_loan_cnt, args.n_lenders, args.seed, log)

    negatives, neg_occasions = sample_negatives(
        positives, borrowers, sampled_lenders_df, args.k, args.overlap_mode, args.min_overlap_days,
        args.seed, log)

    pairs = assemble_features(positives, negatives, borrowers, sampled_lenders_df, mapping, pair_lookup, log)
    pairs = add_splits(pairs, args.test_frac, args.seed, log)

    checks = run_validations(pairs, positives, borrowers, sampled_lenders_df, neg_occasions, args.k, log)

    pairs.to_parquet(out_dir / "pairs_dataset.parquet", index=False)
    log.log("step8", f"wrote {out_dir / 'pairs_dataset.parquet'}", len(pairs))

    summary = {
        "params": vars(args),
        "log": log.to_list(),
        "validation_checks": checks,
    }
    with open(out_dir / "build_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.log("step8", f"wrote {out_dir / 'build_summary.json'}", None)


if __name__ == "__main__":
    main()
