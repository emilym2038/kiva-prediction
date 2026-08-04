"""
Remove Kiva records whose country has no match in the gravity dataset
(see distance_scripts/match_countries.py and distance_data/kiva_country_mapping.csv).

Drops directly by nation_code from borrower_info.csv / lender_info.csv, then
cascades to the tables keyed off lender id / loan_id (lender_gender.csv,
lender_loan_funding.csv, lender_loan_pairs.csv) so no dangling references to a
removed borrower or lender remain.
"""
import pandas as pd

mapping = pd.read_csv("distance_data/kiva_country_mapping.csv")
unmatched_codes = set(mapping.loc[~mapping["matched"], "nation_code"])
print(f"Unmatched nation_codes: {sorted(unmatched_codes)}")

# --- borrower_info.csv ---
borrowers = pd.read_csv("cleaned_csv/borrower_info.csv")
drop_mask = borrowers["nation_code"].isin(unmatched_codes)
removed_loan_ids = set(borrowers.loc[drop_mask, "loan_id"])
print(f"borrower_info.csv: removing {drop_mask.sum()} of {len(borrowers)} rows")
borrowers[~drop_mask].to_csv("cleaned_csv/borrower_info.csv", index=False)
del borrowers

# --- lender_info.csv ---
lenders = pd.read_csv("cleaned_csv/lender_info.csv")
drop_mask = lenders["nation_code"].isin(unmatched_codes)
removed_lender_ids = set(lenders.loc[drop_mask, "lender"])
print(f"lender_info.csv: removing {drop_mask.sum()} of {len(lenders)} rows")
lenders[~drop_mask].to_csv("cleaned_csv/lender_info.csv", index=False)
del lenders

# --- lender_gender.csv (keyed by lender) ---
gender = pd.read_csv("cleaned_csv/lender_gender.csv")
drop_mask = gender["lender"].isin(removed_lender_ids)
print(f"lender_gender.csv: removing {drop_mask.sum()} of {len(gender)} rows")
gender[~drop_mask].to_csv("cleaned_csv/lender_gender.csv", index=False)
del gender

# --- lender_loan_funding.csv / lender_loan_pairs.csv (keyed by lender, loan_id) ---
for path in ["cleaned_csv/lender_loan_funding.csv", "cleaned_csv/lender_loan_pairs.csv"]:
    df = pd.read_csv(path)
    drop_mask = df["lender"].isin(removed_lender_ids) | df["loan_id"].isin(removed_loan_ids)
    print(f"{path}: removing {drop_mask.sum()} of {len(df)} rows")
    df[~drop_mask].to_csv(path, index=False)
    del df

print("Done.")
