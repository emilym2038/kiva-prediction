"""
Determine which countries appear in the Kiva data (borrower_info.csv and
lender_info.csv) and match each one to a country_id in the gravity dataset
(distance_data/Countries_V202211.csv).

Kiva's own country table (KivaMatlabData/01_N_Country.mat) is the source of
truth for nation_code -> ISO2 code, NOT the exported raw_csv/01_Country_N.csv.
That CSV was built from N['var']['name']/N['var']['code'], which are only
populated for the first 75 rows (borrower countries) -- rows 75-229 (the
lender-only countries, e.g. Australia, Canada, UK, ...) are blank in those
two fields. N['map']['idx2id'], however, is fully populated for all 230
countries and is what nation_code actually indexes into (nation_code is
MATLAB's 1-based row number, so idx = nation_code - 1).
"""
import scipy.io as sio
import pandas as pd

# Kiva uses a couple of non-standard ISO2 codes that don't exist in the
# gravity dataset. Map them to the code the gravity data actually uses.
MANUAL_ISO2_OVERRIDES = {
    "QS": "SS",  # Kiva's code for South Sudan; real ISO 3166-1 alpha-2 is 'SS'
    "GZ": "PS",  # Kiva tracks Gaza separately from the West Bank, but the
                 # gravity dataset only has one unified Palestine ('PS') entry
}

# --- 1. Load Kiva's nation_code -> ISO2 table straight from the .mat file ---
country_mat = sio.loadmat("KivaMatlabData/01_N_Country.mat", simplify_cells=True)["N"]
idx2iso2 = country_mat["map"]["idx2id"]
idx2name = country_mat["var"]["name"]

kiva_countries = pd.DataFrame({
    "nation_code": range(1, len(idx2iso2) + 1),
    "kiva_iso2": idx2iso2,
    "kiva_name": idx2name,
})

# --- 2. Find which nation_codes actually show up in the borrower/lender data ---
borrower_codes = set(
    pd.read_csv("cleaned_csv/borrower_info.csv", usecols=["nation_code"])
    ["nation_code"].dropna().astype(int)
)
lender_codes = set(
    pd.read_csv("cleaned_csv/lender_info.csv", usecols=["nation_code"])
    ["nation_code"].dropna().astype(int)
)

kiva_countries["in_borrowers"] = kiva_countries["nation_code"].isin(borrower_codes)
kiva_countries["in_lenders"] = kiva_countries["nation_code"].isin(lender_codes)
kiva_countries = kiva_countries[kiva_countries["in_borrowers"] | kiva_countries["in_lenders"]].copy()

kiva_countries["match_iso2"] = kiva_countries["kiva_iso2"].replace(MANUAL_ISO2_OVERRIDES)

# --- 3. Load the gravity country table and resolve iso2 -> country_id ---
gravity = pd.read_csv("distance_data/Countries_V202211.csv", keep_default_na=False,
                       na_values=[""])
# Every iso2 value in the source file has a stray leading space (e.g. " AF").
gravity["iso2"] = gravity["iso2"].str.strip()

# Some iso2 codes appear on multiple rows (e.g. historical splits/mergers like
# "Sudan + South Sudan" vs. present-day "Sudan"). Prefer the row with no
# last_year (still active today); otherwise take the one with the latest
# last_year.
gravity_sorted = gravity.sort_values("last_year", na_position="first", ascending=False)
gravity_current = gravity_sorted.drop_duplicates(subset="iso2", keep="first")

gravity_lookup = gravity_current.set_index("iso2")[["country_id", "iso3", "country"]]
gravity_lookup = gravity_lookup.rename(columns={"country": "gravity_country_name"})

result = kiva_countries.merge(
    gravity_lookup, left_on="match_iso2", right_index=True, how="left"
)
result["matched"] = result["country_id"].notna()

result = result[[
    "nation_code", "kiva_iso2", "kiva_name", "in_borrowers", "in_lenders",
    "country_id", "iso3", "gravity_country_name", "matched",
]].rename(columns={"country_id": "gravity_country_id", "iso3": "gravity_iso3"})
result = result.sort_values("nation_code")

output_path = "distance_data/kiva_country_mapping.csv"
result.to_csv(output_path, index=False)
# Note: Namibia's kiva_iso2 is the literal string "NA" -- re-reading this file
# with plain pd.read_csv() will silently turn it back into a missing value
# unless you pass keep_default_na=False.

# --- 4. Summary ---
n_total = len(result)
n_matched = int(result["matched"].sum())
print(f"Kiva countries found in borrower/lender data: {n_total}")
print(f"  in borrowers only: {int((result['in_borrowers'] & ~result['in_lenders']).sum())}")
print(f"  in lenders only:   {int((result['in_lenders'] & ~result['in_borrowers']).sum())}")
print(f"  in both:           {int((result['in_borrowers'] & result['in_lenders']).sum())}")
print(f"Matched to a gravity country_id: {n_matched}/{n_total}")
print(f"Wrote {output_path}")

unmatched = result[~result["matched"]]
if len(unmatched):
    print("\nUnmatched Kiva countries (no gravity country_id found):")
    print(unmatched[["nation_code", "kiva_iso2", "kiva_name", "in_borrowers", "in_lenders"]]
          .to_string(index=False))
