import pandas as pd
import gender_guesser.detector as gender

df = pd.read_csv('cleaned_csv/lender_info.csv')

# Detector only looks up single first names, so take the first whitespace
# token of 'name' (handles "First Last", "First & Second", "First and
# Second", etc). case_sensitive=False avoids manual capitalization handling
# for names like "mCkenzie" or "DAVID".
first_name = df['name'].str.split().str[0]

d = gender.Detector(case_sensitive=False)
df['gender'] = first_name.apply(d.get_gender)

# Keep the detector's own categories (male, female, mostly_male,
# mostly_female, andy, unknown) rather than collapsing to binary.
print("--- Gender Distribution (lender_info.csv 'name') ---")
counts = df['gender'].value_counts()
for category, count in counts.items():
    print(f"{category}: {count} ({count / len(df) * 100:.2f}%)")

# Collapsed grouping: mostly_female -> female, mostly_male -> male, and
# andy (androgynous names) -> unknown, since neither is a confident single-
# gender read.
gender_group_map = {
    'female': 'female',
    'mostly_female': 'female',
    'male': 'male',
    'mostly_male': 'male',
    'andy': 'unknown',
    'unknown': 'unknown',
}
df['gender_group'] = df['gender'].map(gender_group_map)

print("\n--- Grouped Gender Distribution ---")
group_counts = df['gender_group'].value_counts()
for category, count in group_counts.items():
    print(f"{category}: {count} ({count / len(df) * 100:.2f}%)")

#df[['lender', 'name', 'gender', 'gender_group']].to_csv('lender_gender.csv', index=False)

df[df['gender_group'] == 'unknown']['name'].to_csv('unknown_names.csv', index=False)

# --- Is 'unknown' concentrated in specific countries? ---
# 01_Country_N.csv's row index is nation_code - 1 (MATLAB's 1-indexed
# idx2id carried straight into a 0-indexed pandas row on export). Many
# lender nation_codes have no entry here since this table only covers
# Kiva's borrowing-country partners, not every donor country.
countries = pd.read_csv('raw_csv/01_Country_N.csv', index_col=0)
unknown_mask = df['gender_group'] == 'unknown'

nation_rate = (
    df.assign(unknown=unknown_mask)
    .groupby('nation_code')['unknown']
    .agg(['sum', 'count'])
    .assign(pct=lambda d: d['sum'] / d['count'] * 100)
)
nation_rate.columns = ['unknown_count', 'total_count', 'pct_unknown']
nation_rate['country'] = countries['name'].reindex((nation_rate.index - 1).astype(int)).values
nation_rate = nation_rate.sort_values('pct_unknown', ascending=False)

print("\n=== Nation codes with highest unknown-gender COUNT (top 20) ===")
print(nation_rate.sort_values('unknown_count', ascending=False).head(20))

print("\n=== Nations with highest unknown-gender PERCENTAGE (min 20 lenders) ===")
print(nation_rate[nation_rate['total_count'] >= 20].sort_values('pct_unknown', ascending=False).head(20))

#nation_rate.to_csv('stat_csv/unknown_gender_by_nation.csv')
print("\nSaved: lender_gender.csv, stat_csv/unknown_gender_by_nation.csv")