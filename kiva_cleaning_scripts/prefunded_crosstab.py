import pandas as pd

df = pd.read_csv('raw_csv/05_L_loan_L.csv')

columns_drop_borrowers = [
    'amnt_basket',
    'amnt_curloss',
    'desc_es',
    'desc_fr',
    'desc_ru',
    'desc_pt',
    'desc_ar',
    'desc_id',
    'desc_vi',
    'desc_mn',
    'vdd_utb_id',
    'vid_id',
]

columns_required = ['activity', 'amnt', 'amnt_funded', 'brwr_female', 'brwr_male', 'brwr_pic', 'brwr_nopic', 'nation_code', 'partner', 'sector', 'status', 'time_posted', 'use']
df_dropped = df.drop(columns=columns_drop_borrowers)

null_mask = df_dropped[columns_required].isnull() | (df_dropped[columns_required] == '[]')
df_cleaned = df_dropped[~null_mask.any(axis=1)]

prefunded_mask = df_cleaned['time_funded'] < df_cleaned['time_posted']
prefunded = df_cleaned[prefunded_mask]
kept = df_cleaned[~prefunded_mask]

print(f"Total cleaned rows: {len(df_cleaned)}")
print(f"Prefunded rows removed: {len(prefunded)} ({len(prefunded)/len(df_cleaned)*100:.2f}%)")

# Overall prefunded rate per partner
partner_rate = (
    df_cleaned.assign(prefunded=prefunded_mask)
    .groupby('partner')['prefunded']
    .agg(['sum', 'count'])
    .assign(pct=lambda d: d['sum'] / d['count'] * 100)
    .sort_values('pct', ascending=False)
)
partner_rate.columns = ['prefunded_count', 'total_count', 'pct_prefunded']

print("\n=== Prefunded rate by partner (top 20 by count of prefunded loans) ===")
print(partner_rate.sort_values('prefunded_count', ascending=False).head(20))

print("\n=== Partners with highest prefunded PERCENTAGE (min 20 total loans) ===")
print(partner_rate[partner_rate['total_count'] >= 20].sort_values('pct_prefunded', ascending=False).head(20))

# Overall prefunded rate per nation_code
nation_rate = (
    df_cleaned.assign(prefunded=prefunded_mask)
    .groupby('nation_code')['prefunded']
    .agg(['sum', 'count'])
    .assign(pct=lambda d: d['sum'] / d['count'] * 100)
    .sort_values('pct', ascending=False)
)
nation_rate.columns = ['prefunded_count', 'total_count', 'pct_prefunded']

print("\n=== Prefunded rate by nation_code (top 20 by count of prefunded loans) ===")
print(nation_rate.sort_values('prefunded_count', ascending=False).head(20))

print("\n=== Nations with highest prefunded PERCENTAGE (min 20 total loans) ===")
print(nation_rate[nation_rate['total_count'] >= 20].sort_values('pct_prefunded', ascending=False).head(20))

# Cross-tab: partner x nation_code counts of prefunded loans (top combos)
cross = (
    prefunded.groupby(['partner', 'nation_code'])
    .size()
    .reset_index(name='prefunded_count')
    .sort_values('prefunded_count', ascending=False)
)
print("\n=== Top partner x nation_code combos among prefunded loans ===")
print(cross.head(30).to_string(index=False))

# How concentrated is the loss? Gini-ish check: top N partners share of prefunded
top_partner_share = partner_rate.sort_values('prefunded_count', ascending=False)['prefunded_count'].cumsum() / partner_rate['prefunded_count'].sum() * 100
print("\n=== Cumulative share of prefunded loans by top partners ===")
print(top_partner_share.head(10))

partner_rate.to_csv('prefunded_by_partner.csv')
nation_rate.to_csv('prefunded_by_nation.csv')
cross.to_csv('prefunded_partner_nation_crosstab.csv', index=False)
print("\nSaved: prefunded_by_partner.csv, prefunded_by_nation.csv, prefunded_partner_nation_crosstab.csv")
