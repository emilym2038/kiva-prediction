import h5py
import pandas as pd

df = pd.read_csv('raw_csv/05_L_loan_L.csv')

# Row order in 05_L_loan_L.csv matches the loan index used in
# 11_gUL_GraphLenderLoan (and thus lender_loan_funding.csv's 'loan_id' column
# is this same idx2id). Attach it before filtering so borrower_info.csv stays
# joinable to the funding pair table after rows are dropped.
with h5py.File('KivaMatlabData/05_L_Loan.mat', 'r') as f:
    df['loan_id'] = f['L']['map']['idx2id'][:].flatten()

"""
Ran on 7/18 to create borrower_info.csv

55,288 rows (10.1%) were dropped as prefunded
This was concentrated among specific partners, and to some extent for specific nations (but only 6-27%)
In total, 70179 rows were removed, or about 12.44%
"""

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

# Drop unneeded columns
columns_required = ['activity', 'amnt', 'amnt_funded', 'brwr_female', 'brwr_male', 'brwr_pic', 'brwr_nopic', 'nation_code', 'partner', 'sector', 'status', 'time_posted', 'use']
df_dropped = df.drop(columns=columns_drop_borrowers)

# Drop rows without required column values
null_mask = df_dropped[columns_required].isnull() | (df_dropped[columns_required] == '[]')
df_cleaned = df_dropped[~null_mask.any(axis=1)]

# Drop pre-funded rows
df_final = df_cleaned[~(df_cleaned['time_funded'] < df_cleaned['time_posted'])]
print(f"Rows removed due to prefunded: {len(df_cleaned) - len(df_final)}")

# Final analysis
rows_removed = len(df) - len(df_final)
pct_removed = rows_removed / len(df) * 100
print(f"Rows removed: {rows_removed}")
print(f"Percentage of total rows removed: {pct_removed:.2f}%")

df_final.to_csv('borrower_info.csv', index=False)