import pandas as pd
import scipy.io as sio

df = pd.read_csv('raw_csv/02_U_Lender_L.csv')

# Row order in 02_U_Lender_L.csv matches the lender index used in
# 11_gUL_GraphLenderLoan (and thus lender_loan_funding.csv's 'lender' column
# is this same idx2id). Attach it before filtering so lender_info.csv stays
# joinable to the funding pair table after rows are dropped.
U = sio.loadmat('KivaMatlabData/02_U_Lender.mat', simplify_cells=True)['U']
df['lender'] = U['map']['idx2id']

"""
Initially Null Counts:
name: 2703 null rows (0.24%)
memb_since: 0 null rows (0.00%)
whereabouts: 524543 null rows (47.33%)
nation_code: 485332 null rows (43.79%)
occu: 635003 null rows (57.30%)
occu_info: 1010761 null rows (91.20%)
invitee_cnt: 0 null rows (0.00%)
loan_cnt: 0 null rows (0.00%)
loan_bcuz: 961408 null rows (86.75%)

Ran on 7/18 to create lender_info.csv
In total 552384 rows removed, or 47.04%
"""

columns_required = ['name', 'memb_since', 'loan_cnt', 'nation_code']
columns_drop_lenders = [
    'img_id',
    'img_tmpl_id', 
    'inviter',
]

# Dropping unneeded columns
df_dropped = df.drop(columns=columns_drop_lenders)

#Dropping rows without columns_required 
null_mask = df_dropped[columns_required].isnull() | (df_dropped[columns_required] == '[]')
df_cleaned = df_dropped[~null_mask.any(axis=1)]



#Final analysis
rows_removed = len(df) - len(df_cleaned)
pct_removed = rows_removed / len(df) * 100

print(f"Rows removed: {rows_removed}")
print(f"Percentage of total rows removed: {pct_removed:.2f}%")

df_cleaned.to_csv('lender_info.csv', index=False)

