import pandas as pd

df = pd.read_csv('05_L_Loan_L_dropped.csv')
df

required_cols = ['activity', 'amnt', 'amnt_funded', 'brwr_female', 'brwr_male', 'brwr_pic', 'delnq', 'desc_en', 'geox', 'geoy', 'nation_code', 'partner', 'sector', 'status', 'time_funded', 'time_posted', 'trm_amnt', 'trm_dsbsl_amnt', 'trm_dsbsl_curncy', 'trm_liab_curexch', 'trm_liab_nonpay', 'use']

all_cols = df.columns.to_list()

total_rows = len(df)
#print(len(df))

"""
null_counts = df[all_cols].isnull().sum()
for col, count in null_counts.items():
    print(f"{col}: {count} null rows ({count / total_rows * 100:.2f}%)")
"""

df = df.dropna(subset=required_cols)
rows_removed = total_rows - len(df)
pct_removed = rows_removed / total_rows * 100

print(f"Rows removed: {rows_removed}")
print(f"Percentage of total rows removed: {pct_removed:.2f}%")

df.to_csv('05_L_Loan_cleaned.csv')


"""
print(df['status'].unique()) == [ 1.  2.  3. nan  4.  5.  6.  7.  8.  9. 10. 11. 12.]

print(len(df[df['amnt_funded'] == df['amnt']]) / len(df['amnt'])) == 0.97272

df_c = df[(df['time_funded'] - df['time_posted']) >= 0]
df_neg = df[(df['time_funded'] - df['time_posted']) < 0]
time_to_fund = df_c['time_funded'] - df_c['time_posted']
time_to_fund_neg = df_neg['time_funded'] - df_neg['time_posted']


print(len(df['amnt']) - len(df_c['amnt']))
print("Based on: df[(df['time_funded'] - df['time_posted']) >= 0] ")
print("---------------------------------------------------")
print(f"# of Rows with Positive Time Lengths: {len(df_c)}")
print(f"# of Rows with Negative Time Lengths: {len(df_neg)}")

print(df_neg['time_funded'][1], df_neg['time_posted'][1])

print(time_to_fund_neg.describe())
print(time_to_fund.describe())
"""