import pandas as pd

df = pd.read_csv('distance_data/Gravity_V202211.csv')
df_t = df[(df['year'] > 2005) & (df['year'] < 2014)]

df_t.to_csv('gravity.csv')

#print(df.head())
#print(df.columns.to_list())