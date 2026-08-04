import pandas as pd

columns_keep = ['year', 'country_id_o', 'country_id_d', 'iso3_o', 'iso3_d', 'iso3num_o', 'iso3num_d', 'distw_harmonic', 'distw_arithmetic',
                'dist', 'contig', 'comlang_off', 'comlang_ethno', 'col_dep', 'col45', 'col_dep_ever', 'comcol', 'comrelig', 'diplo_disagreement']

df = pd.read_csv('distance_data/gravity_filtered.csv')

df_fil = df[columns_keep]

df_fil.to_csv('gravity_filtered_more.csv')

