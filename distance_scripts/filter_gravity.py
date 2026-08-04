"""
Filter distance_data/gravity.csv down to only country-pair rows where both
the origin and destination are countries that appear in Kiva's data (i.e.
have a gravity_iso3 in distance_data/kiva_country_mapping.csv).
"""
import pandas as pd

mapping = pd.read_csv("distance_data/kiva_country_mapping.csv",
                       keep_default_na=False, na_values=[""])
kiva_iso3 = set(mapping["gravity_iso3"].dropna())

gravity = pd.read_csv("distance_data/gravity.csv", index_col=0)
filtered = gravity[gravity["iso3_o"].isin(kiva_iso3) & gravity["iso3_d"].isin(kiva_iso3)]

output_path = "distance_data/gravity_filtered.csv"
filtered.to_csv(output_path, index=False)

print(f"Kiva countries with a gravity_iso3: {len(kiva_iso3)}")
print(f"gravity.csv rows: {len(gravity)}")
print(f"gravity_filtered.csv rows: {len(filtered)}")
print(f"Wrote {output_path}")
