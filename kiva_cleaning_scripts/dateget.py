import pandas as pd

df = pd.read_csv('cleaned_csv/borrower_info.csv')

df_min = df['time_posted'].min()
df_max = df['time_posted'].max()


def matlab_serial_to_date(serial):
    """Convert a MATLAB datenum (as found in the Kiva .mat exports) to a pandas Timestamp.

    MATLAB's datenum epoch is day 1 = 0000-01-01, which is 366 days before
    Python's ordinal day 1 (0001-01-01) - hence the `- 366` offset. This is
    different from Excel's serial date epoch (1899-12-30); using the Excel
    epoch on these MATLAB values is what produced bogus year-3906 dates.
    """
    if not isinstance(serial, (int, float)):
        raise ValueError("Serial date must be an integer or float.")
    days = int(serial)
    frac = serial - days
    return pd.Timestamp.fromordinal(days - 366) + pd.to_timedelta(frac, unit='D')

# Example usage
print(matlab_serial_to_date(df_min))       
print(matlab_serial_to_date(df_max))

# Earliest: 2006-02-15 00:59:59.999996650
# Latest: 2013-05-29 12:40:01.999996973

"""
date_cols = ['time_posted', 'time_funded', 'time_paid', 'time_planexp']
for col in date_cols:
    df[col] = df[col].apply(lambda x: matlab_serial_to_date(x) if pd.notna(x) else pd.NaT)

print(df[date_cols])
"""
