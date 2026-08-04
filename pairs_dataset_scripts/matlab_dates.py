"""
Convert MATLAB datenum floats (as stored in Kiva's .mat exports -- see
kiva_cleaning_scripts/dateget.py) to real pandas Timestamps.

MATLAB's datenum epoch is day 1 = 0000-01-01, which is 366 days before
Python's ordinal day 1 (0001-01-01). This is NOT the Excel serial-date epoch
(1899-12-30) -- using the Excel epoch on these values silently produces
bogus year-3906 dates.
"""
import numpy as np
import pandas as pd

MATLAB_EPOCH_OFFSET_DAYS = 366


def matlab_serial_to_datetime(serial: pd.Series) -> pd.Series:
    serial = pd.to_numeric(serial)
    days = np.floor(serial)
    frac = serial - days
    base = pd.Timestamp("0001-01-01") + pd.to_timedelta(days - MATLAB_EPOCH_OFFSET_DAYS, unit="D")
    out = base + pd.to_timedelta(frac, unit="D")
    return out
