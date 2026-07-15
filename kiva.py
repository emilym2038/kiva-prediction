import scipy.io as sio
import pandas as pd
import numpy as np

mat_data = sio.loadmat('KivaMatlabData/05_L_Loan.mat')
lender_var = mat_data['L']['var'][0, 0][0, 0]

fields = lender_var.dtype.names
print("Fields:", fields)

def unwrap_cell(x):
    if isinstance(x, np.ndarray):
        return x[0] if x.size else None
    return x


data_dict = {}
for field in fields:
    col = lender_var[field].ravel() 
    if col.dtype == object:
        data_dict[field] = [unwrap_cell(v) for v in col]
    else:
        data_dict[field] = col

df_lender = pd.DataFrame(data_dict)

"""df_lender['memb_since'] = pd.to_datetime(
    df_lender['memb_since'] - 719529, unit='D', origin='unix', errors='coerce'
)"""

print(df_lender.shape)
print(df_lender.dtypes)
print(df_lender.head())


df_lender.to_csv('05_L_loan_L.csv')