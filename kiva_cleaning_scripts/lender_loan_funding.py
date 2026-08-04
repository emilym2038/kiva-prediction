import h5py
import scipy.io as sio
import pandas as pd

# gUL['graph'] is a sparse lender x loan adjacency matrix: row = lender index,
# col = loan index, True = that lender funded that loan. The indices line up
# with the row order in U['var'] (02_U_Lender.mat) and L['var'] (05_L_Loan.mat).
# U/L['map']['idx2id'] translates an index back to the real-world ID
# (lender username, numeric Kiva loan ID).

U = sio.loadmat('KivaMatlabData/02_U_Lender.mat', simplify_cells=True)['U']
graph = sio.loadmat('KivaMatlabData/11_gUL_GraphLenderLoan.mat', simplify_cells=True)['gUL']['graph'].tocoo()

with h5py.File('KivaMatlabData/05_L_Loan.mat', 'r') as f:
    L_idx2id = f['L']['map']['idx2id'][:].flatten()
    L_amnt = f['L']['var']['amnt'][:].flatten()
    L_sector = f['L']['var']['sector'][:].flatten()

funding = pd.DataFrame({
    'lender': U['map']['idx2id'][graph.row],
    'loan_id': L_idx2id[graph.col],
    'loan_amnt': L_amnt[graph.col],
    'loan_sector': L_sector[graph.col],
})

print("\n--- Funding Edge List Summary ---")
print(funding.shape)
print(funding.head())

funding.to_csv('lender_loan_funding.csv', index=False)
