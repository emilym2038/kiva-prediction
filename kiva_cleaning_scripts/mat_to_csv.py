import h5py
import scipy.io as sio
import pandas as pd
import numpy as np

file_path = 'KivaMatlabData/11_gUL_GraphLenderLoan.mat'
struct_key = 'gUL'  # top-level struct name inside the .mat file (L, U, N, P, T, gUL, or gUT)
output_path = '11_gUL_GraphLenderLoan.csv'

# 01_N_Country.mat, 02_U_Lender.mat, 03_P_Partner.mat, and 04_T_Team.mat are
# MATLAB v5 files (plain binary), while 05_L_Loan.mat is v7.3 (HDF5-based).
# scipy.io.loadmat only handles v5; h5py only handles v7.3+. Detect which one
# we're dealing with by sniffing the file header.
with open(file_path, 'rb') as fh:
    is_hdf5 = fh.read(8) == b'\x89HDF\r\n\x1a\n'

if is_hdf5:
    with h5py.File(file_path, 'r') as f:
        print("Top-level HDF5 keys:", list(f.keys()))

        struct_group = f[struct_key]
        print(f"Keys inside '{struct_key}':", list(struct_group.keys()))

        var_group = struct_group['var']
        fields = list(var_group.keys())
        print(f"Detected Fields in {struct_key}['var']:", fields)

        data_dict = {}

        for field in fields:
            print(f"Processing field: {field}...")
            dataset = var_group[field]
            raw_data = dataset[:]

            if h5py.check_dtype(ref=dataset.dtype) is not None:
                resolved_column = []
                for ref in raw_data.flatten():
                    try:
                        ref_obj = f[ref]
                        val = np.array(ref_obj).tobytes().decode('utf-16', errors='ignore').strip()
                        resolved_column.append(val)
                    except Exception:
                        resolved_column.append(None)
                data_dict[field] = resolved_column
            else:
                data_dict[field] = raw_data.flatten()
else:
    mat = sio.loadmat(file_path, simplify_cells=True)
    print("Top-level keys:", list(mat.keys()))

    struct = mat[struct_key]
    print(f"Keys inside '{struct_key}':", list(struct.keys()))

    if 'graph' in struct:
        # The gUL/gUT files store a sparse adjacency matrix (lender x loan,
        # lender x team) rather than a struct-of-arrays table. Represent it
        # as an edge list: one row per non-zero entry.
        graph = struct['graph'].tocoo()
        print(f"Graph shape: {graph.shape}, non-zero entries: {graph.nnz}")

        data_dict = {
            'row': graph.row,
            'col': graph.col,
            'value': graph.data,
        }
    else:
        var = struct['var']
        fields = list(var.keys())
        print(f"Detected Fields in {struct_key}['var']:", fields)

        data_dict = {field: var[field] for field in fields}

# 6. Convert to DataFrame
df_loan = pd.DataFrame(data_dict)

print("\n--- DataFrame Summary ---")
print(df_loan.shape)
print(df_loan.head())

df_loan.to_csv(output_path, index=False)
