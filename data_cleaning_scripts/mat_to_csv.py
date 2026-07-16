import h5py
import pandas as pd
import numpy as np

file_path = 'KivaMatlabData/05_L_Loan.mat'

# 1. Open the HDF5 file in read-only mode
with h5py.File(file_path, 'r') as f:
    # Let's inspect the top-level keys
    print("Top-level HDF5 keys:", list(f.keys()))  # Typically ['#refs#', 'L']
    
    # 2. Access the 'L' group
    L_group = f['L']
    
    # In HDF5, MATLAB structures are accessed via keys
    # Let's see what's inside 'L'
    print("Keys inside 'L':", list(L_group.keys()))  # Typically contains 'var'
    
    # 3. Access 'var'
    L_var = L_group['var']
    
    # 4. Get the field names
    # In h5py, fields of a MATLAB struct are represented as sub-datasets
    fields = list(L_var.keys())
    print("Detected Fields in L['var']:", fields)
    
    data_dict = {}
    
    # 5. Extract and resolve each field
    for field in fields:
        print(f"Processing field: {field}...")
        dataset = L_var[field]
        
        # Read the raw data
        raw_data = dataset[:]
        
        # If the dataset contains HDF5 object references (like pointers to strings)
        if h5py.check_dtype(ref=dataset.dtype) is not None:
            resolved_column = []
            # We iterate through and dereference each pointer back to its actual value
            for ref in raw_data.flatten():
                try:
                    # Dereference the object pointer
                    ref_obj = f[ref]
                    # If it's character/string data, convert it to a string
                    val = np.array(ref_obj).tobytes().decode('utf-16', errors='ignore').strip()
                    resolved_column.append(val)
                except Exception:
                    resolved_column.append(None)
            data_dict[field] = resolved_column
        else:
            # If it's standard numerical data (like IDs or amounts), flatten it directly
            data_dict[field] = raw_data.flatten()

# 6. Convert to DataFrame
df_loan = pd.DataFrame(data_dict)

print("\n--- DataFrame Summary ---")
print(df_loan.shape)
print(df_loan.head())

df_loan.to_csv('05_L_loan_L.csv', index=False)