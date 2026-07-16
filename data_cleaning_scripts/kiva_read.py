from pymatreader import read_mat

# This library is optimized for handling MATLAB's unique container objects
data = read_mat('KivaMatlabData/05_L_Loan.mat')

print(data.keys())