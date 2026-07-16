import scipy.io as sio

mat_data = sio.loadmat('KivaMatlabData/02_U_loan_bcuz_stemmed.mat')

print(mat_data.keys())
