import numpy as np
from p_eigenvectors import p_eigens 

data = np.load("data/db_train_norm.npy")
labels = np.load("data/labels_train.npy")

eigval, eigvec = p_eigens(data)

# Dataset original shape is 7x40x10304 (flattening it to 280x10304)
data = data.reshape(-1, data.shape[2])

eigspc_data = data @ eigvec

print(str(eigspc_data.shape))