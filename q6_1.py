import numpy as np
from p_eigenvectors import p_eigens

# Initial Data
init_data_train = np.load("data/db_train.npy")
init_data_test = np.load("data/db_test.npy")

# Normalized Data
norm_data_train = np.load("data/db_train_norm.npy")
norm_data_test = np.load("data/db_test_norm.npy")

h = 112
w = 92
image_size = h*w
num_classes = 40
img_per_class = 10
t_images_per_class = norm_data_train.shape[0]

# Flatten centered database to 2D
norm_data_train = norm_data_train.reshape(t_images_per_class*num_classes, image_size)
norm_data_test = norm_data_test.reshape((img_per_class - t_images_per_class) * num_classes, image_size)

# Mean Face
mean = np.average(init_data_train, axis=(0,1))

_, eigvec = p_eigens(norm_data_train, p=2)

# Form one projection of the maximum eigenvectors
test_proj = norm_data_test @ eigvec

X_test = init_data_test.reshape(init_data_test.shape[0] * num_classes, image_size)

for p in [2, 5, 20, 30, 50, 100, 300, 500, 1000]:
    # Use part of the calculated projection and reconstruct the face
    reconst_data = eigvec[:, :p] @ test_proj[:, :p].T + mean[:, None]
    X_hat = reconst_data.T

    err = np.sum(np.sum((X_test - X_hat)**2, axis=1))
    print(f"Reconst Error({p} eigenvectors): {err}")
