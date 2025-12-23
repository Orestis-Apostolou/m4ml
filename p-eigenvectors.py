import numpy as np

data = np.load("data/db_train_norm.npy")
# labels = np.load("data/labels_train.npy")

# Number of eigenvectors to keep
p = 2

num_classes = 40
images_per_class = 10
t_images_per_class = data.shape[0]
image_size = 112 * 92

# Flatten train set into 2D array and then rearrange axis
data = data.reshape(image_size, t_images_per_class*num_classes)

U, S, V = np.linalg.svd(data)
p_evalues = S[:p]**2 / (t_images_per_class)
p_evectors = U[:, :p]
print(str(p_evalues.shape))
print(str(p_evectors.shape))