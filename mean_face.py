import numpy as np
from matplotlib import pyplot as plt

h = 112
w = 92
data_train = np.load("data/db_train.npy")
data_test = np.load("data/db_test.npy")
# labels = np.load("data/labels_train.npy")

# Find train set mean
mean = np.average(data_train, axis=(0,1))

# Normalize train / test sets with train set mean
data_train_norm = np.subtract(data_train, mean)
np.save("data/db_train_norm", data_train_norm)

data_test_norm = np.subtract(data_test, mean)
np.save("data/db_test_norm", data_test_norm)

# Plot mean face image
plt.imshow(mean.reshape(h, w), cmap="gray")

# Debug, example of a normalized image with its corresponding label
# plt.imshow(data_norm[0,1].reshape(h,w), cmap='gray')
# print(str(labels[0][1]))

# Enable line below to save the image in the script dir
# plt.savefig("debug.png")
plt.show()