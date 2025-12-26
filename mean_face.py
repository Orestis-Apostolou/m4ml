import numpy as np
from matplotlib import pyplot as plt

h = 112
w = 92
data = np.load("data/db_train.npy")
labels = np.load("data/labels_train.npy")

# Normalize and save database
mean = np.average(data, axis=(0,1))
data_norm = np.subtract(data, mean)
np.save("data/db_train_norm", data_norm)

# Plot mean face image
plt.imshow(mean.reshape(h, w), cmap="gray")

# Debug, example of a normalized image with its corresponding label
# plt.imshow(data_norm[0,1].reshape(h,w), cmap='gray')
# print(str(labels[0][1]))

# Enable line below to save the image in the script dir
# plt.savefig("debug.png")
plt.show()