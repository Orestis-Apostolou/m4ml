import numpy as np
from matplotlib import pyplot as plt
from p_eigenvectors import p_eigens

init_data_train = np.load("data/db_train.npy")
init_data_test = np.load("data/db_test.npy")

norm_data_train = np.load("data/db_train_norm.npy")
norm_data_test = np.load("data/db_test_norm.npy")

h, w = 112, 92
image_size = h*w
num_classes = 40
img_per_class = 10

t_images_per_class = norm_data_train.shape[0]
num_test_imgs = img_per_class - t_images_per_class

# Flatten datasets
norm_data_train_flat = norm_data_train.reshape(t_images_per_class*num_classes, image_size)
norm_data_test_flat = norm_data_test.reshape((img_per_class - t_images_per_class) * num_classes, image_size)
X_test_flat = init_data_test.reshape((img_per_class - t_images_per_class) * num_classes, image_size)

mean = np.mean(init_data_train, axis=(0,1))

_, eigvec = p_eigens(norm_data_train_flat, p=None)
test_proj = norm_data_test_flat @ eigvec

p_list = [2, 20, 50, 100, 500]
sel_classes = [0, 1, 2, 3, 4]

plt.figure(figsize=(15,8))
for i, c_id in enumerate(sel_classes):

    # Find the first image for each class in flattened set
    img_idx = c_id * num_test_imgs

    # Plot the original image for comparison
    plt.subplot(len(sel_classes), len(p_list)+1, i*(len(p_list)+1) + 1)
    plt.imshow(X_test_flat[img_idx].reshape(h, w), cmap="gray")

    if i == 0:
        plt.title("Original")
    plt.axis("off")

    for j, p in enumerate(p_list):
        # Reconstruct all test images
        X_hat = (eigvec[:, :p] @ test_proj[:,:p].T).T + mean

        # Plot image reconstruction from class #i using p eigenvectors
        plt.subplot(len(sel_classes), len(p_list)+1, i*(len(p_list)+1) + j + 2)
        plt.imshow(X_hat[img_idx].reshape(h, w), cmap="gray")

        # Add titles to the first row for column names
        if i == 0:
            plt.title(f"p={p}")
        plt.axis("off")

plt.show()
plt.savefig("6_2_out.png")
