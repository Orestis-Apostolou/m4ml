import numpy as np
from p_eigenvectors import p_eigens
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt

data_train = np.load("data/db_train_norm.npy")
labels_train = np.load("data/labels_train.npy")

data_test = np.load("data/db_test_norm.npy")
labels_test = np.load("data/labels_test.npy")

eigval, eigvec = p_eigens(data_train)

# Dataset original shape is tx40x10304 (flattening it to (t*40)x10304)
data_train = data_train.reshape(-1, data_train.shape[2])
data_test = data_test.reshape(-1, data_test.shape[2])

# Label original shape is tx40 (flattening it to (t*40)x1)
labels_train = labels_train.reshape(-1)
labels_test = labels_test.reshape(-1)

print(str(labels_test.shape))
print(str(labels_train.shape))

for p in [2, 5, 20, 30, 50]:
    # Project data into p eigenspace
    train_proj = data_train @ eigvec[:, :p]
    test_proj = data_test @ eigvec[:, :p]

    dist_euclidean = cdist(test_proj, train_proj, metric='euclidean')
    dist_cosine = cdist(test_proj, train_proj, metric='cosine')

    if p == 5:
        plt.figure(0)
        plt.imshow(dist_euclidean, cmap='Blues', interpolation='nearest', \
                aspect='auto')
        plt.colorbar()
        plt.savefig("debug1.png")
    

    prediction_euclidean = labels_train[np.argmin(dist_euclidean, axis=1)]
    prediction_cosine = labels_train[np.argmin(dist_cosine, axis=1)]

    acc_euclidean = np.mean(prediction_euclidean == labels_test)
    acc_cosine = np.mean(prediction_cosine == labels_test)

    print("Accuracy (euclidean) with top " + str(p) + "-eigenvectors: " + str(acc_euclidean))
    print("Accuracy (cosine) with top " + str(p) + "-eigenvectors: " + str(acc_cosine))
    
    