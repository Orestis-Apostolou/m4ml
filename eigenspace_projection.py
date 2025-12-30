import numpy as np
from p_eigenvectors import p_eigens
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt

"""

data_train: train data, must be 2D array with features on axis=1

labels_train: train labels, must be vector with same size as data_train.shape[0]

Same goes for test set. Both sets must be centered by train set mean.

p: number of eigenvectors to keep, 'None' means it will use all eigenvectors.

method: method for scipy.spatial.distance.cdist to use (ex. euclidean, cosine)

"""
def predict(data_train, data_test, labels_train, labels_test, p=None, method='euclidean'):
    eigval, eigvec = p_eigens(data_train, p=p)
    
    # Project the data into p-eigenspace
    train_proj = data_train @ eigvec
    test_proj = data_test @ eigvec
    
    # Calculate distances and find nearest neighbor
    distances = cdist(test_proj[:, :p], train_proj[:, :p], metric=method)
    prediction = labels_train[np.argmin(distances, axis=1)]
    accuracy = np.mean(prediction == labels_test)
    
    #print("Accuracy (" + method + ") with top " + str(p) + "-eigenvectors: " + str(accuracy))
    return accuracy, distances

if __name__ == "__main__":
    data_train = np.load("data/db_train_norm.npy")
    labels_train = np.load("data/labels_train.npy")

    data_test = np.load("data/db_test_norm.npy")
    labels_test = np.load("data/labels_test.npy")
    
    # Dataset original shape is tx40x10304 (flattening it to (t*40)x10304)
    data_train = data_train.reshape(-1, data_train.shape[2])
    data_test = data_test.reshape(-1, data_test.shape[2])
    
    # Label original shape is tx40 (flattening it to (t*40)x1)
    labels_train = labels_train.reshape(-1)
    labels_test = labels_test.reshape(-1)
    
    # Debug
    #print(str(labels_test.shape))
    #print(str(labels_train.shape))

    _, dist_euclidean = predict(data_train, data_test, labels_train, labels_test)
    _, dist_cosine = predict(data_train, data_test, labels_train, labels_test, method='cosine')
    
    # Plot distance matrix heatmaps
    plt.figure(0)
    plt.imshow(dist_euclidean, cmap='Blues', interpolation='nearest', \
            aspect='auto')
    plt.colorbar()
    
    plt.figure(1)
    plt.imshow(dist_cosine, cmap='Blues', interpolation='nearest', \
            aspect='auto')
    plt.colorbar()
    plt.show()