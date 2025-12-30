import numpy as np
from p_eigenvectors import p_eigens
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split

# An accumulation of other .py files that performs train / test splits, normalization
# and reshaping of the data to fit predict method's criteria
def prep_data(t, data, labels):
    num_classes = 40
    img_per_class = 10

    labels = labels.repeat(img_per_class).reshape(num_classes, img_per_class)

    # Switch axis to satisfy 'train_test_split' input format
    data = data.transpose(1, 2, 0)
    labels = labels.transpose(1, 0)
    
    data_train, data_test, labels_train, labels_test = train_test_split(
        data, labels, train_size=t, random_state=15
    )
    
    mean = np.average(data_train, axis=(0,1))
    # Normalize train / test sets with train set mean
    data_train_norm = np.subtract(data_train, mean)
    data_test_norm = np.subtract(data_test, mean)
    
    # Dataset original shape is tx40x10304 (flattening it to (t*40)x10304)
    data_train_norm = data_train_norm.reshape(-1, data_train_norm.shape[2])
    data_test_norm = data_test_norm.reshape(-1, data_test_norm.shape[2])
    
    # Label original shape is tx40 (flattening it to (t*40)x1)
    labels_train = labels_train.reshape(-1)
    labels_test = labels_test.reshape(-1)
    
    return data_train_norm, data_test_norm, labels_train, labels_test

data = np.load("data/db_3d.npy")
labels = np.load("data/labels_3d.npy")

# Training percentage
for t in [0.2, 0.5, 0.7, 0.9]:
    data_train, data_test, labels_train, labels_test = prep_data(t, data, labels)
    eigval, eigvec = p_eigens(data_train, p=50)
    
    print("-------------- t = " + str(t) + " --------------")
    # Number of eigenvectors to use
    for p in [2, 5, 20, 30, 50]:
        
        train_proj = data_train @ eigvec
        test_proj = data_test @ eigvec
        
        # Calculate distances and find nearest neighbor
        distances = cdist(test_proj[:, :p], train_proj[:, :p], metric='euclidean')
        prediction = labels_train[np.argmin(distances, axis=1)]
        accuracy = np.mean(prediction == labels_test)
        
        print("Accuracy (euclidean) with top " + str(p) + "-eigenvectors: " + str(accuracy) + '\n')
    
    print("-------------------------------------\n")