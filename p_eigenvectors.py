import numpy as np
from matplotlib import pyplot as plt

"""

p is number of eigenvectors to use
h and w are the height and width of the image respectively
data is a 3D dataset with shape (train size param * images per class, #classes, h*w)

"""
def p_eigens(data, p=50, h=112, w=92, num_classes=40, images_per_class=10):
    
    t_images_per_class = data.shape[0]
    image_size = h*w
    
    # Flatten train set into 2D array and then rearrange axis
    data = data.reshape(t_images_per_class*num_classes, image_size).transpose(1, 0)
    
    # Calculate S and V and then the eigenvalue/vector pairs
    U, S, _ = np.linalg.svd(data)
    p_evalues = S**2 / (t_images_per_class - 1)
    p_evalues = p_evalues[:p]
    p_evectors = U[:, :p]
    
    return (p_evalues, p_evectors)

if __name__ == "__main__":
    data = np.load("data/db_train_norm.npy")
    eigval, eigvec = p_eigens(data, p=1000)

    h=112
    w=92

    for i in range(4):
        plt.figure(i)
        img = eigvec[:, i].reshape(h, w)
        plt.imshow(img, cmap='gray')
        
    plt.show()

    np.save("data/evectors-1000-07", eigvec)
    np.save("data/evalues-1000-07", eigval)
