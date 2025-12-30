import numpy as np
from matplotlib import pyplot as plt

"""

p: number of eigenvectors to use

h: image height in pixels

w: image width in pixels

data: dataset with shape (train size param * images per class, #classes, h*w)

"""
def p_eigens(data, p=50):
    
    t_images_per_class = data.shape[0]
    
    # Rearrange axis
    data = data.transpose(1, 0)
    
    #print(str(data.shape))
    
    # Calculate S and V and then the eigenvalue/vector pairs
    U, S, _ = np.linalg.svd(data)
    p_evalues = S**2 / (t_images_per_class - 1)
    p_evalues = p_evalues[:p]
    p_evectors = U[:, :p]
    
    return (p_evalues, p_evectors)

if __name__ == "__main__":
    data = np.load("data/db_train_norm.npy")
    
    h = 112
    w = 92
    image_size = h*w
    num_classes = 40
    t_images_per_class = data.shape[0]
    
    data = data.reshape(t_images_per_class*num_classes, image_size)
    eigval, eigvec = p_eigens(data, p=50)
    
    for i in range(4):
        plt.figure(i)
        img = eigvec[:, i].reshape(h, w)
        plt.imshow(img, cmap='gray')
    plt.show()