import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

db = np.load("data/db_3d.npy")
labels = np.load("data/labels_3d.npy")

num_classes = 40
img_per_class = 10

labels = labels.repeat(img_per_class).reshape(num_classes, img_per_class)

# Switch axis to satisfy 'train_test_split' input format
db = db.transpose(1, 2, 0)
labels = labels.transpose(1, 0)

# Training percentage
t = 0.7

im_train, im_test, label_train, label_test = train_test_split(
    db, labels, train_size=t, random_state=15
)

# Debug
print("TRAIN: " + str(im_train.shape) + " TEST: " + str(im_test.shape))
print("TRAIN: " + str(label_train.shape) + " TEST: " + str(label_test.shape))

Path("data").mkdir(exist_ok=True)

# Save split results
np.save("data/db_train", im_train)
np.save("data/labels_train", label_train)
np.save("data/db_test", im_test)
np.save("data/labels_test", label_test)
