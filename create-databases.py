import numpy as np 
import cv2 as cv2
from pathlib import Path

path = Path("./orl_faces_png")

data = {}

for impath in path.glob("s*/*.png"):
    img = np.array(cv2.imread(str(impath), cv2.IMREAD_GRAYSCALE))
    
    class_name = impath.parent.name
    if class_name not in data:
        data[class_name] = []
    data[class_name].append(img)

# Debug
# print([d + ":" + str(len(data[d])) for d in data])

classes = np.array(list(data))
# print(classes)
num_classes = len(classes)

img_per_class = len(data[classes[0]])
h, w = data[classes[0]][0].shape

# 4D Database
db_4d = np.zeros((h, w, img_per_class, num_classes), dtype=np.uint8)
labels_4d = classes

for i, cname in enumerate(classes):
    for j, img in enumerate(data[cname]):
        db_4d[:, :, j, i] = img

# 3D Database
db_3d = db_4d.reshape(h*w, img_per_class, num_classes)
labels_3d = labels_4d.reshape(1, 1, num_classes)

# 2D Database
db_2d = db_3d.reshape(h*w, img_per_class*num_classes)
labels_2d = np.repeat(np.arange(num_classes), img_per_class).reshape(1, num_classes*img_per_class)

Path("data").mkdir(exist_ok=True)

# Save the databases and their corresponding labels
np.save("data/db_4d", db_4d)
np.save("data/labels_4d", labels_4d)

np.save("data/db_3d", db_3d)
np.save("data/labels_3d", labels_3d)

np.save("data/db_2d", db_2d)
np.save("data/labels_2d", labels_2d)

# Debug
"""
print("\n4D")
print(str(db_4d.shape))
print(str(labels_4d.shape))

print("\n3D")
print(str(db_3d.shape))
print(str(labels_3d.shape))

print("\n2D")
print(str(db_2d.shape))
print(str(labels_2d.shape))
"""
