import cv2 as cv2
import numpy as np
from pathlib import Path

path = Path("./orl_faces")

# Read all .pgm images at ./orl_faces/s*
for impath in path.glob("s*/*.pgm"):

    # Read image as grayscale
    img = cv2.imread(str(impath), cv2.IMREAD_GRAYSCALE)

    Path("./orl_faces_png/" + impath.parent.name).mkdir(parents=True, exist_ok=True)

    #Create .png version following the dataset's original format
    target_path = "orl_faces_png/"+ impath.parent.name + "/" + impath.stem + ".png"
    st = cv2.imwrite(target_path, img)
    if not st:
        print("There was an error writing image " + impath.stem)
    else:
        print("Successfully saved " + target_path)
    


