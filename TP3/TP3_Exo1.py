
import cvkit.utils as ut
from pathlib import Path
import cv2

imgpath = Path.home() / "data" / "Images"

im1 = cv2.imread("{}/compteur.jpg".format(imgpath), cv2.IMREAD_GRAYSCALE)
if im1 is None:
    raise ValueError("No image!")
