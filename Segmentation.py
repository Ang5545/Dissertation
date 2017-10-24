import cv2
import numpy as np


# -- load image -
imgPath = '/home/ange/Python/workplace/Dissertation/resources/img2/img.JPG'
img = cv2.imread(imgPath, 3)







# -- show results --
cv2.imshow("Original image", img)
cv2.waitKey()