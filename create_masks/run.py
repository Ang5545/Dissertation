import cv2
import numpy as np

imgPath = '/home/ange/Python/workplace/Dissertation/resources/manualSegmented/manualSegment2.bmp'
img = cv2.imread(imgPath, 3)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

colors = set()
for i in range(len(img)):
    for j in range(len(img[0])):
        color = tuple(img[i][j])  # this row prints an array of RGB color for each pixel in the image
        if color not in colors:
            colors.add(color)

masks = []
for color in colors:
    scope = np.array([color[0], color[1], color[2]])
    mask = cv2.inRange(img, scope, scope)
    count = cv2.countNonZero(mask)
    if count > 30 : # check for small interference
        masks.append(mask)


print('end')
cv2.waitKey()



