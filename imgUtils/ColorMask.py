import cv2
import numpy as np


def getMaskFromColors(img):
    masks = []
    if img.ndim != 3:
        masks.append(img)
        masks.append(cv2.bitwise_not(img))

    else:
        colors = set()
        for i in range(len(img)):
            for j in range(len(img[0])):
                color = tuple(img[i][j])  # this row prints an array of RGB color for each pixel in the image
                if color not in colors:
                    colors.add(color)

        for color in colors:
            scope = np.array([color[0], color[1], color[2]])
            mask = cv2.inRange(img, scope, scope)
            count = cv2.countNonZero(mask)
            if count > 30:  # check for small interference
                masks.append(mask)

    return masks