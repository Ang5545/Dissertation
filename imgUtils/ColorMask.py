import cv2
import numpy as np


def getMaskFromColors(img):
    height = img.shape[0]
    width = img.shape[1]
    colors = set()
    masks = []

    if img.ndim != 3:
        for i in range(height):
            for j in range(width):
                color = img[i][j]
                if color not in colors:
                    colors.add(color)
        for color in colors:
            scope = np.array(color)
            mask = cv2.inRange(img, scope, scope)
            count = cv2.countNonZero(mask)
            if count > 30:  # check for small interference
                masks.append(mask)

    else:
        for i in range(height):
            for j in range(width):
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