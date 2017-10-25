import cv2
import numpy as np


def __getObjectsMaksByGrayLayers(img):
    # TODO создать функцию для автоматического распарса по разным оттенка серого на разные объекты

    ret, colorLayer_1 = cv2.threshold(img, 250, 250, cv2.THRESH_BINARY)
    ret, colorLayer_2 = cv2.threshold(img, 10, 250, cv2.THRESH_BINARY_INV)
    ret, colorLayer_3 = cv2.threshold(img, 150, 250, cv2.THRESH_BINARY_INV)
    colorLayer_3 = cv2.absdiff(colorLayer_2, colorLayer_3)

    colorLayers = [colorLayer_2, colorLayer_3]
    objects = [colorLayer_1]

    for layer in colorLayers:
        im2, contours, hierarchy = cv2.findContours(layer, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            result = np.zeros(img.shape, np.uint8)
            cv2.drawContours(result, [cnt], -1, (255, 255, 255), -1)
            objects.append(result)

    return objects



def getQuality(edges, template):

    cv2.imshow("template", template)

    # objects = __getObjectsMaksByGrayLayers(template)
    # for obj in objects:
    #     cv2.imshow("obj", obj)
    #     cv2.waitKey()

    result = np.zeros(edges.shape, np.uint8)

    im2, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        cv2.drawContours(result, [cnt], -1, (255, 255, 255), -1)

    cv2.imshow("result", result)
    cv2.waitKey()





    return 1

