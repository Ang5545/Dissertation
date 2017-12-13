import cv2
import imgUtils.ImgLoader as iml
import numpy as np
import matplotlib.pyplot as plt
from segmentationQuality.yasnoff import Yasnoff
from segmentationQuality.yasnoff_moments import YasnoffMoments


project_dir = iml.getParamFromConfig('projectdir')

img_path = project_dir + '/resources/paint_test_3/template.png'
img = cv2.imread(img_path, 3)

height = img.shape[0]
width = img.shape[1]


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, contours, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnt = contours[0]

for y in range(0, height):
    for x in range(0, width):
        dist = cv2.pointPolygonTest(cnt, (x, y), True)
        if dist > 0:
            img[y, x] = (dist, 0, 0)
        else:
            img[y, x] = (0, 0, abs(dist))



cv2.drawContours(img, [cnt], -1, (255, 255, 255), 2)
cv2.imshow("img", img)
cv2.waitKey()





