import cv2
import segmentation.Yasnoff as ql
import numpy as np

def repaint(img, threshold1, threshold2):
    edges = cv2.Canny(img, threshold1, threshold2)
    result = np.zeros(img.shape, np.uint8)

    im2, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        cv2.drawContours(result, [cnt], -1, (255, 255, 255), -1)

    cv2.imshow("Original image", result)
    print('threshold1 = %s' %threshold1)
    print('threshold2 = %s' %threshold2)
    print('---------------------------')



# -- load image -
imgPath = '/home/ange/Dropbox/Учеба/Опыты/Примеры сегментации/original.jpg'
img = cv2.imread(imgPath, 0)

# -- load segmented image -
templatePath = '/home/ange/Dropbox/Учеба/Опыты/Примеры сегментации/manualSegment2.bmp'
template = cv2.imread(templatePath, 0)

# edges = cv2.Canny(img, 10, 200)
# qualityLevel = ql.getQuality(edges, template)





# edges = cv2.Canny(img, 100, 200)
# cv2.imshow("Original image", edges)
#

key = 0
threshold1 = 100
threshold2 = 200
step = 10

repaint(img, threshold1, threshold2)

while key != 13 and key != 10:
    key = cv2.waitKey()

    if (key == 82) or (key == 0):
        threshold1 = threshold1 + step
        repaint(img, threshold1, threshold2)

    elif (key == 84) or (key == 1):
        if (threshold1 > 0 ):
            threshold1 = threshold1 - step
        repaint(img, threshold1, threshold2)

    elif (key == 81) or (key == 1):
        if (threshold2 > 0):
            threshold2 = threshold2 - step
        repaint(img, threshold1, threshold2)

    elif (key == 83) or (key == 1):
        threshold2 = threshold2 + step
        repaint(img, threshold1, threshold2)


# edges = cv2.Canny(img, 100, 200)

# -- show results --
# cv2.imshow("Original image", edges)


cv2.waitKey()