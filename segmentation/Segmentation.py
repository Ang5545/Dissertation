import cv2
from SRM import SRM
from matplotlib import pyplot
from scipy.misc import imread


def repaint(img, thres):
    ret, result = cv2.threshold(img, thres, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow("Result", result)
    print('thres = %s' %thres)
    print('---------------------------')


print('----------------------------------------------------------------------')


im = imread("/home/ange/Desktop/images/original.jpg")

srm = SRM(im, 256)
segmented = srm.run()

pyplot.imshow(segmented/256)
pyplot.show()

'''
# -- load image -
imgPath = '/home/ange/Desktop/images/original.jpg'
img = cv2.imread(imgPath, 0)

# -- load segmented image -
# templatePath = '/home/ange/Dropbox/Учеба/Опыты/Примеры сегментации/manualSegment2.bmp'
# template = cv2.imread(templatePath, 0)

# edges = cv2.Canny(img, 10, 200)
# qualityLevel = ql.getQuality(edges, template)





# edges = cv2.Canny(img, 100, 200)
# cv2.imshow("Original image", edges)
#

key = 0
thres = 100
step = 10

repaint(img, thres)

while key != 13 and key != 10:
    key = cv2.waitKey()

    if (key == 82) or (key == 0):
        if (thres < 255):
            thres = thres + step
            repaint(img, thres)

    elif (key == 84) or (key == 1):
        if (thres > 0 ):
            thres = thres - step
            repaint(img, thres)



# edges = cv2.Canny(img, 100, 200)

# -- show results --
# cv2.imshow("Original image", edges)


# cv2.waitKey()
'''
