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


