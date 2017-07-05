import cv2
import numpy as np

def createSegmentingImg(img, mask):
    maskInv = cv2.bitwise_not(mask)

    grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    red = np.zeros(img.shape, np.uint8)
    red[:, :, 2] = grayimg

    blue = np.zeros(img.shape, np.uint8)
    blue[:, :, 0] = grayimg

    object = cv2.bitwise_and(blue, blue, mask=mask)
    back = cv2.bitwise_and(red, red, mask=maskInv)

    result = cv2.add(object, back)

    edges = cv2.Canny(mask, 100, 200)
    im2, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (255, 255, 255), 2)
    return result

def getThreshold(th):
    grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thres = cv2.threshold(grayimg, th, 255, 0)
    return thres



imgPath = '/home/ange/Python/workplace/Dissertation/resources/simple/2.JPG'
samplePath = '/home/ange/Python/workplace/Dissertation/resources/simple/etalon.bmp'

img = cv2.imread(imgPath, 3)
sampleMask = cv2.imread(samplePath, 0)


segmentImg = createSegmentingImg(img, sampleMask)
cv2.imshow("segmentImg", segmentImg)

cv2.imshow("img", img)

minPrc = 100
bestThres = img.copy
step = 0
greath = 0

while (step != 255):
    thres = getThreshold(step)
    segmentThres = createSegmentingImg(img, thres)
    diff = cv2.absdiff(mask, thres)
    allPtCount = img.shape[0] * img.shape[1]
    errPtCount = cv2.countNonZero(diff)
    prc = errPtCount / allPtCount * 100

    if minPrc > prc:
        minPrc = prc
        bestThres = thres
        greath = step

    step = step +1

print('minPrc %s' % minPrc)
print('greath %s' % greath)
cv2.imshow("bestThres", bestThres)

key = 0
th = 0
step = 5

while (key != 27):
    key = cv2.waitKey()
    neddProcess = False

    if (key == 82):
        th = th + step
        neddProcess = True

    elif (key == 84):
        th = th - step
        neddProcess = True

    if (neddProcess) :
        thres = getThreshold(th)
        segmentThres = createSegmentingImg(img, thres)

        diff = cv2.absdiff(mask, thres)
        allPtCount = img.shape[0] * img.shape[1]
        errPtCount = cv2.countNonZero(diff)
        prc = errPtCount / allPtCount * 100

        print("allPtCount %s" %allPtCount)
        print("errPtCount %s" %errPtCount)
        print("prc %s" %prc)
        print("----------------------")
        neddProcess = False

