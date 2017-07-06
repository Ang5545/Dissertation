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


def resizeImage(image, winHeight):
    size = tuple(image.shape[1::-1])
    ratio = size[0] / winHeight
    height = int(size[0] // ratio)
    width = int(size[1] // ratio)
    result = cv2.resize(image, (height, width), interpolation = cv2.INTER_CUBIC)
    return result

imgPath = '/home/ange/Python/workplace/Dissertation/resources/img2/img.JPG'
samplePath = '/home/ange/Python/workplace/Dissertation/resources/img2/sampleMask.bmp'
saveDirPath = '/home/ange/Python/workplace/Dissertation/resources/img2/'

origin = cv2.imread(imgPath, 3)
originSampleMask = cv2.imread(samplePath, 0)


img = resizeImage(origin, 800)
sampleMask = resizeImage(originSampleMask, 800)
cv2.imshow("Original image", img)
cv2.imshow("Sample mask", sampleMask)

# etalon segmentation
segmentImg = createSegmentingImg(img, sampleMask)
cv2.imshow("Segmentation by sample mask", segmentImg)

# find best parameters
minPrc = 100
bestTh = 0
step = 0

while (step != 255):
    thres = getThreshold(step)
    diff = cv2.absdiff(sampleMask, thres)
    allPtCount = img.shape[0] * img.shape[1]
    errPtCount = cv2.countNonZero(diff)
    prc = errPtCount / allPtCount * 100

    if minPrc > prc:
        minPrc = prc
        bestTh = step

    step = step +1

print('-------------------------------')
print('-- best Threshold -------------')
print('-------------------------------')
print('bestTh = %s' % bestTh)
print('minPrc = %s' % minPrc)
print('-------------------------------')
print('-------------------------------')


# show and tunig parametrs
key = 0
th = bestTh
step = 1


while (key != 27):
    key = cv2.waitKey()
    neddProcess = False

    if (key == 82):
        th = th + step
        neddProcess = True

    elif (key == 84):
        th = th - step
        neddProcess = True

    elif (key == ord('r')):
        mask = getThreshold(th)
        segm = createSegmentingImg(img, mask)
        diff = cv2.absdiff(sampleMask, mask)



        cv2.imwrite(saveDirPath + 'sampleSegment.bmp', segmentImg)
        cv2.imwrite(saveDirPath + 'mask.bmp', mask)
        cv2.imwrite(saveDirPath +'segm.bmp', segm)
        cv2.imwrite(saveDirPath +'diff.bmp', diff)

    if (neddProcess):
        mask = getThreshold(th)
        segm = createSegmentingImg(img, mask)
        diff = cv2.absdiff(sampleMask, mask)

        allPtCount = img.shape[0] * img.shape[1]
        errPtCount = cv2.countNonZero(diff)
        prc = errPtCount / allPtCount * 100

        cv2.imshow("Mask", mask)
        cv2.imshow("Segmentation", segm)
        cv2.imshow("difference image", diff)

        print("th  = %s" %th)
        print("prc  = %s" %prc)
        print('-------------------------------')
        neddProcess = False

