import cv2
import numpy as np


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

def printTable(data, names):
    row_format = "{:>15}" * (len(names) + 1)
    print(row_format.format("", *names))

    for team, row in zip(names, data):
        print(row_format.format(team, *row))


imgPath    = '/home/ange/Python/workplace/Dissertation/resources/img4/img.JPG'
samplePath = '/home/ange/Python/workplace/Dissertation/resources/img4/sampleMask.bmp'

originSample = cv2.imread(samplePath, 0)
sample = resizeImage(originSample, 800)

origin = cv2.imread(imgPath, 3)
img = resizeImage(origin, 800)


#  -- count sample pixels for all objects --
edges = cv2.Canny(sample, 100, 200)
im2, contours, hierarchy = cv2.findContours(sample, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
sampleObjects = []
objects = []
for cnt in contours :
    cntImg = np.zeros(sample.shape, np.uint8)
    cv2.drawContours(cntImg, [cnt], -1, (255, 255, 255), -1)
    allPt = cv2.countNonZero(cntImg)
    sampleObjects.append(allPt)
    objects.append(cntImg)
print(sampleObjects)


# искомый объект - 2
testObject = objects[2]


# -- segmentation --
key = 0
step = 1
th = 200
grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thres = cv2.threshold(grayimg, th, 255, 0)
cv2.imshow("thres", thres)

while (key != 10):
    key = cv2.waitKey()
    neddProcess = False

    if (key == 82):
        th = th + 1
        neddProcess = True

    elif (key == 84):
        th = th - 1
        neddProcess = True

    if (neddProcess) :
        grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thres = cv2.threshold(grayimg, th, 255, 0)
        cv2.imshow("thres", thres)


# -- count progress --
im2, thresContours, hierarchy = cv2.findContours(thres, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
minErr = 10000000
minDiff = np.zeros(img.shape, np.uint8)
searchTestImg = np.zeros(img.shape, np.uint8)

print('thresContours len %s' %len(thresContours))
i = 0
for cnt in thresContours :
    cntImg = np.zeros(sample.shape, np.uint8)
    cv2.drawContours(cntImg, [thresContours[i]], -1, (255, 255, 255), -1)
    diff = cv2.absdiff(testObject, cntImg)
    errPtCount = cv2.countNonZero(diff)
    print(errPtCount)
    if errPtCount < minErr :
        minDiff = diff
        minErr = errPtCount
        searchTestImg = cntImg
    i = i + 1
#     cntImg = np.zeros(sample.shape, np.uint8)
#     cv2.drawContours(cntImg, [cnt], -1, (255, 255, 255), -1)
#     i = i + 1
#     # cv2.imshow("cntImg %s" %i, cntImg)
#
print(minErr)
cv2.imshow("minDiff", minDiff)
cv2.imshow("searchTestImg", searchTestImg)
cv2.waitKey()








# cv2.imshow("Edges", edges)
#
#
# confMatrix = np.zeros((len(contours), len(contours)))
# objNames = []
#
# i=0
#
#
#
#
#
#     confMatrix[i][i] = allPt
#
#     objNames.append('Object %s' %i)
#     objName = 'Object %s' % i
#     cv2.imshow(objNames[i], img)
#     i = i + 1
#
#     print("allPt %s" % allPt)
#     print("objName %s" %objName)
#
#
# printTable(confMatrix, objNames)
#
# cv2.imshow("Sample mask", sample)
# cv2.waitKey()
#
#
