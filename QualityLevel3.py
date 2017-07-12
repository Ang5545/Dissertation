import cv2
import numpy as np
import ImgLoader as ld

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


def resizeImage(image, winHeight):
    size = tuple(image.shape[1::-1])
    ratio = size[0] / winHeight
    height = int(size[0] // ratio)
    width = int(size[1] // ratio)
    result = cv2.resize(image, (height, width), interpolation = cv2.INTER_CUBIC)
    return result

projectDir = ld.getParamFromConfig('projectdir')
imgPath = '%s/resources/img6/img.bmp' %projectDir
patternPath = '%s/resources/img6/pattern.bmp' %projectDir

pattern = cv2.imread(patternPath, 0)
img = cv2.imread(imgPath, 3)

# img = resizeImage(bigImg, 800)
# resizePattern = resizeImage(pattern, 800)
# cv2.imwrite('%s/resources/img6/img.JPG' %projectDir, resizeImage)
# cv2.imwrite('%s/resources/img6/img.bmp' %projectDir, img)

cv2.imshow("pattern", pattern)
cv2.imshow("img", img)


#  -- count sample pixels for all objects --
edges = cv2.Canny(pattern, 100, 200)
patternBack = np.zeros(pattern.shape, np.uint8)
patternBack[::] = 255

im2, contours, hierarchy = cv2.findContours(pattern, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
patternObjects = []

for cnt in contours :
    cntImg = np.zeros(pattern.shape, np.uint8)
    cv2.drawContours(cntImg, [cnt], -1, (255, 255, 255), -1)
    cv2.drawContours(patternBack, [cnt], -1, (0, 0, 0), -1)
    patternObjects.append(cntImg)




# -- segmentation --
key = 0
step = 1
th = 200
grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thres = cv2.threshold(grayimg, th, 255, 0)
cv2.imshow("thres", thres)



while (key != 13 | key != 10):

    key = cv2.waitKey()
    neddProcess = False

    if (key == 82) | (key == 0):
        th = th + 1
        neddProcess = True

    elif (key == 84) | (key == 1):
        th = th - 1
        neddProcess = True

    if (neddProcess) :
        grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thres = cv2.threshold(grayimg, th, 255, 0)
        cv2.imshow("thres", thres)



# -- count progress --
im2, thresContours, hierarchy = cv2.findContours(thres, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
searchObjects = []
objects = []

objectBack = np.zeros(pattern.shape, np.uint8)
objectBack[::] = 255
usesIndex = []


for pattern in patternObjects:

    minErr = img.shape[0] * img.shape[1]
    searchObj = np.zeros(img.shape, np.uint8)
    minIndex = 0
    i = 0

    for cnt in thresContours :
        area = cv2.contourArea(cnt)
        if area > 500:
            cntImg = np.zeros(pattern.shape, np.uint8)
            cv2.drawContours(objectBack, [cnt], -1, (0, 0, 0), -1)
            cv2.drawContours(cntImg, [cnt], -1, (255, 255, 255), -1)
            objects.append(cntImg)
            diff = cv2.absdiff(pattern, cntImg)
            errPtCount = cv2.countNonZero(diff)
            if errPtCount < minErr :
                minErr = errPtCount
                searchObj = cntImg
                minIndex = i

            i = i + 1

    searchObjects.append(searchObj)
    usesIndex.append(minIndex)

# -- create confMatrix --
confMatrix = np.zeros((len(objects), len(objects)))

# add search data
objNames = []
i = 0
for pattern in patternObjects:
    diff = cv2.absdiff(pattern, searchObjects[i])

    j = 0
    for object in searchObjects:
        if i == j:
            confMatrix[i][i] = cv2.countNonZero(pattern)
        else:
            intersec = cv2.bitwise_and(diff, object)
            ptCount = cv2.countNonZero(intersec)
            confMatrix[i][j] = ptCount

        j = j + 1

    i = i + 1


# add not search data
i = 0
for obj in objects:
    objNames.append('Object %s' % i)

    j = 0
    for obj in objects:
        intersec = cv2.bitwise_and(diff, object)
        ptCount = cv2.countNonZero(intersec)
        confMatrix[i][j] = ptCount
        j = j + 1

    i = i + 1

printTable(confMatrix, objNames)



