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

# cv2.imshow("pattern", pattern)
# cv2.imshow("img", img)


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
th = 105
grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thres = cv2.threshold(grayimg, th, 255, 0)
cv2.imshow("thres", thres)



# while ( (key != 13) or (key != 10)):
while key != 13 and  key != 10:

    key = cv2.waitKey()
    neddProcess = False

    if (key == 82) or (key == 0):
        th = th + 1
        neddProcess = True

    elif (key == 84) or (key == 1):
        th = th - 1
        neddProcess = True

    if (neddProcess) :
        grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thres = cv2.threshold(grayimg, th, 255, 0)
        # cv2.imshow("thres", thres)


# -- count progress --
im2, thresContours, hierarchy = cv2.findContours(thres, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
searchObjects = []
objects = []

objectBack = np.zeros(pattern.shape, np.uint8)
objectBack[::] = 255
usesIndex = []


# draw find contours and add to object collection
for cnt in thresContours:
    area = cv2.contourArea(cnt)
    if area > 700:
        cntImg = np.zeros(pattern.shape, np.uint8)
        cv2.drawContours(objectBack, [cnt], -1, (0, 0, 0), -1)
        cv2.drawContours(cntImg, [cnt], -1, (255, 255, 255), -1)
        objects.append(cntImg)





# associate object with patterns
for pattern in patternObjects:
    minErr = img.shape[0] * img.shape[1]
    searchObj = np.zeros(img.shape, np.uint8)
    minIndex = 0
    index = 0

    for obj in objects:
        diff = cv2.absdiff(pattern, obj)
        errPtCount = cv2.countNonZero(diff)
        if errPtCount < minErr:
            minErr = errPtCount
            searchObj = obj
            minIndex = index

        index = index + 1

    searchObjects.append(searchObj)
    usesIndex.append(minIndex)




# add bacground images
objects.append(objectBack)
searchObjects.append(objectBack)

patternObjects.append(patternBack)
usesIndex.append(len(objects)-1)



cc = 0
for obj in objects:
    cv2.imshow("Object %s" % cc, obj)
    cc = cc + 1




# -- create confMatrix --
confMatrix = np.zeros((len(objects), len(objects)))
objNames = []
pttrnCnt = 0
pttrnCnt2 = 0
pttrnCnt3 = 0

row = 0
for rowObj in objects:
    objNames.append('Object %s' % row)
    diff = np.zeros(img.shape, np.uint8)

    if row in usesIndex:
        diff = cv2.absdiff(rowObj, patternObjects[pttrnCnt2])
        pttrnCnt2 = pttrnCnt2 + 1

        cell = 0
        for cellObj in objects:
            print('row  %s' % row)
            print('cell %s' % cell)

            if row == cell:
                confMatrix[row][cell] = cv2.countNonZero(patternObjects[pttrnCnt])
                cv2.imshow("pttrnCnt", patternObjects[pttrnCnt])
                print('pattern count')
                print('ptCount %s' % cv2.countNonZero(patternObjects[pttrnCnt]))
                pttrnCnt = pttrnCnt + 1
            else:
                intersec = cv2.bitwise_and(diff, cellObj)
                ptCount = cv2.countNonZero(intersec)
                confMatrix[row][cell] = ptCount
                cv2.imshow("pttrnCnt", intersec)
                print('intersec')
                print('ptCount %s' % ptCount)

            print('--------------')
            cv2.waitKey()
            cell = cell + 1

    else:
        diff = rowObj

        cell = 0
        for cellObj in objects:
            print('row  %s' % row)
            print('cell %s' % cell)
            if row == cell:
                confMatrix[row][cell] = 0
                cv2.imshow("pttrnCnt", np.zeros(img.shape, np.uint8))
                print('its null')
                print('ptCount %s' % 0)
            else:
                if cell in usesIndex:
                    confMatrix[row][cell] = 4
                    cv2.imshow("TEST", patternObjects[cell])
                    pttrnCnt3 = pttrnCnt3 + 1
                else :
                    confMatrix[row][cell] = 5
                print('-- else --')

            print('--------------')
            cv2.waitKey()
            cell = cell + 1




    row = row + 1





cv2.waitKey()
print("end")

printTable(confMatrix, objNames)





# cc = 0
# for obj in searchObjects:
#     cv2.imshow("searchObjects %s" % cc, obj)
#     cc = cc + 1
#
# cc = 0
# for obj in objects:
#     cv2.imshow("objects %s" % cc, obj)
#     cc = cc + 1

# cc = 0
# for obj in patternObjects:
#     # cv2.imshow("patternObjects %s" % cc, obj)
#     cc = cc + 1



    # for obj in objects:
    #     diff = cv2.absdiff(pattern, obj)
    #     errPtCount = cv2.countNonZero(diff)
        # if errPtCount < minErr :
        #     minErr = errPtCount
        #     searchObj = cntImg
        #     minIndex = i

    #     i = i + 1
    # searchObjects.append(searchObj)
    # usesIndex.append(minIndex)
    #

# objects.append(objectBack)















# for pattern in patternObjects:
#     minErr = img.shape[0] * img.shape[1]
#     searchObj = np.zeros(img.shape, np.uint8)
#     minIndex = 0
#     i = 0
#
#
#
#
# objects.append(objectBack)
# patternObjects.append(patternBack)
#
# usesIndex.append(len(objects)-1)
#
# # patternObjects.append(patternBack)
# # searchObjects.append(objectBack)
# # usesIndex.append(len(patternObjects))
#
#
# # -- create confMatrix --
# confMatrix = np.zeros((len(objects), len(objects)))
# objNames = []
#

#
# #
# # row = 0
# # for obji in objects:
# #     objNames.append('Object %s' % row)
# #
# #     cell = 0
# #     for objj in objects:
# #
# #         if row == cell:
# #             if row in usesIndex:
# #                 confMatrix[cell][row] = cv2.countNonZero(patternObjects[0])
# #                 del patternObjects[0]
# #             else:
# #                 confMatrix[cell][row] = 0
# #         else:
# #             print('row  %s'  %row)
# #             print('cell %s' %cell)
# #             print("-----------")
# #
# #             # cv2.imshow("obji", obji)
# #             # cv2.imshow("objj", objj)
# #             # cv2.waitKey()
# #
# #
# #
# #             confMatrix[cell][row] = cell
#
#
#         #     if i in usesIndex:
#         #         print()
#         #     intersec = cv2.bitwise_and(obji, objj)
#         #     cv2.imshow("obji", obji)
#         #     cv2.imshow("objj", objj)
#         #     cv2.imshow("intersec", intersec)
#         #     print("i = %s" % i)
#         #     print("j = %s" % j)
#         #     print("-----------")
#         #
#         #
#         #     cv2.waitKey()
#
#
#
#
#
# #         cell = cell + 1
# #
# #     row = row + 1
# #
# # cv2.waitKey()
#
# # cv2.imshow("obj 1", objects[0])
# # cv2.imshow("obj 1", objects[1])
# # cv2.imshow("obj 1", objects[2])
# # cv2.waitKey()
# #
# #
#
# # printTable(confMatrix, objNames)
#
#
#
#
#
#
# # # add search data
# # objNames = []
# # i = 0
# # for pattern in patternObjects:
# #     objNames.append('Object %s' % i)
# #     diff = cv2.absdiff(pattern, searchObjects[i])
# #
# #     j = 0
# #     for object in searchObjects:
# #         if i == j:
# #             confMatrix[i][i] = cv2.countNonZero(pattern)
# #         else:
# #             intersec = cv2.bitwise_and(diff, object)
# #             ptCount = cv2.countNonZero(intersec)
# #             confMatrix[i][j] = ptCount
# #
# #         j = j + 1
# #
# #     i = i + 1
#
#
# # add not search data
# # i = len(patternObjects)
# # for obj in objects:
# #     objNames.append('Object %s' % i)
# #
# #     j = len(patternObjects)
# #     for obj in objects:
# #         intersec = cv2.bitwise_and(diff, object)
# #         ptCount = cv2.countNonZero(intersec)
# #         confMatrix[i][j] = ptCount
# #         j = j + 1
# #
# #     i = i + 1
# #
# # print( len(objects) )
# # printTable(confMatrix, objNames)
#
#
#
