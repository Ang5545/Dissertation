import cv2
import numpy as np
import ImgLoader as ld

def getThreshold(th):
    grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thres = cv2.threshold(grayimg, th, 255, 0)
    return thres


def resizeImage(image, winHeight) :
    size = tuple(image.shape[1::-1])
    ratio = size[0] / winHeight
    height = int(size[0] // ratio)
    width = int(size[1] // ratio)
    result = cv2.resize(image, (height, width), interpolation = cv2.INTER_CUBIC)
    return result

def printTable(data):
    names = []
    size = data.shape[0]
    i = 0
    while i < size :
        names.append('Object %s' %i)
        i = i + 1

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


def createConfMatrix(img, pattern):

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

    while key != 13 and key != 10:

        key = cv2.waitKey()
        neddProcess = False

        if (key == 82) or (key == 0):
            th = th + step
            neddProcess = True

        elif (key == 84) or (key == 1):
            th = th - step
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

    # -- create confMatrix --
    confMatrix = np.zeros((len(objects), len(objects)))

    row = 0
    diffPttrnCnt = 0
    rowPtrnCnt = 0

    # проходим по всем строкам объектов
    for rowObj in objects:

        if row in usesIndex: # для текущего объекта строки ЕСТЬ соотвествующий паттерн

            # рассчитываем количесвто не правильно сеuментированных пикселей как разницу
            # между пттерном рассмариваекмого объекта и его найденым представлением
            diff = cv2.absdiff(rowObj, patternObjects[diffPttrnCnt])
            diffPttrnCnt = diffPttrnCnt + 1

            # проходим по всме объектам в строке
            cell = 0
            for cellObj in objects:
                # объекту в строке, найденому на изображении сответсвует объект реального мира
                if row == cell:
                    # рассматривается пересечение строки и столбца с одним и тем же объектом
                    # в таблицу пишутется сумма белых пикселей в соотвествующем паттерне
                    confMatrix[row][cell] = cv2.countNonZero(patternObjects[rowPtrnCnt])
                    rowPtrnCnt = rowPtrnCnt + 1
                else:
                    # рассчитываем сколько из неверно распознаных пикселей относятся к
                    # текущему найденому объекту
                    intersec = cv2.bitwise_and(diff, cellObj)
                    ptCount = cv2.countNonZero(intersec)
                    confMatrix[row][cell] = ptCount

                cell = cell + 1

        else: # для текущего объекта строки НЕТ соотвествующего паттерна

            # проходим по всме объектам в строке
            cell = 0
            pttrnCnt2 = 0
            for cellObj in objects:

                if row == cell or not cell in usesIndex:
                    # рассматривается пересечение строки и столбца с одним и тем же объектом
                    confMatrix[row][cell] = 0
                else:
                    # так как самого объекта изначально присутсвовало в реальном объекте
                    # рассматриваем сколько пикселей изображения относятся к соотвествующему паттерну
                    intersec = cv2.bitwise_and(patternObjects[pttrnCnt2], rowObj)
                    ptCount = cv2.countNonZero(intersec)
                    confMatrix[row][cell] = ptCount
                    pttrnCnt2 = pttrnCnt2 + 1

                cell = cell + 1

        row = row + 1
    return confMatrix









# ===================================================================================================


projectDir = ld.getParamFromConfig('projectdir')
imgPath = '%s/resources/img6/img.bmp' %projectDir
patternPath = '%s/resources/img6/pattern.bmp' %projectDir

img = cv2.imread(imgPath, 3)
pattern = cv2.imread(patternPath, 0)

confMatrix = createConfMatrix(img, pattern)
printTable(confMatrix)





