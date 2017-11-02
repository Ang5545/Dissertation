import cv2
import numpy as np
from utils import ColorMask as colMaks



def __creatAssociateObj__(name, template, obj):
    assObj = []
    assObj.append(name)
    assObj.append(template)
    assObj.append(obj)
    return assObj



def __createConfMatrix__(associateObjs):

    confMatrix = np.zeros((len(associateObjs), len(associateObjs)))

    # проходим по всем строкам объектов
    rowIndex = 0
    for row in associateObjs:
        template = row[1]

        # проходим по всем столбцам объектов
        cellIndex = 0
        for cell in associateObjs:
            segmObj = cell[2]

            intersec = cv2.bitwise_and(template, segmObj)
            ptCount = cv2.countNonZero(intersec)
            confMatrix[rowIndex][cellIndex] = ptCount

            cellIndex = cellIndex + 1

        rowIndex = rowIndex + 1

    return confMatrix


def __getIncorrectlyClassifiedPixels__(confMatrix):
    # TODO проверить как счтается

    result = [0] * len(confMatrix)

    rowIndex = 0
    for row in confMatrix:
        c_ik = 0
        c_kk = 0

        cellIndex = 0
        for cell in row:
            c_ik = c_ik + cell

            if rowIndex == cellIndex :
                c_kk = cell

            cellIndex = cellIndex + 1

        if c_ik != 0 :
            result[rowIndex] = ((c_ik - c_kk) / c_ik) * 100

        rowIndex = rowIndex + 1

    return result


# def __getWronglyAssignedToClass__(confMatrix):
#     esult = [0] * len(confMatrix)
#
#     rowIndex = 0
#     for row in confMatrix:
#         c_ik = 0
#         c_kk = 0
#         c_ki = 0
#
#         cellIndex = 0
#         for cell in row:
#             c_ik = c_ik + cell
#
#             if rowIndex == cellIndex:
#                 c_kk = cell
#
#             cellIndex = cellIndex + 1
#
#         if c_ik != 0:
#             result[rowIndex] = ((c_ik - c_kk) / c_ik) * 100
#
#         rowIndex = rowIndex + 1
#
#     return result

def printConfMatrix(confMatrix):

    # создание новой матрицы с Total
    withTotalSize = len(confMatrix)+1
    confMatrixWithTotal = np.zeros((withTotalSize, withTotalSize))

    # создание и расчет массивов с суммами
    rowTotals = [0] * len(confMatrix)
    cellTotals = [0] * len(confMatrix)

    rowIndex = 0
    for row in confMatrix:
        cellTotal = 0
        cellIndex = 0
        for cell in row:
            cellTotal = cellTotal + cell
            cellTotals[cellIndex] = cellTotals[cellIndex] + cell
            cellIndex = cellIndex + 1

        rowTotals[rowIndex] = cellTotal
        rowIndex = rowIndex + 1

    # Наполнение значенями из сторой матрицы
    rowIndex = 0
    for row in confMatrix:
        cellIndex = 0
        for cell in row:
            confMatrixWithTotal[rowIndex, cellIndex] = cell
            cellIndex = cellIndex + 1
        rowIndex = rowIndex +1

    # Добавление сумм
    rowIndex = 0
    for rowTotal in rowTotals:
        confMatrixWithTotal[rowIndex, len(confMatrixWithTotal)-1] = rowTotal
        rowIndex = rowIndex + 1

    cellIndex = 0
    for cellTotal in cellTotals:
        confMatrixWithTotal[len(confMatrixWithTotal)-1, cellIndex] = cellTotal
        cellIndex = cellIndex + 1

    names = []
    size = confMatrix.shape[0]
    i = 0
    while i < size :
        names.append('Object %s' %i)
        i = i + 1

    names.append('Total')

    row_format = "{:>15}" * (len(names) + 1)
    print(row_format.format("", *names))

    for team, row in zip(names, confMatrixWithTotal):
        print(row_format.format(team, *row))




def getQuality(template, segm):

    # cv2.imshow("template", template)
    # cv2.imshow("segm", segm)

    templateObjs = colMaks.getMaskFromColors(template)
    segmObjsObjs = colMaks.getMaskFromColors(segm)

    # -- ассоцияция объектов с их шаблонными масками --
    associateObjs = [] # выходной массив

    i = 0
    for templ in templateObjs: # проходимся по всем шаблонам

        minErr = template.shape[0] * template.shape[1] # минимальная ошибка изначально равна количеству всех пикселей
        searchObj = np.zeros(template.shape, np.uint8) # по умолчанию искомый объект = пустое поле
        searchIndex = -1 # индекс найденого объекта

        index = 0
        for obj in segmObjsObjs: # проходимся по всем найденым при сегментации объектам
            diff = cv2.absdiff(templ, obj) # кадровая разница между объектами
            errPtCount = cv2.countNonZero(diff) # количество точек в разнице
            objPtCount = cv2.countNonZero(obj)  # количество точек в самом обхекте

            if (errPtCount < minErr) and (errPtCount < objPtCount): # еслиошибка минимальна и не весь объект ошибочный
                # приравниваем текущий объект исомому
                minErr = errPtCount
                searchObj = obj
                searchIndex = index

            index = index + 1

        if searchIndex != -1: # если найден объект
            segmObjsObjs.pop(searchIndex) # удаляем его из коллекции

        name =  'Object {0}'.format(i) # имя объекта в ассоцированном списке
        associateObjs.append(__creatAssociateObj__(name, templ, searchObj)) # объект с шаблоном в резуьтат

        i = i + 1

    for segmObjsObj in segmObjsObjs:  # проходимся по всем отставшимся сегментированным объектам

        templ = np.zeros(segmObjsObj.shape, np.uint8) # создаем пустое изображение как шаблон
        name =  'Object {0}'.format(len(associateObjs)) # имя объекта в ассоцированном списке
        associateObjs.append(__creatAssociateObj__(name, templ, segmObjsObj)) # дабавляем объект с шаблоном в выводу


    # for obj in associateObjs:
    #     cv2.imshow("template " + obj[0], obj[1])
    #     cv2.imshow("obj " + obj[0], obj[2])


    confMatrix = __createConfMatrix__(associateObjs)
    printConfMatrix(confMatrix)
    print('-------------------------------')

    m1 = __getIncorrectlyClassifiedPixels__(confMatrix)
    print(m1)
    print('-------------------------------')

    m2 = __getWronglyAssignedToClass__(confMatrix)
    print(m2)
    print('-------------------------------')


    return 1

