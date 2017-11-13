import cv2
import numpy as np
from imgUtils import ColorMask as colMaks


class Yasnoff:

    def __init__(self, template, img):
        # init parameters
        self._template = template
        self._img = img

        # calculation
        associateObjs = self._getAssociateObjs()
        self._confMatrix = self.__createConfMatrix__(associateObjs)


    def __creatAssociateObj__(self, name, template, obj):
        assObj = []
        assObj.append(name)
        assObj.append(template)
        assObj.append(obj)
        return assObj


    def __createConfMatrix__(self, associateObjs):

        confMatrix = np.zeros((len(associateObjs), len(associateObjs)))

        # проходим по всем строкам объектов
        rowIndex = 0
        for row in associateObjs:
            segmObj = row[2]

            # проходим по всем столбцам объектов
            cellIndex = 0
            for cell in associateObjs:
                template = cell[1]

                intersec = cv2.bitwise_and(template, segmObj)
                ptCount = cv2.countNonZero(intersec)
                confMatrix[rowIndex][cellIndex] = ptCount

                cellIndex = cellIndex + 1

            rowIndex = rowIndex + 1

        return confMatrix


    def __getIncorrectlyClassifiedPixels__(self, confMatrix):
        result = [0] * len(confMatrix)
        c_kk = [0] * len(confMatrix)
        c_ik = [0] * len(confMatrix)

        rowIndex = 0
        for row in confMatrix:

            cellIndex = 0
            for cell in row:

                c_ik[cellIndex] = c_ik[cellIndex] + cell
                if rowIndex == cellIndex :
                    c_kk[rowIndex] = cell

                cellIndex = cellIndex + 1

            rowIndex = rowIndex + 1

        index = 0
        while index < len(confMatrix) :

            c_ik_val = c_ik[index]
            c_kk_val = c_kk[index]

            if c_ik[index] != 0 :
                res_val = ((c_ik_val - c_kk_val) / c_ik_val) * 100
                result[index] = res_val

            index = index + 1

        return result


    def __getWronglyAssignedToClass__(self, confMatrix):
        result = [0] * len(confMatrix)
        c_kk = [0] * len(confMatrix)
        c_ik = [0] * len(confMatrix)
        c_ki = [0] * len(confMatrix)
        total = 0

        rowIndex = 0
        for row in confMatrix:

            cellIndex = 0
            for cell in row:
                c_ki[rowIndex] = c_ik[rowIndex] + cell
                c_ik[cellIndex] = c_ik[cellIndex] + cell
                total = total + cell
                if rowIndex == cellIndex:
                    c_kk[rowIndex] = cell

                cellIndex = cellIndex + 1

            rowIndex = rowIndex + 1

        index = 0
        while index < len(confMatrix):

            c_ik_val = c_ik[index]
            c_kk_val = c_kk[index]
            c_ki_val = c_ki[index]

            if c_ik[index] != 0 :
                res_val = ((c_ki_val - c_kk_val ) / (total - c_ik_val)) * 100
                result[index] = res_val

            index = index + 1

        return result

    def _getAssociateObjs(self):
        template = self._template
        segm = self._img

        templateObjs = colMaks.getMaskFromColors(template)
        segmObjsObjs = colMaks.getMaskFromColors(segm)

        # -- ассоцияция объектов с их шаблонными масками --
        associateObjs = []  # выходной массив

        i = 0
        for templ in templateObjs:  # проходимся по всем шаблонам

            minErr = template.shape[0] * template.shape[
                1]  # минимальная ошибка изначально равна количеству всех пикселей
            searchObj = np.zeros(template.shape, np.uint8)  # по умолчанию искомый объект = пустое поле
            searchIndex = -1  # индекс найденого объекта

            index = 0
            for obj in segmObjsObjs:  # проходимся по всем найденым при сегментации объектам
                diff = cv2.absdiff(templ, obj)  # кадровая разница между объектами
                errPtCount = cv2.countNonZero(diff)  # количество точек в разнице
                objPtCount = cv2.countNonZero(obj)  # количество точек в самом обхекте

                if (errPtCount < minErr) and (
                    errPtCount < objPtCount):  # еслиошибка минимальна и не весь объект ошибочный
                    # приравниваем текущий объект исомому
                    minErr = errPtCount
                    searchObj = obj
                    searchIndex = index

                index = index + 1

            if searchIndex != -1:  # если найден объект
                segmObjsObjs.pop(searchIndex)  # удаляем его из коллекции

            name = 'Object {0}'.format(i)  # имя объекта в ассоцированном списке
            associateObjs.append(self.__creatAssociateObj__(name, templ, searchObj))  # объект с шаблоном в резуьтат

            i = i + 1

        return associateObjs


    def printConfMatrix(self):
        confMatrix = self._confMatrix

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
        total = 0
        for rowTotal in rowTotals:
            confMatrixWithTotal[rowIndex, len(confMatrixWithTotal)-1] = rowTotal
            total = total + rowTotal
            rowIndex = rowIndex + 1

        cellIndex = 0
        for cellTotal in cellTotals:
            confMatrixWithTotal[len(confMatrixWithTotal)-1, cellIndex] = cellTotal
            cellIndex = cellIndex + 1

        confMatrixWithTotal[len(confMatrixWithTotal) - 1, len(confMatrixWithTotal)-1] = total


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




    def getQuality(self):
        confMatrix = self._confMatrix

        m1 = self.__getIncorrectlyClassifiedPixels__(confMatrix)
        m2 = self.__getWronglyAssignedToClass__(confMatrix)

        result = sum(m1) / len(m2)
        return result

