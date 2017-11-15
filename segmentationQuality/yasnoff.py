import cv2
import numpy as np
import math

from pexpect.ANSI import term

from imgUtils import ColorMask as colMaks


class Yasnoff:

    def __init__(self, template, img):
        # init parameters
        self._template = template
        self._img = img

        self._height = img.shape[0]
        self._width = img.shape[1]

        # calculation
        templateObjs = colMaks.getMaskFromColors(template)
        segmObjsObjs = colMaks.getMaskFromColors(img)


        self._templ_len = len(templateObjs)
        self._segm_len = len(segmObjsObjs)

        associateObjs = self._getAssociateObjs(templateObjs, segmObjsObjs)
        self._confMatrix = self.__createConfMatrix__(associateObjs)


    def __creatAssociateObj__(self, name, template, obj):
        assObj = []
        assObj.append(name)
        assObj.append(template)
        assObj.append(obj)
        return assObj



    def __createConfMatrix__(self, associateObjs):

        confMatrix = np.zeros((len(associateObjs), len(associateObjs)))

        # проходим по всем
        #  объектов
        rowIndex = 0
        for row in associateObjs:
            segmObj = row[2]

            # если изображение не одноканальное - преобразуем в оттенки серого
            if segmObj.ndim == 3:
                segmObj = cv2.cvtColor(segmObj, cv2.COLOR_BGR2GRAY)

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


    def _getIncorrectlyClassifiedPixels(self, confMatrix):
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
                # res_val = ((sum(c_ik) - c_kk_val) / sum(c_ik)) * 100


                result[index] = res_val

            index = index + 1

        # test



        return result


    def _getWronglyAssignedToClass(self, confMatrix):
        result = [0] * len(confMatrix)
        c_kk = [0] * len(confMatrix)
        c_ik = [0] * len(confMatrix)
        c_ki = [0] * len(confMatrix)
        total = 0

        rowIndex = 0
        for row in confMatrix:

            cellIndex = 0
            for cell in row:
                c_ki[rowIndex] = c_ki[rowIndex] + cell
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

    def _getAssociateObjs(self, templateObjs, segmObjsObjs):
        template = self._template
        height = self._height
        width = self._width


        # -- ассоцияция объектов с их шаблонными масками --
        associateObjs = []  # выходной массив

        print(' ** tempaltes **')

        i = 0
        for templ in templateObjs:  # проходимся по всем шаблонам

            # test
            templPtCount = cv2.countNonZero(templ)
            print('index = {0}; c_ik = {1}'.format(i, templPtCount))

            minErr = template.shape[0] * template.shape[1]  # минимальная ошибка изначально равна количеству всех пикселей
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

        for segmObj in segmObjsObjs: # если остались не ассоциированны объекты - проходимся по ним
            name = 'Object {0}'.format(len(associateObjs))  # имя объекта в ассоцированном списке
            blank_image = np.zeros((height, width, 1), np.uint8)

            associateObjs.append(self.__creatAssociateObj__(name, blank_image, segmObj))  # объект с пустым шаблоном в резуьтат

        print(' **--**')

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


    def getIncorrecClassPixels(self):
        confMatrix = self._confMatrix
        m1 = self._getIncorrectlyClassifiedPixels(confMatrix)
        result = sum(m1) / len(m1)
        return result

    def getWronglyAssigneToClass(self):
        confMatrix = self._confMatrix
        m2 = self._getWronglyAssignedToClass(confMatrix)
        result = sum(m2) / len(m2)
        return result


    def getFrags(self, a = 0.16, b = 2):
        templ_len = self._templ_len
        segm_len = self._segm_len

        frag = 1 / 1 + (a * math.fabs((segm_len - templ_len)) ** b)
        return frag



# ------------------------------
# --------- test method --------
# ------------------------------


# project_dir = '/home/ange/Python/workplace/Dissertation'
# print('project_dir = {0}'.format(project_dir))
#
# template = cv2.imread(project_dir + "/resources/applePears/1/template.png", 3)
# segm = cv2.imread(project_dir + "/resources/applePears/1/segmented/java/val_12_0.png", 3)
#
# print('start qual calc')
# yasn = Yasnoff(template, segm)
# yasn.printConfMatrix()
#
# print('qual = {0}'.format(yasn.getQuality()))


