import cv2
import numpy as np
import math

from sympy.functions.special.polynomials import hermite

from imgUtils import ColorMask as colMaks


class Yasnoff:


    def __init__(self, template, img, pixelDistError=False):
        self._template = template
        self._img = img

        self._height = img.shape[0]
        self._width = img.shape[1]

        templObjs = colMaks.getMaskFromColors(template)
        segmObjs = colMaks.getMaskFromColors(img)

        self._templ_len = len(templObjs)
        self._segm_len = len(segmObjs)

        sortTemplObjs = self._sortBySize(templObjs)
        sortSegmObjs = self._sortByBitwise(sortTemplObjs, segmObjs)

        self._confMatrix = self._createDiffMatrix(sortTemplObjs, sortSegmObjs)

        if (pixelDistError):
            self._pixelDistError = self._createPixelDistError(sortTemplObjs, sortSegmObjs)
        else:
            self._pixelDistError = None




    def _sortBySize(self, templObjs):
        # сортируем массив с шаблонами по размеру найденных элементов
        def sortBySize(img):
            count = cv2.countNonZero(img)
            return 0 - count  # для сортировки по убыванию

        templObjs.sort(key=sortBySize)
        return templObjs


    def _sortByBitwise(self, templObjs, segmObjs):
        templ_len = self._templ_len
        segm_len = self._segm_len

        diff_matrix = self._createDiffMatrix(templObjs, segmObjs)

        # получаем по порядку индексы элементов, имеющих минимальную разницу с шаблонами
        row_indexes = []
        for cell_idx in range(0, templ_len):  # переменная индекса столбца
            max_row_val = -1
            max_row_idx = -1

            for j in range(0, segm_len):
                row = diff_matrix[j]
                val = row[cell_idx]

                if val > max_row_val and j not in row_indexes:
                    max_row_val = val
                    max_row_idx = j

            if max_row_idx != -1:
                row_indexes.append(max_row_idx)

        # добавляем по порядку все найденные соотвесвующие объекты
        sorted_segmObjs = []
        for idx in row_indexes:
            sorted_segmObjs.append(segmObjs[idx])

        # добавляем все оставшиеся объекты
        for i in range(0, len(segmObjs)):
            if i not in row_indexes:
                sorted_segmObjs.append(segmObjs[i])

        return sorted_segmObjs


    def _createDiffMatrix(self, templObjs, segmObjs):
        # заполняем двумерный массив пиксельной разницей
        max_length = max(len(segmObjs), len(templObjs))
        diff_matrix = np.zeros((max_length, max_length))

        for i in range(0, len(segmObjs)):
            segm = segmObjs[i]

            if segm.ndim == 3:  # если изображение не одноканальное
                segm = cv2.cvtColor(segm, cv2.COLOR_BGR2GRAY)  # преобразуем в оттенки серого

            for j in range(0, len(templObjs)):
                templ = templObjs[j]
                intersec = cv2.bitwise_and(templ, segm)
                ptCount = cv2.countNonZero(intersec)
                diff_matrix[i][j] = ptCount

        return diff_matrix


    def _createPixelDistError(self, templObjs, segmObjs):
        height = self._height
        width = self._width

        max_length = max(len(segmObjs), len(templObjs))
        pixelDistError = np.zeros((max_length, max_length))

        for i in range(0, len(segmObjs)):
            segm = segmObjs[i]

            if segm.ndim == 3:  # если изображение не одноканальное
                segm = cv2.cvtColor(segm, cv2.COLOR_BGR2GRAY)  # преобразуем в оттенки серого

            for j in range(0, len(templObjs)):

                if i != j: # ошибочно квалифицированне пиксели
                    templ = templObjs[j]

                    # находим контуры шаблона
                    _, contours, _ = cv2.findContours(templ, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    cnt = contours[0]

                    # нахоим неправлиьно классифицированные пиксели (белым)
                    errorPt = cv2.bitwise_and(templ, segm)
                    dists = []

                    y_s, x_s = (errorPt > 0).nonzero()
                    for x, y in zip(x_s, y_s):
                        dist = cv2.pointPolygonTest(cnt, (x, y), True)

                        # если пиксель не лежит контуре добаляем к массиву
                        if dist > 0:
                            dists.append(dist ** 2)

                    summ = sum(dists)
                    sqrt = math.sqrt(summ)
                    a = height * width

                    result = (sqrt / a * 100)
                    pixelDistError[i][j] = result

        return pixelDistError



    def _getIncorrectlyClassifiedPixels(self, confMatrix):
        height = self._segm_len
        width = self._templ_len

        result = []
        for i in range(0, len(confMatrix)):

            # правильно класифицированные пиксели
            c_kk = 0
            if i < width and i < height:
                c_kk = confMatrix[i][i]

            # сумма всех пикселей полученого шаблона
            c_ik = 0
            for row in confMatrix:
                if i < len(row):
                    val = row[i]
                    c_ik = c_ik + val

            # расчет значения
            if c_ik != 0:
                res_val = ((c_ik - c_kk) / c_ik) * 100
                result.append(res_val)
            else:
                result.append(0)

        return result


    def _getWronglyAssignedToClass(self, confMatrix):
        height = self._segm_len
        width = self._templ_len

        result = []
        for i in range(0, len(confMatrix)):

            # правильно класифицированные пиксели
            c_kk = 0
            if i < width and i < height:
                c_kk = confMatrix[i][i]

            # сумма всех пикселей полученого шаблона
            c_ik = 0
            for row in confMatrix:
                if i < len(row):
                    val = row[i]
                    c_ik = c_ik + val

            # сумма всех пикселей полученого объекта
            c_ki = 0
            if i < height:
                row = confMatrix[i]
                for val in row:
                    c_ki = c_ki + val

            # общая сумма
            total = 0
            for row in confMatrix:
                total = total + sum(row)

            # расчет значения
            res_val = ((c_ki - c_kk) / (total - c_ik)) * 100
            result.append(res_val)

        return result



    def printMatrixWithTotal(self):
        confMatrix = self._confMatrix

        # получаем количество шаблонов и объектов
        height = self._segm_len
        width = self._templ_len

        # создание и расчет массивов с суммами
        rowTotals = [0] * (height + 1)
        cellTotals = [0] * (width + 1)

        rowIndex = 0
        for i in range(0, height):
            row = confMatrix[i]
            cellTotal = 0
            cellIndex = 0

            for j in range(0, width):
                cell = row[j]
                cellTotal = cellTotal + cell
                cellTotals[cellIndex] = cellTotals[cellIndex] + cell
                cellIndex = cellIndex + 1

            rowTotals[rowIndex] = cellTotal
            rowIndex = rowIndex + 1

        row_names = []
        for i in range(0, height):
            row_names.append('Object %s' % i)

        cell_names = []
        for i in range(0, width):
            cell_names.append('Object %s' % i)

        row_names.append('Total')
        cell_names.append('Total')

        # создание новой матрицы с Total
        resultMatrix = np.zeros((height + 1, width + 1))

        # Наполнение значенями из сторой матрицы
        rowIndex = 0
        for i in range(0, height):
            row = confMatrix[i]
            cellIndex = 0

            for j in range(0, width):
                cell = row[j]
                resultMatrix[rowIndex, cellIndex] = cell
                cellIndex = cellIndex + 1
            rowIndex = rowIndex + 1

        # Добавление сумм
        for j in range(0, height):
            resultMatrix[j][width] = rowTotals[j]

        for i in range(0, width):
            resultMatrix[height][i] = cellTotals[i]

        resultMatrix[height][width] = sum(cellTotals)

        # Создаем архивы для хранения имен столбцов и колонок
        rows_names = []
        for i in range(0, height):
            rows_names.append('Object %s' % i)

        cell_names = []
        for i in range(0, width):
            cell_names.append('Temaple %s' % i)

        row_names.append('Total')
        cell_names.append('Total')

        # Печать значений
        row_format = "{:>15}" * (width + 2)
        print(row_format.format("", *cell_names))

        for name, row in zip(row_names, resultMatrix):
            print(row_format.format(name, *row))


    def getIncorrecClassPixels(self):
        confMatrix = self._confMatrix
        m1 = self._getIncorrectlyClassifiedPixels(confMatrix)

        result = sum(m1) / len(m1)
        return result


    def getWronglyAssigneToClass(self):
        confMatrix = self._confMatrix
        m2 = self._getWronglyAssignedToClass(confMatrix)

        result = sum(m2)
        return result

    def getFrags(self, a=0.16, b=2):
        templ_len = self._templ_len
        segm_len = self._segm_len

        frag = 1 / 1 + (a * math.fabs((segm_len - templ_len)) ** b)
        return frag

    def printPixelDistError(self):
        pixelDistError = self._pixelDistError

        if pixelDistError is not None:

            # получаем количество шаблонов и объектов
            height = self._segm_len
            width = self._templ_len

            row_names = []
            for i in range(0, height):
                row_names.append('Object %s' % i)

            cell_names = []
            for i in range(0, width):
                cell_names.append('Object %s' % i)

            # Создаем архивы для хранения имен столбцов и колонок
            rows_names = []
            for i in range(0, height):
                rows_names.append('Object %s' % i)

            cell_names = []
            for i in range(0, width):
                cell_names.append('Template %s' % i)

            # Печать значений
            row_format = "{:>22}" * (width + 1)
            print(row_format.format("", *cell_names))

            for name, row in zip(row_names, pixelDistError):
                print(row_format.format(name, *row))


    def getPixelDistError(self):
        pixelDistError = self._pixelDistError
        errors = []
        if pixelDistError is not None:
            for row in pixelDistError:
                row_average = sum(row) / len(row)
                errors.append(row_average)
            result = sum(errors)
            return result
        else:
            return 0
