import cv2
import numpy as np
import math

from sympy.functions.special.polynomials import hermite

from imgUtils import ColorMask as colMaks


class Yasnoff_Moments:

    def __init__(self, template, img):
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
        self._contMomentMatrix = self._createMommentsMatrix(sortTemplObjs, sortSegmObjs)


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


    def _createMommentsMatrix(self, templObjs, segmObjs):

        # расчитываем моменты
        temp_res = self._getMoments(templObjs)
        segm_res = self._getMoments(segmObjs)

        temp_moments = temp_res[0]
        segm_moments = segm_res[0]

        # используются только пространствунный моменты
        used_m_keys = ['m00', 'm10', 'm01', 'm20', 'm11', 'm02', 'm30', 'm21', 'm12', 'm03']

        max_length = max(len(segmObjs), len(templObjs))
        contMomentMatrix = np.zeros((max_length, max_length))

        # расчитываем разницу моментов
        for i in range(0, len(temp_moments)):
            temp_moment = temp_moments[i]
            temp_count = cv2.countNonZero(templObjs[i])

            for j in range(0, len(segm_moments)):
                segm_moment = segm_moments[j]
                segm_count = cv2.countNonZero(segmObjs[j])

                diff_moments = []
                for m_key in used_m_keys:
                    obj_m = segm_moment[m_key]
                    temp_m = temp_moment[m_key]

                    # diff = abs((1 / temp_m) - (1 / obj_m))  # CV_CONTOURS_MATCH_I1
                    # diff = abs(temp_m - obj_m)            # CV_CONTOURS_MATCH_I2
                    diff = abs((temp_m - obj_m) / temp_m) # CV_CONTOURS_MATCH_I3

                    diff_moments.append(diff)

                contour_diff = sum(diff_moments) / len(diff_moments)
                dss = contour_diff ** (1 / 4)

                contMomentMatrix[j][i] = dss

                # if i == j:
                #
                #     print(' num = {0}; contour_diff = {1}; dss = {2}'.format(i, contour_diff, dss))
                #     temp_cnt_img = temp_res[1][i]
                #     segm_cnt_img = segm_res[1][i]
                #
                #     cv2.imshow("temp_cnt_img", temp_cnt_img)
                #     cv2.imshow("segm_cnt_img", segm_cnt_img)
                #     cv2.waitKey()

        # # нормализация значений разницы моментов по шаблонам
        # height = self._segm_len
        # width = self._templ_len
        #
        # def normalize(array):
        #     sqr = []
        #     for val in array:
        #         sqr.append(val ** 2)
        #
        #     length = math.sqrt(sum(sqr))
        #     result = []
        #     for val in array:
        #         if length != 0:
        #             result.append(val / length)
        #         else:
        #             result.append(val)
        #
        #     return result
        #
        # for cell_idx in range(0, width):
        #     array = []
        #     for row in contMomentMatrix:
        #         array.append(row[cell_idx])
        #
        #     norm_array = normalize(array)
        #
        #     for j in range(0, height):
        #         val = norm_array[j]
        #         contMomentMatrix[j][cell_idx] = (1 - val)
        #
        #     # print('norm_array = {0};'.format(norm_array))

        return contMomentMatrix



    def _getMoments(self, images):
        img_height = self._height
        img_width = self._width

        # проходим по всем сегментированным объектам и считаем моменты
        moments = []
        cnt_imgs = []

        for img in images:
            if img.ndim == 3:  # если изображение не одноканальное
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # преобразуем в оттенки серого

            _, contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) > 0:
                cnt = contours[0]
                moment = cv2.moments(cnt)
                moments.append(moment)

                # рисуем полученые контуры и добавдяем в коллекция (для дальнейшей визуализации)
                cnt_img = np.zeros((img_height, img_width, 3), np.uint8)
                cv2.drawContours(cnt_img, [cnt], -1, (0, 255, 0), 1)
                cnt_imgs.append(cnt_img)

            else:
                moments.append(0.0)

                # добавдяем в коллекцию с контурами  (для дальнейшей визуализации)
                cnt_img = np.zeros((img_height, img_width, 3), np.uint8)
                cnt_imgs.append(cnt_img)

        res = []
        res.append(moments)
        res.append(cnt_imgs)

        # return (moments)
        return res

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




    def printMatrix(self):
        contMomentMatrix = self._contMomentMatrix
        confMatrix = self._confMatrix

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

        for name, conf_row, mom_row in zip(row_names, confMatrix, contMomentMatrix):
            str_row = []
            for idx in range(0, len(conf_row)):
                cell = conf_row[idx]
                moment = round(mom_row[idx], 4)
                moment_str = '{:>8}'.format(moment)
                val = '{0} : {1}'.format(cell, moment_str)
                str_row.append(val)
            print(row_format.format(name, *str_row))


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

        # print('m1 = {0};'.format(m1))

        result = sum(m1) / len(m1)
        return result

    def getWronglyAssigneToClass(self):
        confMatrix = self._confMatrix
        m2 = self._getWronglyAssignedToClass(confMatrix)

        # print('m2 = {0};'.format(m2))

        result = sum(m2)
        return result

    def getFrags(self, a=0.16, b=2):
        templ_len = self._templ_len
        segm_len = self._segm_len

        frag = 1 / 1 + (a * math.fabs((segm_len - templ_len)) ** b)
        return frag


    def get_m3(self):
        height = self._segm_len
        width = self._templ_len
        contMomentMatrix = self._contMomentMatrix

        min_l = min(height, width)

        result = []
        for i in range(0, min_l):
            i_kk = contMomentMatrix[i][i]
            result.append(i_kk)

        print('m4s = {0};'.format(result))

        return sum(result) / len(result)

    def get_m4(self):
        height = self._segm_len
        width = self._templ_len
        contMomentMatrix = self._contMomentMatrix
        confMatrix = self._confMatrix

        min_l = min(height, width)

        result = []
        for i in range(0, min_l):
            i_kk = contMomentMatrix[i][i]

            # сумма всех пикселей полученого шаблона
            c_ik = 0
            for row in confMatrix:
                if i < len(row):
                    val = row[i]
                    c_ik = c_ik + val

            result.append(i_kk * i_kk)

        print('m4s = {0};'.format(result))

        return sum(result) / len(result)