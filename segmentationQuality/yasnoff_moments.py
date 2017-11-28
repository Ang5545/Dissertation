import cv2
import numpy as np
import math

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

        # print('len sortTemplObjs = {0}'.format(len(sortTemplObjs)))
        # print('len sortSegmObjs = {0}'.format(len(sortSegmObjs)))
        #
        # for i in range(0, len(sortSegmObjs)):
        #     if i > len(sortTemplObjs):
        #         cv2.imshow("Template", sortTemplObjs[i])
        #         cv2.imshow("Object", sortSegmObjs[i])
        #         cv2.waitKey()
        #     else:
        #         cv2.imshow("Object", sortSegmObjs[i])
        #         cv2.waitKey()

        # cv2.imshow("Template 1", sortTemplObjs[1])
        # cv2.imshow("Object 1", sortSegmObjs[1])
        # cv2.waitKey()

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
            else:
                print('not find max!!')

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
        diff_matrix = []
        for segm in segmObjs:
            row = []

            if segm.ndim == 3:  # если изображение не одноканальное
                segm = cv2.cvtColor(segm, cv2.COLOR_BGR2GRAY)  # преобразуем в оттенки серого

            for templ in templObjs:
                intersec = cv2.bitwise_and(templ, segm)
                ptCount = cv2.countNonZero(intersec)
                row.append(ptCount)

            diff_matrix.append(row)
        return diff_matrix


    def _createMommentsMatrix(self, templObjs, segmObjs):

        # расчитываем моменты
        temp_moments = self._getMoments(templObjs)
        segm_moments = self._getMoments(segmObjs)

        # используются только пространствунный моменты
        used_m_keys = ['m00', 'm10', 'm01', 'm20', 'm11', 'm02', 'm30', 'm21', 'm12', 'm03']


        contMomentMatrix = np.zeros((len(segm_moments), len(temp_moments)))


        # расчитываем разницу моментов
        for i in range(0, len(temp_moments)):
            temp_moment = temp_moments[i]

            for j in range(0, len(segm_moments)):
                segm_moment = segm_moments[j]

                if i == 4 and j == 4:
                    for m_key in used_m_keys:
                        obj_m = segm_moment[m_key]
                        temp_m = temp_moment[m_key]
                        diff = abs((temp_m - obj_m) / temp_m)

                        print('m_key = {0}; obj_m = {1}; temp_m = {2}; diff = {3};'.format(m_key, obj_m, temp_m, diff) )

                diff_moments = []
                for m_key in used_m_keys:
                    obj_m = segm_moment[m_key]
                    temp_m = temp_moment[m_key]

                    # diff = abs((1 / temp_m) - (1 / obj_m))  # CV_CONTOURS_MATCH_I1
                    # diff = abs(temp_m - obj_m)            # CV_CONTOURS_MATCH_I2
                    diff = abs((temp_m - obj_m) / temp_m) # CV_CONTOURS_MATCH_I3

                    diff_moments.append(diff)


                contour_diff = sum(diff_moments) / len(diff_moments)
                contMomentMatrix[j][i] = contour_diff


        # # нормализация значений разницы моментов по объектам
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
        # for i in range(0, height):
        #     array = []
        #     for j in range(0, width):
        #         array.append(contMomentMatrix[i][j])
        #
        #     norm_array = normalize(array)
        #
        #     for j in range(0, width):
        #         val = norm_array[j]
        #         contMomentMatrix[i][j] = val

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

        # for cnt_img in cnt_imgs:
        #     cv2.imshow("cnt", cnt_img)
        #     cv2.waitKey()

        # cv2.imshow("cnt_img 1", cnt_imgs[1])
        # cv2.waitKey()

        # cv2.imshow("cnt_img 4", cnt_imgs[4])
        # cv2.waitKey()

        return(moments)


    def _getIncorrectlyClassifiedPixels(self, confMatrix):
        height = self._segm_len
        width = self._templ_len

        result = []
        for i in range(0, height):

            # правильно класифицированные пиксели
            c_kk = 0
            if i < width:
                c_kk = confMatrix[i][i]

            # сумма всех пикселей полученого объекта
            row = confMatrix[i]
            c_ki = 0
            for val in row:
                c_ki = c_ki + val

            # сумма всех пикселей полученого шаблона
            c_ik = 0
            for row in confMatrix:
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
        for i in range(0, height):

            # правильно класифицированные пиксели
            c_kk = 0
            if i < width:
                c_kk = confMatrix[i][i]

            # сумма всех пикселей полученого объекта
            row = confMatrix[i]
            c_ki = 0
            for val in row:
                c_ki = c_ki + val

            # сумма всех пикселей полученых шаблонов
            c_ik = 0
            for row in confMatrix:
                val = row[i]
                c_ik = c_ik + val

            # общая сумма
            total = 0
            for row in confMatrix:
                for val in row:
                    total = total + val

            print('total = {0}; c_ik = {1}'.format(total, c_ik))

            # расчет значения
            if c_ik != 0:
                res_val = ((c_ki - c_kk) / (total - c_ik)) * 100
                result.append(res_val)
            else:
                result.append(0)

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

    def get_m3(self):
        contMomentMatrix = self._contMomentMatrix
        confMatrix = self._confMatrix

        height = self._segm_len
        width = self._templ_len

        i_ik = [0] * width
        i_kk = [0] * height
        i_ki = [0] * height
        total = 0

        for i in range(0, width):
            cell = []
            for j in range(0, height):
                val = contMomentMatrix[j][i]
                cell.append(val)
                i_ki[j] = i_ki[j] + val
                total = total + val
                if i == j:
                    i_kk[i] = val

            i_ik[i] = sum(cell)

        print('i_kk = {0}'.format(i_kk))



        result = []
        m2 = self._getWronglyAssignedToClass(confMatrix)

        for i_kk_val, i_ik_val, i_ki_val, m2_val in zip(i_kk, i_ik, i_ki, m2):
            # res_val = i_kk_val / i_ki_val
            # res_val = i_ki_val
            res_val = i_kk_val
            result.append(res_val)

        # print('i_ki = {0};'.format(i_ki))
        # print('i_kk = {0};'.format(i_kk))
        # print('i_ik = {0};'.format(i_ik))
        #
        # print('m2 = {0};'.format(m2))
        # print('m3 = {0};'.format(result))

        return sum(result)