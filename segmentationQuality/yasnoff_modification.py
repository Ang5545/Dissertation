import cv2
import numpy as np
import math
import matplotlib.pyplot as plt




from imgUtils import ColorMask as colMaks


class Yasnoff_Modification:

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

        self._associateObjs = self._getAssociateObjs(templateObjs, segmObjsObjs)
        self._confMatrix = self.__createConfMatrix__()
        self._contMomentMatrix = self.__createContourMomentMatrix__()

    def __creatAssociateObj__(self, name, template, obj):
        assObj = []
        assObj.append(name)
        assObj.append(template)
        assObj.append(obj)
        return assObj



    def __createConfMatrix__(self):
        associateObjs = self._associateObjs

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

                if template is not None:

                    intersec = cv2.bitwise_and(template, segmObj)
                    ptCount = cv2.countNonZero(intersec)
                    confMatrix[rowIndex][cellIndex] = ptCount

                cellIndex = cellIndex + 1

            rowIndex = rowIndex + 1

        return confMatrix


    def __createContourMomentMatrix__(self):

        associateObjs = self._associateObjs
        height = self._height
        width = self._width

        contMomentMatrix = np.zeros((len(associateObjs), len(associateObjs)))

        # проходим по всем шаблонам и считаем моменты (чтобы делать это один раз)
        temp_moments = []
        temp_cont_images = []
        for assObj in associateObjs:
            template = assObj[1]

            if template is not None:

                #получаем контуры шаблона
                _, templ_contours, _ = cv2.findContours(template, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                if (len(templ_contours) != 0):
                    templ_cnt = templ_contours[0]
                    templ_moment = cv2.moments(templ_cnt)
                    temp_moments.append(templ_moment)

                    # # Показать контуры шаблона
                    temp_cnt_image = np.zeros((height, width, 3), np.uint8)
                    cv2.drawContours(temp_cnt_image, [templ_cnt], -1, (0, 255, 0), 1)
                    temp_cont_images.append(temp_cnt_image)

                    # cv2.imshow("temp_cnt_image", temp_cnt_image)
                    # cv2.waitKey()

                else:
                    temp_moments.append(0)

        # проходим полученным моментам
        for i in range(0, len(temp_moments)):
            temp_moment = temp_moments[i]
            temp_cont_image = temp_cont_images[i]

            # проходим по всем сегментам
            for j in range(0, len(associateObjs)):
                assObj = associateObjs[j]
                segm = assObj[2]

                # если изображение не одноканальное - преобразуем в оттенки серого
                if segm.ndim == 3:
                    segm = cv2.cvtColor(segm, cv2.COLOR_BGR2GRAY)

                _, segm_contours, _ = cv2.findContours(segm, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                if len(segm_contours) > 0:
                    segm_cnt = segm_contours[0]
                    segm_moment = cv2.moments(segm_cnt)

                    # используются только пространствунный моменты
                    used_m_keys = ['m00', 'm10', 'm01', 'm20', 'm11', 'm02', 'm30', 'm21', 'm12', 'm03']

                    diff_moments = []
                    for m_key in used_m_keys:
                        obj_m = segm_moment[m_key]
                        temp_m = temp_moment[m_key]

                        diff = abs((temp_m - obj_m) / temp_m)
                        diff_moments.append(diff)

                    contour_diff = sum(diff_moments) / len(diff_moments)
                    # print('contour_diff = {0}'.format(contour_diff))
                    contMomentMatrix[j][i] = contour_diff

                    # # # Показать контуры сегментирвоаного объекта
                    # segm_cnt_image = np.zeros((height, width, 3), np.uint8)
                    # cv2.drawContours(segm_cnt_image, [segm_cnt], -1, (0, 255, 0), 1)
                    # cv2.imshow("segm_cnt_image", segm_cnt_image)
                    # cv2.imshow("temp_cont_image", temp_cont_image)
                    # cv2.waitKey()
                    #

        return contMomentMatrix



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
        while index < len(confMatrix):

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

            res_val = ((c_ki_val - c_kk_val ) / (total - c_ik_val)) * 100
            result[index] = res_val

            # print('Object {5}: c_ik_val = {0}; c_kk_val = {1}; c_ki_val = {2}; total = {3}; res_val = {4};'.format(
            #                                                                                                   c_ik_val,
            #                                                                                                   c_kk_val,
            #                                                                                                   c_ki_val,
            #                                                                                                   total,
            #                                                                                                   res_val,
            #                                                                                                   index))

            index = index + 1

        return result

    def _getAssociateObjs(self, templateObjs, segmObjsObjs):
        template = self._template
        height = self._height
        width = self._width

        # -- ассоцияция объектов с их шаблонными масками --
        associateObjs = []  # выходной массив

        i = 0
        for templ in templateObjs:  # проходимся по всем шаблонам

            minErr = template.shape[0] * template.shape[1]  # минимальная ошибка изначально равна количеству всех пикселей
            searchObj = np.zeros(template.shape, np.uint8)  # по умолчанию искомый объект = пустое поле
            searchIndex = -1  # индекс найденого объекта

            index = 0
            for obj in segmObjsObjs:  # проходимся по всем найденым при сегментации объектам
                diff = cv2.absdiff(templ, obj)  # кадровая разница между объектами
                errPtCount = cv2.countNonZero(diff)  # количество точек в разнице
                objPtCount = cv2.countNonZero(obj)  # количество точек в самом обхекте


                if (errPtCount < minErr) and (
                    errPtCount < objPtCount):  # если ошибка минимальна и не весь объект ошибочный
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

            associateObjs.append(self.__creatAssociateObj__(name, None, segmObj))  # объект с пустым шаблоном в резуьтат

        return associateObjs


    def printMatrix(self):
        contMomentMatrix = self._contMomentMatrix
        confMatrix = self._confMatrix
        associateObjs = self._associateObjs

        # получаем количество шаблонов и объектов
        height = len(associateObjs)
        width = 0
        for ass in associateObjs:
            template = ass[1]
            if template is not None:
                width = width + 1


        row_names = []
        for i in range (0, height):
            row_names.append('Object %s' %i)

        cell_names = []
        for i in range (0, width):
            cell_names.append('Object %s' %i)


        # # Создаем архивы для хранения имен столбцов и колонок
        # rows_names = []
        # for i in range(0, height):
        #     rows_names.append('Object %s' % i)
        #
        # cell_names = []
        # for i in range(0, width + 1):
        #     cell_names.append('Temaple %s' % i)
        #
        #
        # # Печать значений
        # row_format = "{:>22}" * (height + 1)
        # print(row_format.format("", *cell_names))
        #
        # for i in range(0, height):
        #     name = row_names[i]
        #     matrRow = confMatrix[i]
        #     momRow = contMomentMatrix[i]
        #
        #     str_row = []
        #
        #     for j in range(0, width):
        #         cell = matrRow[j]
        #         moment = round(momRow[j], 4)
        #         moment_str = '{:>8}'.format(moment)
        #         val = '{0} : {1}'.format(cell, moment_str)
        #         str_row.append(val)
        #
        #     print(row_format.format(name, *str_row))


        # Создаем архивы для хранения имен столбцов и колонок
        rows_names = []
        for i in range(0, height):
            rows_names.append('Object %s' % i)

        cell_names = []
        for i in range(0, width):
            cell_names.append('Temaple %s' % i)


        # Печать значений
        row_format = "{:>22}" * (width + 1)
        print(row_format.format("", *cell_names))

        for i in range(0, height):
            name = row_names[i]
            matrRow = confMatrix[i]
            momRow = contMomentMatrix[i]
            str_row = []

            for j in range(0, width):
                cell = matrRow[j]
                moment = round(momRow[j], 4)
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


    def getFrags(self, a = 0.16, b = 2):
        templ_len = self._templ_len
        segm_len = self._segm_len

        frag = 1 / 1 + (a * math.fabs((segm_len - templ_len)) ** b)
        return frag



    def get_m3(self):
        confMatrix = self._confMatrix
        contMomentMatrix = self._contMomentMatrix


        # получаем критерии по йаснову
        m2 = self._getWronglyAssignedToClass(confMatrix)

        # получаем критерии по моментам
        m_diffs = []
        for i in range(0, len(contMomentMatrix)):
            row = contMomentMatrix[i]
            all = sum(row)
            for j in range(0, len(row)):
                cell = row[j]
                if i == j:
                    if cell != 0:
                        m_diffs.append(all / cell)
                    else:
                        m_diffs.append(all)

        # нормализуем
        def normalize(vec):
            sqr = []
            for val in vec:
                sqr.append(val ** 2)

            length = math.sqrt(sum(sqr))

            result = []
            for val in vec:
                result.append(val / length)

            return result

        m2_norm =  normalize(m2)
        m_diffs_norm = normalize(m_diffs)

        result = []
        for i in range(0, len(m2_norm)):
            val_1 = m2_norm[i]
            val_2 = m_diffs_norm[i]
            result.append((val_1 + val_2) / 2)

        return (sum(result) / len(result))


    def getTemplateComut(self):
        return self._templ_len

    def getSegmentsComut(self):
        return self._segm_len




