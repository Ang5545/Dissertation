import cv2
import numpy as np
import math
import matplotlib.pyplot as plt




from imgUtils import ColorMask as colMaks


class Yasnoff_Mod:

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
        img_height = self._height
        img_width = self._width

        # получаем количество шаблонов и объектов
        height = len(associateObjs)
        width = 0
        for ass in associateObjs:
            if ass[1] is not None:
                width = width + 1

        contMomentMatrix = np.zeros((len(associateObjs), len(associateObjs)))

        # проходим по всем шаблонам и считаем моменты
        temp_moments = []
        # temp_cnt_imgs = []
        for assObj in associateObjs:
            template = assObj[1]

            if template is not None:
                _, templ_contours, _ = cv2.findContours(template, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                if len(templ_contours) > 0:
                    templ_cnt = templ_contours[0]
                    templ_moment = cv2.moments(templ_cnt)
                    temp_moments.append(templ_moment)

                    # # рисуем полученые контуры и добавдяем в коллекция (для дальнейшей визуализации)
                    # temp_cnt_img = np.zeros((img_height, img_width, 3), np.uint8)
                    # cv2.drawContours(temp_cnt_img, [templ_cnt], -1, (0, 255, 0), 1)
                    # temp_cnt_imgs.append(temp_cnt_img)
                else:
                    temp_moments.append(0.0)

                    # # добавдяем в коллекцию с контурами  (для дальнейшей визуализации)
                    # temp_cnt_img = np.zeros((height, width, 3), np.uint8)
                    # temp_cnt_imgs.append(temp_cnt_img)


        # проходим по всем сегментированным изображениям и сравниваем моменты
        for i in range(0, width):
            temp_moment = temp_moments[i]
            # temp_cnt_img = temp_cnt_imgs[i]

            for j in range(0, height):
                assObj = associateObjs[j]
                segmObj = assObj[2]

                # если изображение не одноканальное - преобразуем в оттенки серого
                if segmObj.ndim == 3:
                    segmObj = cv2.cvtColor(segmObj, cv2.COLOR_BGR2GRAY)

                _, segm_contours, _ = cv2.findContours(segmObj, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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
                    contMomentMatrix[j][i] = contour_diff

                    # # Показываем объекты для сравнения
                    # print('contour_diff = {0};'.format(contour_diff))
                    #
                    # segm_cnt_image = np.zeros((img_height, img_width, 3), np.uint8)
                    # cv2.drawContours(segm_cnt_image, [segm_cnt], -1, (0, 255, 0), 1)
                    #
                    # cv2.imshow("temp_cnt_img", temp_cnt_img)
                    # cv2.imshow("segm_cnt_image", segm_cnt_image)
                    # cv2.waitKey()


        # нормализация значений разницы моментов по объектам
        def normalize(array):
            sqr = []
            for val in array:
                sqr.append(val ** 2)

            length = math.sqrt(sum(sqr))
            result = []
            for val in array:
                if length != 0:
                    result.append(val/length)
                else:
                    result.append(val)

            return result

        for i in range(0, height):
            array = []
            for j in range(0, width):
                array.append(contMomentMatrix[i][j])

            norm_array = normalize(array)

            for j in range(0, width):
                val = norm_array[j]
                contMomentMatrix[i][j] = val


            # # print('array = {0};'.format(array))
            # # print('norm_array = {0};'.format(norm_array))

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

        # print(' m2 c_kk = {0}'.format(c_kk))

        index = 0
        while index < len(confMatrix):

            c_ik_val = c_ik[index]
            c_kk_val = c_kk[index]
            c_ki_val = c_ki[index]

            res_val = ((c_ki_val - c_kk_val ) / (total - c_ik_val)) * 100
            result[index] = res_val

            index = index + 1

        return result

    def _getAssociateObjs(self, templateObjs, segmObjsObjs):

        # TODO сделать алгоритм инвариатный к длиннам коллекций шаблонов и объектов

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
            for idx in range (0, len(conf_row)):
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


    def getFrags(self, a = 0.16, b = 2):
        templ_len = self._templ_len
        segm_len = self._segm_len

        frag = 1 / 1 + (a * math.fabs((segm_len - templ_len)) ** b)
        return frag

    def get_m3(self):
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

        i_kk = [1] * height
        i_ik = [0] * height
        i_ki = [0] * height
        total = 0
        print('def i_kk = {0};'.format(i_kk))

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

        result = []
        m2 = self._getWronglyAssignedToClass(confMatrix)

        for i_kk_val, i_ik_val, i_ki_val, m2_val in zip(i_kk, i_ik, i_ki, m2):
            res_val = i_kk_val / i_ki_val
            result.append(m2_val * res_val)

        print('i_ki = {0};'.format(i_ki))
        print('i_kk = {0};'.format(i_kk))
        print('m2 = {0};'.format(m2))
        print('m3 = {0};'.format(result))

        return sum(result)


    # def get_m4(self):
    #     contMomentMatrix = self._contMomentMatrix
    #     confMatrix = self._confMatrix
    #     associateObjs = self._associateObjs
    #
    #     # получаем количество шаблонов и объектов
    #     height = len(associateObjs)
    #     width = 0
    #     for ass in associateObjs:
    #         template = ass[1]
    #         if template is not None:
    #             width = width + 1
    #
    #     c_kk = [0] * height
    #     c_ik = [0] * height
    #     c_ki = [0] * height
    #     total = 0
    #
    #     for i in range(0, width):
    #         cell = []
    #         for j in range(0, height):
    #             val = confMatrix[j][i]
    #             mom = contMomentMatrix[j][i]
    #             cell.append(val)
    #             c_ki[j] = c_ki[j] + val
    #             total = total + val
    #             if i == j:
    #                 kk = val * mom
    #                 c_kk[i] = kk
    #
    #
    #         c_ik[i] = sum(cell)
    #
    #     result = []
    #     for i in range(0, height):
    #         c_ik_val = c_ik[i]
    #         c_kk_val = c_kk[i]
    #         c_ki_val = c_ki[i]
    #
    #         res_val = ((c_ki_val - c_kk_val) / (total - c_ik_val)) * 100
    #         result.append(res_val)
    #
    #     return sum(result)


    def getTemplateComut(self):
        return self._templ_len

    def getSegmentsComut(self):
        return self._segm_len









