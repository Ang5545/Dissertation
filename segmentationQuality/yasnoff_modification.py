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

        # проходим по всем
        #  объектов
        rowIndex = 0
        for row in associateObjs:
            segmObj = row[2]

            # если изображение не одноканальное - преобразуем в оттенки серого
            if segmObj.ndim == 3:
                segmObj = cv2.cvtColor(segmObj, cv2.COLOR_BGR2GRAY)

            _, segm_contours, _ = cv2.findContours(segmObj, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            segm_cnt = segm_contours[0]
            segm_moment = cv2.moments(segm_cnt)


            # проходим по всем столбцам объектов
            cellIndex = 0
            for cell in associateObjs:
                template = cell[1]

                if template is not None:

                    _, templ_contours, _ = cv2.findContours(template, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    templ_cnt = templ_contours[0]
                    templ_moment = cv2.moments(templ_cnt)

                    # используются только пространствунный моменты
                    used_m_keys = ['m00', 'm10', 'm01', 'm20', 'm11', 'm02', 'm30', 'm21', 'm12', 'm03']

                    diff_moments = []
                    for m_key in used_m_keys:
                        obj_m = segm_moment[m_key]
                        temp_m = templ_moment[m_key]

                        diff = abs((temp_m - obj_m) / temp_m)
                        diff_moments.append(diff)
                        # print('moment: {0}; obj_val = {1}; temp_val = {2}; diff = {3};'.format(m_key, obj_m, temp_m, diff))

                    contour_diff = sum(diff_moments) / len(diff_moments)
                    contMomentMatrix[rowIndex][cellIndex] = contour_diff

                    # print('contour_diff = {0};'.format(contour_diff))

                    '''
                    # Показать сравниваемые контуры
                    segm_cnt_image = np.zeros((height, width, 3), np.uint8)
                    temp_cnt_image = np.zeros((height, width, 3), np.uint8)

                    cv2.drawContours(segm_cnt_image, [segm_cnt], -1, (0, 255, 0), 1)
                    cv2.drawContours(temp_cnt_image, [templ_cnt], -1, (0, 255, 0), 1)

                    cv2.imshow("segm_cnt_image", segm_cnt_image)
                    cv2.imshow("temp_cnt_image", temp_cnt_image)
                    cv2.waitKey()
                    '''

                cellIndex = cellIndex + 1

            rowIndex = rowIndex + 1

        from sklearn.preprocessing import normalize

        x = np.random.rand(1000) * 10
        norm1 = x / np.linalg.norm(x)
        norm2 = normalize(x[:, np.newaxis], axis=0).ravel()
        # print(x)
        # print(norm1)
        print(norm2)

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

            # # image, temp_contours, hierarchy = cv2.findContours(templ, 1, 2)
            # _, temp_contours, _ = cv2.findContours(templ, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # temp_cnt = temp_contours[0]
            #
            # temp_moment = cv2.moments(temp_cnt)
            # # print('temp_moment = {0};'.format(temp_moment))
            #
            # temp_cnt_image = np.zeros((height, width, 3), np.uint8)
            # cv2.drawContours(temp_cnt_image, [temp_cnt], -1, (0, 255, 0), 1)
            #
            # # cv2.imshow("Template", templ)
            # cv2.imshow("Template contours", temp_cnt_image)
            # # cv2.waitKey()
            # errPtCounts = []
            #
            # min_moment = 500;
            # best_obj = segmObjsObjs[0]

            index = 0
            for obj in segmObjsObjs:  # проходимся по всем найденым при сегментации объектам
                diff = cv2.absdiff(templ, obj)  # кадровая разница между объектами
                errPtCount = cv2.countNonZero(diff)  # количество точек в разнице
                objPtCount = cv2.countNonZero(obj)  # количество точек в самом обхекте

                # _, obj_contours, _ = cv2.findContours(obj, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                # obj_cnt = obj_contours[0]
                #
                # obj_moment = cv2.moments(obj_cnt)
                # # print('obj_moment = {0};'.format(obj_moment))
                #
                # obj_cnt_image = np.zeros((height, width, 3), np.uint8)
                # cv2.drawContours(obj_cnt_image, [obj_cnt], -1, (0, 255, 0), 1)


                # # используются только пространствунный моменты
                # used_m_keys = ['m00', 'm10', 'm01', 'm20', 'm11', 'm02', 'm30', 'm21', 'm12', 'm03']
                #
                # diff_moments = []
                # for m_key in used_m_keys:
                #     obj_m = obj_moment[m_key]
                #     temp_m = temp_moment[m_key]
                #
                #     diff = abs((temp_m - obj_m) / temp_m)
                #     diff_moments.append(diff)
                #     print('moment: {0}; obj_val = {1}; temp_val = {2}; diff = {3};'.format(m_key, obj_m, temp_m, diff))
                #
                #
                # contour_diff = sum(diff_moments) / len(diff_moments)
                # print('contour_diff = {0};'.format(contour_diff))
                # print('---------------------------------------------------------------------------')
                #
                # # cv2.imshow("Object", templ)
                # cv2.imshow("Object contours", obj_cnt_image)
                # cv2.waitKey()

                    # print('obj_m = {0}; obj_m = {1}; temp_m = {1}'.format(m_key, obj_m, temp_m))

                    # diff = abs((1 / obj_m) - (1 / temp_m))
                    # diff_moment = diff_moment + diff

                    # if (obj_m > 0 and obj_m <= 1) and (temp_m > 0 and temp_m <= 1):
                    #     m_a = np.sign(obj_m) * math.log(obj_m)
                    #     m_b = np.sign(temp_m) * math.log(temp_m)
                    #     diff = abs((1/m_a) - (1/m_b))
                    #     diff_moment = diff_moment + diff
                    #     print('diff = {0};'.format(diff))

                # if diff_moment < min_moment:
                #     min_moment = diff_moment
                #     best_obj = obj
                #
                # diff_moments.append(diff_moment)
                # errPtCounts.append(errPtCount)

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

            # plt.plot(errPtCounts, label=name)
            # plt.plot(diff_moments, label='diff_moments')

            # cv2.imshow("best by moments", best_obj)
            # cv2.waitKey()


            # plt.figure(1)
            # plt.subplot(211)
            # plt.plot(errPtCounts, label=name)
            #
            # plt.subplot(212)
            # plt.plot(diff_moments, label='diff_moments')
            # plt.show()

            plt.show()

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

        # создание новой матрицы с результатом
        # resultMatrix =  [[''] * height for i in range(width)]

        # # Наполнение значенями из сторой матрицы
        # rowIndex = 0
        # for i in range(0, height):
        #     conf_row = confMatrix[i]
        #     moment_row = contMomentMatrix[i]
        #     cellIndex = 0
        #
        #     for j in range(0, width):
        #         conf_cell = conf_row[j]
        #         moment_cell = moment_row[j]
        #         resultMatrix[rowIndex, cellIndex] = 'test'
        #         cellIndex = cellIndex + 1
        #     rowIndex = rowIndex +100


        # Создаем архивы для хранения имен столбцов и колонок
        rows_names = []
        for i in range(0, height):
            rows_names.append('Object %s' % i)

        cell_names = []
        for i in range(0, width):
            cell_names.append('Temaple %s' % i)

        # row_names.append('Total')
        # cell_names.append('Total')

        # Печать значений
        row_format = "{:>22}" * (width + 1)
        print(row_format.format("", *cell_names, ''))

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
        confMatrix = self._confMatrix

        result = []
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

            res_val = (c_ki_val * (c_ik_val - c_kk_val)) / (c_ik_val + total) ** 2 * 1000
            result.append(res_val)

            index = index + 1

        return sum(result)

    def getTemplateComut(self):
        return self._templ_len

    def getSegmentsComut(self):
        return self._segm_len


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



