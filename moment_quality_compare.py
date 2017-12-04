import cv2
# from sympy.solvers.tests.test_diophantine import m3

import imgUtils.ImgLoader as iml
import numpy as np
from imgUtils import ColorMask as colMaks


def find_cnt(img):
    if img.ndim == 3:  # если изображение не одноканальное
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # преобразуем в оттенки серого

    _, contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print('len(contours) = {0}'.format(len(contours)))
    cnt = contours[len(contours)-1]
    return cnt


def showCnt(cnt, winNmae, height, width):
    cnt_img = np.zeros((height, width, 3), np.uint8)
    cv2.drawContours(cnt_img, [cnt], -1, (0, 255, 0), 1)
    cv2.imshow(winNmae, cnt_img)


def count_moment_diff(m_1, m_2):
    used_moments = ['m00', 'm10', 'm01', 'm20', 'm11', 'm02', 'm30', 'm21', 'm12', 'm03']

    diff_moments = []
    for m_key in used_moments:
        tt_m = m_1[m_key]
        ss_m = m_2[m_key]

        diff = abs((tt_m - ss_m) / tt_m)
        diff_moments.append(diff)

    contour_diff = sum(diff_moments) / len(diff_moments)
    dss = contour_diff ** (1 / 4)

    return dss

def count_moment_diff_invar(m_1, m_2):
    used_moments = ['m00', 'm10', 'm01', 'm20', 'm11', 'm02', 'm30', 'm21', 'm12', 'm03']

    diff_moments = []
    for m_key in used_moments:
        tt_m = m_1[m_key]
        ss_m = m_2[m_key]
        max_m = max(tt_m, ss_m)
        min_m = min(tt_m, ss_m)

        diff = abs((max_m - min_m) / max_m)
        diff_moments.append(diff)

    contour_diff = sum(diff_moments) / len(diff_moments)
    dss = contour_diff ** (1 / 4)

    return dss

def count_moment_diff_invar_mm(m_1, m_2):
    used_moments = ['m00', 'm10', 'm01', 'm20', 'm11', 'm02', 'm30', 'm21', 'm12', 'm03']

    diff_moments = []
    for m_key in used_moments:
        tt_m = m_1[m_key]
        ss_m = m_2[m_key]
        max_m = max(tt_m, ss_m)
        min_m = min(tt_m, ss_m)

        diff = abs((min_m - max_m) / min_m)
        diff_moments.append(diff)

    contour_diff = sum(diff_moments) / len(diff_moments)
    dss = contour_diff ** (1 / 4)

    return dss

def get_moment(img):
    height = img.shape[0]
    width = img.shape[1]

    if img.ndim == 3:  # если изображение не одноканальное
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # преобразуем в оттенки серого

    _, contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    moments = cv2.moments(cnt)

    cnt_img_1 = np.zeros((height, width, 3), np.uint8)
    cv2.drawContours(cnt_img_1, [cnt], -1, (0, 255, 0), 1)
    cv2.imshow("cnt", cnt_img_1)

    return moments

def compare_two_image(img1, img2):

    height = img1.shape[0]
    width = img1.shape[1]

    cnt_1 = find_cnt(img1)
    cnt_2 = find_cnt(img2)

    showCnt(cnt_1, "cnt_1", height, width)
    showCnt(cnt_2, "cnt_2", height, width)
    cv2.waitKey()

    moments_1 = cv2.moments(cnt_1)
    moments_2 = cv2.moments(cnt_2)

    used_moments = ['m00', 'm10', 'm01', 'm20', 'm11', 'm02', 'm30', 'm21', 'm12', 'm03']
    diff_moments = []
    for m_key in used_moments:
        obj_m = moments_1[m_key]
        temp_m = moments_2[m_key]

        max_m = max(obj_m, temp_m)
        min_m = min(obj_m, temp_m)

        diff = abs((max_m - min_m) / max_m)
        diff_moments.append(diff)

    contour_diff = sum(diff_moments) / len(diff_moments)
    dss = contour_diff ** (1 / 20)


    print('dss = {0}'.format(dss))

    # cv2.waitKey()
    #
    # diff_1 = count_moment_diff(moments_1, moments_2)
    # diff_2 = count_moment_diff(moments_2, moments_1)
    # diff_3 = count_moment_diff_invar(moments_2, moments_1)
    # diff_4 = count_moment_diff_invar_mm(moments_1, moments_2)
    #
    # print('diff_1 = {0};'.format(diff_1))
    # print('diff_2 = {0};'.format(diff_2))
    # print('diff_3 = {0};'.format(diff_3))
    # print('diff_4 = {0};'.format(diff_4))

    # cnt_img_1 = np.zeros((height, width, 3), np.uint8)
    # cnt_img_2 = np.zeros((height, width, 3), np.uint8)
    #
    # cv2.drawContours(cnt_img_1, [cnt_1], -1, (0, 255, 0), 1)
    # cv2.drawContours(cnt_img_2, [cnt_2], -1, (0, 255, 0), 1)
    #
    # cv2.imshow("cnt_img_1", cnt_img_1)
    # cv2.imshow("cnt_img_2", cnt_img_2)
    # cv2.waitKey()


def templ_segm(templ_path, segm_path):

    templ = cv2.imread(templ_path, 3)
    segm = cv2.imread(segm_path, 3)

    height = templ.shape[0]
    width = templ.shape[1]

    templObjs = colMaks.getMaskFromColors(templ)
    segmObjs = colMaks.getMaskFromColors(segm)

    for tt in templObjs:

        # tt_moment = get_moment(tt)

        for ss in segmObjs:

            # ss_moment = get_moment(ss)
            # diff = count_moment_diff(tt_moment, ss_moment)

            compare_two_image(tt, ss)

            # print('---------------------------------------------------')
            # print('m1 = {0}'.format(tt_moment))
            # print('m2 = {0}'.format(ss_moment))
            # print('---------------------------------------------------')

            # print('diff = {0};'.format(diff))









project_dir = iml.getParamFromConfig('projectdir')

# pearTempl = project_dir + '/resources/pears/template.bmp'
all_white_path = project_dir + '/resources/pears/all_white11.png'
one_point_path = project_dir + '/resources/pears/one_point_left.bmp'


all_white = cv2.imread(all_white_path, 3)
one_point = cv2.imread(one_point_path, 3)

cv2.imshow("Test1", all_white)
cv2.imshow("Test2", one_point)
cv2.waitKey()

compare_two_image(one_point, all_white)



# templ_path = project_dir + '/resources/pears/template.bmp'
# segm_path = project_dir + '/resources/pears/segmented/java/val_18_0.png'
#
# templ_segm(templ_path, segm_path)



# # test
#
# m_tt_1 = {'nu02': 0.5659659744145829, 'm21': 38205543403.11667, 'mu11': -105355.31956982613, 'nu30': 0.008404948721255528, 'mu03': 6176749.094060898, 'nu12': 0.04687637071197581, 'm12': 20949151667.916668, 'm20': 183014664.5833333, 'nu03': 0.1191780091705673, 'mu12': 2429505.0936256647, 'mu30': 435611.064163208, 'nu20': 0.043321704893562754, 'nu21': -0.02121514421612119, 'm00': 1218.5, 'mu20': 64321.56559750438, 'm30': 70965548981.5, 'm01': 254921.0, 'm11': 98672470.04166666, 'm10': 472149.3333333333, 'm02': 54172046.25, 'mu02': 840313.5942757502, 'nu11': -0.07095865936988466, 'mu21': -1099536.934114689, 'm03': 11691051074.6}
# m_tt_2 = {'nu02': 0.02126851438435427, 'm21': 10511989602001.916, 'mu11': 465286059.4685974, 'nu30': 0.14635301598315822, 'mu03': -8947539369.029297, 'nu12': -0.010298650021192823, 'm12': 10695937365777.516, 'm20': 18761648743.583332, 'nu03': -0.0033090139010955775, 'mu12': -27847443155.78975, 'mu30': 395737041736.791, 'nu20': 0.6907219765239421, 'nu21': -0.0026014940417585395, 'm00': 93929.5, 'mu20': 6094067988.549606, 'm30': 11761627550649.55, 'm01': 50952195.5, 'm11': 19176791824.958332, 'm10': 34494340.5, 'm02': 27826740235.25, 'mu02': 187646805.92033768, 'nu11': 0.0527370727154745, 'mu21': -7034412986.062988, 'm03': 15289287222602.25}
#
# m_ss_1 = {'mu12': -27847443155.78975, 'nu12': -0.010298650021192823, 'mu11': 465286059.4685974, 'm30': 11761627550649.55, 'mu02': 187646805.92033768, 'm01': 50952195.5, 'm20': 18761648743.583332, 'm00': 93929.5, 'mu20': 6094067988.549606, 'nu11': 0.0527370727154745, 'm02': 27826740235.25, 'mu03': -8947539369.029297, 'm11': 19176791824.958332, 'nu02': 0.02126851438435427, 'mu21': -7034412986.062988, 'nu20': 0.6907219765239421, 'nu21': -0.0026014940417585395, 'nu03': -0.0033090139010955775, 'm21': 10511989602001.916, 'm10': 34494340.5, 'nu30': 0.14635301598315822, 'm12': 10695937365777.516, 'mu30': 395737041736.791, 'm03': 15289287222602.25}
# m_ss_2 = {'mu12': 2429505.0936256647, 'nu12': 0.04687637071197581, 'mu11': -105355.31956982613, 'm30': 70965548981.5, 'mu02': 840313.5942757502, 'm01': 254921.0, 'm20': 183014664.5833333, 'm00': 1218.5, 'mu20': 64321.56559750438, 'nu11': -0.07095865936988466, 'm02': 54172046.25, 'mu03': 6176749.094060898, 'm11': 98672470.04166666, 'nu02': 0.5659659744145829, 'mu21': -1099536.934114689, 'nu20': 0.043321704893562754, 'nu21': -0.02121514421612119, 'nu03': 0.1191780091705673, 'm21': 38205543403.11667, 'm10': 472149.3333333333, 'nu30': 0.008404948721255528, 'm12': 20949151667.916668, 'mu30': 435611.064163208, 'm03': 11691051074.6}
#
# m_3_1 = {}
# m_3_2 = {}
#
# for key in m_tt_1.keys():
#     val_1 = m_tt_1[key]
#     val_2 = m_ss_1[key]
#     m_3_1[key] = abs(val_1 - val_2)
#
# for key in m_tt_1.keys():
#     val_1 = m_tt_2[key]
#     val_2 = m_ss_2[key]
#     m_3_2[key] = abs(val_1 - val_2)
#
# for key in m_3_1:
#     print('-- m_3_1.{0} = {1}'.format(key, m_3_1[key]))
#
# # for key in m_3_2:
# #     print('-- m_3_2.{0} = {1}'.format(key, m_3_2[key]))
#
