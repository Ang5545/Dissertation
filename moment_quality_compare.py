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

    diff_moment = get_moments_diff(moments_1, moments_2)
    print('diff_moment = {0}'.format(diff_moment))



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


def get_moments_diff(moment1, moment2):
    # используются только пространствунный моменты
    used_moments = ['m00', 'm10', 'm01', 'm20', 'm11', 'm02', 'm30', 'm21', 'm12', 'm03']

    diff_moments = []
    for m_key in used_moments:
        obj_m = moment1[m_key]
        temp_m = moment2[m_key]

        max_m = max(obj_m, temp_m)
        min_m = min(obj_m, temp_m)

        # # diff = abs((1 / temp_m) - (1 / obj_m))  # CV_CONTOURS_MATCH_I1
        # # diff = abs(temp_m - obj_m)              # CV_CONTOURS_MATCH_I2
        diff = abs((max_m - min_m) / max_m)         # CV_CONTOURS_MATCH_I3
        diff_moments.append(diff)

    contour_diff = sum(diff_moments) / len(diff_moments)
    result = contour_diff ** (1 / 4)
    return result






project_dir = iml.getParamFromConfig('projectdir')

# pearTempl = project_dir + '/resources/pears/template.bmp'
all_white_path = project_dir + '/resources/paint_tests_2/big_point.bmp'
one_point_path = project_dir + '/resources/paint_tests_2/small_point.bmp'


all_white = cv2.imread(all_white_path, 3)
one_point = cv2.imread(one_point_path, 3)

cv2.imshow("Test1", all_white)
cv2.imshow("Test2", one_point)
cv2.waitKey()

compare_two_image(one_point, all_white)

