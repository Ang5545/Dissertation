import cv2
import imgUtils.ImgLoader as iml
import numpy as np
from imgUtils import ColorMask as colMaks


def find_cnt(img):
    if img.ndim == 3:  # если изображение не одноканальное
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # преобразуем в оттенки серого


    _, contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    return cnt

def showCnt(cnt, winNmae, height, width):
    cnt_img = np.zeros((height, width, 3), np.uint8)
    cv2.drawContours(cnt_img, [cnt], -1, (0, 255, 0), 1)
    cv2.imshow(winNmae, cnt_img)
    
    
project_dir = iml.getParamFromConfig('projectdir')
pearTempl = project_dir + '/resources/pears/template.bmp'
pearAllWhite = project_dir + '/resources/pears/all_white.bmp'

templImage = cv2.imread(pearTempl, 3)
templates = colMaks.getMaskFromColors(templImage)

all_white = cv2.imread(pearAllWhite, 0)

height = all_white.shape[0]
width = all_white.shape[1]



for template in templates:
    cv2.imshow("Template", template)

    templ_cnt = find_cnt(template)
    img_snt = find_cnt(all_white)

    showCnt(templ_cnt, 'Tempolate cnt', height, width)
    showCnt(img_snt, 'Img cnt', height, width)

    templ_moment = cv2.moments(templ_cnt)
    segm_moment = cv2.moments(img_snt)

    # используются только пространствунный моменты
    used_m_keys = ['m00', 'm10', 'm01', 'm20', 'm11', 'm02', 'm30', 'm21', 'm12', 'm03']

    temp_m_summ = 0
    obj_m_summ = 0
    for m_key in used_m_keys:
        temp_m = templ_moment[m_key]
        obj_m = segm_moment[m_key]

        temp_m_summ = temp_m_summ + temp_m
        obj_m_summ = obj_m_summ + obj_m

    print('temp_m_summ = {0};'.format(temp_m_summ))
    print('obj_m_summ = {0};'.format(obj_m_summ))

        # diff = abs((temp_m - obj_m) / temp_m)  # CV_CONTOURS_MATCH_I3
        #
        # m_diffs.append(diff)
        # print('diff = {0};'.format(diff))




    cv2.waitKey()



#
#
# templ_path = project_dir + '/resources/paint_test/template.png'
# img_dir_path = project_dir + '/resources/paint_test/segmented/'
#
#
# templ = cv2.imread(templ_path, 0)
# images = iml.getNamedImages(img_dir_path)
#
# height = templ.shape[0]
# width = templ.shape[1]
#
#
# for imgObj in images:
#     name = imgObj[0]
#     img = imgObj[1]
#
#     print(' --- name = {0} --- '.format(name))
#
#     templ_cnt = find_cnt(templ)
#     img_snt = find_cnt(img)
#
#
#     showCnt(templ_cnt, 'Tempolate cnt', height, width)
#     showCnt(img_snt, 'Img cnt', height, width)
#
#     templ_moment = cv2.moments(templ_cnt)
#     segm_moment = cv2.moments(img_snt)
#
#     # используются только пространствунный моменты
#     used_m_keys = ['m00', 'm10', 'm01', 'm20', 'm11', 'm02', 'm30', 'm21', 'm12', 'm03']
#
#     m_diffs = []
#     for m_key in used_m_keys:
#         temp_m = templ_moment[m_key]
#         obj_m = segm_moment[m_key]
#         diff = abs((temp_m - obj_m) / temp_m)  # CV_CONTOURS_MATCH_I3
#
#         m_diffs.append(diff)
#         print('diff = {0};'.format(diff))
#
#
#     cont_diff = sum(m_diffs)
#     print('cont_diff = {0};'.format(cont_diff))
#
#     cv2.waitKey()








