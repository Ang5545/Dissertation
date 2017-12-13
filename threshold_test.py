import cv2
import imgUtils.ImgLoader as iml
import numpy as np
import matplotlib.pyplot as plt
from segmentationQuality.yasnoff import Yasnoff
from segmentationQuality.yasnoff_moments import YasnoffMoments



def test_one_th(path):
    img = cv2.imread(path, 0)

    th = 100
    key = 10
    step = 1

    while key != 27:

        if key == 1 and th < 255:
            th = th + step
        if key == 0 and th > 0:
            th = th - step

        _, thresh = cv2.threshold(img, th, 255, cv2.THRESH_BINARY)
        cv2.imshow('thresh', thresh)
        key = cv2.waitKey()
        print('th = {0}'.format(th))


def get_thresholded(img, step = 10, init_th = 0):
    th = init_th
    results = []
    while th < 255:
        _, thres = cv2.threshold(img, th, 255, cv2.THRESH_BINARY)
        results.append(thres)
        th = th + step
    return results


def get_two_thresholded(img):
    height = img.shape[0]
    width = img.shape[1]

    step = 10
    th_1 = 100
    th_2 = 100
    results = []

    while th_1 < 250:
        th_2 = th_1
        while th_2 < 250:

            print('th_1 = {0}; th_2 = {1};'.format(th_1, th_2))

            _, main_img = cv2.threshold(img, th_1, 255, cv2.THRESH_BINARY)
            _, thresh = cv2.threshold(img, th_2, 255, cv2.THRESH_BINARY_INV)

            object = cv2.bitwise_and(main_img, thresh)
            _, object_gray = cv2.threshold(object, 250, 150, cv2.THRESH_BINARY)

            result = np.zeros([height, width, 1], dtype=np.uint8)
            for y in range(0, height):
                for x in range(0, width):
                    main_val = main_img[y, x]
                    if main_val == 255:
                        obj_val = object[y, x]
                        if obj_val == 255:
                            result[y, x] = 100
                        else:
                            result[y, x] = 0
                    else:
                        result[y, x] = 200

            results.append(result)
            th_2 = th_2 + step

        th_1 = th_1 + step

    return results


def get_range_thresholded(img):
    step = 10
    th_1 = 1

    # create first
    while th_1 <= 255:
        th_2 = 1
        while th_2 <= th_1:
            lower = np.array(th_2)
            upper = np.array(th_1)

            mask = cv2.inRange(img, lower, upper)
            th_2 = th_2 + step

        th_1 = th_1 + step






# --------------- MAIN ---------------
print(' - start work - ')

project_dir = iml.getParamFromConfig('projectdir')

img_path = project_dir + '/resources/orange/original.png'
temp_path = project_dir + '/resources/orange/template.png'
img = cv2.imread(img_path, 0)
template = cv2.imread(temp_path, 3)


# threses = get_range_thresholded(img) # get_two_thresholded(img)

'''
_, thres = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)

yasn_m = YasnoffMoments(template, thres)
yasn_m.printMatrixWithTotal()
yasn_m.printMatrix()
m3 = yasn_m.get_m3()

print('m3 = {0}'.format(m3))
'''





threses = get_thresholded(img, 1) # get_two_thresholded(img)

m1s = []
m2s = []
m3s = []
pxDistErrs = []

best_yasn_res = 255
best_yasn_thres = threses[0]
best_yasn_idx = 0

best_yasn_m_res = 255
best_yasn_m_thres = threses[0]
best_yasn_m_idx = 0

for idx in range(0, len(threses)):
    print('idx = {0}'.format(idx))

    thres = threses[idx]

    yasn = Yasnoff(template, thres, True)
    m1 = yasn.getIncorrecClassPixels()
    m2 = yasn.getWronglyAssigneToClass()
    pxDistErr = yasn.getPixelDistError()
    frag = yasn.getFrags()

    # res = (m1 + m2 + frag) / 3
    res = (m1 + m2 + pxDistErr) / 3
    m1s.append(m1)
    m2s.append(m2)
    pxDistErrs.append(pxDistErr)


    if pxDistErr < best_yasn_res:
        best_yasn_res = pxDistErr
        best_yasn_thres = thres
        best_yasn_idx = idx

    yasn_m = YasnoffMoments(template, thres)
    m3 = yasn_m.get_m3()
    m3s.append(m3)

    if m3 < best_yasn_m_res:
        best_yasn_m_res = m3
        best_yasn_m_thres = thres
        best_yasn_m_idx = idx

    print('-------------------------------------------')

print('best_yasn_res = {0}; best_yasn_m_res = {1}'.format(best_yasn_res,best_yasn_m_res ))
print('best_yasn_idx = {0}; best_yasn_m_idx = {1}'.format(best_yasn_idx, best_yasn_m_idx))


# нормализация вектров

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm

m1s_norm = normalize(m1s)
m2s_norm = normalize(m2s)
pxDistErrs_norm = normalize(pxDistErrs)

results = []
for m1, pxDistErr in zip(m1s_norm, pxDistErrs_norm):
    val = ((m1 + pxDistErr) / 2)
    results.append(val)


m3s_norm = normalize(m3s)

plt.figure(1)
plt.subplot(211)
plt.plot(m1s_norm, label="m1")
plt.plot(m2s_norm, label="m2")
plt.plot(pxDistErrs_norm, label="pxDistErr")
plt.plot(results, label="result")

plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)

plt.subplot(212)
plt.plot(m3s_norm, label="m1s")



plt.show()

cv2.imshow("best_yasn_thres", best_yasn_thres)
cv2.imshow("best_yasn_m_thres", best_yasn_m_thres)
cv2.waitKey()


print(' - end - ')
