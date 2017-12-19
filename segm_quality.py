import cv2
import numpy as np
import matplotlib.pyplot as plt
import imgUtils.ImgLoader as iml
from segmentationQuality.yasnoff import Yasnoff
from segmentationQuality.yasnoff_moments import YasnoffMoments


def sort_by_name(array):
    def sortByVal(inp):
        name = inp[0]
        return float(name[4:len(name)].replace('_', '.'))
    array.sort(key=sortByVal)


def getSortImages(img_dir_path):
    images = iml.getNamedImages(img_dir_path)
    sort_by_name(images)
    return images



def yasnoff_one_img(img, template):
    yasn = Yasnoff(template, img, True)

    print('---------------------------------------------------------------------------------------')
    yasn.printMatrixWithTotal()

    print('---------------------------------------------------------------------------------------')
    yasn.printPixelDistError()

    print('---------------------------------------------------------------------------------------')
    m1 = yasn.getIncorrecClassPixels()
    m2 = yasn.getWronglyAssigneToClass()
    distErr = yasn.getPixelDistError()

    frag = yasn.getFrags()
    print('m1  = {0}; m2 = {1}; frag = {2}; distErr = {3}'.format(m1, m2, frag, distErr))

    cv2.imshow('img', img)
    cv2.imshow('template', template)
    cv2.waitKey()



def yasnoff_chart(images, template, limit = -1):
    m1s = []
    m2s = []
    frags = []
    results = []
    minRes = 5000
    bestImg = images[0][1]
    best_idx = 0

    i = 0
    while (limit == -1 and i < len(images)) or (limit > 0 and i <= limit):
        img = images[i]
        name = img[0]
        image = img[1]

        yasn = Yasnoff(template, image)
        m1 = yasn.getIncorrecClassPixels()
        m2 = yasn.getWronglyAssigneToClass()
        frag = yasn.getFrags()
        res = (m1 + m2 + frag) / 3

        m1s.append(m1)
        m2s.append(m2)
        frags.append(frag)
        results.append(res)

        if res < minRes:
            minRes = res
            bestImg = image
            best_idx = i

        print('name = {0};'.format(name))
        print('m1  = {0}; m2 = {1}; frag = {2}; res = {3}'.format(m1, m2, frag, res))

        i = i + 1

    plt.plot(m1s, label="m1")
    plt.plot(m2s, label="m2")
    plt.plot(frags, label="frag")
    plt.plot(results, label="result")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0.)
    plt.show()

    print('best_idx = {0};'.format(best_idx))
    cv2.imshow("Best Result Image", bestImg)
    cv2.imshow("Temlate", template)
    cv2.waitKey()



def yasnoff_mom_one_img(img, template):
    yasn = YasnoffMoments(template, img)
    yasn.printMatrix()

    print('---------------------------------------------------------------------------------------')
    m3 = yasn.get_m3()
    print('m3  = {0};'.format(m3))

    cv2.imshow('img', img)
    cv2.imshow('template', template)
    cv2.waitKey()



def yasnoff_mom_chart(images, template, limit = -1):

    m3s = []
    minRes = 5
    bestImg = images[0]

    i = 1
    while i < len(images) and (limit == -1 or i <= limit):
        img = images[i]
        name = img[0]
        image = img[1]

        yasn = YasnoffMoments(template, image)
        m3 = yasn.get_m3()
        m3s.append(m3)

        if m3 < minRes:
            minRes = m3
            bestImg = image

        print('name = {0};'.format(name))
        print('m3  = {0};'.format(m3))

        i = i + 1

    plt.plot(m3s, label="m3")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0.)
    plt.show()

    cv2.imshow("Best Result Image", bestImg)
    cv2.imshow("Temlate", template)
    cv2.waitKey()


def compare_yasnoff_charts(images, template, limit = -1, frg = True):
    m1s = []
    m2s = []
    frags = []
    pxDistErrs = []
    m3s = []

    i = 1
    while i < len(images) and (limit == -1 or i <= limit):
        img = images[i]
        name = img[0]
        image = img[1]

        yasn = Yasnoff(template, image, True)
        m1 = yasn.getIncorrecClassPixels()
        m2 = yasn.getWronglyAssigneToClass()
        pxDistErr = yasn.getPixelDistError()
        frag = yasn.getFrags()

        m1s.append(m1)
        m2s.append(m2)
        pxDistErrs.append(pxDistErr)
        frags.append(frag)

        yasn_m = YasnoffMoments(template, image)
        m3 = yasn_m.get_m3()
        m3s.append(m3)

        print('name = {0};'.format(name))
        print('m1 = {0}; m2 = {1}; m3 = {2}; frag = {3};'.format(m1, m2, m3, frag))
        print('-----------------------')

        i = i + 1

    # нормализация вектров
    def normalize(v):
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm

    m1s_norm = normalize(m1s)
    m2s_norm = normalize(m2s)
    pxDistErr_norm = normalize(pxDistErrs)
    frag_norm = normalize(frags)

    results = []
    for m1, m2, frag, pxDistErr in zip(m1s_norm, m2s_norm, frag_norm, pxDistErr_norm):
        if frg:
            res =  (m1 + m2 + frag + pxDistErr) / 4
            results.append(res)
        else:
            res = (m1 + m2 + pxDistErr) / 3
            results.append(res)

    m3s_norm = normalize(m3s)

    plt.figure(1)
    plt.subplot(211)
    plt.plot(m1s_norm, label="m1")
    plt.plot(m2s_norm, label="m2")
    plt.plot(pxDistErr_norm, label="pxDistErr")

    if frg:
        plt.plot(frag_norm, label="frags")

    plt.legend(loc="upper right")

    plt.subplot(212)
    plt.plot(results, label="yasnoff")
    plt.plot(m3s_norm, label="yasnoff moments")
    plt.legend(loc="upper right")

    plt.show()

    best_yasn_idx = np.argmin(results)
    best_mom_idx = np.argmin(m3s_norm)

    print('best_yasn_idx = {0}'.format(best_yasn_idx))
    print('best_mom_idx = {0}'.format(best_mom_idx))


    best_yasn_img = images[best_yasn_idx][1]
    best_mom_img = images[best_mom_idx][1]

    best_yasn_name = images[best_yasn_idx][0]
    best_mom_name = images[best_mom_idx][0]

    print('best_yasn = {0}; best_mom = {1};'.format(results[best_yasn_idx], m3s_norm[best_mom_idx]))
    print('best_yasn img = {0}; best_mom img = {1};'.format(best_yasn_name, best_mom_name))

    cv2.imshow("best_yasn", best_yasn_img)
    cv2.imshow("best_mom", best_mom_img)
    cv2.waitKey()



def get_thresholded(img, step = 10, init_th = 240, init_th_lower = 0, limit = -1):
    results = []
    th_upper = init_th
    i = 0

    while (limit == -1 and th_upper <= 255) or (limit >  0 and i < limit):
        th_lower = init_th_lower

        while (limit == -1 and th_lower <= th_upper) or (limit > 0 and i < limit):

            name = 'i = {2}: th_upper = {0}; th_lower = {1}'.format(th_upper, th_lower, i)
            print(name)

            _, upper_threshold = cv2.threshold(img, th_upper, 255, cv2.THRESH_BINARY)
            _, lower_threshold = cv2.threshold(img, th_lower, 150, cv2.THRESH_BINARY)

            res_img = lower_threshold
            white_px = np.argwhere(upper_threshold == 255)
            for px in white_px:
                y = px[0]
                x = px[1]
                res_img[y, x] = 255

            result = []
            result.append(name)
            result.append(res_img)

            results.append(result)

            i = i + 1
            th_lower = th_lower + step
        th_upper = th_upper + step

    return results



# --------------- MAIN ---------------
print(' - start work - ')

project_dir = iml.getParamFromConfig('projectdir')



# =====================
# ======== SRM ========
# =====================

'''
# -- all experiment data paths --
pear_templ = project_dir + '/resources/pears/template.bmp'
pear_segm_dir = project_dir + '/resources/pears/segmented/java/'

apple_pear_templ = project_dir + '/resources/applePears/1/template.png'
apple_pear_segm_dir = project_dir + '/resources/applePears/1/segmented/java/'

# -- used paths --
template_path = apple_pear_templ
segm_dir_path = apple_pear_segm_dir

template = cv2.imread(template_path, 3)
images = getSortImages(segm_dir_path)


# -- Only yasnoff --
# yasnoff_one_img(images[0][1], template)
yasnoff_chart(images, template, 30)


# -- Only moments --
# yasnoff_mom_one_img(images[20][1], template)
# yasnoff_mom_chart(images, template, 5)


#  -- Compare  moments --
# compare_yasnoff_charts(images, template, 50)
'''



# ===== TRESHOLD =====

# -- all experiment data paths --
lime_templ = project_dir + '/resources/lime/template.png'
lime_img = project_dir + '/resources/lime/new_color_2.png'


# -- used paths --
template_path = lime_templ
img_path = lime_img

template = cv2.imread(template_path, 3)
img = cv2.imread(lime_img, 0)

# threses = get_thresholded(img)
threses = get_thresholded(img, 10, 240, 0, -1)


print('----------------------------------------')


# -- Only yasnoff --
# yasnoff_one_img(threses[0][1], template)
# yasnoff_chart(threses, template)


# -- Only moments --
# yasnoff_mom_one_img(images[20][1], template)
yasnoff_mom_chart(threses, template, -1)



#  -- Compare  moments --
# compare_yasnoff_charts(threses, template, -1, False)




'''
# TODO тестировать на другом изображении / проверить почему не работает на минимальном th
heart_img = project_dir + '/resources/heart/img.JPG'
heart_templ = project_dir + '/resources/heart/sampleMask.bmp'

# threshold_yasnoff_one_img(heart_img, heart_templ, 190)
# threshold_yasnoff_chart(heart_img, heart_templ)

# threshold_yasnoff_mom_one_img(heart_img, heart_templ, 190)
# threshold_yasnoff_mom_chart(heart_img, heart_templ)
'''






print(' - end - ')