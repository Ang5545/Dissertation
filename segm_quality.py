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



def srm_yasnoff_one_img(img_path, template_path):
    img = cv2.imread(img_path, 3)
    template = cv2.imread(template_path, 3)

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



def srm_yasnoff_chart(img_dir_path, template_path):
    template = cv2.imread(template_path, 3)
    images = iml.getNamedImages(img_dir_path)
    sort_by_name(images)

    m1s = []
    m2s = []
    frags = []
    results = []
    minRes = 100
    bestImg = images[0]

    for img in images:
        print('name = {0};'.format(img[0]))
        name = img[0]
        srm_val = float(name[4:len(name)].replace('_', '.'))
        image = img[1]

        yasn = Yasnoff(template, image)
        m1 = yasn.getIncorrecClassPixels()
        m2 = yasn.getWronglyAssigneToClass()
        frag = yasn.getFrags()
        res = (m1 + m2 + frag) / 3

        m1s.append([srm_val, m1])
        m2s.append([srm_val, m2])
        frags.append([srm_val, frag])
        results.append([srm_val, res])

        print('m1  = {0}; m2 = {1}; frag = {2}; res = {3}'.format(m1, m2, frag, res))

        if res < minRes:
            minRes = res
            bestImg = image

    plt.plot(*zip(*m1s), label="m1")
    plt.plot(*zip(*m2s), label="m2")
    plt.plot(*zip(*frags), label="frag")
    plt.plot(*zip(*results), label="result")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0.)
    plt.show()

    cv2.imshow("Best Result Image", bestImg)
    cv2.imshow("Temlate", template)
    cv2.waitKey()



def srm_yasnoff_mom_one_img(img_path, template_path):
    img = cv2.imread(img_path, 3)
    template = cv2.imread(template_path, 3)

    yasn = YasnoffMoments(template, img)
    yasn.printMatrix()

    print('---------------------------------------------------------------------------------------')
    m3 = yasn.get_m3()
    print('m3  = {0};'.format(m3))

    cv2.imshow('img', img)
    cv2.imshow('template', template)
    cv2.waitKey()



def srm_yasnoff_mom_chart(img_dir_path, template_path):
    template = cv2.imread(template_path, 3)
    images = iml.getNamedImages(img_dir_path)
    sort_by_name(images)

    m3s = []
    minRes = 5
    bestImg = images[0]

    for img in images:
        print('name = {0};'.format(img[0]))
        name = img[0]
        srm_val = float(name[4:len(name)].replace('_', '.'))
        image = img[1]

        yasn = YasnoffMoments(template, image)
        m3 = yasn.get_m3()
        m3s.append([srm_val, m3])

        print('m3  = {0};'.format(m3))

        if m3 < minRes:
            minRes = m3
            bestImg = image

    plt.plot(*zip(*m3s), label="m3")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0.)
    plt.show()

    cv2.imshow("Best Result Image", bestImg)
    cv2.imshow("Temlate", template)
    cv2.waitKey()



def threshold_yasnoff_one_img(img_path, template_path, th=200):
    template = cv2.imread(template_path, 3)
    img = cv2.imread(img_path, 3)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(img, th, 255, cv2.THRESH_BINARY)

    yasn = Yasnoff(template, thresh)
    yasn.printMatrixWithTotal()

    print('---------------------------------------------------------------------------------------')
    m1 = yasn.getIncorrecClassPixels()
    m2 = yasn.getWronglyAssigneToClass()
    frag = yasn.getFrags()
    print('m1  = {0}; m2 = {1}; frag = {2};'.format(m1, m2, frag))

    cv2.imshow('thresh', thresh)
    cv2.imshow('template', template)
    cv2.waitKey()



def threshold_yasnoff_chart(img_path, template_path):
    template = cv2.imread(template_path, 3)
    img = cv2.imread(img_path, 3)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    step = 10
    th = 40

    bestImg = img
    m1s = []
    m2s = []
    frags = []
    results = []
    minRes = 100

    while (th < 255):
        _, thresh = cv2.threshold(img, th, 255, cv2.THRESH_BINARY)
        yasn = YasnoffMoments(template, thresh)
        m1 = yasn.getIncorrecClassPixels()
        m2 = yasn.getWronglyAssigneToClass()
        frag = yasn.getFrags()
        res = (m1 + m2 + frag) / 3

        m1s.append([th, m1])
        m2s.append([th, m2])
        frags.append([th, frag])
        results.append([th, res])

        print('m1  = {0}; m2 = {1}; frag = {2}; res = {3}'.format(m1, m2, frag, res))

        if res < minRes:
            minRes = res
            bestImg = thresh

        th = th + step

    plt.plot(*zip(*m1s), label="m1")
    plt.plot(*zip(*m2s), label="m2")
    plt.plot(*zip(*frags), label="frag")
    plt.plot(*zip(*results), label="result")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0.)
    plt.show()

    print('best result  = {0};'.format(minRes))

    cv2.imshow("Best Result Image", bestImg)
    cv2.imshow("Temlate", template)
    cv2.waitKey()



def threshold_yasnoff_mom_one_img(img_path, template_path, th=200):
    template = cv2.imread(template_path, 3)
    img = cv2.imread(img_path, 3)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(img, th, 255, cv2.THRESH_BINARY)

    yasn = YasnoffMoments(template, thresh)
    yasn.printMatrix()

    print('---------------------------------------------------------------------------------------')
    m3 = yasn.get_m3()
    print('m3  = {0};'.format(m3))

    cv2.imshow('thresh', thresh)
    cv2.imshow('template', template)
    cv2.waitKey()



def threshold_yasnoff_mom_chart(img_path, template_path):
    template = cv2.imread(template_path, 3)
    img = cv2.imread(img_path, 3)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    step = 10
    th = 40

    bestImg = img
    minRes = 5
    m3s = []

    while (th < 255):
        _, thresh = cv2.threshold(img, th, 255, cv2.THRESH_BINARY)
        yasn = YasnoffMoments(template, thresh)
        m3 = yasn.get_m3()
        m3s.append([th, m3])

        print('m3  = {0};'.format(m3))

        if m3 < minRes:
            minRes = m3
            bestImg = thresh

        th = th + step

    plt.plot(*zip(*m3s), label="m3")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0.)
    plt.show()

    cv2.imshow("Best Result Image", bestImg)
    cv2.imshow("Temlate", template)
    cv2.waitKey()



def srm_compare_yasnoff_charts(img_dir_path, template_path):
    template = cv2.imread(template_path, 3)
    images = iml.getNamedImages(img_dir_path)
    sort_by_name(images)

    m1s = []
    m2s = []
    frags = []
    pxDistErrs = []
    m3s = []

    for img in images:
    # for i in range(0, 30):
    #     img = images[i]

        name = img[0]
        print('name = {0};'.format(name))

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

        print('m1 = {0}; m2 = {1}; m3 = {2}; frag = {3};'.format(m1, m2, m3, frag))

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
        res =  (m1 + frag + pxDistErr) / 3
        results.append(res)

    m3s_norm = normalize(m3s)

    plt.figure(1)
    plt.subplot(211)
    plt.plot(m1s_norm, label="m1")
    plt.plot(m2s_norm, label="m2")
    plt.plot(frag_norm, label="frags")
    plt.plot(pxDistErr_norm, label="pxDistErr")

    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)

    plt.subplot(212)
    plt.plot(results, label="result")
    plt.plot(m3s_norm, label="m3s")

    plt.show()

    best_yasn_idx = np.argmin(results)
    best_mom_idx = np.argmin(m3s_norm)

    best_yasn_img = images[best_yasn_idx][1]
    best_mom_img = images[best_mom_idx][1]

    best_yasn_name = images[best_yasn_idx][0]
    best_mom_name = images[best_mom_idx][0]

    print('best_yasn = {0}; best_mom = {1};'.format(results[best_yasn_idx], m3s_norm[best_mom_idx]))
    print('best_yasn img = {0}; best_mom img = {1};'.format(best_yasn_name, best_mom_name))

    cv2.imshow("best_yasn", best_yasn_img)
    cv2.imshow("best_mom", best_mom_img)
    cv2.waitKey()




# --------------- MAIN ---------------
print(' - start work - ')

project_dir = iml.getParamFromConfig('projectdir')

pear_templ = project_dir + '/resources/pears/template.bmp'
pear_segm_dir = project_dir + '/resources/pears/segmented/java/'

apple_pear_templ = project_dir + '/resources/applePears/1/template.png'
apple_pear_segm_dir = project_dir + '/resources/applePears/1/segmented/java/'


lime_templ = project_dir + '/resources/lime/template.png'
lime_segm_dir = project_dir + '/resources/lime/segmented/'




# Only yasnoff



# Only moments



# Compare  moments

# srm_yasnoff_one_img(lime_segm_dir + 'thres_303.png', lime_templ)
# srm_yasnoff_chart(apple_pear_segm_dir, apple_pear_templ)

# srm_yasnoff_mom_one_img(apple_pear_segm_dir + 'val_20_0.png', apple_pear_templ)
# srm_yasnoff_mom_chart(pear_segm_dir, pear_templ)


srm_compare_yasnoff_charts(lime_segm_dir, lime_templ)

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

