import cv2
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




# --------------- MAIN ---------------
print(' - start work - ')

project_dir = iml.getParamFromConfig('projectdir')

pear_templ = project_dir + '/resources/pears/template.bmp'
pear_segm_dir = project_dir + '/resources/pears/segmented/java/'

apple_pear_templ = project_dir + '/resources/applePears/1/template.png'
apple_pear_segm_dir = project_dir + '/resources/applePears/1/segmented/java/'

srm_yasnoff_one_img(apple_pear_segm_dir + 'val_20_0.png', apple_pear_templ)
# srm_yasnoff_chart(apple_pear_segm_dir, apple_pear_templ)

# srm_yasnoff_mom_one_img(apple_pear_segm_dir + 'val_20_0.png', apple_pear_templ)
# srm_yasnoff_mom_chart(pear_segm_dir, pear_templ)


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

