import cv2
import matplotlib.pyplot as plt
import imgUtils.ImgLoader as iml
from segmentationQuality.yasnoff import Yasnoff
from segmentationQuality.yasnoff_moments import Yasnoff_Moments




def testSrm(templatePath, srmREsultDir):
    template = cv2.imread(templatePath, 3)
    images = iml.getNamedImages(srmREsultDir)

    m1s = []
    m2s = []
    frags = []
    results = []


    minRes = 100
    bestImg = images[0]

    def sortByVal(inp):
        name = inp[0]
        return float(name[4:len(name)].replace('_', '.'))

    images.sort(key=sortByVal)

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

        frags.append([srm_val, frag])
        m1s.append([srm_val, m1])
        m2s.append([srm_val, m2])
        results.append([srm_val, res])

        # tc = yasn.getTemplateComut()
        # sc = yasn.getSegmentsComut()
        #
        # tcs.append([srm_val, tc])
        # scs.append([srm_val, sc])

        if res < minRes:
            minRes = res
            bestImg = image


    plt.plot(*zip(*m1s), label="m1")
    plt.plot(*zip(*m2s), label="m2")
    plt.plot(*zip(*frags), label="frag")
    plt.plot(*zip(*results), label="result")

    cv2.imshow("Best Result Image", bestImg)
    cv2.imshow("Temlate", template)
    cv2.waitKey()

    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0.)
    plt.show()

    print('------------------')

def testSrmMod(templatePath, srmREsultDir):
    template = cv2.imread(templatePath, 3)
    images = iml.getNamedImages(srmREsultDir)

    m1s = []
    m2s = []
    frags = []
    results = []

    def sortByVal(inp):
        name = inp[0]
        return float(name[4:len(name)].replace('_', '.'))

    images.sort(key=sortByVal)

    for img in images:
        print('name = {0};'.format(img[0]))
        name = img[0]
        srm_val = float(name[4:len(name)].replace('_', '.'))
        image = img[1]

        yasn = Yasnoff_Moments(template, image)
        # yasn.printMatrix()
        # m3 = yasn.get_m3()

        m1 = yasn.getIncorrecClassPixels()
        m2 = yasn.getWronglyAssigneToClass()
        frag = yasn.getFrags()
        res = (m1 + m2 + frag) / 3

        m1s.append([srm_val, m1])
        m2s.append([srm_val, m2])
        frags.append([srm_val, frag])
        results.append([srm_val, res])

        # print('m1 = {0}; m2 = {1}; frag = {2}; result = {3};'.format(m1, m2, frag, res))
        print('m1 = {0}; m2 = {1}; frag = {2}; result = {3};'.format(m1, m2, frag, res))


    plt.plot(*zip(*m1s), label="m1")
    plt.plot(*zip(*m2s), label="m2")
    plt.plot(*zip(*frags), label="frags")
    plt.plot(*zip(*results), label="result")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0.)
    plt.show()

    print('------------------')


def testSrmMod_m4_plot(templatePath, srmREsultDir):
    template = cv2.imread(templatePath, 3)
    images = iml.getNamedImages(srmREsultDir)

    m3s = []

    def sortByVal(inp):
        name = inp[0]
        return float(name[4:len(name)].replace('_', '.'))

    images.sort(key=sortByVal)

    for img in images:
        print('name = {0};'.format(img[0]))
        name = img[0]
        srm_val = float(name[4:len(name)].replace('_', '.'))
        image = img[1]

        yasn = Yasnoff_Moments(template, image)
        m3 = yasn.get_m3()
        m3s.append([srm_val, m3])

        print('!m3 = {0};'.format(m3))


    plt.plot(*zip(*m3s), label="m3s")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0.)
    plt.show()

    print('------------------')


def testSrmOneImage(img_path, template_path):

    img = cv2.imread(img_path, 3)
    template = cv2.imread(template_path, 3)

    # cv2.imshow('img', img)
    # cv2.imshow('template', template)
    # cv2.waitKey()

    yasn = Yasnoff_Moments(template, img)
    yasn.printMatrix()
    print('---------------------------------------')
    yasn.printMatrixWithTotal()
    print('---------------------------------------')

    # m1 = yasn.getIncorrecClassPixels()
    # m2 = yasn.getWronglyAssigneToClass()
    # frag = yasn.getFrags()
    m3 = yasn.get_m3()
    print('in the end m3 = {0}'.format(m3))
    # print('m1 = {0}; m2 = {1}; frag = {2}; m4 = {3}'.format(m1, m2, frag, m4))



def testThreshold():
    # test Threshold
    template = cv2.imread(project_dir + '/resources/heart/sampleMask.bmp')
    img = cv2.imread(project_dir + '/resources/heart/img.JPG')

    cv2.imshow("template", template)
    cv2.imshow("img", img)

    # print('start')


    # th = 0
    # step = 10
    # m1s = []
    # m2s = []
    # frags= []
    #
    # while (th < 255):
    #     th = th + step
    #     threshold = Threshold(img, th)
    #     segm = threshold.run()
    #
    #     yasn = Yasnoff(template, segm)
    #
    #     m1 = yasn.getIncorrecClassPixels()
    #     m2 = yasn.getWronglyAssigneToClass()
    #     frag = yasn.getFrags()
    #
    #     # frags.append(frag)
    #     m1s.append(m1)
    #     m2s.append(m2)
    #
    #     print('th = {0}; m1 = {1}; m2 = {2}; frag = {3};'.format(th, m1, m2, frags))
    #
    #     # cv2.imshow("Threshold", segm)
    #     # cv2.waitKey()
    #
    # plt.plot(m1s)
    # plt.plot(m2s)
    # plt.plot(frags)
    #
    # plt.ylabel('some numbers')
    # plt.show()

    # th = 190
    # threshold = Threshold(img, th)
    # segm = threshold.run()
    #
    # yasn = Yasnoff(template, segm)
    # m1 = yasn.getIncorrecClassPixels()
    # m2 = yasn.getWronglyAssigneToClass()
    # frag = yasn.getFrags()
    #
    # yasn.printConfMatrix()
    # print('m1 = {0}; m2 = {1}; frag = {2};'.format(m1, m2, frag))
    #
    # print('end')


# --------------- MAIN ---------------
print(' - start work - ')

project_dir = iml.getParamFromConfig('projectdir')

pearTempl = project_dir + '/resources/pears/template.bmp'
pearSegmDir = project_dir + '/resources/pears/segmented/java/'

applePearTempl = project_dir + '/resources/applePears/1/template.png'
applePearSegmDir = project_dir + '/resources/applePears/1/segmented/java/'


# tempPath = project_dir + "/resources/applePears/1/template.png"
# imgPath = project_dir + "/resources/applePears/1/segmented/java/"

# imgPath = project_dir + "/resources/pears/segmented/java/"
# tempPath = project_dir + '/resources/pears/template.bmp'
#
# testSrm(tempPath, imgPath)

# testSrmOneImage(applePearSegmDir + 'val_3_0.png', applePearTempl)

# testSrmMod(pearTempl, pearSegmDir)

testSrmMod_m4_plot(pearTempl, pearSegmDir)


print(' - end - ')

