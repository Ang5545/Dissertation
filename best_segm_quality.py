import imgUtils.ImgLoader as iml
from segmentationQuality.yasnoff import Yasnoff
from segmentationQuality.yasnoff_modification import Yasnoff_Modification
from segmentationQuality.yasnoff_mod_2 import Yasnoff_Mod
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from segmentation.threshold.threshold import Threshold
import cv2



def testSrm(templatePath, srmREsultDir):
    template = cv2.imread(templatePath, 3)
    images = iml.getNamedImages(srmREsultDir)

    m1s = []
    m2s = []
    m3s = []
    frags = []
    results = []
    tcs = []
    scs = []

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
        m3 = yasn.get_m3()
        print('m3 = {0}'.format(m3))

        frags.append([srm_val, frag])
        m1s.append([srm_val, m1])
        m2s.append([srm_val, m2])
        m3s.append([srm_val, m3])
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
    plt.plot(*zip(*m3s), label="m3")
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

        yasn = Yasnoff_Mod(template, image)
        m3 = yasn.get_m3()

        print('m3 = {0};'.format(m3))
        m3s.append([srm_val, m3])


    plt.plot(*zip(*m3s), label="m2")


    # cv2.imshow("Best Result Image", bestImg)
    # cv2.imshow("Temlate", template)
    # cv2.waitKey()

    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0.)
    plt.show()

    print('------------------')



def testSrmOneImage(img_path, template_path):

    img = cv2.imread(img_path, 3)
    template = cv2.imread(template_path, 3)

    # cv2.imshow('img', img)
    # cv2.imshow('template', template)
    # cv2.waitKey()

    yasn = Yasnoff_Mod(template, img)
    yasn.printMatrix()

    print(' -- getIncorrecClassPixels --')
    m1 = yasn.getIncorrecClassPixels()


    print(' -- getWronglyAssigneToClass --')
    m2 = yasn.getWronglyAssigneToClass()
    frag = yasn.getFrags()

    m3 = yasn.get_m3()

    print('m1 = {0}; m2 = {1}; frag = {2}; m3 = {3};'.format(m1, m2, frag, m3))





# --------------- MAIN ---------------

project_dir = iml.getParamFromConfig('projectdir')

# tempPath = project_dir + "/resources/applePears/1/template.png"
# imgPath = project_dir + "/resources/applePears/1/segmented/java/"

# imgPath = project_dir + "/resources/pears/segmented/java/"
# tempPath = project_dir + '/resources/pears/template.bmp'
#
# testSrm(tempPath, imgPath)


# imgPath = project_dir + "/resources/pears/segmented/java/val_13_0.png"
# tempPath = project_dir + '/resources/pears/template.bmp'
#
# testSrmOneImage(imgPath, tempPath)


imgPath = project_dir + "/resources/pears/segmented/java/"
tempPath = project_dir + '/resources/pears/template.bmp'

testSrmMod(tempPath, imgPath)





'''
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

'''


