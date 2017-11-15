import imgUtils.ImgLoader as iml
from segmentationQuality.yasnoff import Yasnoff
import matplotlib.pyplot as plt
from segmentation.threshold.threshold import Threshold
import cv2


project_dir = iml.getParamFromConfig('projectdir')
path = project_dir + "/resources/applePears/1/segmented/java/"

# test SRM

template = cv2.imread(project_dir + "/resources/applePears/1/template.png", 3)
images = iml.getImages(path)

quals = []
i = 0
m1s = []
m2s = []
frags= []


for img in images:
    yasn = Yasnoff(template, img)

    m1 = yasn.getIncorrecClassPixels()
    m2 = yasn.getWronglyAssigneToClass()
    frag = yasn.getFrags()

    frags.append(frag)
    m1s.append(m1)
    m2s.append(m2)

plt.plot(m1s)
plt.plot(m2s)
# plt.plot(frags)

plt.ylabel('some numbers')
plt.show()

print('------------------')



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