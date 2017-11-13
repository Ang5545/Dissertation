import imgUtils.ImgLoader as iml
from segmentationQuality.yasnoff import Yasnoff

import cv2


project_dir = iml.getParamFromConfig('projectdir')
path = project_dir + "/resources/applePears/1/segmented/java/"

template = cv2.imread(project_dir + "/resources/applePears/1/template.png", 3)
images = iml.getImages(path)

print(project_dir + "/resources/applePears/1/template.png")



yqual = Yasnoff(template, images[0])
yqual.printConfMatrix()

# for img in images:
#     yqual = Yasnoff(template, img)

#     yqual.printConfMatrix()

    # print("qual = {0}".format(yqual.getQuality()))

