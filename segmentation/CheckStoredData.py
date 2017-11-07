import cv2
from segmentation import Yasnoff as yasn

templatePath = '/home/ange/Python/workplace/Dissertation/resources/applePears/1/template.png'
segmPath = '/home/ange/Python/workplace/Dissertation/resources/applePears/1/segmented/4_0_2.png'


template = cv2.imread(templatePath, 3)
segm = cv2.imread(segmPath, 3)

cv2.imshow("template", template)
cv2.imshow("segm", segm)
cv2.waitKey()


qual = yasn.getQuality(template, segm)
print('qual = {0}'.format(qual))


print('end')
cv2.waitKey()