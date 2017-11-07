from matplotlib import pyplot
from scipy.misc import imread
import cv2

from StatisticalRegionMerging.statistical_region_merging.srm import SRM

imgPath = "/home/ange/Dropbox/Учеба/Опыты/Примеры сегментации/original.png"
img = cv2.imread(imgPath, 3)


srm = SRM(img, 32)
segmented = srm.run()
# cv2.imshow("segmented", segmented / 256)


# i = 1
# step = 3
# while i <= 30:
#     val = i * step
#     print(' i = %s' % i)
#     print(' val = %s' % val)
#     srm = SRM(img, val)
#     segmented = srm.run()
#     cv2.imwrite("/home/ange/Dropbox/Учеба/Опыты/Примеры сегментации/segmented/apple-pear5/" +str(val) +".png", segmented)
#     cv2.imshow("segmented", segmented / 256)
#     i = i + 1
#     print('------------------------------')

cv2.waitKey()



