# from SrmNew.srm import SRM
from SrmNew.srm2 import SRM
import cv2


imgPath = '/home/ange/Desktop/cv_experiments/segmentation_test_50x50.png'
img = cv2.imread(imgPath, 3)

# cv2.imshow("Test", img)
# cv2.waitKey()


srm = SRM(img)
segmented = srm.run()

cv2.imshow("Origin", img)
cv2.imshow("Result", segmented)
cv2.waitKey()


