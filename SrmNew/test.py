from SrmNew.srm import SRM
import cv2


imgPath = '/home/ange/Python/workplace/Dissertation/resources/applePears/1/original.png'
img = cv2.imread(imgPath, 3)

# cv2.imshow("Test", img)
# cv2.waitKey()


srm = SRM(img)
srm.run()

