import cv2
import numpy as np

imgPath = '/home/ange/Desktop/2222.jpg'
img = cv2.imread(imgPath, 3)

cv2.imshow("img", img)
cv2.waitKey()



# -- segmentation --
key = 0
step = 1
th = 45

grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thres = cv2.threshold(grayimg, th, 255, 0)


while (key != 10):
    key = cv2.waitKey()
    neddProcess = False

    if (key == 82):
        th = th + 1
        neddProcess = True
        print(th)

    elif (key == 84):
        th = th - 1
        neddProcess = True
        print(th)

    if (neddProcess) :
        grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thres = cv2.threshold(grayimg, th, 255, 0)
        cv2.imshow("thres", thres)



im2, contours, hierarchy = cv2.findContours(thres, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
i = 0
for cnt in contours :
    cntImg = np.zeros(img.shape, np.uint8)
    cv2.drawContours(cntImg, [cnt], -1, (255, 255, 255), 1)

    cv2.imshow("cnt %s" %i, cntImg)
    cv2.waitKey()
    i = i + 1


cv2.waitKey()
