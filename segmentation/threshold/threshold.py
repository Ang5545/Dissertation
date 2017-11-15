import cv2

class Threshold:

    def __init__(self, img, th):
        self._th = th
        self._img = img


    def run(self):
        th = self._th
        img = self._img

        grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thres = cv2.threshold(grayimg, th, 255, 0)
        return thres

