import cv2
import numpy as np

from utils import ImgLoader as loader

img = loader.getImages('imgdirpath')[0]
sampleMask = loader.getImages('segmentsdirpath')[0]

ret, th = cv2.threshold(img, 50, 255, 0)

# key = 0
# thVal = 0
#
# while (key != 27): # esc/enter
#     key = cv2.waitKey()
#
#     if (key == 82):
#         # up
#         thVal = thVal + 10
#
#     elif (key == 84):
#         # down
#         thVal = thVal - 10
#
#     ret, th = cv2.threshold(img, thVal, 255, cv2.THRESH_BINARY)
#     cv2.imshow("fresult", th)

def minBTrackCallBack(pos):
    global minB
    minB = pos
    colorSelect()

def maxBTrackCallBack(pos):
    global maxB
    maxB = pos
    colorSelect()

def minGTrackCallBack(pos):
    global minG
    minG = pos
    colorSelect()

def maxGTrackCallBack(pos):
    global maxG
    maxG = pos
    colorSelect()

def minRTrackCallBack(pos):
    global minR
    minR = pos
    colorSelect()

def maxRTrackCallBack(pos):
    global maxR
    maxR = pos
    colorSelect()

def colorSelect():
    global minB, minG, minR, maxB, maxG, maxR, mask
    lower_blue = np.array([minB, minG, minR])
    upper_blue = np.array([maxB, maxG, maxR])

    mask = cv2.inRange(img, lower_blue, upper_blue)
    res = cv2.bitwise_and(sampleMask, sampleMask, mask=mask)

    cv2.imshow('res', res)
    cv2.imshow('mask', mask)



minB = 0
minG = 0
minR = 0

maxB = 0
maxG = 0
maxR = 0

mask = img.copy()

colorSelect()
cv2.imshow('frame', img)

cv2.createTrackbar("Blue min", "frame", minB, 255, minBTrackCallBack);
cv2.createTrackbar("Blue max", "frame", maxB, 255, maxBTrackCallBack);

cv2.createTrackbar("Green min", "frame", minG, 255, minGTrackCallBack);
cv2.createTrackbar("Green max", "frame", maxG, 255, maxGTrackCallBack);

cv2.createTrackbar("Red min", "frame", minR, 255, minRTrackCallBack);
cv2.createTrackbar("Red max", "frame", maxR, 255, maxRTrackCallBack);

key = 0
while (key != 27):
    key = cv2.waitKey()
    # if (key == ord('r')):
    #     cv2.imwrite("/home/ange/Desktop/mask.bmp", mask)

print('------------------')