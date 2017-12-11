import cv2
import imgUtils.ImgLoader as iml


def test_one_th(path):
    img = cv2.imread(path, 0)

    th = 100
    key = 10
    step = 1

    while key != 27:

        if key == 1 and th < 255:
            th = th + step
        if key == 0 and th > 0:
            th = th - step

        _, thresh = cv2.threshold(img, th, 255, cv2.THRESH_BINARY)
        cv2.imshow('thresh', thresh)
        key = cv2.waitKey()
        print('th = {0}'.format(th))


# --------------- MAIN ---------------
print(' - start work - ')

project_dir = iml.getParamFromConfig('projectdir')

path = project_dir + '/resources/one_green_apple/original.bmp'
img = cv2.imread(path, 0)
cv2.imshow('img', img)







step = 10
th_1 = 0
th_2 = 0
th_3 = 0
results = []

while th_1 < 255:
    th_2 = th_1
    while th_2 < 255:
        th_3 = th_2
        while th_3 < 255:
            _, thresh_1 = cv2.threshold(img, th_1, 255, cv2.THRESH_BINARY)
            _, thresh_2 = cv2.threshold(img, th_2, 100, cv2.THRESH_BINARY)
            _, thresh_3 = cv2.threshold(img, th_3, 150, cv2.THRESH_BINARY)


            result = cv2.bitwise_or(thresh_1, thresh_2)
            result = cv2.bitwise_or(result, thresh_3)

            results.append(result)
            cv2.imshow('result', result)
            cv2.imshow('thresh_1', thresh_1)
            cv2.imshow('thresh_2', thresh_2)
            cv2.imshow('thresh_3', thresh_3)
            key = cv2.waitKey()
            print('th_1 = {0}; th_2 = {1}; th_3 = {2}'.format(th_1, th_2, th_3))

            th_3 = th_3 + step

        th_2 = th_2 + step

    th_1 = th_1 + step




print(' - end - ')