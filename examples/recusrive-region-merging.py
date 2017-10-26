import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt

sys.setrecursionlimit(15000)

original_image = cv2.imread('Peppers.jpg', 0)
original_image_boundaries = cv2.imread('Peppers.jpg', 0)
height, width = original_image.shape;

R = np.zeros((height, width), dtype='int32')

neighbors = []
label_mean = []
visited = []


def main():
    label = 1
    for i in range(0, height):
        for j in range(0, width):
            if (R[i][j] == 0):
                recursive_label(i, j, label, original_image[i][j])
                label = label + 1;

    label = label - 1
    print
    'Value of labels >> '
    print
    label

    # calculating neighbours of each pixel
    for x in range(1, label + 1):
        neighbors.append([])
        r_pixel = np.where(R == x)
        for y in range(0, len(r_pixel[0])):
            i = r_pixel[0][y]
            j = r_pixel[1][y]
            if (i > 0):
                if (R[i - 1][j] != R[i][j] and (R[i - 1][j] not in neighbors[x - 1])):
                    neighbors[x - 1].append(R[i - 1][j])
            if (i < 511):
                if (R[i + 1][j] != R[i][j] and (R[i + 1][j] not in neighbors[x - 1])):
                    neighbors[x - 1].append(R[i + 1][j])
            if (j < 511):
                if (R[i][j + 1] != R[i][j] and (R[i][j + 1] not in neighbors[x - 1])):
                    neighbors[x - 1].append(R[i][j + 1])
            if (j > 0):
                if (R[i][j - 1] != R[i][j] and (R[i][j - 1] not in neighbors[x - 1])):
                    neighbors[x - 1].append(R[i][j - 1])

    print
    'Done with calculating neighbors! >> stored in neighbors=[][]'
    print
    'Length of neighbors list >> '
    print
    len(neighbors)

    # getting mean values for each label
    label_list = []
    for i in range(1, label + 1):
        r_pixel = np.where(R == i)
        for j in range(0, len(r_pixel[0])):
            x = r_pixel[0][j]
            y = r_pixel[1][j]
            label_list.append(original_image[x][y])
        label_mean.append(sum(label_list) / len(label_list))
        del label_list[:]

    print
    'Done with calculating mean of all Labels! >> stored in label_mean=[]'
    print
    'Length of label_mean list >> '
    print
    len(label_mean)

    for i in range(1, label + 1):
        if (i not in visited):
            recursive_merge(i, i)

    print
    R[height - 1][width - 1]

    for i in range(1, R[height - 1][width - 1] + 1):
        r_pixel = np.where(R == i)
        for j in range(0, len(r_pixel[0])):
            x = r_pixel[0][j]
            y = r_pixel[1][j]
            if (x > 0):
                if (R[x - 1][y] != R[x][y]):
                    original_image_boundaries[x][y] = 511
            if (x < 511):
                if (R[x + 1][y] != R[x][y]):
                    original_image_boundaries[x][y] = 511
            if (y < 511):
                if (R[x][y + 1] != R[x][y]):
                    original_image_boundaries[x][y] = 511
            if (y > 0):
                if (R[x][y - 1] != R[x][y]):
                    original_image_boundaries[x][y] = 511


def recursive_label(i, j, label, intensity):
    if (i < 0 or j < 0 or i > 511 or j > 511 or R[i][j] != 0 or np.absolute(original_image[i][j] - intensity) > 20):
        return
    R[i][j] = label
    recursive_label(i, j + 1, label, original_image[i][j])
    recursive_label(i, j - 1, label, original_image[i][j])
    recursive_label(i - 1, j, label, original_image[i][j])
    recursive_label(i + 1, j, label, original_image[i][j])


def recursive_merge(i, label_value):
    if (i in visited):
        return
    visited.append(i)
    for neighbor_pixel in neighbors[i - 1]:
        if (np.absolute(label_mean[label_value - 1] - label_mean[neighbor_pixel - 1]) < 20):
            replace_pixel = np.where(R == neighbor_pixel)
            for k in range(0, len(replace_pixel[0])):
                x = replace_pixel[0][k]
                y = replace_pixel[1][k]
                R[x][y] = label_value
            recursive_merge(neighbor_pixel, label_value)


main()

plt.figure('Original Image')
plt.imshow(original_image, cmap="Greys_r")
plt.show()

R = R % 255
plt.figure('Merged Image')
plt.imshow(R, cmap="Greys_r")
plt.show()

plt.figure('Original Image with boundaries')
plt.imshow(original_image_boundaries, cmap="Greys_r")
plt.show()

print("tets")