import math

import numpy as np

from segmentation.srm import UnionFind


class SRM:

    def __init__(self, image, Q=32.0):
        self._img = image

        height = image.shape[0]
        width = image.shape[1]

        self._height = height
        self._width = width

        n = image.shape[0] * image.shape[1]
        self._n = n

        if image.ndim == 3:
            self._depth = image.shape[2]
        else:
            self._depth = 1

        self._q = Q
        self._smallregion = (int)(0.001 * n) # small regions are less than 0.1 % of image pixels


        self._logdelta =  2 * math.log(6 * n)
        self._uf = UnionFind(n)

        self._pixels = image.reshape((n, image.shape[2])).astype('int16')

        self._g = 256 #color chanels count


    def run(self):

        rmpairs = self._getRmPairs()
        sortRmPairs = self.__bucketSort(rmpairs)

        result = self._segmentation(sortRmPairs)


        # result = self._getImage(segmentedPixels)

        return result

    def _getRmPairs(self):
        rmPairs = []

        height = self._height
        width = self._width
        img = self._img
        n = self._n
        pixels = self._pixels

        i = 0

        # -- using a C4-connectivity --

        # for all pixels except max height and max width border lines
        for i in range(height - 1):
            for j in range(width - 1):
                idx = i * width + j
                pt = pixels[idx]

                pt_left = pixels[idx + 1]
                diff_left = self._getMaxColorDiff(pt, pt_left)
                rmPairs.append(Rmpair(idx, idx + 1, diff_left))

                # below
                pt_bellow = pixels[idx + width]
                diff_bellow = self._getMaxColorDiff(pt, pt_bellow)
                rmPairs.append(Rmpair(idx, idx + width, diff_bellow))

        # for max height border line
        for i in range(width - 1):
            idx = ((height-1) * width) + i
            pt = pixels[idx]

            pt_left = pixels[idx + 1]
            diff_left = self._getMaxColorDiff(pt, pt_left)
            rmPairs.append(Rmpair(idx, idx + 1, diff_left))


        # for max width border line
        for i in range(height - 2):
            idx = (width - 1) + (i * width)
            pt = pixels[idx]

            pt_bellow = pixels[idx + width]
            diff_bellow = self._getMaxColorDiff(pt, pt_bellow)
            rmPairs.append(Rmpair(idx, idx + width, diff_bellow))

        return rmPairs

    def _getMaxColorDiff(slef, pt_1, pt_2):
        b_1 = pt_1[0]
        g_1 = pt_1[1]
        r_1 = pt_1[2]

        b_2 = pt_2[0]
        g_2 = pt_2[1]
        r_2 = pt_2[2]

        max_diff = max(abs(b_1 - b_2), abs(g_1 - g_2), abs(r_1 - r_2))
        return max_diff


    def _segmentation(self, rmpairs):
        pixels = self._pixels
        n = self._n
        height = self._height
        width = self._width

        uf = UnionFind(len(rmpairs))
        nn = [1] * n

        for rmpair in rmpairs:
            reg_1 = rmpair.get_r1()
            reg_2 = rmpair.get_r2()

            c_1 = uf.find(reg_1)
            c_2 = uf.find(reg_2)

            if (c_1 != c_2) and self._mergePredicate(c_1, c_2, nn[c_1], nn[c_2]):
                reg = uf.unionRoot(c_1, c_2)
                nreg = nn[c_1] + nn[c_2]

                pt_1 = pixels[c_1]
                pt_2 = pixels[c_2]

                b_avg = (nn[c_1] * pt_1[0] + nn[c_2] * pt_2[0]) / nreg
                g_avg = (nn[c_1] * pt_1[1] + nn[c_2] * pt_2[1]) / nreg
                r_avg = (nn[c_1] * pt_1[2] + nn[c_2] * pt_2[2]) / nreg

                nn[reg] = nreg
                pixels[reg] = [b_avg, g_avg, r_avg]


        # output result
        img = np.zeros((height, width, 3), np.uint8)

        for i in range(height):
            for j in range(width):
                idx = i * width + j
                indexb = uf.find(idx) # Get the root index

                pt = pixels[indexb]
                img[i, j] = pt

        return img


    def __bucketSort(self, rmpairs, bucketSize=256):

        def default_compare(a, b):
            if a < b:
                return -1
            elif a > b:
                return 1
            return 0

        def rmpair_compare(a, b):
            a_diff = a.get_diff()
            b_diff = b.get_diff()

            if a_diff < b_diff:
                return -1
            elif a_diff > b_diff:
                return 1
            return 0

        def sort(array, compare=default_compare):
            for i in range(1, len(array)):
                item = array[i]
                indexHole = i
                while indexHole > 0 and compare(array[indexHole - 1], item) > 0:
                    array[indexHole] = array[indexHole - 1]
                    indexHole -= 1
                array[indexHole] = item
            return array

        if len(rmpairs) == 0:
            return rmpairs

        sortRmPairs = [Rmpair()] * len(rmpairs)
        nbe = [0] * 255

        # Determine minimum and maximum values
        minValue = rmpairs[0].get_diff()
        maxValue = rmpairs[0].get_diff()

        for i in range(1, len(rmpairs)):
            diff = rmpairs[i].get_diff()
            if diff < minValue:
                minValue = diff
            elif diff > maxValue:
                maxValue = diff

        # Initialize buckets
        bucketCount = math.floor((maxValue - minValue) / bucketSize) + 1
        buckets = []
        for i in range(0, bucketCount):
            buckets.append([])

        # Distribute input array values into buckets
        for i in range(0, len(rmpairs)):
            diff = rmpairs[i].get_diff()
            buckets[math.floor((diff - minValue) / bucketSize)].append(rmpairs[i])

        # Sort buckets and place back into input array
        array = []
        for i in range(0, len(buckets)):
            sort(buckets[i], rmpair_compare)
            for j in range(0, len(buckets[i])):
                array.append(buckets[i][j])

        return array


    def _mergePredicate(self, c_1, c_2, nn_c_1, nn_c_2):
        pixels = self._pixels
        logdelta = self._logdelta
        g = self._g
        n = self._n
        q = self._q

        pt_1 = pixels[c_1]
        pt_2 = pixels[c_2]

        dB = (pt_1[0] - pt_2[0]) ** 2
        dG = (pt_1[1] - pt_2[1]) ** 2
        dR = (pt_1[2] - pt_2[2]) ** 2

        logreg_1 = min(g, nn_c_1) * math.log(1 + nn_c_1)
        logreg_2 = min(g, nn_c_2) * math.log(1 + nn_c_2)

        dev_1 = ((g * g) / (2 * q * nn_c_1)) * (logreg_1 + logdelta)
        dev_2 = ((g * g) / (2 * q * nn_c_2)) * (logreg_2 + logdelta)

        dev = dev_1 + dev_2
        result = ((dB < dev) and (dG < dev) and (dR < dev))

        return result



class Rmpair:

    # An edge: two indices and a difference value

    def __init__(self, r1=0, r2=0, diff=0):
        # print('create new Rmpair: r1 = {0}; r2 = {1}; diff = {2}'.format(r1, r2, diff))

        self._r1 = r1
        self._r2 = r2
        self._diff = diff

    def get_r1(self):
        return self._r1

    def get_r2(self):
        return self._r2

    def get_diff(self):
        return self._diff

    def set_r1(self, r1):
        self._r1 = r1

    def set_r2(self, r2):
        self._r2 = r2

    def set_diff(self, diff):
        self._diff = diff



