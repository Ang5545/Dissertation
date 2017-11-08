import numpy as np
import math
from SrmNew.unionfind import UnionFind

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


        self._logdelta =  2.0 * math.log(6.0 * n)
        self._uf = UnionFind(n)

        # -- init auxiliary buffers for union-find operations --
        blue_avg = np.arange(n, dtype=np.int16)
        green_avg = np.arange(n, dtype=np.int16)
        red_avg = np.arange(n, dtype=np.int16)
        nn = [0] * n
        class_numbers = [0] * n

        for y in range(0, height):
            for x in range(0, width):
                index = y * width + x;
                pt = self._img[y][x]

                blue_avg[index] = pt[0]
                green_avg[index] = pt[1]
                red_avg[index] = pt[2]
                nn[index] = 1
                class_numbers[index] = index

        print('red_avg')
        print(red_avg)
        print('------')

        self._blue_avg = blue_avg
        self._green_avg = green_avg
        self._red_avg = red_avg
        self._nn = nn
        self._class_numbers = class_numbers

        self._mergePredicateCount = 0

    def run(self):

        n = self._n
        height = self._height
        width = self._width
        depth = self._depth
        uf = self._uf

        # Consider C4 - connectivity here
        npair = 2 * (width - 1) * (height - 1) + (height - 1) + (width - 1)
        orders = [[0] * 3 for i in range(npair)]

        img = self._img
        # self._data = np.empty([n, depth + 2])
        # self._sizes = np.ones(n)


        # pixels = self._img.reshape((self._n, self._img.shape[2]))
        pixels = img.reshape((n, img.shape[2])).astype('int16')
        pairs = []

        for i in range(height - 1):
            for j in range(width - 1):
                idx = i * width + j

                pt = pixels[idx]

                # -- C4 left --
                pt_left = pixels[idx+1]
                diff_left = self._getMaxColorDiff(pt, pt_left)

                order_left = [idx, idx + 1, diff_left]
                pairs.append(order_left)


                # -- C4 bellow --
                pt_bellow = pixels[idx + width]

                diff_bellow = self._getMaxColorDiff(pt, pt_bellow)
                order_bellow = [idx, idx + width, diff_bellow]
                pairs.append(order_bellow)


        # The two border lines

        for i in range(height - 1):
            idx = i * width + width - 1
            pt = pixels[idx]

            pt_bellow = pixels[idx + width]
            diff = self._getMaxColorDiff(pt, pt_bellow)

            order_left = [idx, idx + width, diff]
            pairs.append(order_left)

        for i in range(width - 1):
            idx = (height - 1) * width + j;
            pt = pixels[idx]

            pt_left = pixels[idx + 1]
            diff = self._getMaxColorDiff(pt, pt_left)

            order_left = [idx, idx + 1, diff]
            pairs.append(order_left)

        sortPairs = self._bucketSort(pairs)


        # -- Main segmentation algorithm --

        for pair in sortPairs:

            req_1 = pair[0]
            req_2 = pair[1]
            c_1 = self._uf.find(req_1)
            c_2 = self._uf.find(req_2)

            print('c_1 = {0}; c_2 = {1};'.format(c_1, c_2))

            # print('c_1 = {0}; c_2 = {1}; (c_1!=c_2) = {2}'.format(c_1, c_2, (c_1!=c_2)))

            if c_1 != c_2 and self._mergePredicate(req_1, req_2):
                self._mergeRegions(c_1, c_2)

            if self._mergePredicate(req_1, req_2):
                self._mergePredicateCount = self._mergePredicateCount + 1

        print('self._mergePredicateCount = {0}'.format(self._mergePredicateCount))

        # # -- Merdge small regions --
        # nn = self._nn
        # smallregion = self._smallregion
        #
        # for i in range(height - 1):
        #     for j in range(width - 1):
        #         index = i * width + j;
        #         c_1 = uf.find(index)
        #         c_2 = uf.find(index-1)
        #
        #         if c_2 != c_1:
        #             if (nn[c_2] < smallregion) or (nn[c_2] < smallregion):
        #                 self._mergeRegions(c_1, c_2)


        result = self._outPutSegmentation()
        return result

    def _getMaxColorDiff(slef, pt_1, pt_2):

        b_1 = pt_1[0]
        g_1 = pt_1[1]
        r_1 = pt_1[2]

        b_2 = pt_2[0]
        g_2 = pt_2[1]
        r_2 = pt_2[2]

        return  max(abs(b_1 - b_2), abs(g_1 - g_2), abs(r_1 - r_2))


    def _bucketSort(self, pairs):
        n = len(pairs)
        nbe = [0] * n
        cnbe = [0] * n
        result = [[0] * 3 for i in range(n)]

        # class all elements according to their family
        for i in range(0, n):
            pair = pairs[i]
            df = pair[2]
            nbe[df] = nbe[df] + 1

        # cumulative histogram
        for i in range(1, 255):
            cnbe[i] = cnbe[i - 1] + nbe[i - 1] # index of first element of category i

        # allocation
        for i in range(0, n):
            pair = pairs[i]
            df = pair[2]
            result[cnbe[df]] = pairs[i];
            cnbe[df] = cnbe[df] + 1

        return result


    def _mergePredicate(self, req_1, req_2):

        blue_avg = self._blue_avg
        green_avg = self._green_avg
        red_avg = self._red_avg
        nn = self._nn
        q = self._q
        logdelta = self._logdelta

        g = 256 # number of levels in a color channel

        dB = blue_avg[req_1] - blue_avg[req_2]
        dG = green_avg[req_1] - green_avg[req_2]
        dR = red_avg[req_1] - red_avg[req_2]

        # print('dB = {0}; dG = {1}; dR = {2}'.format(dB,dG, dR))

        dB = dB * dB
        dG = dG * dG
        dR = dR * dR

        logreg_1 = min(g, nn[req_1]) * math.log(1.0 + nn[req_1])
        logreg_2 = min(g, nn[req_1]) * math.log(1.0 + nn[req_2])

        dev_1 = ((g * g) / (2.0 * q * nn[req_1])) * (logreg_1 + logdelta);
        dev_2 = ((g * g) / (2.0 * q * nn[req_2])) * (logreg_2 + logdelta)
        dev = dev_1 + dev_2


        return (dR < dev) and (dG < dev) and (dB < dev)


    def _mergeRegions(self, c_1, c_2):
        print('_mergeRegions')

        nn = self._nn
        blue_avg = self._blue_avg
        green_avg = self._green_avg
        red_avg = self._red_avg

        reg = self._uf.unionRoot(c_1, c_2)

        print('reg = {0}'.format(reg))

        nreg = nn[c_1] + nn[c_2]

        bavg = (nn[c_1] * blue_avg[c_1] + nn[c_2] * blue_avg[c_2]) / nreg
        gavg = (nn[c_1] * green_avg[c_1] + nn[c_2] * green_avg[c_2]) / nreg
        ravg = (nn[c_1] * red_avg[c_1] + nn[c_2] * red_avg[c_2]) / nreg

        self._nn[reg] = nreg
        self._red_avg[reg] = ravg
        self.green_avg[reg] = gavg
        self.blue_avg[reg] = bavg


    def _outPutSegmentation(self):
        height = self._height
        width = self._width
        uf = self._uf
        blue_avg = self._blue_avg
        green_avg = self._green_avg
        red_avg = self._red_avg

        result = np.zeros((height, width, 3), np.uint8)

        for y in range(0, height):
            for x in range(0, width):
                index = y * width + x
                indexb = uf.find(index) # get the root index

                b = blue_avg[indexb]
                g = green_avg[indexb]
                r = red_avg[indexb]

                result[y, x] = [b, g, r]

        return result



'''
class Rmpair:
    # An edge: two indices and a difference value

    def __init__(self, r1=0, r2=0, diff=0):
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

'''



