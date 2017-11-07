import numpy as np


class SRM:
    def __init__(self, image, Q=32.0):

        self._img = image

        self._height = image.shape[0]
        self._width = image.shape[1]

        n = image.shape[0] * image.shape[1]
        self._n = n

        self._blue_avg = [0] * n
        self._green_avg = [0] * n
        self._red_avg = [0] * n

        if image.ndim == 3:
            self._depth = image.shape[2]
        else:
            self._depth = 1


    def run(self):

        # init auxiliary buffers for union-find operations
        for y in range(0, self._height):
            for x in range(0, self._width):
                index = y + x
                pt = self._img[y][x]

                self._blue_avg[index] = pt[0]
                self._green_avg[index] = pt[1]
                self._red_avg[index] = pt[2]

        # Consider C4 - connectivity here
        npair = 2 * (self._width - 1) * (self._height - 1) + (self._height - 1) + (self._width - 1)
        orders = [[0] * 3 for i in range(npair)]

        n = self._n
        height = self._height
        width = self._width
        depth = self._depth

        img = self._img
        self._data = np.empty([n, depth + 2])
        self._sizes = np.ones(n)


        # pixels = self._img.reshape((self._n, self._img.shape[2]))
        pixels = self._img.reshape((self._n, self._img.shape[2])).astype('int16')
        pairs = []


        for i in range(height - 1):
            for j in range(width - 1):
                idx = i * width + j

                pt = pixels[idx]

                # -- C4 left --
                pt_left = pixels[idx+1]
                diff_left = getMaxColorDiff(pt, pt_left)

                order_left = [idx, idx + 1, diff_left]
                pairs.append(order_left)


                # -- C4 bellow --
                pt_bellow = pixels[idx + width]

                diff_bellow =  getMaxColorDiff(pt, pt_bellow)
                order_bellow = [idx, idx + width, diff_bellow]
                pairs.append(order_bellow)


        # The two border lines

        for i in range(height - 1):
            idx = i * width + width - 1
            pt = pixels[idx]

            pt_bellow = pixels[idx + width]
            diff = getMaxColorDiff(pt, pt_bellow)

            order_left = [idx, idx + width, diff]
            pairs.append(order_left)

        for i in range(width - 1):
            idx = (height - 1) * width + j;
            pt = pixels[idx]

            pt_left = pixels[idx + 1]
            diff = getMaxColorDiff(pt, pt_left)

            order_left = [idx, idx + 1, diff]
            pairs.append(order_left)

        sortPairs = bucketSort(pairs)

        for i in range (0, len(sortPairs)):
            pair = sortPairs[i]
            req_1 = pair[0]
            req_2 = pair[1]


# for (i = 0; i < npair; i++) {
# 			reg1 = order[i].r1;
# 			C1 = UF.Find(reg1);
# 			reg2 = order[i].r2;
# 			C2 = UF.Find(reg2);
# 			if ((C1 != C2) && (MergePredicate(C1, C2)))
# 				MergeRegions(C1, C2);
# 		}


def getMaxColorDiff(pt_1, pt_2):

    b_1 = pt_1[0]
    g_1 = pt_1[1]
    r_1 = pt_1[2]

    b_2 = pt_2[0]
    g_2 = pt_2[1]
    r_2 = pt_2[2]

    return  max(abs(b_1 - b_2), abs(g_1 - g_2), abs(r_1 - r_2))


'''
        print(' -- Building the initial image RAG ({0} edges) --'.format(npair))

        pixels = self._img.reshape((self._n, self._img.shape[2]))

        index = 0
        i = 0

        for y in range(0, self._height):
            for x in range(0, self._width):


        for pt_1 in range(len(pixels), len(pixels)-1):
            b1 = pt_1[0]
            g1 = pt_1[1]
            r1 = pt_1[2]

            pt_2 = pixels[index + 1]
            b2 = pt_2[0]
            g2 = pt_2[1]
            r2 = pt_2[2]

            diff = np.max(np.abs(b1-b2), np.abs(g1-g2), np.abs(r1-r2))

            orders[0] = index
            orders[1] = index + 1
            orders[2] = diff

        print('end')

        i = 0
        for y in range(0, self._height):
            for x in range(0, self._width):

                index = self._height * self._width
                order = orders[i]

                pt = self._img[y][x]


                # C4 left
                order[0] = index;
                order[1] = index + 1;

                order[3] = biggest()















class Rmpair:

    def __init__(self):
        self._r1 = 0
        self._r2 = 0
        self._diff = 0

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

def bucketSort(pairs):

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
