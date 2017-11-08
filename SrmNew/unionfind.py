class UnionFind:


    def __init__(self, n):

        self._rank = [0] * n
        self._parent = [0] * n


    def find(self, k):
        parent = self._parent
        while (parent[k] != k):
            k = parent[k];
        return k;

    def unionRoot(self, x, y):
        rank = self._rank
        parent = self._parent
        if (x == y):
            return -1
        if (rank[x] > rank[y]):
            parent[y] = x;
            return x

        else:
            parent[x] = y
            if (rank[x] == rank[y]):
                rank[y] = rank[y] +1

        return y
