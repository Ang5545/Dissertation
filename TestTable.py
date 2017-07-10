import numpy as np

def printTable(data):
    names = []
    a = np.arange(data.shape[0])
    for x in np.nditer(a.T):
        names.append('Object %s' % (x + 1))


    row_format = "{:>15}" * (len(names) + 1)
    print(row_format.format("", *names))

    for team, row in zip(names, data):
        print(row_format.format(team, *row))


data = np.array([[1, 2, 1, 1, 1],
                 [0, 1, 0, 1, 1],
                 [0, 1, 0, 1, 1],
                 [0, 1, 0, 1, 1],
                 [2, 4, 2, 1, 1]])

printTable(data)

# for x in range(0, data.shape[0]):
#     headerList = np.


