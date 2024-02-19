from numpy import array, ndarray, double, uint32
from numpy.linalg import det, solve
from pyparsing import List

def generate_feasable_set(A: List[List[double]], b: List[double]) -> List[ndarray[double]]:
    n = len(b)
    m = len(A)

    b = array(b)
    bfs = []
    indexes = []

    for i in range(m):
        indexes.append(i)

        search(i, A, b, indexes, bfs, n, m)

        indexes.pop()

    return bfs


def search(
    i: uint32, 
    A: List[List[double]], b: ndarray[double], 
    indexes: List[uint32], bfs: List[ndarray[double]], 
    n: uint32, m: uint32
):
    if len(indexes) == n:
        d = []

        for j in indexes:
            d.append(A[j].copy())

        B = array(d)
        B.transpose()

        if abs(det(B)) < 10**-9:
            print(f'{indexes}:')
            print(f'Матрица B = {B} вырождена')
            print('---\n')
        else:
            x = solve(B, b)
            f = True

            for k in range(n):
                if x[k] < 0:
                    f = False

                    print(f'{indexes}: k = {k}, x[k] = {x[k]} отрицательна')
                    print('---\n')

                    break

            if f:
                bfs.append(x)
                print(f'{indexes}: x = {x} является базовым допустим решением')
                print('---\n')
    else:
        for j in range(i + 1, m):
            indexes.append(j)

            search(j, A, b, indexes, bfs, n, m)

            indexes.pop()

