from typing import List
from numpy import ndarray, double, delete, arange, zeros

def exclude_columns(A: ndarray[double, double], indexes: List[int]) -> List[int]:
    return delete(arange(A.shape[1]), indexes)

def sort_vector(v: ndarray[double], indexes: List[int]) -> ndarray[double]:
    r = zeros(len(v))

    for i in range(len(indexes)):
        j = indexes[i]
        r[j] = v[i]

    return r