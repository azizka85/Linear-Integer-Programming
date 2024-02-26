from typing import List, Tuple
from numpy import ndarray, double, zeros, concatenate, dot, where, abs
from numpy.linalg import solve
from modules.matrix import exclude_columns, sort_vector

epsilon = 10**-9

def simplex_method(
    c: ndarray[double], 
    A: ndarray[double, double], b: ndarray[double],
    B_set: List[int]
) -> Tuple[ndarray[double], double, ndarray[double], int, bool]:
    B = A[:, B_set]
    
    c_B = c[B_set]
    x_B = solve(B, b)

    N_set = exclude_columns(A, B_set)
    
    N = A[:, N_set]
    
    c_N = c[N_set]
    x_N = zeros(len(N_set))

    x = concatenate([x_B, x_N])
    obj = dot(concatenate([c_B, c_N]), x)
        
    dir = None
    flag = False
    k = 0

    while True:
        p = solve(B.transpose(), c_B)
        r_N = c_N - dot(p, N)

        ri = where(r_N < -epsilon * abs(r_N).max())[0]        

        if len(ri) == 0:
            x = sort_vector(x, concatenate([B_set, N_set]))

            if flag:
                dir = sort_vector(dir, concatenate([B_set, N_set]))

            break
        else:
            ri = ri[0]
            e = zeros(len(N_set))
            e[ri] = 1.
            d = -solve(B, N[:, ri])
            dir = concatenate([d, e])

            di = where(dir < -epsilon * abs(dir).max())[0]            

            if len(di) == 0:
                flag = True
                break
            else:
                steps = -x[di] / dir[di]
                step = steps.min()
                x_d = x + step * dir
                leaves = where(abs(x_d) <= epsilon * abs(x_d).max())[0]
                leave = leaves[0]

                li = B_set[leave]
                ei = N_set[ri]

                B_set[leave] = ei
                N_set[ri] = li

                x = x_d

                x[leave] = x[len(B_set) + ri]
                x[len(B_set) + ri] = 0

                B = A[:, B_set]
                c_B = c[B_set]

                N = A[:, N_set]
                c_N = c[N_set]

                obj = dot(concatenate([c_B, c_N]), x)
                
        k += 1

    return x, obj, dir, k, flag        
