from main_rot import fitness_function, calculate_c, gfm
from readonly import _matrix_xor_cost, _matrix_xtime_cost, isMDS
import numpy as np
#import gf2m

"""[6, 14, 12, 7],
        [11, 8, 2, 9],
        [1, 13, 1, 8],
        [2, 6, 13, 1]"""

"""mat = np.array([
    [1, 1, 9, 12, 9, 3], 
    [1, 9, 12, 9, 3, 1], 
    [9, 12, 9, 3, 1, 1], 
    [12, 9, 3, 1, 1, 9], 
    [9, 3, 1, 1, 9, 12], 
    [3, 1, 1, 9, 12, 9]
    ])"""
"""mat = np.array([   
    [1, 2, 3, 2, 1],
    [2, 3, 2, 1, 1],
    [3, 2, 1, 1, 2],
    [2, 1, 1, 2, 3],
    [1, 1, 2, 3, 2]


])"""
mat = gfm(np.array([3,2,1,1]))
FIELD_ARG = 8

print("Melhor solução: " + str(mat))
print("É MDS: " + str(isMDS(mat, FIELD_ARG).result))
print("Custo: " + str(calculate_c(mat)))
print("XOR: "+ str(_matrix_xor_cost(mat, FIELD_ARG)) + ", XTIME: "+ str(_matrix_xtime_cost(mat, FIELD_ARG)))
print("Fitness: " + str(fitness_function(mat, True, direct=True)))

"""inv = gf2m.invert_matrix(mat,len(mat))

print("Inv melhor solução: " + str(inv))
print("É MDS: " + str(isMDS(inv, FIELD_ARG).result))
print("Custo: " + str(calculate_c(inv)))
print("XOR: "+ str(_matrix_xor_cost(inv, FIELD_ARG)) + ", XTIME: "+ str(_matrix_xtime_cost(inv, FIELD_ARG)))
print("Fitness: " + str(fitness_function(inv, True, direct=True)))
"""
"""
227

1 1 9 12 9 3
1 9 12 9 3 1
9 12 9 3 1 1
12 9 3 1 1 9
9 3 1 1 9 12
3 1 1 9 12 9

1 2 8 5 8 2
2 5 1 2 6 12
12 9 15 8 8 13
13 5 11 3 10 1
1 15 13 14 11 8
8 2 3 3 2 8
"""