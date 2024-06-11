from main_rot import fitness_function, calculate_c, gfm
from readonly import _matrix_xor_cost, _matrix_xtime_cost, isMDS
import numpy as np

"""mat = np.array([
        [1, 5, 3, 2, 1],
        [5, 3, 2, 1, 1],
        [3, 2, 1, 1, 5],
        [2, 1, 1, 5, 3],
        [1, 1, 5, 3, 2]
    ])"""
mat = gfm(np.array([1,2,3,2,1]))
FIELD_ARG = 8

print("Melhor solução: " + str(mat))
print("É MDS: " + str(isMDS(mat, FIELD_ARG).result))
print("Custo: " + str(calculate_c(mat)))
print("XOR: "+ str(_matrix_xor_cost(mat, FIELD_ARG)) + ", XTIME: "+ str(_matrix_xtime_cost(mat, FIELD_ARG)))
print("Fitness: " + str(fitness_function(mat, True, direct=True)))