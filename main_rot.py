import numpy as np
import random
from typing import List
from helpers.datasetDict import *
from readonly import validIntegers, isMDS, combinedCost, existsInDataset, _matrix_xor_cost, _matrix_xtime_cost
from customization import isMDSCountDistance
import galois
import numpy as np
import scipy
from helpers.countsHelper import _matrix_xtime_cost, _matrix_xor_cost

k1 = 1000.0
k2_0 = 2000.0
k2 = 2000.0
k3 = 10.0
FIELD_ARG = 8  

max_mut = 0.3
min_mut = 0.05

def gfm(arr, right = True):
    n = len(arr)
    matrix = np.zeros((n,n), dtype=arr.dtype)
    mult = 1
    if right:
        mult = -1
    for i in range(n):
        matrix[i] = np.roll(arr, mult*i)
    return matrix

def generate_all_k_submatrices(A, k):
    """Gera todas as submatrizes k x k de uma matriz A."""
    n = A.shape[0]
    submatrices = []
    for i in range(n - k + 1):
        for j in range(n - k + 1):
            submatrices.append(A[i:i + k, j:j + k])
    return submatrices

def calculate_d(A, verbose = False):
    result = isMDSCountDistance(A,FIELD_ARG, verbose)
    if result.error == None:
        return result.distance
    return float('inf')
    """Calcula o valor de d para a matriz A no campo finito GF(2^FIELD_ARG)."""
    """GF = galois.GF(2**FIELD_ARG)
    A = np.array(A) if not isinstance(A, np.ndarray) else A
    A_gf = GF(A)
    
    n = A.shape[0]
    total_submatrices = 0
    zero_det_submatrices = 0

    for k in range(1, n + 1):
        submatrices = generate_all_k_submatrices(A, k)
        for submatrix in submatrices:
            submatrix_gf = GF(submatrix)
            det = np.linalg.det(submatrix_gf)
            if det == 0:
                zero_det_submatrices += 1
        total_submatrices += len(submatrices)

    return zero_det_submatrices / total_submatrices if total_submatrices > 0 else 0"""

def calculate_c(A):
    xtime_cost = _matrix_xtime_cost(A, FIELD_ARG)
    xor_cost = _matrix_xor_cost(A, FIELD_ARG)
    return 3 * xtime_cost + xor_cost

def fitness_function(A_line, verbose = False, direct = False):

    if direct:
        A = A_line
    else:
        A = gfm(A_line)
    d = calculate_d(A, verbose)
    c = calculate_c(A)
    #return (1/c)*(1-d)"""
    if verbose:
        print("d: "+ str(d) + ", c: "+ str(c))

    if d > 0:
        return k1 / (1 + c + k3 * d)
    else:
        return k2 / (1 + c)

def generate_random_matrix(size):
    # Retorna uma matriz NumPy em vez de uma lista de listas
    valid_numbers = validIntegers(FIELD_ARG)
    return np.array(random.choices(valid_numbers[1:], k=size))

def initialize_population(pop_size, matrix_size):
    return [generate_random_matrix(matrix_size) for _ in range(pop_size)]

def variety_checker(population):
    population = np.array(population)
    vectorized_children = population.reshape(population.shape[0], -1)

    print(vectorized_children)

    _, count = scipy.stats.mode(vectorized_children)
    sum_count = sum(count)

    return sum_count
    
def diminish_mutation(A, limit_to_1 = False):
    indexes = np.array(np.where(A>1))[0]
    #print(indexes)
    n = min(len(A),len(indexes))+1
    swaps = np.random.randint(1,n)

    if limit_to_1:
        swaps = 1

    selected_swaps = random.sample(list(indexes), swaps)
    #print(list(indexes), selected_swaps)
    result = (np.copy(A))
    for x in selected_swaps:
        result[x] = np.random.randint(1,result[x])
    #print(A)
    #print(result)
    return result

# def fitness_function(matrix):
#     checker_result = isMDS(matrix, FIELD_ARG)
#     if checker_result.result:
#         cost_result = combinedCost(matrix, FIELD_ARG)
#         if cost_result.error is None:
#             return 1 / (1 + cost_result.cost)
#         else:
#             raise ValueError("Erro ao calcular o custo: " + cost_result.error)
#     else:
#         if checker_result.error:
#             raise ValueError("Erro na verificação MDS: " + checker_result.error)
#         return 0

def tournament_selection(population, k=3):
    return max(random.sample(population, k), key=fitness_function)

def two_point_crossover(parent1, parent2):
    point1, point2 = sorted(random.sample(range(len(parent1)), 2))
    child = np.copy(parent1)
    for i in range(point1, point2):
        child[i] = parent2[i]
    return child

def mutate(matrix):
    i = random.randint(0, len(matrix)-1)
    valid_numbers = validIntegers(FIELD_ARG)
    matrix[i] = random.choice(valid_numbers[1:])
    return matrix

def genetic_algorithm(pop_size, generations, matrix_size):
    population = initialize_population(pop_size, matrix_size)
    print("Initial population ", population)
    for _ in range(generations):
        print("Generation ", _)
        best_solution = max(population, key=fitness_function)
        best_solution_mat = gfm(best_solution)
        print("Melhor solução: " + str(best_solution_mat))
        print("É MDS: " + str(isMDS(best_solution_mat, FIELD_ARG).result))
        print("Custo: " + str(calculate_c(best_solution_mat)))
        print("XOR: "+ str(_matrix_xor_cost(best_solution_mat, FIELD_ARG)) + ", XTIME: "+ str(_matrix_xtime_cost(best_solution_mat, FIELD_ARG)))
        print("Fitness: " + str(fitness_function(best_solution, True)))

        equal_values = variety_checker(population)
        max_equal = pop_size*matrix_size
        min_equal = matrix_size
        mutation_rate = min_mut + (max_mut-min_mut)*((equal_values-min_equal)/(max_equal-min_equal))**2
        print(mutation_rate)

        little_bro = diminish_mutation(best_solution)
        #little_bro = best_solution//2
        #little_bro[little_bro == 0] = 1
        little_bro2 = diminish_mutation(best_solution,True)
        print(little_bro,little_bro2)
        new_population = [best_solution, little_bro, little_bro2]
        while len(new_population) < pop_size:
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            child = two_point_crossover(parent1, parent2)
            if random.random() < mutation_rate:
                child = mutate(child)
            if(isMDS(gfm(child), FIELD_ARG)):
                new_population.append(child)
        
        population = new_population
        print("New population ", population)
        
    
    best_solution = max(population, key=fitness_function)
    print("Melhor solução: " + str(best_solution))
    print("É MDS: " + str(isMDS(best_solution, FIELD_ARG).result))
    print("Custo: " + str(calculate_c(best_solution)))
    return best_solution

if __name__ == "__main__":
    best_matrix = genetic_algorithm(10, 10000, 6)
    print("Melhor matriz encontrada: ", best_matrix, " é mds: ", isMDS(best_matrix, FIELD_ARG).result, " já foi encontrada? ", existsInDataset(best_matrix).exists, " nome: ", existsInDataset(best_matrix).name)