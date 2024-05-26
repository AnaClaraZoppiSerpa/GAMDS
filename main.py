import numpy as np
import random
from typing import List
from helpers.datasetDict import *
from readonly import validIntegers, isMDS, combinedCost, existsInDataset
import galois
import numpy as np
from helpers.countsHelper import _matrix_xtime_cost, _matrix_xor_cost

k1 = 1.0
k2 = 1.0
k3 = 10.0
FIELD_ARG = 8  

def generate_all_k_submatrices(A, k):
    """Gera todas as submatrizes k x k de uma matriz A."""
    n = A.shape[0]
    submatrices = []
    for i in range(n - k + 1):
        for j in range(n - k + 1):
            submatrices.append(A[i:i + k, j:j + k])
    return submatrices

def calculate_d(A):
    """Calcula o valor de d para a matriz A no campo finito GF(2^FIELD_ARG)."""
    GF = galois.GF(2**FIELD_ARG)
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

    return zero_det_submatrices / total_submatrices if total_submatrices > 0 else 0

def calculate_c(A):
    xtime_cost = _matrix_xtime_cost(A, FIELD_ARG)
    xor_cost = _matrix_xor_cost(A, FIELD_ARG)
    return 3 * xtime_cost + xor_cost

def fitness_function(A):
    d = calculate_d(A)
    c = calculate_c(A)
    return (1/c)*(1-d)
    # if d > 0:
    #     return k1 / (1 + c + k3 * d)
    # else:
    #     return k2 / (1 + c)

def generate_random_matrix(size):
    # Retorna uma matriz NumPy em vez de uma lista de listas
    valid_numbers = validIntegers(FIELD_ARG)
    return np.array([random.choices(valid_numbers[1:], k=size) for _ in range(size)])

def initialize_population(pop_size, matrix_size):
    return [generate_random_matrix(matrix_size) for _ in range(pop_size)]


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
    return min(random.sample(population, k), key=fitness_function)

def two_point_crossover(parent1, parent2):
    point1, point2 = sorted(random.sample(range(len(parent1)), 2))
    child = [list(row) for row in parent1]
    for i in range(point1, point2):
        child[i] = parent2[i]
    return child

def mutate(matrix):
    i, j = random.randint(0, len(matrix)-1), random.randint(0, len(matrix)-1)
    valid_numbers = validIntegers(FIELD_ARG)
    matrix[i][j] = random.choice(valid_numbers[1:])
    return matrix

def genetic_algorithm(pop_size, generations, matrix_size):
    population = initialize_population(pop_size, matrix_size)
    print("Initial population ", population)
    for _ in range(generations):
        print("Generation ", _)
        new_population = []
        while len(new_population) < pop_size:
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            child = two_point_crossover(parent1, parent2)
            # if random.random() < 0.1:
            #     child = mutate(child)
            if(isMDS(child, FIELD_ARG)):
                new_population.append(child)
        
        population = new_population
        print("New population ", population)
    
    best_solution = min(population, key=fitness_function)
    print(best_solution)
    return best_solution

best_matrix = genetic_algorithm(10, 10, 4)
print("Melhor matriz encontrada: ", best_matrix, " é mds: ", isMDS(best_matrix, FIELD_ARG).result, " já foi encontrada? ", existsInDataset(best_matrix).exists, " nome: ", existsInDataset(best_matrix).name)