import numpy as np
import random
from random import choice
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
k3 = 100.0
FIELD_ARG = 8  

max_mut = 0.7
min_mut = 0.05

def gfm(arr, right = True):
    if arr is None:
        return []
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

def generate_random_matrix(size, max_value = 12):
    # Retorna uma matriz NumPy em vez de uma lista de listas
    valid_numbers = [i for i in validIntegers(FIELD_ARG) if i <= max_value]
    return np.array(random.choices(valid_numbers[1:], k=size))

def initialize_population(pop_size, matrix_size, max_value = 12):
    return [generate_random_matrix(matrix_size, max_value=max_value) for _ in range(pop_size)]

def variety_checker(population):
    
    population = np.array([j for (i,j) in population])
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

def mini_local_search(A, max_test = 12, random_order = False):
    if random_order:
        index_max = random.randint(0,len(A)-1)
    else:
        index_max = np.argmax(A)

    max_val = min(max_test, A[index_max] + 1)
    min_val = 1#max(1,A[index_max] - 3)
    a_next = np.copy(A)
    
    best_fitness = fitness_function(A)
    a_next = np.copy(A)
    for val in range(min_val, max_val+1):
        a_next[index_max] = val
        fitness = fitness_function(a_next)
        if fitness > best_fitness:
            return fitness, a_next
    return best_fitness, A

def local_search(A, max_test = 12, random_order = False):
    indexes = np.argsort(A)
    if random_order:
        random.shuffle(indexes)
    best_fitness = fitness_function(A)
    for i in range(len(indexes)-1, -1, -1):
        index_max = indexes[i]
        max_val = min(max_test, A[index_max] + 1)
        min_val = 1#max(1,A[index_max] - 3)
        a_next = np.copy(A)
        for val in range(min_val, max_val+1):
            a_next[index_max] = val
            fitness = fitness_function(a_next)
            if fitness > best_fitness:
                return fitness, a_next
    """point1 = random.sample(range(len(A)), 1)[0]
    p2Range = [i for i in range(len(A)) if i != point1]
    print(p2Range)
    random.shuffle(p2Range)
    for point2 in p2Range:
        a_next = np.copy(A)
        a_next[point1], a_next[point2] = a_next[point2], a_next[point1]
        fitness = fitness_function(a_next)
        if fitness > best_fitness:
            return a_next"""
    a_shuffle = np.copy(A)
    random.shuffle(a_shuffle)
    indexes = np.argsort(a_shuffle)
    if random_order:
        random.shuffle(indexes)
    for i in range(len(indexes)-1, -1, -1):
        index_max = indexes[i]
        max_val = min(max_test, a_shuffle[index_max] + 3)
        min_val = max(1,a_shuffle[index_max] - 3)
        a_next = np.copy(a_shuffle)
        for val in range(min_val, max_val+1):
            a_next[index_max] = val
            fitness = fitness_function(a_next)
            if fitness >= best_fitness:
                return fitness, a_next
    return best_fitness, A
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
    return max(random.sample(population, k), key = lambda x: x[0])#, key=fitness_function)

"""def stochastic_universal_sampling(population, num_offspring):

    fitnesses = np.vectorize(fitness_function)(population)
    # Calculate the total fitness
    total_fitness = np.sum(fitnesses)
    
    # Determine the distance between pointers
    pointer_distance = total_fitness / num_offspring
    
    # Select a random starting point
    start_point = np.random.uniform(0, pointer_distance)
    
    # Create pointers
    pointers = [start_point + i * pointer_distance for i in range(num_offspring)]
    
    # Traverse the population to select individuals
    selected_indices = []
    cumulative_fitness = 0.0
    current_member = 0
    
    for pointer in pointers:
        while cumulative_fitness + fitnesses[current_member] < pointer:
            cumulative_fitness += fitnesses[current_member]
            current_member += 1
        selected_indices.append(current_member)
    
    return selected_indices"""

def two_point_crossover(parent1, parent2):
    point1, point2 = sorted(random.sample(range(len(parent1)), 2))
    child = np.copy(parent1)
    for i in range(point1, point2):
        child[i] = parent2[i]
    return child

def mutate(matrix):
    """i = random.randint(0, len(matrix)-1)
    valid_numbers = validIntegers(FIELD_ARG)
    matrix[i] = random.choice(valid_numbers[1:])"""
    maxPos = max(validIntegers(FIELD_ARG))
    for i in range(len(matrix)):
        if bool(random.getrandbits(1)):
            continue
        if matrix[i] == 1:
            matrix[i] += 1
        elif matrix[i] == maxPos:
            matrix[i] -= 1
        elif bool(random.getrandbits(1)):
            matrix[i] += 1
        else:
            matrix[i] -= 1
    if bool(random.getrandbits(1)):
        return matrix
    point1, point2 = random.sample(range(len(matrix)), 2)
    matrix[point1], matrix[point2] = matrix[point2], matrix[point1]

    return matrix


    return matrix

def full_local_search(A, max_test = 10):
    old_A = np.copy(A)
    while True:
        fitness, new_A = local_search(old_A, max_test=max_test)
        if np.array_equal(new_A, old_A):
            print("best: " + str(new_A) + ", MDS? "+ str(isMDS(gfm(new_A),FIELD_ARG).result)+ ", cost: "+ str(calculate_c(gfm(new_A))))
            return fitness, new_A
        old_A = new_A

def genetic_algorithm(pop_size, generations, matrix_size):
    #population = [full_local_search(a) for a in initialize_population(pop_size, matrix_size, max_value=12)]
    population = [(fitness_function(a),a) for a in initialize_population(pop_size, matrix_size, max_value=15)]

    incumbent = None
    incumbent_cost = np.inf
    inc_is_mds = False
    inc_xor = np.inf
    inc_xtime = np.inf

    print("Initial population ", population)
    for _ in range(generations):
        print("Generation ", _)
        best_fitness, best_solution = max(population, key = lambda x: x[0])

        best_solution_mat = gfm(best_solution)

        best_cost = calculate_c(best_solution_mat)
        best_is_mds = isMDS(best_solution_mat, FIELD_ARG).result
        best_xor = _matrix_xor_cost(best_solution_mat, FIELD_ARG)
        best_xtime = _matrix_xtime_cost(best_solution_mat, FIELD_ARG)

        if (best_is_mds and not inc_is_mds) or (best_cost < incumbent_cost and (best_is_mds or not inc_is_mds)):
            incumbent = best_solution
            incumbent_cost = best_cost
            inc_is_mds = best_is_mds
            inc_xor = best_xor
            inc_xtime = best_xtime

        print("Incumbent: "+ str(gfm(incumbent)))
        print("É MDS: " + str(inc_is_mds))
        print("Custo: " + str(incumbent_cost))
        print("XOR: "+ str(inc_xor) + ", XTIME: "+ str(inc_xtime))

        print("Melhor solução da geração: " + str(best_solution_mat))
        print("É MDS: " + str(best_is_mds))
        print("Custo: " + str(best_cost))
        print("XOR: "+ str(best_xor) + ", XTIME: "+ str(best_xtime))
        print("Fitness: " + str(best_fitness))

        equal_values = variety_checker(population)
        max_equal = pop_size*matrix_size
        min_equal = matrix_size
        mutation_rate = min_mut + (max_mut-min_mut)*((equal_values-min_equal)/(max_equal-min_equal))**2
        print(mutation_rate)

        #little_bro = diminish_mutation(best_solution)
        #little_bro = best_solution//2
        #little_bro[little_bro == 0] = 1
        #little_bro2 = diminish_mutation(best_solution,True)
        new_population = [local_search(best_solution)]
        #print("start_fit")
        #fitnesses = [(fitness_function(individual), individual) for individual in population]
        #print("end_fit")
        while len(new_population) < pop_size:
            _, parent1 = tournament_selection(population)
            _, parent2 = tournament_selection(population)
            child = two_point_crossover(parent1, parent2)
            if random.random() < mutation_rate:
                child = mutate(child)
                child_pair = mini_local_search(child, random_order=True, max_test=15)
            else:
                child_pair = (fitness_function(child), child)
            new_population.append(child_pair)
        
        population = new_population
        print("New population ", population)
        
    
    best_solution = incumbent
    print("Melhor solução: " + str(best_solution))
    print("É MDS: " + str(isMDS(gfm(best_solution), FIELD_ARG).result))
    print("Custo: " + str(calculate_c(gfm(best_solution))))
    return gfm(best_solution)

if __name__ == "__main__":
    best_matrix = genetic_algorithm(10, 10000, 8)
    print("Melhor matriz encontrada: ", best_matrix, " é mds: ", isMDS(best_matrix, FIELD_ARG).result, " já foi encontrada? ", existsInDataset(best_matrix).exists, " nome: ", existsInDataset(best_matrix).name)
