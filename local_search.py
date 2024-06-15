from main_rot import initialize_population, full_local_search

matrix_size = 5
pop_size = 10
population = [full_local_search(a) for a in initialize_population(pop_size, matrix_size, max_value=12)]