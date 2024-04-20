def _poly_xtime_cost(poly, ORDER):
    if poly == 0:
       return 0
    degree = ORDER
    degree_mask = 2**ORDER

    while (poly & degree_mask) == 0:
        degree_mask = degree_mask >> 1
        degree -= 1
    return degree

def _matrix_xtime_cost(mat, ORDER):
    total_cost = 0
    for row in range(len(mat)):
        row_cost = 0
        for col in range(len(mat[row])):
            row_cost += _poly_xtime_cost(mat[row][col], ORDER)
        total_cost += row_cost
    return total_cost

def _poly_xor_cost(poly, ORDER):
    mask = 1
    set_bits = 0
    current_bit = 0
    while current_bit < ORDER:
        if (poly & mask) != 0:
            set_bits += 1
        mask = mask << 1
        current_bit += 1
    return set_bits - 1

def _matrix_xor_cost(mat, ORDER):
    total_cost = 0
    for row in range(len(mat)):
        row_cost = len(mat) - 1
        for col in range(len(mat[row])):
            row_cost += _poly_xor_cost(mat[row][col], ORDER)
        total_cost += row_cost
    return total_cost