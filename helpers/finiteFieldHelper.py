import galois
import numpy as np
import itertools

def _int_to_gf(int_poly):
    coeffs = []
    for x in bin(int_poly)[2:]:
        coeffs.append(int(x))

    return galois.Poly(coeffs, field=galois.GF(2))

def _int_to_gf_mat(int_mat, field):
    rows = len(int_mat)
    cols = len(int_mat[0])

    gf_mat = field.Zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            gf_mat[i][j] = int_mat[i][j]

    return gf_mat

def _is_mds(mat_in_field):
	if np.linalg.det(mat_in_field) == 0:
		return False

	dim = len(mat_in_field)

	dim_list = [i for i in range(dim)]

	z = 1
	while z < dim:
		possibilities = list(itertools.combinations(dim_list, z))

		for rows_to_be_removed in possibilities:
			for columns_to_be_removed in possibilities:
				submat = np.delete(mat_in_field, rows_to_be_removed, axis=0)
				submat = np.delete(submat, columns_to_be_removed, axis=1)
				if np.linalg.det(submat) == 0:
					return False
		z += 1
	return True