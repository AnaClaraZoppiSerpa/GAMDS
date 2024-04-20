def _matrix_equals(mat1, mat2):
    if len(mat1) != len(mat2):
        return False
    
    rows = len(mat1)
    cols = len(mat1[0])

    for i in range(rows):
        for j in range(cols):
            if mat1[i][j] != mat2[i][j]:
                return False
    
    return True


def _exists_in_dataset(dic, mat):
    for id in dic:
        if _matrix_equals(dic[id], mat):
            return (True, id)
    return (False, None)