from typing import List
from helpers.countsHelper import _matrix_xtime_cost, _matrix_xor_cost
from helpers.finiteFieldHelper import _is_mds, _int_to_gf_mat
from helpers.datasetHelper import _exists_in_dataset
from helpers.datasetDict import *
import galois
import readonly
import itertools
import numpy as np

# Place customized functions here.
# Don't change the ones from readonly, copy them and modify here after copying.
# For example, if you want a customized cost function, with different weights, you can define it here.
# Or the MDS gradient score.

class CheckerResultDistance:
    def __init__(self, distance: int, error: str):
        self.distance = distance
        self.error = error

def _is_mds_distance(mat_in_field):

  distance = 0

  if np.linalg.det(mat_in_field) == 0:
    distance += 1

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
          distance += 1
    z += 1
  return distance

# Check if a given matrix is MDS.
# Note that it returns a CheckerResult, to wrap the result.
# This is just to ensure you are not feeding it invalid matrices.
# Use it!
def isMDSCountDistance(intMatrix: List[List[int]], fieldArg: int, verbose: False) -> CheckerResultDistance:
    if verbose:
       print("a")
    if not readonly._allElementsValid(intMatrix, fieldArg):
        return CheckerResultDistance(None, "invalid elements")
    if verbose:
       print("b")
    if not readonly._dimensionsValid(intMatrix):
        return CheckerResultDistance(None, "invalid dimensions")
    if verbose:
       print("c")
    fieldConvertedMatrix = _int_to_gf_mat(intMatrix, galois.GF(2**fieldArg))
    if verbose:
       print("d")
    try:
        if verbose:
          print("e")
        check = _is_mds_distance(fieldConvertedMatrix)
        return CheckerResultDistance(check, None)
    except Exception as e:
        if verbose:
          print("f")
        return CheckerResultDistance(None, "error on old checker:\n"+str(e))