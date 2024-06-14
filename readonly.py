from typing import List
from helpers.countsHelper import _matrix_xtime_cost, _matrix_xor_cost
from helpers.finiteFieldHelper import _is_mds, _int_to_gf_mat
from helpers.datasetHelper import _exists_in_dataset
from helpers.datasetDict import *
import galois
import numpy as np

# Wrapper for the return value of checking if a matrix is MDS or not.
# result: bool - True if the matrix is MDS, False otherwise.
# error: str - Error message, if any. None otherwise.
# Possible errors are:
# - Having an invalid element for the chosen finite field.
# - Having invalid dimensions for the matrix (only square matrices are allowed).
class CheckerResult:
    def __init__(self, result: bool, error: str):
        self.result = result
        self.error = error

# Wrapper for the return value of calculating costs for a matrix.
# Should be used as return value for xtime count, xor count, and weighted cost.
# cost: int - The count (or weighted cost) of the matrix.
# error: str - Error message, if any. None otherwise.
# Possible errors are:
# - Having an invalid element for the chosen finite field.
# - Having invalid dimensions for the matrix (only square matrices are allowed).
class CostResult:
    def __init__(self, cost: int, error: str):
        self.cost = cost
        self.error = error

# Wrapper for the return value of fetching the baseline result for a matrix.
# baseline: int - The baseline cost for the matrix.
# info: str - Message, if any. None otherwise.
# Possible messages are:
# - There are no baselines for this dimension. (Could happen if you're experimenting with e.g. a 9x9 matrix.)
class BaselineResult:
    def __init__(self, baseline: int, info: str):
        self.baseline = baseline
        self.info = info

# Wrapper for the return value of checking if a matrix is already published or not.
# exists: bool - True if the matrix is already published, False otherwise.
# name: str - Name of the matrix, if it is already published. None otherwise.
class ExistsResult:
    def __init__(self, exists: bool, name: str):
        self.exists = exists
        self.name = name

# Upper bound for integers you can choose for your matrix, given your chosen finite field.
# Use it!
def validUpper(fieldArg: int) -> int:
    return 2**fieldArg - 1

# Lower bound for integers you can choose for your matrix. Always zero.
# Use it!
def validLower(fieldArg: int) -> int:
    return 0

# Check if a given integer is valid for the chosen finite field.
# Use it!
def isValidInteger(fieldArg: int, element: int) -> bool:
    return validLower(fieldArg) <= element <= validUpper(fieldArg)

# Return a list of all valid integers for the chosen finite field.
# Use it!
def validIntegers(fieldArg: int) -> List[int]:
    return list(range(validLower(fieldArg), validUpper(fieldArg) + 1))

# Helper - checks if all elements of a matrix are valid.
# Don't call directly.
def _allElementsValid(intMatrix: List[List[int]], fieldArg: int) -> bool:
    for row in intMatrix:
        for element in row:
            if not isValidInteger(fieldArg, element):
                return False
    return True

# Helper - checks if the dimensions of a matrix are valid.
# Don't call directly.
def _dimensionsValid(intMatrix: List[List[int]]) -> bool:
    rows = len(intMatrix)
    cols = len(intMatrix[0])

    for row in intMatrix:
        if len(row) != cols:
            return False

    return rows > 0 and rows == cols

# Calculate the weighted cost of a matrix.
# Note that it returns a CostResult, to wrap the result.
# This is just to ensure you are not feeding it invalid matrices.
def combinedCost(intMatrix: List[List[int]], fieldArg: int) -> CostResult:
    if not _allElementsValid(intMatrix, fieldArg):
        return CostResult(None, "invalid elements")
    
    if not _dimensionsValid(intMatrix):
        return CostResult(None, "invalid dimensions")

    xtimeCount = _matrix_xtime_cost(intMatrix, fieldArg)
    xorCount = _matrix_xor_cost(intMatrix, fieldArg)

    return CostResult(3*xtimeCount + xorCount, None)

# Calculate the xtOnly cost of a matrix.
# Note that it returns a CostResult, to wrap the result.
# This is just to ensure you are not feeding it invalid matrices.
def xtOnlyCost(intMatrix: List[List[int]], fieldArg: int) -> CostResult:
    if not _allElementsValid(intMatrix, fieldArg):
        return CostResult(None, "invalid elements")
    
    if not _dimensionsValid(intMatrix):
        return CostResult(None, "invalid dimensions")
    
    xtimeCount = _matrix_xtime_cost(intMatrix, fieldArg)
    return CostResult(xtimeCount, None)

# Fetch the baseline cost for a dimension.
# Note that it returns a BaselineResult, to wrap the result.
# This way you will know if there are no baselines for the dimension you are experimenting with.
# Use it!
def baselineCombinedCost(dimension: int) -> BaselineResult:
    baselinesMap = {8: 296, 4: 40, 3: 15, 2: 8, 5: 120, 6: 234, 7: 384, 16: 4544, 32: 20032}
    
    if dimension not in baselinesMap:
        return BaselineResult(None, "there are no baselines for this dimension")
    
    return BaselineResult(baselinesMap[dimension], None)

# Fetch the baseline cost for a dimension.
# Note that it returns a BaselineResult, to wrap the result.
# This way you will know if there are no baselines for the dimension you are experimenting with.
# Use it!
def baselineXtOnlyCost(dimension: int) -> BaselineResult:
    baselinesMap = {2: 2, 3: 3, 4: 8, 5: 30, 6: 59, 7: 96, 8: 72, 16: 1248, 32: 5440}
    
    if dimension not in baselinesMap:
        return BaselineResult(None, "there are no baselines for this dimension")
    
    return BaselineResult(baselinesMap[dimension], None)

# Check if a matrix is already present on the dataset of Ana's previous work or not.
# Note that it returns an ExistsResult, to wrap the result.
# This way you will know the ID of the matrix and will be able to ask Ana for more details if you have to.
# def existsInDataset(intMatrix: List[List[int]]) -> ExistsResult:
#     tuple = _exists_in_dataset(LOOKUP_DICT, intMatrix)
#     tupleBool = tuple[0]
#     tupleID = tuple[1]
#     return ExistsResult(tupleBool, tupleID)

def existsInDataset(intMatrix: List[List[int]]) -> ExistsResult:
    # Convert intMatrix to a numpy array for easier manipulation
    np_matrix = np.array(intMatrix)
    dic = LOOKUP_DICT
    # Check for the original and its variations
    variations = generate_variations(np_matrix)
    for var in variations:
        for id, stored_matrix in dic.items():
            if _matrix_equals(stored_matrix, var.tolist()):
                return ExistsResult(True, id)
    return ExistsResult(False, None)

def generate_variations(matrix: np.ndarray) -> List[np.ndarray]:
    """Generate all relevant variations of the matrix."""
    variations = []
    variations.append(matrix)  # Original
    variations.append(matrix.T)  # Transpose
    # Add the inverse if the matrix is square and invertible
    if matrix.shape[0] == matrix.shape[1]:
        try:
            inv = np.linalg.inv(matrix)
            variations.append(inv)
        except np.linalg.LinAlgError:
            pass  # Not invertible, do not add
    # Add rotations if needed
    for k in range(1, 4):
        variations.append(np.rot90(matrix, k))

    return variations

def _matrix_equals(mat1: List[List[int]], mat2: List[List[int]]) -> bool:
    """Check if two matrices are equal."""
    return np.array_equal(mat1, mat2)


# Check if a given matrix is MDS.
# Note that it returns a CheckerResult, to wrap the result.
# This is just to ensure you are not feeding it invalid matrices.
# Use it!
def isMDS(intMatrix: List[List[int]], fieldArg: int) -> CheckerResult:
    if not _allElementsValid(intMatrix, fieldArg):
        return CheckerResult(None, "invalid elements")
    
    if not _dimensionsValid(intMatrix):
        return CheckerResult(None, "invalid dimensions")

    fieldConvertedMatrix = _int_to_gf_mat(intMatrix, galois.GF(2**fieldArg))
    
    try:
        check = _is_mds(fieldConvertedMatrix)
        return CheckerResult(check, None)
    except Exception as e:
        return CheckerResult(None, "error on old checker:\n"+str(e))
