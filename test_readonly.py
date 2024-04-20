from readonly import validUpper, validLower, isValidInteger, validIntegers, _allElementsValid, _dimensionsValid, combinedCost, xtOnlyCost, baselineCombinedCost, baselineXtOnlyCost, existsInDataset, isMDS

from unittest.mock import patch

def test_validUpper():
    assert validUpper(4) == 15
    assert validUpper(8) == 255
    for i in range(0, 33):
        assert validUpper(i) == 2**i - 1

def test_validLower():
    assert validLower(4) == 0
    assert validLower(8) == 0
    for i in range(0, 33):
        assert validLower(i) == 0

def test_isValidInteger():
    assert isValidInteger(4, 0)
    assert isValidInteger(4, 15)
    assert not isValidInteger(4, 16)
    assert not isValidInteger(4, -1)
    assert isValidInteger(8, 0)
    assert isValidInteger(8, 255)
    assert not isValidInteger(8, 256)
    assert not isValidInteger(8, -1)
    for i in range(0, 33):
        assert isValidInteger(i, 0)
        assert isValidInteger(i, 2**i - 1)
        assert not isValidInteger(i, 2**i)
        assert not isValidInteger(i, -1)

def test_validIntegers():
    assert validIntegers(4) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    assert validIntegers(8) == list(range(256))
    for i in range(0, 10):
        assert validIntegers(i) == list(range(2**i))

def test_allElementsValid():
    assert _allElementsValid([[0, 1], [2, 15]], 4)
    assert not _allElementsValid([[0, 1], [2, 300]], 4)
    assert not _allElementsValid([[0, 1], [2, -1]], 4)
    assert not _allElementsValid([[0, 1], [2, 300]], 8)
    assert _allElementsValid([[255, 1], [2, 1]], 8)

def test_dimensionsValid():
    assert _dimensionsValid([[0, 1], [2, 15]])
    assert not _dimensionsValid([[0, 1], [2, 15], [2, 15]])
    assert not _dimensionsValid([[0, 1], [2]])
    assert not _dimensionsValid([[0, 1], [2, 15, 3]])

def test_combinedCost_invalidInput():
    invalidDimensions = [[0, 1], [2, 15], [2, 15]]
    invalidElements = [[0, 300], [2, 15]]

    invalidElementsResult = combinedCost(invalidElements, 4)
    invalidDimensionsResult = combinedCost(invalidDimensions, 4)

    assert invalidElementsResult.cost is None
    assert invalidElementsResult.error == "invalid elements"
    assert invalidDimensionsResult.cost is None
    assert invalidDimensionsResult.error == "invalid dimensions"

def test_xtOnlyCost_invalidInput():
    invalidDimensions = [[0, 1], [2, 15], [2, 15]]
    invalidElements = [[0, 300], [2, 15]]

    invalidElementsResult = xtOnlyCost(invalidElements, 4)
    invalidDimensionsResult = xtOnlyCost(invalidDimensions, 4)

    assert invalidElementsResult.cost is None
    assert invalidElementsResult.error == "invalid elements"
    assert invalidDimensionsResult.cost is None
    assert invalidDimensionsResult.error == "invalid dimensions"

def test_combinedCost_validInput():
    matrixFromSHARK = [[206, 149, 87, 130, 138, 25, 176, 1], [231, 254, 5, 210, 82, 193, 136, 241], [185, 218, 77, 209, 158, 23, 131, 134], [208, 157, 38, 44, 93, 159, 109, 117], [82, 169, 7, 108, 185, 143, 112, 23], [135, 40, 58, 90, 244, 51, 11, 108], [116, 81, 21, 207, 9, 164, 98, 9], [11, 49, 127, 134, 190, 5, 131, 52]]

    costSHARK = combinedCost(matrixFromSHARK, 8)

    assert costSHARK.cost == 235+3*369
    assert costSHARK.error is None

def test_xtOnlyCost_validInput():
    matrixFromSHARK = [[206, 149, 87, 130, 138, 25, 176, 1], [231, 254, 5, 210, 82, 193, 136, 241], [185, 218, 77, 209, 158, 23, 131, 134], [208, 157, 38, 44, 93, 159, 109, 117], [82, 169, 7, 108, 185, 143, 112, 23], [135, 40, 58, 90, 244, 51, 11, 108], [116, 81, 21, 207, 9, 164, 98, 9], [11, 49, 127, 134, 190, 5, 131, 52]]

    costSHARK = xtOnlyCost(matrixFromSHARK, 8)

    assert costSHARK.cost == 369
    assert costSHARK.error is None

def test_baselineCombinedCost():
    assert baselineCombinedCost(8).baseline == 296
    assert baselineCombinedCost(4).baseline == 40
    assert baselineCombinedCost(3).baseline == 15
    assert baselineCombinedCost(2).baseline == 8
    assert baselineCombinedCost(5).baseline == 120
    assert baselineCombinedCost(6).baseline == 234
    assert baselineCombinedCost(7).baseline == 384
    assert baselineCombinedCost(16).baseline == 4544
    assert baselineCombinedCost(32).baseline == 20032
    
    assert baselineCombinedCost(33).baseline is None
    assert baselineCombinedCost(0).info == "there are no baselines for this dimension"

    assert baselineCombinedCost(1).baseline is None
    assert baselineCombinedCost(1).info == "there are no baselines for this dimension"
    
    assert baselineCombinedCost(9).baseline is None
    assert baselineCombinedCost(9).info == "there are no baselines for this dimension"

def test_baselineXtOnlyCost():
    assert baselineXtOnlyCost(2).baseline == 2
    assert baselineXtOnlyCost(3).baseline == 3
    assert baselineXtOnlyCost(4).baseline == 8
    assert baselineXtOnlyCost(5).baseline == 30
    assert baselineXtOnlyCost(6).baseline == 59
    assert baselineXtOnlyCost(7).baseline == 96
    assert baselineXtOnlyCost(8).baseline == 72
    assert baselineXtOnlyCost(16).baseline == 1248
    assert baselineXtOnlyCost(32).baseline == 5440
    
    assert baselineXtOnlyCost(33).baseline is None
    assert baselineXtOnlyCost(0).info == "there are no baselines for this dimension"

    assert baselineXtOnlyCost(1).baseline is None
    assert baselineXtOnlyCost(1).info == "there are no baselines for this dimension"
    
    assert baselineXtOnlyCost(9).baseline is None
    assert baselineXtOnlyCost(9).info == "there are no baselines for this dimension"

def test_existsInDataset():
    matrixFromSHARK = [[206, 149, 87, 130, 138, 25, 176, 1], [231, 254, 5, 210, 82, 193, 136, 241], [185, 218, 77, 209, 158, 23, 131, 134], [208, 157, 38, 44, 93, 159, 109, 117], [82, 169, 7, 108, 185, 143, 112, 23], [135, 40, 58, 90, 244, 51, 11, 108], [116, 81, 21, 207, 9, 164, 98, 9], [11, 49, 127, 134, 190, 5, 131, 52]]

    assert existsInDataset(matrixFromSHARK).exists == True
    assert existsInDataset(matrixFromSHARK).name is not None

    notPresent = [[0, 0], [0, 0]]
    assert existsInDataset(notPresent).exists == False
    assert existsInDataset(notPresent).name is None

def test_isMDS_invalidInput():
    invalidDimensions = [[0, 1], [2, 15], [2, 15]]
    invalidElements = [[0, 300], [2, 15]]

    invalidElementsResult = isMDS(invalidElements, 8)
    invalidDimensionsResult = isMDS(invalidDimensions, 8)

    assert invalidElementsResult.result is None
    assert invalidElementsResult.error == "invalid elements"
    assert invalidDimensionsResult.result is None
    assert invalidDimensionsResult.error == "invalid dimensions"

def test_isMDS_MDS():
    matrixFromAES = [[2, 3, 1, 1], [1, 2, 3, 1], [1, 1, 2, 3], [3, 1, 1, 2]]

    assert isMDS(matrixFromAES, 8).result == True
    assert isMDS(matrixFromAES, 8).error is None

    assert isMDS(matrixFromAES, 1).result is None
    assert isMDS(matrixFromAES, 1).error == "invalid elements"

def test_isMDS_nonMDS():
    nonMDS = [[2, 0, 1, 1], [1, 2, 3, 1], [1, 1, 2, 3], [3, 1, 1, 2]]

    assert isMDS(nonMDS, 8).result == False
    assert isMDS(nonMDS, 8).error is None

    assert isMDS(nonMDS, 1).result is None
    assert isMDS(nonMDS, 1).error == "invalid elements"

def test_isMDS_exception():
    with patch('numpy.linalg.det', side_effect=Exception('Forced exception for testing')):
        result = isMDS([[1, 2], [3, 4]], 8)
        assert result.result is None
        assert result.error is not None