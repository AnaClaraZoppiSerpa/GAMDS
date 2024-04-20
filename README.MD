# Dependencies

```
pip3 install numpy
pip3 install galois
pip3 install pytest
pip3 install pytest-cov
```

# What you should use
Use only the following functions from `readonly.py`.

- `validUpper(fieldArg: int) -> int`
- `validLower(fieldArg: int) -> int`
- `isValidInteger(fieldArg: int, element: int) -> bool`
- `validIntegers(fieldArg: int) -> List[int]`
- `combinedCost(intMatrix: List[List[int]], fieldArg: int) -> CostResult`
- `xtOnlyCost(intMatrix: List[List[int]], fieldArg: int) -> CostResult`
- `baselineCombinedCost(dimension: int) -> BaselineResult`
- `baselineXtOnlyCost(dimension: int) -> BaselineResult`
- `existsInDataset(intMatrix: List[List[int]]) -> ExistsResult`
- `isMDS(intMatrix: List[List[int]], fieldArg: int) -> CheckerResult`

Don't modify them or any other functions from `readonly.py`.

Functions from `readonly.py` prefixed with _ are helpers only, you should not need to call them.

If you want to change anything, do so in `customization.py`, copying the original from `readonly.py` and modifying it there.

For example, you can implement the MDS checker with score there instead of modifying the one from `readonly.py`.

> If you have any questions, reach out to Ana.

Note: `CostResult`, `BaselineResult`, `ExistsResult`, `CheckerResult` are just wrappers for functions that can either return an error, some extra information, or a result when successful. Study `readonly.py` for more details, I've left comments there.

You can ignore the `helpers` directory, it's just finite field arithmetic, XOR, XTIME and dataset stuff that is already wrapped in `readonly.py`.

# When you should use

These will help you if you're generating each matrix element at a time with some strategy but want to make sure you're not generating invalid elements.
For example, if you've generated a random integer x, you must check if it's valid with isValidInteger.
Or you can provide the validUpper and validLower limits to a ranged random function to make sure x stays within valid limits.

```
validUpper(fieldArg: int) -> int
validLower(fieldArg: int) -> int
isValidInteger(fieldArg: int, element: int) -> bool
```

Alternatively, you can just retrieve the list of valid integers for that finite field and work with it.
For example, sampling randomly from this list, or following any criteria e.g. higher elements first, lower elements first etc.

```
validIntegers(fieldArg: int) -> List[int]
```

These are to calculate costs. "Combined cost" and "weighted cost" mean the same thing, I use these terms interchangeably.
combinedCost is 3xtime + 1xor.
xtOnly is xtime.

```
combinedCost(intMatrix: List[List[int]], fieldArg: int) -> CostResult
xtOnlyCost(intMatrix: List[List[int]], fieldArg: int) -> CostResult
```

These are to retrieve baseline costs for each metric.
This way you can compare your newly generated matrix cost with the pre-existing matrices costs.
When your cost < baseline cost means you found a good matrix, if it is MDS.
Make sure you are using the same metrics! Don't mix up combinedCost and xtOnlyCost!

```
baselineCombinedCost(dimension: int) -> BaselineResult
baselineXtOnlyCost(dimension: int) -> BaselineResult
```

This is to check if your matrix is a matrix that has already been reported in the literature before.

```
existsInDataset(intMatrix: List[List[int]]) -> ExistsResult
```

This evaluates if your matrix is MDS or not.

```
isMDS(intMatrix: List[List[int]], fieldArg: int) -> CheckerResult
```

# Suggested parameters

`fieldArgs` in {4,8}: classical cryptographic algorithms applicability. Most recommended.

`fieldArgs` from {2,3,5,6,7,9,10,11,12}: alternative fields. Not very applicable for cryptographic algorithm design, but not very explored. Could be interesting, you will definitely find MDS matrices that haven't been reported in the literature yet.

`dimension` from {2,3,4,5,6,7,8}: bigger matrices are not MDS testable in a personal laptop.

# Testing

It might be helpful to make sure you didn't break anything in the `readonly.py` file once in a while. 

Run the unit tests frequently with `pytest --cov=readonly test_readonly.py` on the root directory.

I also encourage you to write unit tests for anything you choose to customize on `customization.py` to ensure its correct behavior.

