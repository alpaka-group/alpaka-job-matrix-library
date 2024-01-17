from collections import OrderedDict
from copy import deepcopy
from typing import Dict, Tuple, Callable
from dataclasses import dataclass
from typeguard import typechecked
from packaging.version import Version
import packaging.version as pkv

from alpaka_job_coverage.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import

from allpairspy import AllPairs


# The filter adapter convert the type of the row argument of the filter function
# from a list to a OrderedDict
class FilterAdapter:
    @typechecked
    def __init__(self, param_map: Dict[int, str], filter: Callable):
        self.param_map = param_map
        self.filter = filter

    def __call__(self, row) -> bool:
        ordered_row = OrderedDict()
        for index in range(len(row)):
            ordered_row[self.param_map[index]] = row[index]
        return self.filter(ordered_row)


@typechecked
def get_matrix(
    parameter: OrderedDict, filter_func: Callable
) -> List[OrderedDict[str, Tuple[str, Version]]]:
    """Generates a list of test parameter sets via pair wise combination.

    Args:
        parameter (OrderedDict): Combination parameter
        filter_func (Callable): Filter function

    Returns:
        List[OrderedDict[str, Tuple[str, str]]]: List test parameters
    """
    param_map: Dict[int, str] = {}
    for index, key in enumerate(parameter.keys()):
        param_map[index] = key

    filter = FilterAdapter(param_map, filter_func)

    result: List[OrderedDict[str, Tuple[str, Version]]] = []

    # convert AllPair specific data types to a generic Python data structure
    # with OrderedDict to provide easier API
    for parameter_set in AllPairs(parameters=parameters, n=2, filter_func=filter):
        tmp_entry = OrderedDict()
        for param_index in range(len(parameter_set)):
            tmp_entry[param_map[param_index]] = parameter_set[param_index]
        result.append(tmp_entry)

    return result


# no filter rules implemented yet
def custom_filter(row: OrderedDict[str, Tuple[str, Version]]):
    # only for demonstration
    # if (
    #     DEVICE_COMPILER in row
    #     and row[DEVICE_COMPILER][NAME] == NVCC
    #     and row[DEVICE_COMPILER][VERSION] < pkv.parse("12.0")
    # ):
    #     return False
    return True


@dataclass
class ExpectedResult:
    """Only a container object to simplify the code.

    `v1_*` is the first set of expected parameters and `v2_*` is the second set
    of expected parameters for a expected parameter pair.
    """

    v1_parameter_name: str
    v1_name: str
    v1_version: Version
    v2_parameter_name: str
    v2_name: str
    v2_version: Version

    def __str__(self) -> str:
        return (
            f"{self.v1_parameter_name} : ({self.v1_name}, {self.v1_version}) -> "
            + f"{self.v2_parameter_name} : ({self.v2_name}, {self.v2_version})"
        )


@typechecked
def generate_control_matrix(
    parameters: OrderedDict[str, Tuple[str, Version]]
) -> List[ExpectedResult]:
    """Generates a list of expected tuple for verification the algorithm. Pair
    wise testing means, that each pair of two parameter sets needs to appears in
    a least on combination. Therefore we can generate the full matrix of
    expected pairs, if there is no filter rule.

    Args:
        parameters (OrderedDict[str, Tuple[str, str]]): Input parameter for the
            pair wise generation.

    Returns:
        List[ExpectedResult]: List of expected pairs.
    """
    expectedResults: List[ExpectedResult] = []

    number_of_keys: int = len(parameters.keys())
    param_map: Dict[int, str] = {}
    for index, key in enumerate(parameters.keys()):
        param_map[index] = key

    for v1_index in range(number_of_keys):
        for v2_index in range(v1_index + 1, number_of_keys):
            loop_over_parameter_values(
                parameters, expectedResults, param_map[v1_index], param_map[v2_index]
            )

    return expectedResults


@typechecked
def loop_over_parameter_values(
    parameters: OrderedDict[str, Tuple[str, Version]],
    expectedResults: List[ExpectedResult],
    v1_parameter_name: str,
    v2_parameter_name: str,
):
    """Creates all combinations of two parameter sets given by the parameter
    name.

    Args:
        parameters (OrderedDict[str, Tuple[str, str]]): List of parameters
            expectedResults (List[ExpectedResult]): List of expected Tuples.
        New expected Tuples will be added.
        v1_parameter_name (str): Name of the first parameter.
        v2_parameter_name (str): Name of the second parameter.
    """
    for v1_name, v1_version in parameters[v1_parameter_name]:
        for v2_name, v2_version in parameters[v2_parameter_name]:
            expectedResults.append(
                ExpectedResult(
                    v1_parameter_name,
                    v1_name,
                    v1_version,
                    v2_parameter_name,
                    v2_name,
                    v2_version,
                )
            )


@typechecked
def check_for_results(
    parameter_matrix: List[OrderedDict[str, Tuple[str, Version]]],
    expectedResults: List[ExpectedResult],
) -> bool:
    """Check if all expected Results are included in parameter_matrix

    Args:
        parameter_matrix (List[OrderedDict[str, Tuple[str, str]]]): List of test parameters
        expectedResults (List[ExpectedResult]): Expected tuples

    Returns:
        bool: True if all expected tuples was found
    """
    all_right = True

    for eRes in expectedResults:
        v1_name_version = (eRes.v1_name, eRes.v1_version)
        v2_name_version = (eRes.v2_name, eRes.v2_version)
        found = False
        for parameter in parameter_matrix:
            if (
                parameter[eRes.v1_parameter_name] == v1_name_version
                and parameter[eRes.v2_parameter_name] == v2_name_version
            ):
                found = True
                break

        if not found:
            print(f"could not find expected result: \n{eRes}")
            all_right = False

    return all_right


def get_parameters() -> OrderedDict[str, Version]:
    parameters = OrderedDict()
    parameters[HOST_COMPILER] = [
        (GCC, 9),
        (GCC, 10),
        (GCC, 11),
        (GCC, 12),
        (CLANG, 13),
        (CLANG, 14),
        (CLANG, 15),
        (NVCC, 11.2),
        (NVCC, 11.4),
        (NVCC, 12.2),
    ]
    parameters[DEVICE_COMPILER] = deepcopy(parameters[HOST_COMPILER])
    parameters[CMAKE] = [
        (CMAKE, "3.22"),
        (CMAKE, "3.23"),
        (CMAKE, "3.24"),
        (CMAKE, "3.25"),
    ]
    parameters[BOOST] = [
        (BOOST, "1.74.0"),
        (BOOST, "1.75.0"),
        (BOOST, "1.76.0"),
        (BOOST, "1.77.0"),
        (BOOST, "1.78.0"),
    ]

    typed_parameter: OrderedDict[str, Version] = OrderedDict()

    # parse str, int and float to package.version.Version
    for parameter_name in parameters.keys():
        typed_parameter[parameter_name] = []
        for parameter_set in parameters[parameter_name]:
            typed_parameter[parameter_name].append(
                (
                    parameter_set[NAME],
                    pkv.parse(str(parameter_set[VERSION])),
                )
            )

    return typed_parameter


if __name__ == "__main__":
    parameters: OrderedDict[str, Version] = get_parameters()

    cover_matrix = get_matrix(parameter=parameters, filter_func=custom_filter)
    expectedResults = generate_control_matrix(parameters)

    exit(not check_for_results(cover_matrix, expectedResults))
