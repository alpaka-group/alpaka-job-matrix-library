"""Different support functions.

The row type: Several utile functions to do different checks on the row 
parameters. The type of the row is defined by the pairwise library and the 
parameter types.
Each element of the row have ether the type of Tuple[str, str], where the first 
string is a name and the second string is a version number or the type 
List[Tuple[str, str]], where tuple also stores a name and version combination. 
The list type is only used for the backend values of the `backends` parameter.
The position of the value decides, which parameter a value represent. For 
example the first and second value can be ("gcc", "9"). Only the position 
decides if the first value is the host compiler and second value the device 
compiler or vice versa. 
The global variable `param_map` implements a indirection which allows to 
separate the ordering of the parameters from the algorithm. The `param_map` 
needs to be initialized by the user at each application/test start. 
For Example:
    param_map[HOST_COMPILER] = 0
    param_map[DEVICE_COMPILER] = 1

Example for a row:
    [('gcc', '6'), ('nvcc', '11.4'), 
    [
        ('alpaka_ACC_CPU_B_SEQ_T_SEQ_ENABLE', '1.0.0'), 
        ('alpaka_ACC_CPU_B_OMP2_T_SEQ_ENABLE', '0.0.0'), 
        ('alpaka_ACC_GPU_CUDA_ENABLE', '11.4'), 
        ('alpaka_ACC_GPU_HIP_ENABLE', '0.0.0')
    ], 
    ('cmake', '3.19.8'), ('boost', '1.66.0'), ('alpaka', '0.6.1'), 
    ('ubuntu', '20.04')]
"""

import operator, re
from typing import Any, List, Dict, Tuple
from typeguard import typechecked
from packaging import version as pk_version


from alpaka_job_coverage.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import

# maps strings to comparision operators
OPERATOR_MAP = {
    "==": operator.eq,
    "!=": operator.ne,
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
}


# no typechecked, because function is performance critical
def strict_equal(first_value: Any, second_value: Any) -> bool:
    """Compare types and values of a and b. If the types are different,
    throw error. If the types are equal the result is a bool.

    Args:
        first_value (Any): first value to compare
        second_value (Any): second value to compare

    Raises:
        TypeError: Is thrown, if the types of a and b are different.

    Returns:
        bool: True if values are equal, otherwise False.
    """
    if not isinstance(first_value, type(second_value)):
        raise TypeError(
            f"a and b has not the same type: {type(first_value)} != {type(second_value)}"
        )

    return first_value == second_value


def is_in_row(row: List, name: str) -> bool:
    """Check if paramater is in the row.

    Args:
        row (List): Row with parameters.
        name (str): The searched parameter.

    Returns:
        bool: Return True, if parameter is in row.
    """
    return param_map[name] < len(row)


# no typechecked, because function is performance critical
def row_check_name(row: List, colum: str, opr: str, name: str) -> bool:
    """Check if colum is in row and if the name matches or not, depending
    of the operator.

    Args:
        row (List): Row to check.
        colum (str): Colum name in the row.
        opr (str): The operator can be "==" (equal) or "!=" (not equal).
        name (str): Name to compare.

    Raises:
        ValueError: Raise error, if operator does not have the value "==" or "!=".

    Returns:
        bool: Return False, if column is not in the row. If the column is in the row,
        return True if the name matches ("==") or not matches ("!=").
    """
    if not opr in ("==", "!="):
        raise ValueError("op (operator) needs to be == or !=")

    return is_in_row(row, colum) and OPERATOR_MAP[opr](
        row[param_map[colum]][NAME], name
    )


# no typechecked, because function is performance critical
def row_check_version(
    row: List,
    colum: str,
    opr: str,
    version: str,
) -> bool:
    """Check if colum is in row and if the version matches or not, depending
    of the operator.

    Args:
        row (List): Row to check.
        colum (str): Colum name in the row.
        opr (str): The operator can be "==", "!=", "<", "<=", ">" and ">=".
        version (str): Version to compare.

    Raises:
        ValueError: Raise error, if operator does not have the supported value.

    Returns:
        bool: Return False, if column is not in the row. If the column is in the row,
        return True if the version.
    """
    if not opr in OPERATOR_MAP:
        raise ValueError(f"operator needs to be: {', '.join(OPERATOR_MAP.keys())}")

    return is_in_row(row, colum) and OPERATOR_MAP[opr](
        pk_version.parse(row[param_map[colum]][VERSION]), pk_version.parse(version)
    )


def backend_is_not_in_row(row: List, backend: str) -> bool:
    """Returns True, if backend is not in backend list.

    Args:
        row (List): Row to check.
        backend (str): Name of the backend, which version should be compared.

    Returns:
        bool: Return True, if backend is not in the row. If backend is in the row and
        the backend name is in the backend list return False else True.
    """

    if not is_in_row(row, BACKENDS):
        return True
    else:
        for row_backend in row[param_map[BACKENDS]]:
            if row_backend[NAME] == backend:
                return False
    return True


# no typechecked, because function is performance critical
def row_check_backend_version(row: List, backend: str, opr: str, version: str) -> bool:
    """Check, if backend exists and if the backend version matches depending of the operator.

    Args:
        row (List): Row to check.
        backend (str): Name of the backend, which version should be compared.
        opr (str): The operator can be "==", "!=", "<", "<=", ">" and ">=".
        version (str): Version to compare.

    Raises:
        ValueError: Raise error, if operator does not have the supported value.

    Returns:
        bool: Return False, if backend does not exist. If the backend name is in the row, return
        True if the version matches.
    """
    if not opr in OPERATOR_MAP:
        raise ValueError(f"operator needs to be: {', '.join(OPERATOR_MAP.keys())}")

    if not is_in_row(row, BACKENDS):
        return False

    for row_backend in row[param_map[BACKENDS]]:
        if row_backend[NAME] == backend:
            return OPERATOR_MAP[opr](
                pk_version.parse(row_backend[VERSION]), pk_version.parse(version)
            )

    return False


@typechecked
def search_and_move_job(
    job_matrix: List[Dict[str, Tuple[str, str]]],
    searched_job: Dict[str, Tuple[str, str]],
    position: int = 0,
) -> bool:
    """Search job, which contains all items of searched_job and move it to list position.

    Args:
        job_matrix (List[Dict[str, Tuple[str, str]]]): The job matrix.
        searched_job (Dict[str, Tuple[str, str]]): Dict of searched items. If all items matches with
        an entry in job list, move entry to position
        position (int, optional): New position of matched entry in job_matrix. Defaults to 0.

    Raises:
        IndexError: Raise error, if searched_job dict is empty.

    Returns:
        bool: True if found and move entry, otherwise False. If False, job_matrix was not modified.
    """
    if len(searched_job) == 0:
        raise IndexError("searched_job must not be empty")

    for index, job_combination in enumerate(job_matrix):
        matched_attributes = 0
        for attribute_name, attribute_value in searched_job.items():
            if (
                attribute_name in job_combination
                and job_combination[attribute_name] == attribute_value
            ):
                matched_attributes += 1
            if matched_attributes == len(searched_job):
                job_matrix.insert(position, job_matrix.pop(index))
                return True
    return False


@typechecked
def filter_job_list(
    job_matrix: List[Dict[str, Dict]], job_name_regex: str
) -> List[Dict[str, Dict]]:
    """Returns list, where all job names matches the job_name_regex.

    Args:
        job_matrix (List[Dict[str, Dict]]): Input job matrix.
        job_name_regex (str): Regex to match. See Python regex.

    Returns:
        List[Dict[str, Dict]]: Filtered job matrix.
    """
    compiled_regex = re.compile(job_name_regex)
    filtered_jobs: List[Dict[str, Dict]] = []

    for job in job_matrix:
        if compiled_regex.match(list(job.keys())[0]):
            filtered_jobs.append(job)

    return filtered_jobs


@typechecked
def reorder_job_list(
    job_matrix: List[Dict[str, Dict]], job_name_regex: str
) -> List[Dict[str, Dict]]:
    """Reorder list with a list of regex. The ordering of the regex in the job_name_regex will be
    also the ordering of the return job matrix.

    For example, the job_name_regex string "^NVCC ^GCC" has the behavior that all NVCC jobs will be
    the first items in the returned job matrix, than all GCC job will follow and than all jobs,
    which does not match the two regex.

    Args:
        job_matrix (List[Dict[str, Dict]]): Input job matrix.
        job_name_regex (str): List of regex, separated by whitespaces. E.g. "^NVCC ^Clang|^GCC ^HIP"

    Returns:
        List[Dict[str, Dict]]: Reordered job matrix.
    """
    # each regex is separated by a whitespace
    ordering_list = job_name_regex.strip().split(" ")

    # reverse list, because reorder_job_list_single_regex() puts matched jobs in the beginning
    ordering_list.reverse()

    tmp_job_matrix = job_matrix
    for regex in ordering_list:
        tmp_job_matrix = reorder_job_list_single_regex(tmp_job_matrix, regex)

    return tmp_job_matrix


@typechecked
def reorder_job_list_single_regex(
    job_matrix: List[Dict[str, Dict]], job_name_regex: str
) -> List[Dict[str, Dict]]:
    """Reorder list with a regex. Put all jobs in the beginning, which names matches the regex. Then
    all other jobs will following.

    Args:
        job_matrix (List[Dict[str, Dict]]): Input job matrix.
        job_name_regex (str): Regex to match. See Python regex.

    Returns:
        List[Dict[str, Dict]]: Reordered job matrix.
    """
    compiled_regex = re.compile(job_name_regex)
    index_list: List[int] = []
    new_job_list: List[Dict[str, Dict]] = []

    for index, job in enumerate(job_matrix):
        if compiled_regex.match(list(job.keys())[0]):
            index_list.append(index)

    for index, job in enumerate(job_matrix):
        if index in index_list:
            new_job_list.insert(0, job)
        else:
            new_job_list.append(job)

    return new_job_list


@typechecked
def is_supported_sw_version(name: str, version: str, verbose=True) -> bool:
    def warning_text(text: str):
        return "\033[1;33mWARNING: " + text + "\033[0m"

    support_versions: Dict[str, Tuple[str, str]] = {
        GCC: ("5", "13"),
        CLANG: ("6.0", "15"),
        NVCC: ("10.0", "12.1"),
        HIPCC: ("4.3", "5.1"),
        CMAKE: ("3.18", "3.22"),
        BOOST: ("1.66.0", "1.78.0"),
        CXX_STANDARD: ("14", "20"),
    }

    if name not in support_versions:
        if verbose:
            print(warning_text(f"{name} is an unknown software"))
        return False
    else:
        parsed_version = pk_version.parse(version)
        if parsed_version < pk_version.parse(
            support_versions[name][0]
        ) or parsed_version > pk_version.parse(support_versions[name][1]):
            if verbose:
                print(warning_text(f"{name} {version} is not supported"))
            return False

    return True
