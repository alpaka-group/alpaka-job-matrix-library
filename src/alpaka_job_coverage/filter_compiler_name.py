"""Filter rules basing on host and device compiler names.
"""

import io

from alpaka_job_coverage.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from alpaka_job_coverage.util import row_check_name, is_in_row, reason
from typing import List, Tuple, Union, Optional
from typeguard import typechecked


def get_required_parameter() -> List[str]:
    """Returns a list of parameters which are required for using the filter defined by this module.
    Returns:
        List[str]: list of parameters
    """
    return [HOST_COMPILER, DEVICE_COMPILER]


@typechecked
def general_compiler_filter_typed(
    row: List[Union[Tuple[str, str], List[Tuple[str, str]]]],
    output: Optional[Union[io.StringIO, io.TextIOWrapper]] = None,
) -> bool:
    """Type checked version of general_compiler_filter(). Should be only used for
    testing or tooling. The type check adds a big overhead, which slows down
    pair-wise generator by the factor 30.

    Args:
        row (List[Union[Tuple[str, str], List[Tuple[str, str]]]]): Combination
        to verify. The row can contain up to all combination fields and at least
        two items.
        output (Optional[Union[io.StringIO, io.TextIOWrapper]]): Write
        additional information about filter decisions to the IO object
        (io.SringIO, sys.stdout, sys.stderr). If it is None no information is
        generated.

    Returns:
        bool: True, if combination is valid, otherwise False.
    """
    return general_compiler_filter(row, output)


def general_compiler_filter(
    row: List, output: Optional[Union[io.StringIO, io.TextIOWrapper]] = None
) -> bool:
    """Filter rules basing on host and device compiler names.

    Args:
        row (List): Combination to verify. The row can contain
        up to all combination fields and at least two items.
        output (Optional[Union[io.StringIO, io.TextIOWrapper]]): Write
        additional information about filter decisions to the IO object
        (io.SringIO, sys.stdout, sys.stderr). If it is None no information is
        generated.

    Returns:
        bool: True, if combination is valid, otherwise False.
    """

    # it is not allow to use the nvcc as host compiler
    if row_check_name(row, HOST_COMPILER, "==", NVCC):
        reason(output, "nvcc is not allowed as host compiler")
        return False

    # only the nvcc allows to combine different host and device compiler
    if (
        is_in_row(row, HOST_COMPILER)
        and is_in_row(row, DEVICE_COMPILER)
        and (
            row[param_map[DEVICE_COMPILER]][NAME] != NVCC
            and row[param_map[HOST_COMPILER]][NAME]
            != row[param_map[DEVICE_COMPILER]][NAME]
        )
    ):
        reason(output, "host and device compiler must be the same (except for nvcc)")
        return False

    # only clang and gcc are allowed as nvcc host compiler
    if row_check_name(row, DEVICE_COMPILER, "==", NVCC) and not (
        row_check_name(row, HOST_COMPILER, "==", GCC)
        or row_check_name(row, HOST_COMPILER, "==", CLANG)
    ):
        reason(output, "only clang and gcc are allowed as nvcc host compilers")
        return False

    return True
