"""Filter rules basing on host and device compiler names and versions.
"""

from alpaka_job_coverage.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from alpaka_job_coverage.util import row_check_name, row_check_version, is_in_row

from packaging import version as pk_version
from typing import List, Tuple, Union
from typeguard import typechecked


@typechecked
def compiler_version_filter_typed(
    row: List[Union[Tuple[str, str], List[Tuple[str, str]]]]
) -> bool:
    """Type checked version of compiler_version_filter(). Should be only used for
    testing or tooling. The type check adds a big overhead, which slows down
    pair-wise generator by the factor 30.

    Args:
        row (List[Union[Tuple[str, str], List[Tuple[str, str]]]]): Combination
        to verify. The row can contain up to all combination fields and at least
         two items.

    Returns:
        bool: True, if combination is valid, otherwise False.
    """
    return compiler_version_filter(row)


def compiler_version_filter(row: List) -> bool:
    """Filter rules basing on host and device compiler names and versions.

    Args:
        row (List): Combination to verify. The row can contain
        up to all combination fields and at least two items.

    Returns:
        bool: True, if combination is valid, otherwise False.
    """
    # if the same compiler is used for host and device code, it is not possible to use
    # different versions
    if (
        is_in_row(row, HOST_COMPILER)
        and is_in_row(row, DEVICE_COMPILER)
        and (
            # nvcc is a exception, because the host compiler is different to device compiler
            row_check_name(row, DEVICE_COMPILER, "!=", NVCC)
            and row[param_map[HOST_COMPILER]][VERSION]
            != row[param_map[DEVICE_COMPILER]][VERSION]
        )
    ):
        return False

    if row_check_name(row, DEVICE_COMPILER, "==", NVCC):
        cuda_sdk_version = 0
        cuda_host_compiler_version = 1

        # set the and lowest highest supported gcc version for nvcc
        if row_check_name(row, HOST_COMPILER, "==", GCC):
            combinations = [
                # (maximum_CUDA_SDK_version, "maximum_gcc_version")
                ("12.1", "12"),
                ("12.0", "12"),
                ("11.4", "11"),
                ("11.1", "10"),
                ("11.0", "9"),
                ("10.1", "8"),
                ("10.0", "7"),
            ]

            if pk_version.parse(
                row[param_map[DEVICE_COMPILER]][VERSION]
            ) <= pk_version.parse(combinations[0][cuda_sdk_version]):
                # check the maximum supported compiler version
                for combination in combinations:
                    if pk_version.parse(
                        row[param_map[DEVICE_COMPILER]][VERSION]
                    ) >= pk_version.parse(combination[cuda_sdk_version]):
                        if pk_version.parse(
                            row[param_map[HOST_COMPILER]][VERSION]
                        ) > pk_version.parse(combination[cuda_host_compiler_version]):
                            return False
                        else:
                            break

                # since CUDA 11.4, the minimum supported GCC compiler is GCC 6
                if pk_version.parse(
                    row[param_map[DEVICE_COMPILER]][VERSION]
                ) >= pk_version.parse("11.4") and pk_version.parse(
                    row[param_map[HOST_COMPILER]][VERSION]
                ) < pk_version.parse(
                    "6"
                ):
                    return False

        if row_check_name(row, HOST_COMPILER, "==", CLANG):
            # disable clang as host compiler for nvcc 11.3 until 11.5
            if row_check_version(
                row, DEVICE_COMPILER, ">=", "11.3"
            ) and row_check_version(row, DEVICE_COMPILER, "<=", "11.5"):
                return False

            combinations = [
                # (maximum_CUDA_SDK_version, "maximum_clang_version")
                ("12.1", "15"),
                ("12.0", "14"),
                ("11.6", "13"),
                ("11.4", "12"),
                ("11.2", "11"),
                ("11.1", "10"),
                ("11.0", "9"),
                ("10.1", "8"),
                ("10.0", "6"),
            ]

            if pk_version.parse(
                row[param_map[DEVICE_COMPILER]][VERSION]
            ) <= pk_version.parse(combinations[0][cuda_sdk_version]):
                # set the and lowest highest supported clang version for nvcc
                # check the maximum supported compiler version
                for combination in combinations:
                    if pk_version.parse(
                        row[param_map[DEVICE_COMPILER]][VERSION]
                    ) >= pk_version.parse(combination[cuda_sdk_version]):
                        if pk_version.parse(
                            row[param_map[HOST_COMPILER]][VERSION]
                        ) > pk_version.parse(combination[cuda_host_compiler_version]):
                            return False
                        else:
                            break

    if row_check_name(row, DEVICE_COMPILER, "==", CLANG_CUDA):
        # disable all clang versions older than 14 as CUDA Compiler
        if row_check_version(row, DEVICE_COMPILER, "<", "14"):
            return False

    return True
