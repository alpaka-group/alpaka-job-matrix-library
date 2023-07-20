"""Filter rules handling software dependencies and compiler settings.
"""

from alpaka_job_coverage.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from alpaka_job_coverage.util import (
    row_check_backend_version,
    row_check_name,
    row_check_version,
    is_in_row,
)

from packaging import version as pk_version
from typing import List, Tuple, Union
from typeguard import typechecked


@typechecked
def software_dependency_filter_typed(
    row: List[Union[Tuple[str, str], List[Tuple[str, str]]]]
) -> bool:
    """Type checked version of software_dependency_filter(). Should be only used for
    testing or tooling. The type check adds a big overhead, which slows down
    pair-wise generator by the factor 30.

    Args:
        row (List[Union[Tuple[str, str], List[Tuple[str, str]]]]): Combination
        to verify. The row can contain up to all combination fields and at least
         two items.

    Returns:
        bool: True, if combination is valid, otherwise False.
    """
    return software_dependency_filter(row)


def software_dependency_filter(row: List) -> bool:
    """Filter rules handling software dependencies and compiler settings.

    Args:
        row (List): Combination to verify. The row can contain
        up to all combination fields and at least two items.

    Returns:
        bool: True, if combination is valid, otherwise False.
    """

    # GCC 6 and below is not available on Ubuntu 20.04
    if row_check_version(row, UBUNTU, "==", "20.04"):
        if (
            row_check_name(row, HOST_COMPILER, "==", GCC)
            and int(row[param_map[HOST_COMPILER]][VERSION]) <= 6
        ):
            return False

    # GCC 9 and older does not support -std=c++20
    if (
        row_check_version(row, CXX_STANDARD, ">=", "20")
        and row_check_name(row, HOST_COMPILER, "==", GCC)
        and row_check_version(row, HOST_COMPILER, "<=", "9")
    ):
        return False

    if row_check_name(row, DEVICE_COMPILER, "==", NVCC) and is_in_row(
        row, CXX_STANDARD
    ):
        parsed_nvcc_version = pk_version.parse(row[param_map[DEVICE_COMPILER]][VERSION])

        # definition of the tuple values: if the nvcc version of the first
        # tuple is older than the cxx standard of the second value, it is not supported
        nvcc_cxx_versions = [
            ("11.0", 17),  # NVCC versions older than 11.0 does not support C++ 17
            ("12.0", 20),  # NVCC versions older than 12.0 does not support C++ 20
            ("12.2", 23),  # NVCC 12.2 is not released yet, therefore we need to
            # expect that it could support C++23
        ]
        for nvcc_version, cxx_version in nvcc_cxx_versions:
            if (
                parsed_nvcc_version < pk_version.parse(nvcc_version)
                and int(row[param_map[CXX_STANDARD]][VERSION]) >= cxx_version
            ):
                return False

    # clang 11 and 12 are not available in the Ubuntu 18.04 ppa
    if (
        row_check_version(row, UBUNTU, "==", "18.04")
        and (
            row_check_name(
                row,
                HOST_COMPILER,
                "==",
                CLANG,
            )
            or row_check_name(
                row,
                HOST_COMPILER,
                "==",
                CLANG_CUDA,
            )
        )
        and (
            (
                row_check_version(row, HOST_COMPILER, "==", "11")
                or row_check_version(row, HOST_COMPILER, "==", "12")
            )
        )
    ):
        return False

    # Clang 9 and older does not support -std=c++20
    if row_check_version(row, CXX_STANDARD, ">=", "20"):
        for compiler_name in [CLANG, CLANG_CUDA]:
            if row_check_name(
                row, HOST_COMPILER, "==", compiler_name
            ) and row_check_version(row, HOST_COMPILER, "<=", "9"):
                return False

    # ubuntu 18.04 containers are not available for CUDA 11.0 and later
    if (
        row_check_version(row, UBUNTU, "==", "18.04")
        and row_check_backend_version(row, ALPAKA_ACC_GPU_CUDA_ENABLE, "!=", OFF_VER)
        and row_check_backend_version(row, ALPAKA_ACC_GPU_CUDA_ENABLE, ">=", "11.0")
    ):
        return False

    # ubuntu 20.04 containers are not available for CUDA 10.2 and before
    if (
        row_check_version(row, UBUNTU, "==", "20.04")
        and row_check_backend_version(row, ALPAKA_ACC_GPU_CUDA_ENABLE, "!=", OFF_VER)
        and row_check_backend_version(row, ALPAKA_ACC_GPU_CUDA_ENABLE, "<", "11.0")
    ):
        return False

    # all rocm images are Ubuntu 20.04 based
    if (
        row_check_version(row, UBUNTU, "!=", "20.04")
        and row_check_name(row, DEVICE_COMPILER, "==", HIPCC)
        and row_check_backend_version(row, ALPAKA_ACC_GPU_HIP_ENABLE, "!=", OFF_VER)
    ):
        return False

    # a bug in CMAKE 3.18 avoids the correct usage of the variable CMAKE_CUDA_ARCHITECTURE if the
    # CUDA compiler is Clang++
    if row_check_name(row, DEVICE_COMPILER, "==", CLANG_CUDA) and row_check_version(
        row, CMAKE, "<", "3.19"
    ):
        return False

    # disable nvcc 11.0-11.3 + gcc 10 + Ubuntu 20.04
    # Ubuntu 20.04 provides only gcc 10.3 and not 10.4 or 10.5
    # this combination does not work: https://github.com/alpaka-group/alpaka/issues/1297
    if (
        row_check_name(row, DEVICE_COMPILER, "==", NVCC)
        and (
            row_check_version(row, DEVICE_COMPILER, "==", "11.0")
            or row_check_version(row, DEVICE_COMPILER, "==", "11.1")
            or row_check_version(row, DEVICE_COMPILER, "==", "11.2")
            or row_check_version(row, DEVICE_COMPILER, "==", "11.3")
        )
        and row_check_name(row, HOST_COMPILER, "==", GCC)
        and row_check_version(row, HOST_COMPILER, "==", "10")
        and row_check_version(row, UBUNTU, "==", "20.04")
    ):
        return False

    return True
