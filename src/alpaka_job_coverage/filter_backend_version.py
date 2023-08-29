"""Filter rules basing on backend names and versions.
"""

import io

from alpaka_job_coverage.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from alpaka_job_coverage.util import (
    row_check_name,
    row_check_version,
    row_check_backend_version,
    backend_is_not_in_row,
    reason,
)
from typing import List, Tuple, Union, Optional
from typeguard import typechecked


def get_required_parameter() -> List[str]:
    """Returns a list of parameters which are required for using the filter defined by this module.

    Returns:
        List[str]: list of parameters
    """
    return [DEVICE_COMPILER, BACKENDS]


@typechecked
def compiler_backend_filter_typed(
    row: List[Union[Tuple[str, str], List[Tuple[str, str]]]],
    output: Optional[Union[io.StringIO, io.TextIOWrapper]] = None,
) -> bool:
    """Type checked version of compiler_backend_filter(). Should be only used for
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
    return compiler_backend_filter(row, output)


def compiler_backend_filter(
    row: List, output: Optional[Union[io.StringIO, io.TextIOWrapper]] = None
) -> bool:
    """Filter rules basing on backend names and versions.

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
    ###########################
    ## gcc device compiler
    ###########################

    if row_check_name(row, DEVICE_COMPILER, "==", GCC):
        if row_check_backend_version(row, ALPAKA_ACC_GPU_CUDA_ENABLE, "!=", OFF_VER):
            reason(
                output,
                "gcc as device compiler cannot compile with enabled "
                "alpaka_ACC_GPU_CUDA_ENABLE back-end",
            )
            return False

        if row_check_backend_version(row, ALPAKA_ACC_GPU_HIP_ENABLE, "!=", OFF_VER):
            reason(
                output,
                "gcc as device compiler cannot compile with enabled "
                "alpaka_ACC_GPU_HIP_ENABLE back-end",
            )
            return False

        if row_check_backend_version(row, ALPAKA_ACC_SYCL_ENABLE, "!=", OFF_VER):
            reason(
                output,
                "gcc as device compiler cannot compile with enabled "
                "alpaka_ACC_SYCL_ENABLE back-end",
            )
            return False

    ###########################
    ## clang device compiler
    ###########################

    if row_check_name(row, DEVICE_COMPILER, "==", CLANG):
        if row_check_backend_version(row, ALPAKA_ACC_GPU_CUDA_ENABLE, "!=", OFF_VER):
            reason(
                output,
                "clang as device compiler cannot compile with enabled "
                "alpaka_ACC_GPU_CUDA_ENABLE back-end (use clang-cuda instead)",
            )
            return False

        # clang cannot compile with enabled HIP backend
        if row_check_backend_version(row, ALPAKA_ACC_GPU_HIP_ENABLE, "!=", OFF_VER):
            reason(
                output,
                "clang as device compiler cannot compile with enabled "
                "alpaka_ACC_GPU_HIP_ENABLE back-end (use hipcc instead)",
            )
            return False

        if row_check_backend_version(row, ALPAKA_ACC_SYCL_ENABLE, "!=", OFF_VER):
            reason(
                output,
                "clang as device compiler cannot compile with enabled "
                "alpaka_ACC_SYCL_ENABLE back-end (use icpx instead)",
            )
            return False

    ###########################
    ## nvcc device compiler
    ###########################

    if row_check_name(row, DEVICE_COMPILER, "==", NVCC):
        # the nvcc compiler needs the same version, like the backend
        # backend_is_not_in_row() is required in case ALPAKA_ACC_GPU_CUDA_ENABLE
        # is not defined
        if backend_is_not_in_row(
            row, ALPAKA_ACC_GPU_CUDA_ENABLE
        ) or row_check_backend_version(
            row,
            ALPAKA_ACC_GPU_CUDA_ENABLE,
            "!=",
            row[param_map[DEVICE_COMPILER]][VERSION],
        ):
            reason(
                output,
                "the nvcc compiler and the CUDA back-end must have the same version",
            )
            return False

        # it is not allowed to enable the HIP and CUDA backend on the same time
        if row_check_backend_version(row, ALPAKA_ACC_GPU_HIP_ENABLE, "!=", OFF_VER):
            reason(
                output,
                "If nvcc is the device compiler and the CUDA back-end is enabled "
                "it is not allowed to enable the HIP back-end",
            )
            return False

        if row_check_backend_version(row, ALPAKA_ACC_SYCL_ENABLE, "!=", OFF_VER):
            reason(
                output,
                "If nvcc is the device compiler it is not allowed to enable "
                "the SYCL back-end",
            )
            return False

    ###########################
    ## clang-cuda device compiler
    ###########################

    if row_check_name(row, DEVICE_COMPILER, "==", CLANG_CUDA):
        # the CUDA backend needs to be enabled
        # backend_is_not_in_row() is required in case ALPAKA_ACC_GPU_CUDA_ENABLE
        # is not defined
        if backend_is_not_in_row(
            row, ALPAKA_ACC_GPU_CUDA_ENABLE
        ) or row_check_backend_version(row, ALPAKA_ACC_GPU_CUDA_ENABLE, "==", OFF_VER):
            reason(
                output,
                "when CLANG_CUDA is set as device compiler "
                "the CUDA back-end must be enabled",
            )
            return False

        # it is not allowed to enable the HIP and CUDA backend on the same time
        if row_check_backend_version(row, ALPAKA_ACC_GPU_HIP_ENABLE, "!=", OFF_VER):
            reason(
                output,
                "if clang-cuda is the device compiler and the CUDA back-end is enabled "
                "it is not allowed to enable the HIP back-end",
            )
            return False

        # clang-cuda doesn't support the SYCL back-end
        if row_check_backend_version(row, ALPAKA_ACC_SYCL_ENABLE, "!=", OFF_VER):
            reason(
                output,
                "if clang-cuda is the device compiler it is not allowed to "
                "enable the SYCL back-end",
            )
            return False

        # check if clang-cuda supports the CUDA SDK version
        clangcuda_cudasdk_versions = [
            ("7", "9.2"),
            ("8", "10.0"),
            ("10", "10.1"),
            ("12", "11.0"),
            ("13", "11.2"),
            ("16", "11.5"),
        ]

        for clang_cuda_version, cuda_sdk_version in clangcuda_cudasdk_versions:
            if row_check_version(
                row, DEVICE_COMPILER, "<=", clang_cuda_version
            ) and row_check_backend_version(
                row, ALPAKA_ACC_GPU_CUDA_ENABLE, ">", cuda_sdk_version
            ):
                reason(
                    output,
                    f"clang-{clang_cuda_version} supports only up to CUDA {cuda_sdk_version}",
                )
                return False

    ###########################
    ## hipcc device compiler
    ###########################

    # the HIP backend needs to be enabled and has the same version number
    if row_check_name(row, DEVICE_COMPILER, "==", HIPCC):
        if backend_is_not_in_row(
            row, ALPAKA_ACC_GPU_HIP_ENABLE
        ) or row_check_backend_version(
            row,
            ALPAKA_ACC_GPU_HIP_ENABLE,
            "!=",
            row[param_map[DEVICE_COMPILER]][VERSION],
        ):
            reason(
                output,
                "the HIP back-end needs to be enabled and hipcc and the "
                "HIP back-end must have the same version number",
            )
            return False

        # it is not allowed to enable the HIP and CUDA backend on the same time
        if row_check_backend_version(row, ALPAKA_ACC_GPU_CUDA_ENABLE, "!=", OFF_VER):
            reason(
                output,
                "if hipcc is the device compiler and the HIP back-end is enabled, "
                "it is not allowed to enable the CUDA back-end",
            )
            return False

        if row_check_backend_version(row, ALPAKA_ACC_SYCL_ENABLE, "!=", OFF_VER):
            reason(
                output,
                "if hipcc is the device compiler it is not allowed to enable "
                "the SYCL back-end",
            )
            return False

    ###########################
    ## icpx device compiler
    ###########################

    # Don't use icpx for the CUDA and HIP back-ends
    if row_check_name(row, DEVICE_COMPILER, "==", ICPX):
        if row_check_backend_version(row, ALPAKA_ACC_GPU_CUDA_ENABLE, "!=", OFF_VER):
            reason(
                output,
                "if icpx is the device compiler it is not allowed to enable the CUDA "
                "back-end",
            )
            return False

        if row_check_backend_version(row, ALPAKA_ACC_GPU_HIP_ENABLE, "!=", OFF_VER):
            reason(
                output,
                "if icpx is the device compiler it is not allowed to enable the HIP "
                "back-end",
            )
            return False
    return True
