"""Filter rules basing on backend names and versions.
"""

import io

from alpaka_job_coverage.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from alpaka_job_coverage.util import (
    row_check_name,
    row_check_version,
    row_check_backend_version,
    reason,
)
from typing import List, Tuple, Union, Optional
from typeguard import typechecked


def get_required_parameter() -> List[str]:
    """Return a list of parameter names, which are required to use the filter
    of this module.

    Returns:
        List[str]: list of parameter names
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
        (io.SringIO, sys.stdout, sys.stderr). If it is None, no information are
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
        (io.SringIO, sys.stdout, sys.stderr). If it is None, no information are
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
                "gcc as device compiler cannot compile with enabled alpaka_ACC_GPU_CUDA_ENABLE backend",
            )
            return False

        if row_check_backend_version(row, ALPAKA_ACC_GPU_HIP_ENABLE, "!=", OFF_VER):
            reason(
                output,
                "gcc as device compiler cannot compile with enabled alpaka_ACC_GPU_HIP_ENABLE backend",
            )
            return False

    ###########################
    ## clang device compiler
    ###########################

    if row_check_name(row, DEVICE_COMPILER, "==", CLANG):
        if row_check_backend_version(row, ALPAKA_ACC_GPU_CUDA_ENABLE, "!=", OFF_VER):
            reason(
                output,
                "clang as device compiler cannot compile with enabled alpaka_ACC_GPU_CUDA_ENABLE backend",
            )
            return False

        # clang cannot compile with enabled HIP backend
        if row_check_backend_version(row, ALPAKA_ACC_GPU_HIP_ENABLE, "!=", OFF_VER):
            reason(
                output,
                "clang as device compiler cannot compile with enabled alpaka_ACC_GPU_HIP_ENABLE backend",
            )
            return False

    ###########################
    ## nvcc device compiler
    ###########################

    if row_check_name(row, DEVICE_COMPILER, "==", NVCC):
        # the nvcc compiler needs the same version, like the backend
        if row_check_backend_version(
            row,
            ALPAKA_ACC_GPU_CUDA_ENABLE,
            "!=",
            row[param_map[DEVICE_COMPILER]][VERSION],
        ):
            reason(
                output, "nvcc compiler and CUDA backend needs to have the same version"
            )
            return False

        # it is not allowed to enable the HIP and CUDA backend on the same time
        if row_check_backend_version(row, ALPAKA_ACC_GPU_HIP_ENABLE, "!=", OFF_VER):
            reason(
                output,
                "if nvcc is the device compiler and the CUDA backend is enabled, "
                "it is not allowed to enable the HIP backend",
            )
            return False

    ###########################
    ## clang-cuda device compiler
    ###########################

    if row_check_name(row, DEVICE_COMPILER, "==", CLANG_CUDA):
        # the CUDA backend needs to be enabled
        if row_check_backend_version(row, ALPAKA_ACC_GPU_CUDA_ENABLE, "==", OFF_VER):
            reason(
                output,
                "when CLANG_CUDA is set das device compiler, "
                "the CUDA backend needs to be enabled",
            )
            return False

        # it is not allowed to enable the HIP and CUDA backend on the same time
        if row_check_backend_version(row, ALPAKA_ACC_GPU_HIP_ENABLE, "!=", OFF_VER):
            reason(
                output,
                "if clang-cuda is the device compiler and the CUDA backend is enabled, "
                "it is not allowed to enable the HIP backend",
            )
            return False

        # TODO: simplify me
        if row_check_version(
            row, DEVICE_COMPILER, "<=", "7"
        ) and row_check_backend_version(row, ALPAKA_ACC_GPU_CUDA_ENABLE, ">", "9.2"):
            reason(output, "clang 7 supports only up to CUDA 9.2")
            return False

        if row_check_version(
            row, DEVICE_COMPILER, "<=", "8"
        ) and row_check_backend_version(row, ALPAKA_ACC_GPU_CUDA_ENABLE, ">", "10.0"):
            reason(output, "clang 8 supports only up to CUDA 10.0")
            return False

        if row_check_version(
            row, DEVICE_COMPILER, "<=", "10"
        ) and row_check_backend_version(row, ALPAKA_ACC_GPU_CUDA_ENABLE, ">", "10.1"):
            reason(output, "clang 10 supports only up to CUDA 10.1")
            return False

        if row_check_version(
            row, DEVICE_COMPILER, "<=", "12"
        ) and row_check_backend_version(row, ALPAKA_ACC_GPU_CUDA_ENABLE, ">", "11.0"):
            reason(output, "clang 12 supports only up to CUDA 11.0")
            return False

        if row_check_version(
            row, DEVICE_COMPILER, "<=", "13"
        ) and row_check_backend_version(row, ALPAKA_ACC_GPU_CUDA_ENABLE, ">", "11.2"):
            reason(output, "clang 13 supports only up to CUDA 11.2")
            return False

        if row_check_version(
            row, DEVICE_COMPILER, "<=", "15"
        ) and row_check_backend_version(row, ALPAKA_ACC_GPU_CUDA_ENABLE, ">", "11.5"):
            reason(output, "clang 15 supports only up to CUDA 11.5")
            return False
        
        if row_check_version(
            row, DEVICE_COMPILER, "<=", "16"
        ) and row_check_backend_version(row, ALPAKA_ACC_GPU_CUDA_ENABLE, ">", "11.5"):
            return False

    ###########################
    ## hipcc device compiler
    ###########################

    # the HIP backend needs to be enabled and has the same version number
    if row_check_name(row, DEVICE_COMPILER, "==", HIPCC):
        if row_check_backend_version(
            row,
            ALPAKA_ACC_GPU_HIP_ENABLE,
            "!=",
            row[param_map[DEVICE_COMPILER]][VERSION],
        ):
            reason(
                output, "hipcc and HIP backends needs to have the same version number"
            )
            return False

        # it is not allowed to enable the HIP and CUDA backend on the same time
        if row_check_backend_version(row, ALPAKA_ACC_GPU_CUDA_ENABLE, "!=", OFF_VER):
            reason(
                output,
                "if hipcc is the device compiler and the HIP backend is enabled, "
                "it is not allowed to enable the CUDA backend",
            )
            return False

    return True
