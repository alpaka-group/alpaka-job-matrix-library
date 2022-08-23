"""Filter rules basing on backend names and versions.
"""

from alpaka_job_coverage.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from alpaka_job_coverage.util import (
    row_check_name,
    row_check_version,
    row_check_backend_version,
    backend_is_not_in_row,
)


def compiler_backend_filter(row: List) -> bool:
    """Filter rules basing on backend names and versions.

    Args:
        row (List): Combination to verify. The row can contain
        up to all combination fields and at least two items.

    Returns:
        bool: True, if combination is valid, otherwise False.
    """
    ###########################
    ## gcc device compiler
    ###########################

    if row_check_name(row, DEVICE_COMPILER, "==", GCC):
        if row_check_backend_version(row, ALPAKA_ACC_GPU_CUDA_ENABLE, "!=", OFF):
            return False

        if row_check_backend_version(row, ALPAKA_ACC_GPU_HIP_ENABLE, "!=", OFF):
            return False

    ###########################
    ## clang device compiler
    ###########################

    if row_check_name(row, DEVICE_COMPILER, "==", CLANG):
        if row_check_backend_version(row, ALPAKA_ACC_GPU_CUDA_ENABLE, "!=", OFF):
            return False

        # clang cannot compile with enabled HIP backend
        if row_check_backend_version(row, ALPAKA_ACC_GPU_HIP_ENABLE, "!=", OFF):
            return False

    ###########################
    ## nvcc device compiler
    ###########################

    if row_check_name(row, DEVICE_COMPILER, "==", NVCC):
        # the CUDA backend needs to be defined
        if backend_is_not_in_row(row, ALPAKA_ACC_GPU_CUDA_ENABLE):
            return False

        # the nvcc compiler needs the same version, like the backend
        if row_check_backend_version(
            row,
            ALPAKA_ACC_GPU_CUDA_ENABLE,
            "!=",
            row[param_map[DEVICE_COMPILER]][VERSION],
        ):
            return False

        # it is not allowed to enable the HIP and CUDA backend on the same time
        if row_check_backend_version(row, ALPAKA_ACC_GPU_HIP_ENABLE, "!=", OFF):
            return False

    ###########################
    ## clang-cuda device compiler
    ###########################

    if row_check_name(row, DEVICE_COMPILER, "==", CLANG_CUDA):
        # the CUDA backend needs to be defined
        if backend_is_not_in_row(row, ALPAKA_ACC_GPU_CUDA_ENABLE):
            return False

        # the CUDA backend needs to be enabled
        if row_check_backend_version(row, ALPAKA_ACC_GPU_CUDA_ENABLE, "==", OFF):
            return False

        # it is not allowed to enable the HIP and CUDA backend on the same time
        if row_check_backend_version(row, ALPAKA_ACC_GPU_HIP_ENABLE, "!=", OFF):
            return False

        if row_check_version(
            row, DEVICE_COMPILER, "<=", "7"
        ) and row_check_backend_version(row, ALPAKA_ACC_GPU_CUDA_ENABLE, ">", "9.2"):
            return False

        if row_check_version(
            row, DEVICE_COMPILER, "<=", "8"
        ) and row_check_backend_version(row, ALPAKA_ACC_GPU_CUDA_ENABLE, ">", "10.0"):
            return False

        if row_check_version(
            row, DEVICE_COMPILER, "<=", "10"
        ) and row_check_backend_version(row, ALPAKA_ACC_GPU_CUDA_ENABLE, ">", "10.1"):
            return False

        if row_check_version(
            row, DEVICE_COMPILER, "<=", "12"
        ) and row_check_backend_version(row, ALPAKA_ACC_GPU_CUDA_ENABLE, ">", "11.0"):
            return False

        if row_check_version(
            row, DEVICE_COMPILER, "<=", "13"
        ) and row_check_backend_version(row, ALPAKA_ACC_GPU_CUDA_ENABLE, ">", "11.2"):
            return False

        if row_check_version(
            row, DEVICE_COMPILER, "<=", "15"
        ) and row_check_backend_version(row, ALPAKA_ACC_GPU_CUDA_ENABLE, ">", "11.5"):
            return False

    ###########################
    ## hipcc device compiler
    ###########################

    # the HIP backend needs to be enabled and has the same version number
    if row_check_name(row, DEVICE_COMPILER, "==", HIPCC):
        # the HIP backend needs to be defined
        if backend_is_not_in_row(row, ALPAKA_ACC_GPU_HIP_ENABLE):
            return False

        if row_check_backend_version(
            row,
            ALPAKA_ACC_GPU_HIP_ENABLE,
            "!=",
            row[param_map[DEVICE_COMPILER]][VERSION],
        ):
            return False

        # it is not allowed to enable the HIP and CUDA backend on the same time
        if row_check_backend_version(row, ALPAKA_ACC_GPU_CUDA_ENABLE, "!=", OFF):
            return False

    return True
