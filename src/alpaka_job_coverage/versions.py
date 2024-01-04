from typing import Dict, List

from alpaka_job_coverage.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from packaging import version as pk_version
from typeguard import typechecked

# This dict contains all software version, which we handle we the filter rules.
# Not every software version results in a case in the filter rules. For example
# we have no case, where a combination with a specific Boost version needs to be
# disabled (date: 26.7.2023). Maybe in future, a Boost specific case will be
# added.
versions: Dict[str, List[str]] = {
    GCC: ["6", "7", "8", "9", "10", "11", "12", "13"],
    CLANG: ["6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17"],
    NVCC: [
        "10.0",
        "10.1",
        "10.2",
        "11.0",
        "11.1",
        "11.2",
        "11.3",
        "11.4",
        "11.5",
        "11.6",
        "11.7",
        "11.8",
        "12.0",
        "12.1",
        "12.2",
        "12.3",
    ],
    HIPCC: ["5.0", "5.1", "5.2", "5.3", "5.4", "5.5", "5.6", "5.7", "6.0"],
    ICPX: ["2023.1.0", "2023.2.0"],
    UBUNTU: ["18.04", "20.04"],
    CMAKE: ["3.18", "3.19", "3.20", "3.21", "3.22", "3.23", "3.24", "3.25", "3.26"],
    BOOST: [
        "1.74.0",
        "1.75.0",
        "1.76.0",
        "1.77.0",
        "1.78.0",
        "1.79.0",
        "1.80.0",
        "1.81.0",
        "1.82.0",
    ],
    CXX_STANDARD: ["17", "20"],
}


@typechecked
def is_supported_version(name: str, version: str) -> bool:
    """Check if a specific software version is supported by the
    alpaka-job-coverage library.

    Args:
        name (str): Name of the software, e.g. gcc, boost or ubuntu
        version (str): Version of the software.

    Raises:
        ValueError: When the name of the software is not known.

    Returns:
        bool: True if supported otherwise False.
    """
    if (
        name
        not in list(versions.keys())
        + [
            CLANG_CUDA,
        ]
        + BACKENDS_LIST
    ):
        raise ValueError(f"Unknown software name: {name}")

    local_versions = versions.copy()

    local_versions[CLANG_CUDA] = versions[CLANG]
    local_versions[ALPAKA_ACC_GPU_CUDA_ENABLE] = [OFF_VER] + versions[NVCC]
    local_versions[ALPAKA_ACC_GPU_HIP_ENABLE] = [OFF_VER] + versions[HIPCC]

    for backend_name in set(BACKENDS_LIST) - set(
        (ALPAKA_ACC_GPU_CUDA_ENABLE, ALPAKA_ACC_GPU_HIP_ENABLE)
    ):
        local_versions[backend_name] = [ON_VER, OFF_VER]

    for v in local_versions[name]:
        if pk_version.parse(v) == pk_version.parse(version):
            return True

    return False
