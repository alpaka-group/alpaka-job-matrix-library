import unittest

from alpaka_job_coverage.filter_compiler_name import general_compiler_filter_typed
from alpaka_job_coverage.filter_compiler_version import compiler_version_filter_typed
from alpaka_job_coverage.filter_backend_version import compiler_backend_filter_typed
from alpaka_job_coverage.filter_software_dependency import (
    software_dependency_filter_typed,
)
from alpaka_job_coverage.globals import *

# The files contains tests for single, specific rules


def full_filter_chain(row) -> bool:
    return (
        general_compiler_filter_typed(row)
        and compiler_version_filter_typed(row)
        and compiler_backend_filter_typed(row)
        and software_dependency_filter_typed(row)
    )


class TestNVCC11GCC103UBUNTU2004(unittest.TestCase):
    def setUp(self):
        global param_map
        # set param_map, that filters expect the following parameters in the
        # order
        param_map[HOST_COMPILER] = 0
        param_map[DEVICE_COMPILER] = 1
        param_map[UBUNTU] = 2
        param_map[CXX_STANDARD] = 3

    def setDown(self):
        global param_map
        # reset param_map for following up tests
        param_map = {}

    # test if nvcc 11.3 + gcc 10 + Ubuntu 20.04 is disabled
    def test_forbid_combination(self):
        for nvcc_version in ["11.0", "11.1", "11.2", "11.3"]:
            comb = [(GCC, "10"), (NVCC, nvcc_version), (UBUNTU, "20.04")]

            self.assertTrue(
                general_compiler_filter_typed(comb),
                f"HOST_COMPILER: {comb[0][0]} {comb[0][1]}, "
                f"DEVICE_COMPILER: {comb[1][0]} {comb[1][1]}, "
                f"UBUNTU: {comb[2][1]}",
            )

            # nvcc 11.0 does not support GCC 10
            if nvcc_version != "11.0":
                self.assertTrue(
                    compiler_version_filter_typed(comb),
                    f"HOST_COMPILER: {comb[0][0]} {comb[0][1]}, "
                    f"DEVICE_COMPILER: {comb[1][0]} {comb[1][1]}, "
                    f"UBUNTU: {comb[2][1]}",
                )
            else:
                self.assertFalse(
                    compiler_version_filter_typed(comb),
                    f"HOST_COMPILER: {comb[0][0]} {comb[0][1]}, "
                    f"DEVICE_COMPILER: {comb[1][0]} {comb[1][1]}, "
                    f"UBUNTU: {comb[2][1]}",
                )

            self.assertTrue(
                compiler_backend_filter_typed(comb),
                f"HOST_COMPILER: {comb[0][0]} {comb[0][1]}, "
                f"DEVICE_COMPILER: {comb[1][0]} {comb[1][1]}, "
                f"UBUNTU: {comb[2][1]}",
            )
            self.assertFalse(
                software_dependency_filter_typed(comb),
                f"HOST_COMPILER: {comb[0][0]} {comb[0][1]}, "
                f"DEVICE_COMPILER: {comb[1][0]} {comb[1][1]}, "
                f"UBUNTU: {comb[2][1]}",
            )
            self.assertFalse(
                full_filter_chain(comb),
                f"HOST_COMPILER: {comb[0][0]} {comb[0][1]}, "
                f"DEVICE_COMPILER: {comb[1][0]} {comb[1][1]}, "
                f"UBUNTU: {comb[2][1]}",
            )
