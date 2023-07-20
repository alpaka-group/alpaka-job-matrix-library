import unittest

from alpaka_job_coverage.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from alpaka_job_coverage.filter_compiler_name import general_compiler_filter_typed


# Test all compiler names except nvcc and clang-cuda. This is tested in
# test_cuda_sdk.py
class TestHostDeviceCompiler(unittest.TestCase):
    def setUp(self):
        global param_map
        # set param_map, that filters expect the following parameters in the
        # order
        param_map[HOST_COMPILER] = 0
        param_map[DEVICE_COMPILER] = 1

    def setDown(self):
        global param_map
        # reset param_map for following up tests
        param_map = {}

    def test_same_host_device_compiler(self):
        valid_combs = [
            [(GCC, "0"), (GCC, "0")],
            [(CLANG, "0"), (CLANG, "0")],
            [(HIPCC, "0"), (HIPCC, "0")],
        ]

        for comb in valid_combs:
            self.assertTrue(general_compiler_filter_typed(comb))

        invalid_combs = [
            [(GCC, "0"), (CLANG, "0")],
            [(GCC, "0"), (CLANG, "0")],
            [(GCC, "0"), (HIPCC, "0")],
            [(HIPCC, "0"), (CLANG, "0")],
        ]

        for comb in invalid_combs:
            self.assertFalse(general_compiler_filter_typed(comb))
