import unittest

from alpaka_job_coverage.filter_compiler_name import general_compiler_filter_typed
from alpaka_job_coverage.filter_compiler_version import compiler_version_filter_typed
from alpaka_job_coverage.filter_backend_version import compiler_backend_filter_typed
from alpaka_job_coverage.filter_software_dependency import (
    software_dependency_filter_typed,
)
from alpaka_job_coverage.globals import *


def full_filter_chain(row) -> bool:
    return (
        general_compiler_filter_typed(row)
        and compiler_version_filter_typed(row)
        and compiler_backend_filter_typed(row)
        and software_dependency_filter_typed(row)
    )


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

    # test different combinations of NVCC, Clang, GCC and other Compilers as
    # host and device compiler. The version is not important.
    def test_host_device_compiler(self):
        valid_combs = [
            [(GCC, "0"), (NVCC, "0")],
            [(CLANG, "0"), (NVCC, "0")],
            [(CLANG_CUDA, "0"), (CLANG_CUDA, "0")],
        ]

        for comb in valid_combs:
            self.assertTrue(general_compiler_filter_typed(comb))

        invalid_combs = [
            # NVCC is not allowed as host compiler
            [(NVCC, "0"), (NVCC, "0")],
            [(NVCC, "0"), (GCC, "0")],
            [(NVCC, "0"), (CLANG, "0")],
            # only GCC and Clang are allowed as host compiler for nvcc
            [(CLANG_CUDA, "0"), (NVCC, "0")],
            [(HIPCC, "0"), (NVCC, "0")],
        ]

        for comb in invalid_combs:
            self.assertFalse(general_compiler_filter_typed(comb))


class TestGeneralFilterFunctionality(unittest.TestCase):
    def setUp(self):
        global param_map
        # set param_map, that filters expect the following parameters in the
        # order
        param_map[HOST_COMPILER] = 0
        param_map[DEVICE_COMPILER] = 1
        param_map[BACKENDS] = 2
        # UBUNTU and CXX_STANDARD are only required, because
        # software_dependency_filter_typed do a look up in param_map for it
        param_map[CXX_STANDARD] = 3
        param_map[UBUNTU] = 4

    def setDown(self):
        global param_map
        # reset param_map for following up tests
        param_map = {}

    def test_if_backend_is_not_set(self):
        # general_compiler_filter_typed should be always return true, because
        # GCC is a valid host compiler and NVCC a valid device compiler

        # compiler_backend_filter_typed should always return true for the valid
        # combination, because if the backend parameter
        # (ALPAKA_ACC_GPU_CUDA_ENABLE, "11.2") is added, it becomes a valid
        # combination

        # compiler_version_filter_typed filters invalid combination depending of
        # host and device compiler version

        # no rule in software_dependency_filter_typed should affect this test

        comb_valid_1 = [(GCC, "9"), (NVCC, "11.2")]

        self.assertTrue(general_compiler_filter_typed(comb_valid_1))
        self.assertTrue(compiler_backend_filter_typed(comb_valid_1))
        self.assertTrue(compiler_version_filter_typed(comb_valid_1))
        self.assertTrue(software_dependency_filter_typed(comb_valid_1))
        self.assertTrue(full_filter_chain(comb_valid_1))

        comb_invalid_1 = [(GCC, "13"), (NVCC, "11.2")]

        self.assertTrue(general_compiler_filter_typed(comb_invalid_1))
        self.assertTrue(compiler_backend_filter_typed(comb_invalid_1))
        self.assertFalse(compiler_version_filter_typed(comb_invalid_1))
        self.assertTrue(software_dependency_filter_typed(comb_invalid_1))
        self.assertFalse(full_filter_chain(comb_invalid_1))

        comb_valid_2 = [(CLANG, "9"), (NVCC, "11.2")]

        self.assertTrue(general_compiler_filter_typed(comb_valid_2))
        self.assertTrue(compiler_backend_filter_typed(comb_valid_2))
        self.assertTrue(compiler_version_filter_typed(comb_valid_2))
        self.assertTrue(software_dependency_filter_typed(comb_valid_2))
        self.assertTrue(full_filter_chain(comb_valid_2))

        comb_invalid_2 = [(CLANG, "17"), (NVCC, "11.2")]

        self.assertTrue(general_compiler_filter_typed(comb_invalid_2))
        self.assertTrue(compiler_backend_filter_typed(comb_invalid_2))
        self.assertFalse(compiler_version_filter_typed(comb_invalid_2))
        self.assertTrue(software_dependency_filter_typed(comb_invalid_2))
        self.assertFalse(full_filter_chain(comb_invalid_2))

    def test_if_backend_is_disabled(self):
        # because the backend is off, compiler_backend_filter_typed should return
        # False every time

        comb_valid_1 = [
            (GCC, "9"),
            (NVCC, "11.2"),
            [(ALPAKA_ACC_GPU_CUDA_ENABLE, OFF_VER)],
        ]

        self.assertTrue(general_compiler_filter_typed(comb_valid_1))
        self.assertFalse(compiler_backend_filter_typed(comb_valid_1))
        self.assertTrue(compiler_version_filter_typed(comb_valid_1))
        self.assertTrue(software_dependency_filter_typed(comb_valid_1))
        self.assertFalse(full_filter_chain(comb_valid_1))

        comb_valid_2 = [
            (GCC, "9"),
            (NVCC, "11.2"),
            [
                (ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE, ON_VER),
                (ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE, ON_VER),
                (ALPAKA_ACC_GPU_CUDA_ENABLE, OFF_VER),
            ],
        ]

        self.assertTrue(general_compiler_filter_typed(comb_valid_2))
        self.assertFalse(compiler_backend_filter_typed(comb_valid_2))
        self.assertTrue(compiler_version_filter_typed(comb_valid_2))
        self.assertTrue(software_dependency_filter_typed(comb_valid_2))
        self.assertFalse(full_filter_chain(comb_valid_2))

        comb_valid_3 = [
            (CLANG, "9"),
            (NVCC, "11.2"),
            [
                (ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE, ON_VER),
                (ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE, ON_VER),
                (ALPAKA_ACC_GPU_CUDA_ENABLE, OFF_VER),
            ],
        ]

        self.assertTrue(general_compiler_filter_typed(comb_valid_3))
        self.assertFalse(compiler_backend_filter_typed(comb_valid_3))
        self.assertTrue(compiler_version_filter_typed(comb_valid_3))
        self.assertTrue(software_dependency_filter_typed(comb_valid_3))
        self.assertFalse(full_filter_chain(comb_valid_3))

        comb_invalid_1 = [
            (GCC, "13"),
            (NVCC, "11.2"),
            [(ALPAKA_ACC_GPU_CUDA_ENABLE, OFF_VER)],
        ]

        self.assertTrue(general_compiler_filter_typed(comb_invalid_1))
        self.assertFalse(compiler_backend_filter_typed(comb_invalid_1))
        self.assertFalse(compiler_version_filter_typed(comb_invalid_1))
        self.assertTrue(software_dependency_filter_typed(comb_invalid_1))
        self.assertFalse(full_filter_chain(comb_invalid_1))

        comb_invalid_2 = [
            (GCC, "13"),
            (NVCC, "11.2"),
            [
                (ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE, ON_VER),
                (ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE, ON_VER),
                (ALPAKA_ACC_GPU_CUDA_ENABLE, OFF_VER),
            ],
        ]

        self.assertTrue(general_compiler_filter_typed(comb_invalid_2))
        self.assertFalse(compiler_backend_filter_typed(comb_invalid_2))
        self.assertFalse(compiler_version_filter_typed(comb_invalid_2))
        self.assertTrue(software_dependency_filter_typed(comb_invalid_2))
        self.assertFalse(full_filter_chain(comb_invalid_2))

        comb_invalid_3 = [
            (CLANG, "13"),
            (NVCC, "11.2"),
            [
                (ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE, ON_VER),
                (ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE, ON_VER),
                (ALPAKA_ACC_GPU_CUDA_ENABLE, OFF_VER),
            ],
        ]

        self.assertTrue(general_compiler_filter_typed(comb_invalid_3))
        self.assertFalse(compiler_backend_filter_typed(comb_invalid_3))
        self.assertFalse(compiler_version_filter_typed(comb_invalid_3))
        self.assertTrue(software_dependency_filter_typed(comb_invalid_3))
        self.assertFalse(full_filter_chain(comb_invalid_3))

    def test_backend_versions(self):
        # for NVCC, the compiler version needs to be equal to the backend version
        for version in ["10.2", "11.0", "12.2"]:
            comb1 = [
                (GCC, "9"),
                (NVCC, version),
                [(ALPAKA_ACC_GPU_CUDA_ENABLE, version)],
            ]
            self.assertTrue(compiler_backend_filter_typed(comb1))

            comb2 = [
                (CLANG, "9"),
                (NVCC, version),
                [(ALPAKA_ACC_GPU_CUDA_ENABLE, version)],
            ]
            self.assertTrue(compiler_backend_filter_typed(comb2))

            comb3 = [
                (GCC, "9"),
                (NVCC, version),
                [
                    (ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE, ON_VER),
                    (ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE, ON_VER),
                    (ALPAKA_ACC_GPU_CUDA_ENABLE, version),
                ],
            ]
            self.assertTrue(compiler_backend_filter_typed(comb3))

            comb4 = [
                (CLANG, "9"),
                (NVCC, version),
                [
                    (ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE, ON_VER),
                    (ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE, ON_VER),
                    (ALPAKA_ACC_GPU_CUDA_ENABLE, version),
                ],
            ]
            self.assertTrue(compiler_backend_filter_typed(comb4))

        for nvcc_version, backend_version in [
            ("10.1", "10.2"),
            ("11.0", "11.5"),
            ("11.8", "12.2"),
        ]:
            comb1 = [
                (GCC, "9"),
                (NVCC, nvcc_version),
                [(ALPAKA_ACC_GPU_CUDA_ENABLE, backend_version)],
            ]
            self.assertFalse(compiler_backend_filter_typed(comb1))

            comb2 = [
                (CLANG, "9"),
                (NVCC, nvcc_version),
                [(ALPAKA_ACC_GPU_CUDA_ENABLE, backend_version)],
            ]
            self.assertFalse(compiler_backend_filter_typed(comb2))

            comb3 = [
                (GCC, "9"),
                (NVCC, nvcc_version),
                [
                    (ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE, ON_VER),
                    (ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE, ON_VER),
                    (ALPAKA_ACC_GPU_CUDA_ENABLE, backend_version),
                ],
            ]
            self.assertFalse(compiler_backend_filter_typed(comb3))

            comb4 = [
                (CLANG, "9"),
                (NVCC, nvcc_version),
                [
                    (ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE, ON_VER),
                    (ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE, ON_VER),
                    (ALPAKA_ACC_GPU_CUDA_ENABLE, backend_version),
                ],
            ]
            self.assertFalse(compiler_backend_filter_typed(comb4))


class TestNvccGccCompatibility(unittest.TestCase):
    def setUp(self):
        global param_map
        # set param_map, that filters expect the following parameters in the
        # order
        param_map[DEVICE_COMPILER] = 0
        param_map[HOST_COMPILER] = 1

    def setDown(self):
        global param_map
        # reset param_map for following up tests
        param_map = {}

    # General test of the algorithm, which should decide if a GCC and CUDA SDK
    # version combination is valid.
    def test_general_algorithm(self):
        expected_results = [
            ("10.0", "6", True),
            ("10.0", "7", True),
            ("10.0", "8", False),
            ("10.0", "13", False),
            ("10.1", "6", True),
            ("10.1", "7", True),
            ("10.1", "8", True),
            ("10.1", "13", False),
            ("10.2", "6", True),
            ("10.2", "7", True),
            ("10.2", "8", True),
            ("10.2", "13", False),
            ("11.0", "6", True),
            ("11.0", "9", True),
            ("11.0", "10", False),
            ("11.0", "12", False),
        ]

        for nvcc_version, gcc_version, expected_value in expected_results:
            self.assertEqual(
                compiler_version_filter_typed(
                    [(NVCC, nvcc_version), (GCC, gcc_version)]
                ),
                expected_value,
                f"nvcc {nvcc_version} + gcc {gcc_version} -> {expected_value}",
            )

    # Test the combination of GCC version and CUDA SDK version.
    # Validate the combinations, which are defined in the release notes of the
    # CUDA SDK.
    def test_known_version(self):
        expected_results = [
            ("10.0", "7", True),
            ("10.0", "8", False),
            ("10.1", "8", True),
            ("10.1", "9", False),
            ("10.2", "8", True),
            ("10.2", "9", False),
            ("11.0", "9", True),
            ("11.0", "10", False),
            ("11.1", "10", True),
            ("11.1", "11", False),
            ("11.2", "10", True),
            ("11.2", "11", False),
            ("11.3", "10", True),
            ("11.3", "11", False),
            ("11.4", "11", True),
            ("11.4", "12", False),
            ("11.5", "11", True),
            ("11.5", "12", False),
            ("11.6", "11", True),
            ("11.6", "12", False),
            ("11.7", "11", True),
            ("11.7", "12", False),
            ("11.8", "11", True),
            ("11.8", "12", False),
            ("12.0", "12", True),
            ("12.0", "13", False),
            ("12.1", "12", True),
            ("12.1", "13", False),
        ]

        for nvcc_version, gcc_version, expected_value in expected_results:
            self.assertEqual(
                compiler_version_filter_typed(
                    [(NVCC, nvcc_version), (GCC, gcc_version)]
                ),
                expected_value,
                f"nvcc {nvcc_version} + gcc {gcc_version} -> {expected_value}",
            )

    # Test if a unknown CUDA SDK version returns always true.
    # This avoid behavior avoids, that valid combinations are silently disabled
    # if a new CUDA SDK version is released.
    def test_unknown_version(self):
        expected_results = [
            ("42.0", "45", True),
        ]

        # TODO: add is_supported_sw_version to verify that our test version is not supported

        for nvcc_version, gcc_version, expected_value in expected_results:
            self.assertEqual(
                compiler_version_filter_typed(
                    [(NVCC, nvcc_version), (GCC, gcc_version)]
                ),
                expected_value,
                f"nvcc {nvcc_version} + gcc {gcc_version} -> {expected_value}",
            )


class TestNvccClangCompatibility(unittest.TestCase):
    def setUp(self):
        global param_map
        # set param_map, that filters expect the following parameters in the
        # order
        param_map[DEVICE_COMPILER] = 0
        param_map[HOST_COMPILER] = 1

    def setDown(self):
        global param_map
        # reset param_map for following up tests
        param_map = {}

    # Test the combination of Clang version and CUDA SDK version.
    # Validate the combinations, which are defined in the release notes of the
    # CUDA SDK.
    def test_known_version(self):
        expected_results = [
            ("10.0", "6", True),
            ("10.0", "7", False),
            ("10.1", "8", True),
            ("10.1", "9", False),
            ("10.2", "8", True),
            ("10.2", "9", False),
            ("11.0", "9", True),
            ("11.0", "10", False),
            ("11.1", "10", True),
            ("11.1", "11", False),
            ("11.2", "11", True),
            ("11.2", "12", False),
            # because of compiler bugs, clang is disabled for CUDA 11.3 until 11.5
            ("11.3", "11", False),
            ("11.3", "12", False),
            ("11.4", "12", False),
            ("11.4", "13", False),
            ("11.5", "12", False),
            ("11.5", "13", False),
            ("11.6", "13", True),
            ("11.6", "14", False),
            ("11.7", "13", True),
            ("11.7", "14", False),
            ("11.8", "13", True),
            ("11.8", "14", False),
            ("12.0", "14", True),
            ("12.0", "15", False),
            ("12.1", "15", True),
            ("12.1", "16", False),
        ]

        for nvcc_version, clang_version, expected_value in expected_results:
            self.assertEqual(
                compiler_version_filter_typed(
                    [(NVCC, nvcc_version), (CLANG, clang_version)]
                ),
                expected_value,
                f"nvcc {nvcc_version} + clang {clang_version} -> {expected_value}",
            )

    # Test if a unknown CUDA SDK version returns always true.
    # This avoid behavior avoids, that valid combinations are silently disabled
    # if a new CUDA SDK version is released.
    def test_unknown_version(self):
        expected_results = [
            ("42.0", "45", True),
        ]

        # TODO: add is_supported_sw_version to verify that our test version is not supported

        for nvcc_version, clang_version, expected_value in expected_results:
            self.assertEqual(
                compiler_version_filter_typed(
                    [(NVCC, nvcc_version), (CLANG, clang_version)]
                ),
                expected_value,
                f"nvcc {nvcc_version} + clang {clang_version} -> {expected_value}",
            )


class TestNvccCxxStandard(unittest.TestCase):
    def setUp(self):
        global param_map
        # set param_map, that filters expect the following parameters in the
        # order
        param_map[DEVICE_COMPILER] = 0
        param_map[CXX_STANDARD] = 1
        # the following parameters are not used in this test
        param_map[HOST_COMPILER] = 2
        param_map[BACKENDS] = 3
        param_map[UBUNTU] = 4

    def setDown(self):
        global param_map
        # reset param_map for following up tests
        param_map = {}

    # Test the combination of NVCC as device compiler an the C++ standard.
    # If SDK version is unknown, all C++ standards should be allowed.
    def test_nvcc_cxx(self):
        # TODO: add is_supported_sw_version to verify that our test version is not supported

        for nvcc_version, max_cxx in [
            ("11.0", 17),
            ("11.2", 17),
            ("11.8", 17),
            ("12.0", 20),
            ("12.1", 20),
            # not released version
            # therefore they should support all C++ versions
            ("12.2", 32),
            ("13.0", 32),
        ]:
            for cxx_version in [11, 14, 17, 20, 23, 26, 29, 32]:
                comb = [(NVCC, nvcc_version), (CXX_STANDARD, str(cxx_version))]
                if cxx_version <= max_cxx:
                    self.assertTrue(
                        software_dependency_filter_typed(comb),
                        f"NVCC {nvcc_version} + CXX {cxx_version}",
                    )
                else:
                    self.assertFalse(
                        software_dependency_filter_typed(comb),
                        f"NVCC {nvcc_version} + CXX {cxx_version}",
                    )


# tests for Clang as CUDA compiler
class TestClangCUDA(unittest.TestCase):
    def setUp(self):
        global param_map
        # set param_map, that filters expect the following parameters in the
        # order
        param_map[HOST_COMPILER] = 0
        param_map[DEVICE_COMPILER] = 1
        param_map[BACKENDS] = 2
        # the following parameters are not used in this test
        param_map[CXX_STANDARD] = 3
        param_map[UBUNTU] = 4
        param_map[CMAKE] = 5

    def setDown(self):
        global param_map
        # reset param_map for following up tests
        param_map = {}

    # standalone CLANG_CUDA as host compiler should pass all tests
    def test_host_compiler_name(self):
        comb = [(CLANG_CUDA, "99")]

        self.assertTrue(general_compiler_filter_typed(comb))
        self.assertTrue(compiler_version_filter_typed(comb))
        self.assertTrue(compiler_backend_filter_typed(comb))
        self.assertTrue(software_dependency_filter_typed(comb))
        self.assertTrue(full_filter_chain(comb))

    # CLANG_CUDA as host compiler can be only used with CLANG_CUDA as device
    # compiler an vice vera
    def test_host_device_compiler_name(self):
        valid_comb = [(CLANG_CUDA, "99"), (CLANG_CUDA, "99")]

        self.assertTrue(general_compiler_filter_typed(valid_comb))
        self.assertTrue(compiler_version_filter_typed(valid_comb))
        self.assertTrue(compiler_backend_filter_typed(valid_comb))
        self.assertTrue(software_dependency_filter_typed(valid_comb))
        self.assertTrue(full_filter_chain(valid_comb))

        invalid_combs = [
            [(CLANG_CUDA, "99"), (GCC, "99")],
            [(CLANG_CUDA, "99"), (CLANG, "99")],
            [(CLANG_CUDA, "99"), (HIPCC, "99")],
            [(CLANG_CUDA, "99"), (NVCC, "99")],
            [(GCC, "99"), (CLANG_CUDA, "99")],
            [(CLANG, "99"), (CLANG_CUDA, "99")],
            [(HIPCC, "99"), (CLANG_CUDA, "99")],
            [(NVCC, "99"), (CLANG_CUDA, "99")],
        ]

        for comb in invalid_combs:
            self.assertFalse(
                general_compiler_filter_typed(comb),
                f"HOST_COMPILER: {comb[0][0]}, DEVICE_COMPILER: {comb[1][0]}",
            )
            self.assertTrue(
                compiler_version_filter_typed(comb),
                f"HOST_COMPILER: {comb[0][0]}, DEVICE_COMPILER: {comb[1][0]}",
            )
            self.assertTrue(
                compiler_backend_filter_typed(comb),
                f"HOST_COMPILER: {comb[0][0]}, DEVICE_COMPILER: {comb[1][0]}",
            )
            self.assertTrue(
                software_dependency_filter_typed(comb),
                f"HOST_COMPILER: {comb[0][0]}, DEVICE_COMPILER: {comb[1][0]}",
            )
            self.assertFalse(
                full_filter_chain(comb),
                f"HOST_COMPILER: {comb[0][0]}, DEVICE_COMPILER: {comb[1][0]}",
            )

    # host and device compiler needs to have the same version number
    def test_host_device_compiler_version(self):
        valid_combs = [
            [(CLANG_CUDA, "14"), (CLANG_CUDA, "14")],
            [(CLANG_CUDA, "16"), (CLANG_CUDA, "16")],
            [(CLANG_CUDA, "20"), (CLANG_CUDA, "20")],
            [(CLANG_CUDA, "99"), (CLANG_CUDA, "99")],
        ]

        for comb in valid_combs:
            self.assertTrue(
                general_compiler_filter_typed(comb),
                f"HOST_COMPILER: {comb[0][0]} {comb[0][1]}, "
                f"DEVICE_COMPILER: {comb[1][0]} {comb[1][1]}",
            )
            self.assertTrue(
                compiler_version_filter_typed(comb),
                f"HOST_COMPILER: {comb[0][0]} {comb[0][1]}, "
                f"DEVICE_COMPILER: {comb[1][0]} {comb[1][1]}",
            )
            self.assertTrue(
                compiler_backend_filter_typed(comb),
                f"HOST_COMPILER: {comb[0][0]} {comb[0][1]}, "
                f"DEVICE_COMPILER: {comb[1][0]} {comb[1][1]}",
            )
            self.assertTrue(
                software_dependency_filter_typed(comb),
                f"HOST_COMPILER: {comb[0][0]} {comb[0][1]}, "
                f"DEVICE_COMPILER: {comb[1][0]} {comb[1][1]}",
            )
            self.assertTrue(
                full_filter_chain(comb),
                f"HOST_COMPILER: {comb[0][0]} {comb[0][1]}, "
                f"DEVICE_COMPILER: {comb[1][0]} {comb[1][1]}",
            )

        invalid_combs = [
            [(CLANG_CUDA, "16"), (CLANG_CUDA, "17")],
            [(CLANG_CUDA, "21"), (CLANG_CUDA, "20")],
            # only Clang 14 and newer are supported as CUDA compiler
            [(CLANG_CUDA, "9"), (CLANG_CUDA, "9")],
            [(CLANG_CUDA, "13"), (CLANG_CUDA, "13")],
        ]

        for comb in invalid_combs:
            self.assertTrue(
                general_compiler_filter_typed(comb),
                f"HOST_COMPILER: {comb[0][0]} {comb[0][1]}, "
                f"DEVICE_COMPILER: {comb[1][0]} {comb[1][1]}",
            )
            self.assertFalse(
                compiler_version_filter_typed(comb),
                f"HOST_COMPILER: {comb[0][0]} {comb[0][1]}, "
                f"DEVICE_COMPILER: {comb[1][0]} {comb[1][1]}",
            )
            self.assertTrue(
                compiler_backend_filter_typed(comb),
                f"HOST_COMPILER: {comb[0][0]} {comb[0][1]}, "
                f"DEVICE_COMPILER: {comb[1][0]} {comb[1][1]}",
            )
            self.assertTrue(
                software_dependency_filter_typed(comb),
                f"HOST_COMPILER: {comb[0][0]} {comb[0][1]}, "
                f"DEVICE_COMPILER: {comb[1][0]} {comb[1][1]}",
            )
            self.assertFalse(
                full_filter_chain(comb),
                f"HOST_COMPILER: {comb[0][0]} {comb[0][1]}, "
                f"DEVICE_COMPILER: {comb[1][0]} {comb[1][1]}",
            )

    # the CUDA backend needs to enabled and Clang needs to support the CUDA
    # SDK
    def test_host_device_compiler_backend(self):
        valid_combs = [
            [
                (CLANG_CUDA, "14"),
                (CLANG_CUDA, "14"),
                [(ALPAKA_ACC_GPU_CUDA_ENABLE, "11.5")],
            ],
            [
                (CLANG_CUDA, "14"),
                (CLANG_CUDA, "14"),
                [(ALPAKA_ACC_GPU_CUDA_ENABLE, "11.2")],
            ],
            [
                (CLANG_CUDA, "14"),
                (CLANG_CUDA, "14"),
                [
                    (ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE, ON_VER),
                    (ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE, ON_VER),
                    (ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, OFF_VER),
                    (ALPAKA_ACC_GPU_CUDA_ENABLE, "11.2"),
                    (ALPAKA_ACC_GPU_HIP_ENABLE, OFF_VER),
                ],
            ],
        ]

        for comb in valid_combs:
            self.assertTrue(
                general_compiler_filter_typed(comb),
                f"HOST_COMPILER: {comb[0][0]} {comb[0][1]}, "
                f"DEVICE_COMPILER: {comb[1][0]} {comb[1][1]}, "
                f"BACKENDS: {comb[2]}",
            )
            self.assertTrue(
                compiler_version_filter_typed(comb),
                f"HOST_COMPILER: {comb[0][0]} {comb[0][1]}, "
                f"DEVICE_COMPILER: {comb[1][0]} {comb[1][1]}, "
                f"BACKENDS: {comb[2]}",
            )
            self.assertTrue(
                compiler_backend_filter_typed(comb),
                f"HOST_COMPILER: {comb[0][0]} {comb[0][1]}, "
                f"DEVICE_COMPILER: {comb[1][0]} {comb[1][1]}, "
                f"BACKENDS: {comb[2]}",
            )
            self.assertTrue(
                software_dependency_filter_typed(comb),
                f"HOST_COMPILER: {comb[0][0]} {comb[0][1]}, "
                f"DEVICE_COMPILER: {comb[1][0]} {comb[1][1]}, "
                f"BACKENDS: {comb[2]}",
            )
            self.assertTrue(
                full_filter_chain(comb),
                f"HOST_COMPILER: {comb[0][0]} {comb[0][1]}, "
                f"DEVICE_COMPILER: {comb[1][0]} {comb[1][1]}, "
                f"BACKENDS: {comb[2]}",
            )

        invalid_combs = [
            # CUDA backend needs to be enabled
            [
                (CLANG_CUDA, "14"),
                (CLANG_CUDA, "14"),
                [(ALPAKA_ACC_GPU_CUDA_ENABLE, OFF_VER)],
            ],
            # CUDA backend needs to be supported
            [
                (CLANG_CUDA, "14"),
                (CLANG_CUDA, "14"),
                [(ALPAKA_ACC_GPU_CUDA_ENABLE, "12.0")],
            ],
            # HIP backend cannot be enabled, when CLANG_CUDA is used
            [
                (CLANG_CUDA, "14"),
                (CLANG_CUDA, "14"),
                [
                    (ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE, ON_VER),
                    (ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE, ON_VER),
                    (ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, OFF_VER),
                    (ALPAKA_ACC_GPU_CUDA_ENABLE, "11.2"),
                    (ALPAKA_ACC_GPU_HIP_ENABLE, "5.0"),
                ],
            ],
        ]

        for comb in invalid_combs:
            self.assertTrue(
                general_compiler_filter_typed(comb),
                f"HOST_COMPILER: {comb[0][0]} {comb[0][1]}, "
                f"DEVICE_COMPILER: {comb[1][0]} {comb[1][1]}, "
                f"BACKENDS: {comb[2]}",
            )
            self.assertTrue(
                compiler_version_filter_typed(comb),
                f"HOST_COMPILER: {comb[0][0]} {comb[0][1]}, "
                f"DEVICE_COMPILER: {comb[1][0]} {comb[1][1]}, "
                f"BACKENDS: {comb[2]}",
            )
            self.assertFalse(
                compiler_backend_filter_typed(comb),
                f"HOST_COMPILER: {comb[0][0]} {comb[0][1]}, "
                f"DEVICE_COMPILER: {comb[1][0]} {comb[1][1]}, "
                f"BACKENDS: {comb[2]}",
            )
            self.assertTrue(
                software_dependency_filter_typed(comb),
                f"HOST_COMPILER: {comb[0][0]} {comb[0][1]}, "
                f"DEVICE_COMPILER: {comb[1][0]} {comb[1][1]}, "
                f"BACKENDS: {comb[2]}",
            )
            self.assertFalse(
                full_filter_chain(comb),
                f"HOST_COMPILER: {comb[0][0]} {comb[0][1]}, "
                f"DEVICE_COMPILER: {comb[1][0]} {comb[1][1]}, "
                f"BACKENDS: {comb[2]}",
            )

    # test all supported Clang CUDA SDK combinations
    def test_all_clang_cuda_version_combinations(self):
        cuda_versions = [
            9.0,
            9.1,
            9.2,
            10.0,
            10.1,
            10.2,
            11.0,
            11.1,
            11.2,
            11.3,
            11.4,
            11.5,
            11.6,
            11.7,
            11.8,
            12.0,
            12.1,
        ]

        # key: clang version
        # value: max supported CUDA SDK
        supported_versions = {
            7: 9.2,
            8: 10,
            9: 10.1,
            10: 10.1,
            11: 11.0,
            12: 11.0,
            13: 11.2,
            14: 11.5,
            15: 11.5,
            16: 11.5,
        }

        for clang_version in supported_versions.keys():
            for cuda_version in cuda_versions:
                comb = [
                    (CLANG_CUDA, str(clang_version)),
                    (CLANG_CUDA, str(clang_version)),
                    [(ALPAKA_ACC_GPU_CUDA_ENABLE, str(cuda_version))],
                ]

                if cuda_version > supported_versions[clang_version]:
                    self.assertFalse(
                        compiler_backend_filter_typed(comb),
                        f"HOST_COMPILER: {comb[0][0]} {comb[0][1]}, "
                        f"DEVICE_COMPILER: {comb[1][0]} {comb[1][1]}, "
                        f"BACKENDS: {comb[2]}",
                    )
                else:
                    self.assertTrue(
                        compiler_backend_filter_typed(comb),
                        f"HOST_COMPILER: {comb[0][0]} {comb[0][1]}, "
                        f"DEVICE_COMPILER: {comb[1][0]} {comb[1][1]}, "
                        f"BACKENDS: {comb[2]}",
                    )

        # test if combination of non released clang version and released CUDA
        # SDK is allowed
        self.assertTrue(
            compiler_backend_filter_typed(
                [
                    (CLANG_CUDA, "30"),
                    (CLANG_CUDA, "30"),
                    [(ALPAKA_ACC_GPU_CUDA_ENABLE, "11.5")],
                ]
            ),
            f"HOST_COMPILER: {comb[0][0]} {comb[0][1]}, "
            f"DEVICE_COMPILER: {comb[1][0]} {comb[1][1]}, "
            f"BACKENDS: {comb[2]}",
        )

        # test if combination of non released clang version and non released
        # CUDA SDK is allowed
        self.assertTrue(
            compiler_backend_filter_typed(
                [
                    (CLANG_CUDA, "30"),
                    (CLANG_CUDA, "30"),
                    [(ALPAKA_ACC_GPU_CUDA_ENABLE, "30.0")],
                ]
            ),
            f"HOST_COMPILER: {comb[0][0]} {comb[0][1]}, "
            f"DEVICE_COMPILER: {comb[1][0]} {comb[1][1]}, "
            f"BACKENDS: {comb[2]}",
        )

        # test if combination of released clang version and non released
        # CUDA SDK is not allowed
        self.assertFalse(
            compiler_backend_filter_typed(
                [
                    (CLANG_CUDA, "15"),
                    (CLANG_CUDA, "15"),
                    [(ALPAKA_ACC_GPU_CUDA_ENABLE, "30.0")],
                ]
            ),
            f"HOST_COMPILER: {comb[0][0]} {comb[0][1]}, "
            f"DEVICE_COMPILER: {comb[1][0]} {comb[1][1]}, "
            f"BACKENDS: {comb[2]}",
        )

    # Clang 9 and older does not support C++ 20.
    def test_cxx_support(self):
        for clang_version in range(7, 16):
            for cxx_version in [17, 20]:
                comb = [
                    (CLANG_CUDA, str(clang_version)),
                    (CLANG_CUDA, str(clang_version)),
                    [(ALPAKA_ACC_GPU_CUDA_ENABLE, "9.2")],
                    (CXX_STANDARD, str(cxx_version)),
                ]
                self.assertTrue(
                    general_compiler_filter_typed(comb),
                    f"HOST_COMPILER: {comb[0][0]} {comb[0][1]}, "
                    f"DEVICE_COMPILER: {comb[1][0]} {comb[1][1]}, "
                    f"C++: {comb[3]}",
                )

                # does not test compiler_version_filter_typed(), because of the filter
                # rule, that Clang 13 and older is not allowed as CUDA compiler

                self.assertTrue(
                    compiler_backend_filter_typed(comb),
                    f"HOST_COMPILER: {comb[0][0]} {comb[0][1]}, "
                    f"DEVICE_COMPILER: {comb[1][0]} {comb[1][1]}, "
                    f"C++: {comb[3]}",
                )

                if clang_version <= 9 and cxx_version > 17:
                    self.assertFalse(
                        software_dependency_filter_typed(comb),
                        f"HOST_COMPILER: {comb[0][0]} {comb[0][1]}, "
                        f"DEVICE_COMPILER: {comb[1][0]} {comb[1][1]}, "
                        f"C++: {comb[3]}",
                    )
                    self.assertFalse(
                        full_filter_chain(comb),
                        f"HOST_COMPILER: {comb[0][0]} {comb[0][1]}, "
                        f"DEVICE_COMPILER: {comb[1][0]} {comb[1][1]}, "
                        f"C++: {comb[3]}",
                    )
                else:
                    self.assertTrue(
                        software_dependency_filter_typed(comb),
                        f"HOST_COMPILER: {comb[0][0]} {comb[0][1]}, "
                        f"DEVICE_COMPILER: {comb[1][0]} {comb[1][1]}, "
                        f"C++: {comb[3]}",
                    )

    # Clang 11 and 12 is not available in the Ubuntu 18.04 ppa
    def test_ubuntu1804_ppa(self):
        for clang_version in range(7, 16):
            comb = [
                (CLANG_CUDA, str(clang_version)),
                (CLANG_CUDA, str(clang_version)),
                [(ALPAKA_ACC_GPU_CUDA_ENABLE, "9.2")],
                (CXX_STANDARD, "17"),
                (UBUNTU, "18.04"),
            ]

            self.assertTrue(
                general_compiler_filter_typed(comb),
                f"HOST_COMPILER: {comb[0][0]} {comb[0][1]}, "
                f"DEVICE_COMPILER: {comb[1][0]} {comb[1][1]}, "
                f"Ubuntu: {comb[4]}",
            )

            self.assertTrue(
                compiler_backend_filter_typed(comb),
                f"HOST_COMPILER: {comb[0][0]} {comb[0][1]}, "
                f"DEVICE_COMPILER: {comb[1][0]} {comb[1][1]}, "
                f"Ubuntu: {comb[4]}",
            )

            if clang_version == 11 or clang_version == 12:
                self.assertFalse(
                    software_dependency_filter_typed(comb),
                    f"HOST_COMPILER: {comb[0][0]} {comb[0][1]}, "
                    f"DEVICE_COMPILER: {comb[1][0]} {comb[1][1]}, "
                    f"Ubuntu: {comb[4]}",
                )
                self.assertFalse(
                    full_filter_chain(comb),
                    f"HOST_COMPILER: {comb[0][0]} {comb[0][1]}, "
                    f"DEVICE_COMPILER: {comb[1][0]} {comb[1][1]}, "
                    f"Ubuntu: {comb[4]}",
                )
            else:
                self.assertTrue(
                    software_dependency_filter_typed(comb),
                    f"HOST_COMPILER: {comb[0][0]} {comb[0][1]}, "
                    f"DEVICE_COMPILER: {comb[1][0]} {comb[1][1]}, "
                    f"Ubuntu: {comb[4]}",
                )

    # CMake 3.18 and older does not support Clang as CUDA compiler
    def test_cmake_version(self):
        valid_combs = [
            [
                (CLANG_CUDA, "14"),
                (CLANG_CUDA, "14"),
                [(ALPAKA_ACC_GPU_CUDA_ENABLE, "11.2")],
                (CXX_STANDARD, "17"),
                (UBUNTU, "20.04"),
                (CMAKE, "3.19"),
            ],
            [
                (CLANG_CUDA, "14"),
                (CLANG_CUDA, "14"),
                [(ALPAKA_ACC_GPU_CUDA_ENABLE, "11.2")],
                (CXX_STANDARD, "17"),
                (UBUNTU, "20.04"),
                (CMAKE, "3.22"),
            ],
        ]

        for comb in valid_combs:
            self.assertTrue(
                general_compiler_filter_typed(comb),
                f"HOST_COMPILER: {comb[0][0]} {comb[0][1]}, "
                f"DEVICE_COMPILER: {comb[1][0]} {comb[1][1]}, "
                f"CMAKE: {comb[5][1]}",
            )
            self.assertTrue(
                compiler_version_filter_typed(comb),
                f"HOST_COMPILER: {comb[0][0]} {comb[0][1]}, "
                f"DEVICE_COMPILER: {comb[1][0]} {comb[1][1]}, "
                f"CMAKE: {comb[5][1]}",
            )
            self.assertTrue(
                compiler_backend_filter_typed(comb),
                f"HOST_COMPILER: {comb[0][0]} {comb[0][1]}, "
                f"DEVICE_COMPILER: {comb[1][0]} {comb[1][1]}, "
                f"CMAKE: {comb[5][1]}",
            )
            self.assertTrue(
                software_dependency_filter_typed(comb),
                f"HOST_COMPILER: {comb[0][0]} {comb[0][1]}, "
                f"DEVICE_COMPILER: {comb[1][0]} {comb[1][1]}, "
                f"CMAKE: {comb[5][1]}",
            )
            self.assertTrue(
                full_filter_chain(comb),
                f"HOST_COMPILER: {comb[0][0]} {comb[0][1]}, "
                f"DEVICE_COMPILER: {comb[1][0]} {comb[1][1]}, "
                f"CMAKE: {comb[5][1]}",
            )

        invalid_combs = [
            [
                (CLANG_CUDA, "14"),
                (CLANG_CUDA, "14"),
                [(ALPAKA_ACC_GPU_CUDA_ENABLE, "11.2")],
                (CXX_STANDARD, "17"),
                (UBUNTU, "20.04"),
                (CMAKE, "3.18"),
            ],
            [
                (CLANG_CUDA, "14"),
                (CLANG_CUDA, "14"),
                [(ALPAKA_ACC_GPU_CUDA_ENABLE, "11.2")],
                (CXX_STANDARD, "17"),
                (UBUNTU, "20.04"),
                (CMAKE, "3.17"),
            ],
            [
                (CLANG_CUDA, "14"),
                (CLANG_CUDA, "14"),
                [(ALPAKA_ACC_GPU_CUDA_ENABLE, "11.2")],
                (CXX_STANDARD, "17"),
                (UBUNTU, "20.04"),
                (CMAKE, "3.18.3"),
            ],
        ]

        for comb in invalid_combs:
            self.assertTrue(
                general_compiler_filter_typed(comb),
                f"HOST_COMPILER: {comb[0][0]} {comb[0][1]}, "
                f"DEVICE_COMPILER: {comb[1][0]} {comb[1][1]}, "
                f"CMAKE: {comb[5][1]}",
            )
            self.assertTrue(
                compiler_version_filter_typed(comb),
                f"HOST_COMPILER: {comb[0][0]} {comb[0][1]}, "
                f"DEVICE_COMPILER: {comb[1][0]} {comb[1][1]}, "
                f"CMAKE: {comb[5][1]}",
            )
            self.assertTrue(
                compiler_backend_filter_typed(comb),
                f"HOST_COMPILER: {comb[0][0]} {comb[0][1]}, "
                f"DEVICE_COMPILER: {comb[1][0]} {comb[1][1]}, "
                f"CMAKE: {comb[5][1]}",
            )
            self.assertFalse(
                software_dependency_filter_typed(comb),
                f"HOST_COMPILER: {comb[0][0]} {comb[0][1]}, "
                f"DEVICE_COMPILER: {comb[1][0]} {comb[1][1]}, "
                f"CMAKE: {comb[5][1]}",
            )
            self.assertFalse(
                full_filter_chain(comb),
                f"HOST_COMPILER: {comb[0][0]} {comb[0][1]}, "
                f"DEVICE_COMPILER: {comb[1][0]} {comb[1][1]}, "
                f"CMAKE: {comb[5][1]}",
            )
