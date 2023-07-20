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


class TestHipccHostDeviceCompilerBackend(unittest.TestCase):
    def setUp(self):
        global param_map
        # set param_map, that filters expect the following parameters in the
        # order
        param_map[HOST_COMPILER] = 0
        param_map[DEVICE_COMPILER] = 1
        param_map[BACKENDS] = 2

    def setDown(self):
        global param_map
        # reset param_map for following up tests
        param_map = {}

    # test if no combination is forbidden, when only hipcc is the host compiler
    # and no other parameter is set
    def test_only_host_compiler(self):
        valid_combs = [
            [(HIPCC, "0")],
            [(HIPCC, "4.8")],
            [(HIPCC, "5.0")],
            [(HIPCC, "5.5")],
        ]

        for comb in valid_combs:
            self.assertTrue(
                general_compiler_filter_typed(comb),
                f"HOST_COMPILER: {comb[0][0]}",
            )
            self.assertTrue(
                compiler_version_filter_typed(comb),
                f"HOST_COMPILER: {comb[0][0]}",
            )
            self.assertTrue(
                compiler_backend_filter_typed(comb),
                f"HOST_COMPILER: {comb[0][0]}",
            )
            self.assertTrue(
                software_dependency_filter_typed(comb),
                f"HOST_COMPILER: {comb[0][0]}",
            )
            self.assertTrue(
                full_filter_chain(comb),
                f"HOST_COMPILER: {comb[0][0]}",
            )

    # verify that only hipcc can be used, if it set as host and device compiler
    # (ignore compiler version)
    def test_host_device_compiler_name(self):
        valid_combs = [
            [(HIPCC, "0"), (HIPCC, "0")],
        ]

        for comb in valid_combs:
            self.assertTrue(
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
            self.assertTrue(
                full_filter_chain(comb),
                f"HOST_COMPILER: {comb[0][0]}, DEVICE_COMPILER: {comb[1][0]}",
            )

        invalid_combs = [
            # CLANG_CUDA as device compiler is not tested, because a rule
            # forbids to use clang-cuda < 14
            [(HIPCC, "0"), (NVCC, "0")],
            [(HIPCC, "0"), (GCC, "0")],
            [(HIPCC, "0"), (CLANG, "0")],
            [(NVCC, "0"), (HIPCC, "0")],
            [(GCC, "0"), (HIPCC, "0")],
            [(CLANG, "0"), (HIPCC, "0")],
            [(CLANG_CUDA, "0"), (HIPCC, "0")],
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

    # verify that only hipcc can be used, if it set as host and device compiler
    # and has the same compiler version
    def test_host_device_compiler_version(self):
        valid_combs = [
            [(HIPCC, "4.8"), (HIPCC, "4.8")],
            [(HIPCC, "5.0"), (HIPCC, "5.0")],
            [(HIPCC, "5.5"), (HIPCC, "5.5")],
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
            [(HIPCC, "4.4"), (HIPCC, "4.8")],
            [(HIPCC, "5.0"), (HIPCC, "5.2")],
            [(HIPCC, "5.5"), (HIPCC, "4.4")],
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

    # hipcc needs to be set as host and device compiler and the
    # ALPAKA_ACC_GPU_HIP_ENABLE needs to be enabled. All three have to be the
    # same version.
    def test_host_device_compiler_backend(self):
        valid_combs = [
            [(HIPCC, "4.8"), (HIPCC, "4.8"), [(ALPAKA_ACC_GPU_HIP_ENABLE, "4.8")]],
            [(HIPCC, "5.5"), (HIPCC, "5.5"), [(ALPAKA_ACC_GPU_HIP_ENABLE, "5.5")]],
            [
                (HIPCC, "5.0"),
                (HIPCC, "5.0"),
                [
                    (ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE, ON_VER),
                    (ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE, ON_VER),
                    (ALPAKA_ACC_GPU_HIP_ENABLE, "5.0"),
                ],
            ],
            [
                (HIPCC, "5.5"),
                (HIPCC, "5.5"),
                [
                    (ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE, ON_VER),
                    (ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE, ON_VER),
                    (ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, OFF_VER),
                    (ALPAKA_ACC_GPU_HIP_ENABLE, "5.5"),
                    (ALPAKA_ACC_GPU_CUDA_ENABLE, OFF_VER),
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
            # NVCC is not allowed as host compiler
            [(HIPCC, "4.4"), (HIPCC, "4.4"), [(ALPAKA_ACC_GPU_HIP_ENABLE, "4.8")]],
            [(HIPCC, "5.2"), (HIPCC, "5.2"), [(ALPAKA_ACC_GPU_HIP_ENABLE, OFF_VER)]],
            [(HIPCC, "5.3"), (HIPCC, "5.3"), [(ALPAKA_ACC_GPU_HIP_ENABLE, OFF_VER)]],
            [
                (HIPCC, "5.4"),
                (HIPCC, "5.4"),
                [
                    (ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE, ON_VER),
                    (ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE, ON_VER),
                    (ALPAKA_ACC_GPU_HIP_ENABLE, OFF_VER),
                ],
            ],
            # it is forbidden to enable the CUDA and HIP backend on the same
            # time (only for the CI)
            [
                (HIPCC, "5.5"),
                (HIPCC, "5.5"),
                [
                    (ALPAKA_ACC_GPU_CUDA_ENABLE, "11.2"),
                    (ALPAKA_ACC_GPU_HIP_ENABLE, "5.5"),
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
