import unittest

from alpaka_job_coverage.util import backend_is_not_in_row
from alpaka_job_coverage.globals import *

# the shape of the row function parameter is described in the module
# documentation of alpaka_job_coverage.util


class TestBackendIsNotInRowBackendsOnly(unittest.TestCase):
    def setUp(self):
        global param_map
        param_map[BACKENDS] = 0

    def setDown(self):
        global param_map
        param_map = {}

    def test_empty_backend_list(self):
        self.assertTrue(backend_is_not_in_row([], ALPAKA_ACC_GPU_CUDA_ENABLE))

    def test_is_not_in_filled_list_single_element(self):
        self.assertTrue(
            backend_is_not_in_row(
                [[(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, ON_VER)]],
                ALPAKA_ACC_GPU_CUDA_ENABLE,
            )
        )

    def test_is_not_in_filled_list_multi_element(self):
        self.assertTrue(
            backend_is_not_in_row(
                [
                    [
                        (ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, ON_VER),
                        (ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE, ON_VER),
                        (ALPAKA_ACC_GPU_HIP_ENABLE, "5.5"),
                    ],
                ],
                ALPAKA_ACC_GPU_CUDA_ENABLE,
            )
        )

    def test_is_in_filled_list_multi_element(self):
        self.assertFalse(
            backend_is_not_in_row(
                [
                    [
                        (ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, ON_VER),
                        (ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE, ON_VER),
                        (ALPAKA_ACC_GPU_HIP_ENABLE, "5.5"),
                        (ALPAKA_ACC_GPU_CUDA_ENABLE, "11.4"),
                    ],
                ],
                ALPAKA_ACC_GPU_CUDA_ENABLE,
            )
        )

        self.assertFalse(
            backend_is_not_in_row(
                [
                    [
                        (ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, ON_VER),
                        (ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE, ON_VER),
                        (ALPAKA_ACC_GPU_CUDA_ENABLE, OFF_VER),
                        (ALPAKA_ACC_GPU_HIP_ENABLE, "5.5"),
                    ],
                ],
                ALPAKA_ACC_GPU_CUDA_ENABLE,
            )
        )


class TestBackendIsNotInRowMixed(unittest.TestCase):
    def setUp(self):
        global param_map
        param_map[HOST_COMPILER] = 0
        param_map[DEVICE_COMPILER] = 1
        param_map[BACKENDS] = 2
        param_map[CMAKE] = 3

    def setDown(self):
        global param_map
        param_map = {}

    def test_empty_backend_list(self):
        self.assertTrue(backend_is_not_in_row([], ALPAKA_ACC_GPU_CUDA_ENABLE))

    def test_backend_not_included(self):
        self.assertTrue(
            backend_is_not_in_row([(GCC, "9"), (GCC, "9")], ALPAKA_ACC_GPU_CUDA_ENABLE)
        )

    def test_single_backend(self):
        self.assertTrue(
            backend_is_not_in_row(
                [(GCC, "9"), (GCC, "9"), [(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, ON_VER)]],
                ALPAKA_ACC_GPU_CUDA_ENABLE,
            )
        )

        self.assertFalse(
            backend_is_not_in_row(
                [(GCC, "9"), (GCC, "9"), [(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, ON_VER)]],
                ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE,
            )
        )

    def test_multiple_backend(self):
        self.assertTrue(
            backend_is_not_in_row(
                [
                    (GCC, "9"),
                    (GCC, "9"),
                    [
                        (ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, ON_VER),
                        (ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE, ON_VER),
                        (ALPAKA_ACC_GPU_HIP_ENABLE, "5.5"),
                    ],
                ],
                ALPAKA_ACC_GPU_CUDA_ENABLE,
            )
        )

        self.assertFalse(
            backend_is_not_in_row(
                [
                    (GCC, "9"),
                    (GCC, "9"),
                    [
                        (ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, ON_VER),
                        (ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE, ON_VER),
                        (ALPAKA_ACC_GPU_HIP_ENABLE, "5.5"),
                        (ALPAKA_ACC_GPU_CUDA_ENABLE, "11.4"),
                    ],
                ],
                ALPAKA_ACC_GPU_CUDA_ENABLE,
            )
        )

    def test_multiple_backends_with_following_parameter(self):
        self.assertTrue(
            backend_is_not_in_row(
                [
                    (GCC, "9"),
                    (GCC, "9"),
                    [
                        (ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, ON_VER),
                        (ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE, ON_VER),
                        (ALPAKA_ACC_GPU_HIP_ENABLE, "5.5"),
                    ],
                    (CMAKE, "3.25"),
                ],
                ALPAKA_ACC_GPU_CUDA_ENABLE,
            )
        )

        self.assertFalse(
            backend_is_not_in_row(
                [
                    (GCC, "9"),
                    (GCC, "9"),
                    [
                        (ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, ON_VER),
                        (ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE, ON_VER),
                        (ALPAKA_ACC_GPU_HIP_ENABLE, "5.5"),
                        (ALPAKA_ACC_GPU_CUDA_ENABLE, "11.4"),
                    ],
                    (CMAKE, "3.25"),
                ],
                ALPAKA_ACC_GPU_CUDA_ENABLE,
            )
        )
