import unittest


from alpaka_job_coverage.versions import is_supported_version
from alpaka_job_coverage.globals import *
from utils import manual_version_test


class TestSupportedVersion(unittest.TestCase):
    def test_name(self):
        try:
            for name in [
                GCC,
                CLANG,
                NVCC,
                CLANG_CUDA,
                HIPCC,
                ALPAKA_ACC_GPU_CUDA_ENABLE,
                ALPAKA_ACC_GPU_HIP_ENABLE,
            ]:
                is_supported_version(name, "0")
        except ValueError as e:
            self.fail(f'"{name}" should be accepted by is_supported_version()')

        # non_existing_compiler should fail, because the compiler does not exist
        non_existing_compiler = "magic_compiler"
        self.assertRaises(
            ValueError,
            is_supported_version,
            non_existing_compiler,
            "0",
        )

    def test_version(self):
        test_versions: Dict[str, Tuple[List[str], List[str]]] = {
            GCC: (["7", "9", "10"], ["99"]),
            CLANG: (["8", "10", "14"], ["99", "365"]),
            NVCC: (["10.1", "11.4", "12.1"], ["99.9"]),
            HIPCC: (["5.0", "5.3", "5.5"], ["99.9"]),
            UBUNTU: (["18.04", "20.04"], ["99.04", "2004.04"]),
        }

        # manual check of the version
        # the tests only works, if we use the following supported and
        # unsupported versions
        for name, (supported_versions, unsupported_versions) in test_versions.items():
            manual_version_test(self, name, supported_versions, unsupported_versions)

        for name, (supported_versions, unsupported_versions) in test_versions.items():
            for v in supported_versions:
                self.assertTrue(
                    is_supported_version(name, v), f"{name} {v} should be supported."
                )

                # we support the same clang versions also for clang as CUDA compiler
                if name == CLANG:
                    self.assertTrue(
                        is_supported_version(CLANG_CUDA, v),
                        f"{CLANG_CUDA} {v} should be supported.",
                    )

                # the nvcc compiler is bundled with the CUDA SDK and both
                # has the same version
                if name == NVCC:
                    self.assertTrue(
                        is_supported_version(ALPAKA_ACC_GPU_CUDA_ENABLE, v),
                        f"{ALPAKA_ACC_GPU_CUDA_ENABLE} {v} should be supported.",
                    )

                # the hipcc compiler is bundled with the ROCm SDK and both
                # has the same version
                if name == HIPCC:
                    self.assertTrue(
                        is_supported_version(ALPAKA_ACC_GPU_HIP_ENABLE, v),
                        f"{ALPAKA_ACC_GPU_HIP_ENABLE} {v} should be supported.",
                    )

            for v in unsupported_versions:
                self.assertFalse(
                    is_supported_version(name, v),
                    f"{name} {v} should be not supported.",
                )

                # we support the same clang versions also for clang as CUDA compiler
                if name == CLANG:
                    self.assertFalse(
                        is_supported_version(CLANG_CUDA, v),
                        f"{CLANG_CUDA} {v} should be not supported.",
                    )

                # the nvcc compiler is bundled with the CUDA SDK and both
                # has the same version
                if name == NVCC:
                    self.assertFalse(
                        is_supported_version(ALPAKA_ACC_GPU_CUDA_ENABLE, v),
                        f"{ALPAKA_ACC_GPU_CUDA_ENABLE} {v} should be not supported.",
                    )

                # the hipcc compiler is bundled with the ROCm SDK and both
                # has the same version
                if name == HIPCC:
                    self.assertFalse(
                        is_supported_version(ALPAKA_ACC_GPU_HIP_ENABLE, v),
                        f"{ALPAKA_ACC_GPU_HIP_ENABLE} {v} should be not supported.",
                    )

    def test_on_off_backends(self):
        # verify that

        # NVCC and ALPAKA_ACC_GPU_CUDA_ENABLE shares the same versions
        manual_version_test(self, NVCC, [], [ON_VER, "2.7", "3.0"])
        # HIPCC and ALPAKA_ACC_GPU_HIP_ENABLE shares the same versions
        manual_version_test(self, HIPCC, [], [ON_VER, "2.7", "3.0"])

        for backend_name in BACKENDS_LIST:
            if (
                backend_name != ALPAKA_ACC_GPU_CUDA_ENABLE
                and backend_name != ALPAKA_ACC_GPU_HIP_ENABLE
            ):
                for correct_version in [ON_VER, OFF_VER]:
                    self.assertTrue(
                        is_supported_version(backend_name, correct_version),
                        f"{backend_name} {correct_version} should be allowed.",
                    )
            else:
                self.assertTrue(
                    is_supported_version(backend_name, OFF_VER),
                    f"{backend_name} {correct_version} should be allowed.",
                )
                self.assertFalse(
                    is_supported_version(backend_name, ON_VER),
                    f"{backend_name} {correct_version} should be allowed.",
                )

            for correct_version in [
                "2.7",
                "3.0",
            ]:
                self.assertFalse(
                    is_supported_version(backend_name, correct_version),
                    f"{backend_name} {correct_version} is not supported.",
                )
