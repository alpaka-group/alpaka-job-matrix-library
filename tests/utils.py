from alpaka_job_coverage.versions import versions
from typeguard import typechecked
from typing import List


@typechecked
def manual_version_test(
    test_class,
    name: str,
    supported_versions: List[str],
    unsupported_versions: List[str],
):
    """Do a manual check, if software version is included in
    alpaka_job_coverage.versions.versions

    Throws an unittest test failure in the case of a failure.

    Args:
        test_class (_type_): Reference of the unittest class. Normally pass the
        self object inside a unittest test function
        name (str): name of the software
        supported_versions (List[str]): List of version numbers, which have to
        be supported.
        unsupported_versions (List[str]): List of version numbers, which have
        not to be supported.
    """
    for version in supported_versions:
        test_class.assertTrue(
            version in versions[name],
            f"{name} {version} needs to be a supported version for the following up tests.",
        )

    test_class.assertTrue(
        "0" not in versions[name],
        f"{name} 0 exists. If version 0 is used for special cases, "
        "you need to modify the following up tests.",
    )

    for version in unsupported_versions:
        test_class.assertTrue(
            version not in versions[name],
            f"We expect, that {name} {version} is not released. If it is "
            "released, you need to changes the following up tests.",
        )
