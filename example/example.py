"""Generate GitLab-CI test jobs yaml for the example CI."""
import argparse
import sys
from typing import List, Dict, Tuple
from collections import OrderedDict

import alpaka_job_coverage as ajc
from alpaka_job_coverage.util import filter_job_list, reorder_job_list
from alpaka_job_coverage.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import

from example_globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from versions import (
    get_sw_tuple_list,
    get_compiler_versions,
    get_backend_combination_matrix,
    get_backend_single_matrix,
)
from example_filter import example_post_filter
from reorder_jobs import reorder_jobs
from generate_job_yaml import generate_job_yaml_list, write_job_yaml
from verify import verify


def get_args() -> argparse.Namespace:
    """Define and parse the commandline arguments.

    Returns:
        argparse.Namespace: The commandline arguments.
    """
    parser = argparse.ArgumentParser(
        description="Calculate job matrix and create GitLab CI .yml."
    )

    parser.add_argument(
        "version", type=float, help="Version number of the used CI container."
    )
    parser.add_argument(
        "--print-combinations",
        action="store_true",
        help="Display combination matrix.",
    )
    parser.add_argument(
        "--verify", action="store_true", help="Verify generated combination matrix"
    )
    parser.add_argument(
        "-a",
        "--all",
        action="store_true",
        help="Combine flags: --print-combinations and --verify",
    )

    parser.add_argument(
        "-o",
        "--output-path",
        type=str,
        default="./jobs.yml",
        help="Path of the generated jobs yaml.",
    )

    parser.add_argument(
        "--filter",
        type=str,
        default="",
        help="Filter the jobs with a Python regex that checks the job names.",
    )

    parser.add_argument(
        "--reorder",
        type=str,
        default="",
        help="Orders jobs by their names. Expects a string consisting of one or more Python regex. "
        'The regex are separated by whitespaces. For example, the regex "^NVCC ^GCC" has the '
        "behavior that all NVCC jobs are executed first and then all GCC jobs.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    # setup the parameters
    parameters: OrderedDict = OrderedDict()
    parameters[HOST_COMPILER] = get_compiler_versions()
    parameters[DEVICE_COMPILER] = get_compiler_versions()
    # change the comment of the following two lines to use a different backend combination matrix
    # as input
    parameters[BACKENDS] = get_backend_combination_matrix()
    # parameters[BACKENDS] = get_backend_single_matrix()
    parameters[CMAKE] = get_sw_tuple_list(CMAKE)
    parameters[BOOST] = get_sw_tuple_list(BOOST)
    parameters[ALPAKA] = get_sw_tuple_list(ALPAKA)
    parameters[UBUNTU] = get_sw_tuple_list(UBUNTU)
    parameters[CXX_STANDARD] = get_sw_tuple_list(CXX_STANDARD)

    job_matrix: List[Dict[str, Tuple[str, str]]] = ajc.create_job_list(
        parameters=parameters,
        post_filter=example_post_filter,
        pair_size=2,
    )

    if args.print_combinations or args.all:
        print(f"number of combinations before reorder: {len(job_matrix)}")

    ajc.shuffle_job_matrix(job_matrix)
    # it is also possible to read existing jobs from a yaml file and add them to the job_matrix
    reorder_jobs(job_matrix)

    if args.print_combinations or args.all:
        for compiler in job_matrix:
            print(compiler)

        print(f"number of combinations: {len(job_matrix)}")

    job_matrix_yaml = generate_job_yaml_list(
        job_matrix=job_matrix, container_version=args.version
    )

    if args.filter:
        job_matrix_yaml = filter_job_list(job_matrix_yaml, args.filter)

    if args.reorder:
        job_matrix_yaml = reorder_job_list(job_matrix_yaml, args.reorder)

    wave_job_matrix = ajc.distribute_to_waves(job_matrix_yaml, 10)

    if args.verify or args.all:
        if not verify(job_matrix):
            sys.exit(1)

    write_job_yaml(
        job_matrix=wave_job_matrix,
        path=args.output_path,
    )
