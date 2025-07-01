#!/usr/bin/env python3

import argparse
from packaging import version as pk_version
from typing import Dict, Tuple, Union
from typeguard import typechecked
from io import StringIO

from alpaka_job_coverage.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
import alpaka_job_coverage.filter_compiler_name as ajc_compiler_name
import alpaka_job_coverage.filter_compiler_version as ajc_compiler_version
import alpaka_job_coverage.filter_backend_version as ajc_backend_version
import alpaka_job_coverage.filter_software_dependency as ajc_software_dependency
from alpaka_job_coverage.versions import is_supported_version, versions


@typechecked
def cs(text: str, color: str):
    """Prints colored text to the command line. The text printed after
       the function call has the default color of the command line.

    :param text: text to be colored
    :type text: str
    :param color: Name of the color. If color is unknown or empty use default color
                  of the command line.
    :type color: str
    :returns:
    :rtype:

    """
    if color is None:
        return text

    output = ""
    if color == "Red":
        output += "\033[0;31m"
    elif color == "Green":
        output += "\033[0;32m"
    elif color == "Yellow":
        output += "\033[1;33m"
    else:
        return text

    return output + text + "\033[0m"


@typechecked
def exit_error(text: str):
    """Prints error message and exits application with error code 1.

    Args:
        text (str): Error message.
    """
    print(cs("ERROR: " + text, "Red"))
    exit(1)


@typechecked
def validate_args_compiler(arguments: Dict[str, str]) -> Dict[str, Tuple[str, str]]:
    """Checks if the compiler name exists and the version number is parsable.

    Args:
        arguments (Dict[str, str]): Key needs to be HOST_COMPILER or
        DEVICE_COMPILER and values describes the compiler name and version.

    Returns:
        Dict[str, Tuple[str, str]]: Returns parsed compiler name and version
    """
    validated_args = {}
    known_compiler = [GCC, CLANG, NVCC, CLANG_CUDA, HIPCC, ICPX]

    for parameter_name, parameter in arguments.items():
        if "@" not in parameter:
            exit_error(f"@ is missing in {parameter_name}={parameter}")

        splitted_name_version = parameter.split("@", 1)
        name = splitted_name_version[0]
        version = splitted_name_version[1]

        if name not in known_compiler:
            exit_error(f"Unknown compiler: {name}\nKnown compilers: {known_compiler}")

        # use parse() function to validate that the version has a valid shape
        try:
            pk_version.parse(version)
        except Exception as e:
            exit_error(f"Could not parse version number of {name}: {version}")

        validated_args[parameter_name] = (name, version)

    return validated_args


@typechecked
def validate_args_backend(arguments: List[str]) -> Dict[str, List[Tuple[str, str]]]:
    """Checks if back-end names exist and the version numbers are parsable.

    Args:
        arguments (List[str]): List of back-end names with versions.

    Returns:
        Dict[str, List[Tuple[str, str]]]: List of parsed back-end names with
        versions.
    """
    validated_args = {BACKENDS: []}

    for backend in arguments:
        if backend not in BACKENDS_LIST:
            if "@" not in backend:
                exit_error(f"@ is missing in {backend}")

            splitted_name_version = backend.split("@", 1)
            name = splitted_name_version[0]
            version = splitted_name_version[1]

            if name not in BACKENDS_LIST:
                exit_error(
                    f"Unknown back-end: {name}\nKnown back-ends: {BACKENDS_LIST}"
                )

            # use parse() function to validate that the version has a valid shape
            try:
                pk_version.parse(version)
            except Exception as e:
                exit_error(f"Could not parse version number of {name}: {version}")

            validated_args[BACKENDS].append((name, version))

    return validated_args


@typechecked
class VersionAction(argparse.Action):
    # check if argument has valid version shape
    def __call__(self, parser, namespace, values, option_string):
        try:
            pk_version.parse(values)
        except Exception as e:
            print(cs(f"ERROR: wrong version for {option_string}", "Red"))
            raise e
        setattr(namespace, self.dest, values)


@typechecked
def validate_version_support(
    parameters: Dict[str, Union[Tuple[str, str], List[Tuple[str, str]]]],
):
    """Prints a warning if a software version is not officially supported by the
    library. This function does not exit the application because the versions
    can still be tested with the filter functions.

    Args:
        parameters (Dict[str, Union[Tuple[str, str], List[Tuple[str, str]]]]):
        parameter set of the software versions to test
    """
    for param_name, param_value in parameters.items():
        if param_name == BACKENDS:
            for backend_name, backend_version in param_value:
                if not is_supported_version(backend_name, backend_version):
                    print(
                        cs(
                            f"WARNING: {backend_name} {backend_version} is not officially supported.",
                            "Yellow",
                        )
                    )
        else:
            if not is_supported_version(param_value[0], param_value[1]):
                print(
                    cs(
                        f"WARNING: {param_value[0]} {param_value[1]} is not officially supported.",
                        "Yellow",
                    )
                )


@typechecked
def check_single_filter(
    filter: callable,
    req_params: List[str],
    parameters: Dict[str, Union[Tuple[str, str], List[Tuple[str, str]]]],
) -> bool:
    """Applies a parameter set on the filter function if it fulfils all requirements
    and returns the result. The parameter set needs to contain all parameters which
    are defined in `req_params`.

    Args:
        filter (callable): filter function
        req_params (List[str]): list of required parameter names
        parameters (Dict[str, Union[Tuple[str, str], List[Tuple[str, str]]]]):
        parameter set

    Returns:
        bool: returns True if parameter set passes the filter
    """
    missing_parameters = ""

    # get name of the filter for command line output.
    filter_name: str = filter.__name__

    # Each filter function has also type checked version ending with _typed.
    # Remove _typed for better output.
    if filter_name.endswith("_typed"):
        filter_name = filter_name[: -len("_typed")]

    # check if all required parameter are available
    for req_name in req_params:
        if req_name not in parameters.keys():
            missing_parameters += " " + req_name

    if missing_parameters:
        print(
            cs(
                f"skipped {filter_name}(), missing parameters ->" + missing_parameters,
                "Yellow",
            )
        )
        return False
    else:
        global param_map
        row: List[Union[Tuple[str, str], List[Tuple[str, str]]]] = []

        # convert parameter map to list and configure param_map for the
        # lookup
        for i, (param_name, param) in enumerate(parameters.items()):
            param_map[param_name] = i
            row.append(param)

        # the msg object allows to get information from filter function, why
        # parameter combination does not pass the filter
        msg = StringIO()

        if filter(row, msg):
            print(cs(f"{filter_name}() returns True", "Green"))
            # reset param_map
            param_map = {}
            return True
        else:
            print(cs(f"{filter_name}() returns False", "Red"))
            if msg.getvalue() != "":
                print("  " + msg.getvalue())
            # reset param_map
            param_map = {}
            return False


@typechecked
def check_filters(
    parameters: Dict[str, Union[Tuple[str, str], List[Tuple[str, str]]]],
) -> bool:
    """Applies a parameter set to all available filters in the same order, in the same way
    the ajc library is doing it.

    Args:
        parameters (Dict[str, Union[Tuple[str, str], List[Tuple[str, str]]]]):
        parameter set

    Returns:
        bool: returns True if all filters return True
    """
    all_true = 0
    all_true += int(
        check_single_filter(
            ajc_compiler_name.general_compiler_filter_typed,
            ajc_compiler_name.get_required_parameter(),
            parameters,
        )
    )
    all_true += int(
        check_single_filter(
            ajc_compiler_version.compiler_version_filter_typed,
            ajc_compiler_version.get_required_parameter(),
            parameters,
        )
    )
    all_true += int(
        check_single_filter(
            ajc_backend_version.compiler_backend_filter_typed,
            ajc_backend_version.get_required_parameter(),
            parameters,
        )
    )
    all_true += int(
        check_single_filter(
            ajc_software_dependency.software_dependency_filter_typed,
            ajc_software_dependency.get_required_parameter(),
            parameters,
        )
    )

    # each filter add a one, if it was successful
    return all_true == 4


def main():
    parser = argparse.ArgumentParser(
        description="Check if combination of parameters is valid."
    )

    parser.add_argument(
        "--host-compiler",
        type=str,
        help="Define host compiler. Shape needs to be name@version. "
        "For example gcc@10",
    )

    parser.add_argument(
        "--device-compiler",
        type=str,
        help="Define device compiler. Shape needs to be name@version. "
        "For example nvcc@11.3",
    )

    parser.add_argument(
        "--backends",
        nargs="*",
        help="Define back-ends as whitespace separated list. Each element needs "
        "to have the shape of name@version. For example: "
        "alpaka_ACC_CPU_B_SEQ_T_SEQ_ENABLE@1 "
        "ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE@0 "
        "ALPAKA_ACC_GPU_CUDA_ENABLE@11.3\n"
        "Use the values 0 and 1 to disable or enable a back-end.",
    )

    parser.add_argument(
        "--ubuntu",
        type=str,
        help=f"Ubuntu version. (Supported: {', '.join([x for x in versions[UBUNTU]])})",
    )

    parser.add_argument(
        "--cmake", type=str, action=VersionAction, help="Set CMake version."
    )

    parser.add_argument(
        "--boost", type=str, action=VersionAction, help="Set Boost version."
    )

    parser.add_argument(
        "--cxx",
        type=str,
        help=f"C++ version. (Supported: {', '.join([x for x in versions[CXX_STANDARD]])})",
    )

    parser.add_argument(
        "--print-backends", action="store_true", help="Print all available back-ends."
    )

    parser.add_argument(
        "--print-parameters", action="store_true", help="Print validated parameters."
    )

    args = parser.parse_args()

    if args.print_backends:
        for b in BACKENDS_LIST:
            print(b)
        exit(0)

    parameters = {}

    ############################################################################
    ## parse compiler
    ############################################################################

    if args.host_compiler:
        parameters.update(validate_args_compiler({HOST_COMPILER: args.host_compiler}))

    if args.device_compiler:
        parameters.update(
            validate_args_compiler({DEVICE_COMPILER: args.device_compiler})
        )

    if args.backends:
        parameters.update(validate_args_backend(args.backends))

    if args.ubuntu:
        parameters[UBUNTU] = (UBUNTU, args.ubuntu)

    if args.cmake:
        parameters[CMAKE] = (CMAKE, args.cmake)

    if args.boost:
        parameters[BOOST] = (BOOST, args.boost)

    if args.cxx:
        parameters[CXX_STANDARD] = (CXX_STANDARD, args.cxx)

    if args.print_parameters:
        print(parameters)

    validate_version_support(parameters)
    exit(int(not check_filters(parameters)))


if __name__ == "__main__":
    main()
