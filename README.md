# alpaka-job-matrix-library
A library to provide a job generator for CI's for alpaka based projects.

The library provides everything needed to generate a sparse combination matrix for alpaca-based projects, including a set of general-purpose combination rules.

The provision of the input parameters, the reordering of the jobs, the filtering of the job matrix and the generation of the job yaml are project specific. Therefore, the library provides an example of how most parts can be implemented.

# Usage

The main function of the library is `create_job_list()`. It takes a list of parameters, creates the combinations, applies the combination rules, thins them and returns the sparse job matrix.

The thinning is done according to the principle [all-pairs testing](https://en.wikipedia.org/wiki/All-pairs_testing). The principle means that every combination of the values of at least two parameters must be part of a job, unless a filter rule forbids this. The `pair_size` parameter of the `create_job_list()` function decides how large the combination tuple must be. For example, if we have the parameter fields `A, B, C` and `D` and pair size 2, each combination of the values of `AB, AC, AD, BC, BD and CD` must be part of a job. If the parameter is 3, any combination of the values of `ABC, ABD and BCD` must be part of a job. Normally, a larger pairwise factor increases the calculation time and the number of orders.  

The general form of the parameter matrix is an `OrderedDict` of `List[Tuples[str, str]]`. The first value of a tuple is the name of the software and the second value is the version. An exception is the parameter field `BACKENDS`. `BACKENDS` is a `list[list[tuple[str, str]]`. The inner list contains a combination of alpaka backends. This can be a complete combination matrix of all backends (the inner list contains n entries), or it can be only one backend (size of the inner list is 1), as required for [cupla](https://github.com/alpaka-group/cupla). A mixture of both is also possible, e.g. `[(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, ON), (ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, ON), (ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE, ON)],[(ALPAKA_ACC_GPU_CUDA_ENABLE, "11. 0")],[(ALPAKA_ACC_GPU_CUDA_ENABLE, "11.0")] ...]`.

In order to apply the filter rules correctly, it is necessary to use the variables defined in `alpaka_job_coverage.global`.

There are 3 parameters with special meaning:
* `HOST_COMPILER`: Compiler for the host code, or if there is no host device code separation, the compiler for all the code.
* `DEVICE_COMPILER`: Compiler for the device code, or if there is no host device code separation, the compiler is the same as the host compiler.
* `BACKENDS`: See description above.

If one of these 3 parameter fields is missing, it is not guaranteed that the generator will provide correct results. All other parameter fields provided by the `alpaka_job_coverage.global` are optional.

## Adding own parameter and rules

If you want to use a project-specific parameter, you can simply add it to the parameter list and the library will apply it. To limit the possible combinations of your new parameter, you need to add a new filter function. The `create_job_list()` function applies a chain of filter functions to each possible combination. Before and after each filter function is a function hook where you can insert your own filter function. The order is described in the doc string of `create_job_list()`. When you create a new filter rule, you must comply with the following rules:

* The filter returns `True` if a combination is valid and `False` if not. All filters in the library follow the rule that every combination is valid until it is forbidden (blacklist).
* The input of the filter is a combination of the values of the individual parameter fields, and the combination does not have to be complete. The list can contain at least 2 parameter fields up to all. You must check whether a parameter field is included in the current combination.
* If a parameter field is not included in the current combination, it means that it can contain any possible value of this parameter. In practice, this means that if you only check for the presence of the parameter and return `False`, if the parameter is not present, no combination is possible.

# Running the example

First you have to install the required packages via `pip3 install -r requirements.txt`.

Then you can run the example via `python3 3.0 example/example.py`. By default, it creates a `job.yml` in the current directory. To get more output, you can run `python3 3.0 example/example.py -a`. Run `python3 --help` to see all the options.
