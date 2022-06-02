import setuptools
import os


# the setup.py is executed in an environment, therefore files can not be open via relative
cwd = os.path.abspath(os.path.dirname(__file__))

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# attention, the version.txt needs to be added to the MANIFEST.in, otherwise the file is not
# available during build
with open("version.txt", "r", encoding="utf-8") as fh:
    version = fh.read()

setuptools.setup(
    name="alpaka-job-coverage",
    version=version,
    author="Simeon Ehrig",
    author_email="s.ehrig@hzdr.de",
    description="The library provides everything needed to generate a sparse combination matrix for alpaca-based projects, including a set of general-purpose combination rules.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/alpaka-group/alpaka-job-matrix-library",
    project_urls={
        "Bug Tracker": "https://github.com/alpaka-group/alpaka-job-matrix-library/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages('src'),
    package_dir={'': 'src'},
    python_requires=">=3.8",
    install_requires=[
        "allpairspy == 2.5.0",
        "typeguard",
        "pyaml",
        "types-PyYAML",
        "packaging",
    ]
)

