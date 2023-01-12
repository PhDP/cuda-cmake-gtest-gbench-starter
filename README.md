# CUDA/C++14 cmake starter with google test and google benchmark
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

A cross-platform CUDA/C++14 starter project with google test (1.12.1) and
google benchmark (v1.7.1) support. [See this
project](https://github.com/PhDP/cmake-gtest-gbench-starter) for a similar
template without CUDA support.

This project template is using git submodule to include Google Benchmark and Google Test so you
need either to do

    $ git clone --recursive git@github.com:PhDP/cuda-cmake-gtest-gbench-starter.git

or

    $ git clone git@github.com:PhDP/cuda-cmake-gtest-gbench-starter.git
    $ cd cuda-cmake-gtest-gbench-starter
    $ git submodule init
    $ git submodule update

# Build

On Linux/Unix, to build and make the test:

    $ mkdir build && cd $_
    $ cmake ..
    $ make

CUDA is strict about compiler version, on UNIX, cmake will honor the CXX variable, so for example
to use gcc 8 you can write (before calling cmake):

    $ export CC=gcc-8
    $ export CXX=g++-8

or

    $ CC=gcc-8 CXX=g++-8 cmake ..

By default, the makefiles will build the library, executable, tests, and benchmarks. The commands

    $ ./test/test_deepgreen
    $ ./bench/bench_deepgreen

...will run the tests and benchmarks.

On Windows, you can use cmake to generate Visual Studio build files with the same 'cmake ..'
command.

By default, the project will be built in RELEASE mode, use

    $ cmake .. -DCMAKE_BUILD_TYPE=DEBUG

to build in DEBUG mode.

See the CMakeLists.txt file to see all the options.

# License

[MIT](http://opensource.org/licenses/MIT)

