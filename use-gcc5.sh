#!/usr/bin/env bash

GCC5_HOME="/home/zhangj25/dcqcn-tuning/gcc5install"

export GCC5_HOME
export PATH="$GCC5_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$GCC5_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CC="$GCC5_HOME/bin/gcc"
export CXX="$GCC5_HOME/bin/g++"
export CPP="$GCC5_HOME/bin/cpp"
export AR="$GCC5_HOME/bin/gcc-ar"
export NM="$GCC5_HOME/bin/gcc-nm"
export RANLIB="$GCC5_HOME/bin/gcc-ranlib"

hash -r 2>/dev/null || true