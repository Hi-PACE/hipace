#! /usr/bin/env bash

# Copyright 2025
#
# This file is part of HiPACE++.
#
# Authors: Luca Fedeli
# License: BSD-3-Clause-LBNL

# This script is a developer's tool to perform
# checks with clang-tidy.
#
# Note: this script is only tested on Linux

set -eu -o pipefail

echo "============================================="
echo
echo "This script is a developer's tool to perform"
echo "checks with clang-tidy."
echo "_____________________________________________"

# Check source dir
REPO_DIR=$(cd $(dirname ${BASH_SOURCE})/../ && pwd)
echo
echo "Your current source directory is: ${REPO_DIR}"
echo "_____________________________________________"

# Set number of jobs to use for compilation
PARALLEL="${HIPACE_TOOLS_LINTER_PARALLEL:-4}"
echo
echo "${PARALLEL} jobs will be used for compilation."
echo "This can be overridden by setting the environment"
echo "variable HIPACE_TOOLS_LINTER_PARALLEL, e.g.: "
echo
echo "$ export HIPACE_TOOLS_LINTER_PARALLEL=8"
echo "$ ./tools/runClangTidy.sh"
echo "_____________________________________________"

# Check clang version
export CC="${CLANG:-"clang"}"
export CXX="${CLANGXX:-"clang++"}"
export CTIDY="${CLANGTIDY:-"clang-tidy"}"
echo
echo "The following versions of the clang compiler and"
echo "of the clang-tidy linter will be used:"
echo
echo "clang version:"
which ${CC}
${CC} --version
echo
echo "clang++ version:"
which ${CXX}
${CXX} --version
echo
echo "clang-tidy version:"
which ${CTIDY}
${CTIDY} --version
echo
echo "This can be overridden by setting the environment"
echo "variables CLANG, CLANGXX, and CLANGTIDY e.g.: "
echo "$ export CLANG=clang-20"
echo "$ export CLANGXX=clang++-20"
echo "$ export CTIDCLANGTIDYY=clang-tidy-20"
echo "$ ./tools/runClangTidy.sh"
echo "_____________________________________________"

# Prepare clang-tidy wrapper
echo
echo "Prepare clang-tidy wrapper"
echo "The following wrapper ensures that only source files"
echo "in hipace/src/* are actually processed by clang-tidy"
echo
cat > ${REPO_DIR}/clang_tidy_wrapper << EOF
#!/bin/bash
REGEX="[a-z_A-Z0-9\/]*hipace\/src[a-z_A-Z0-9\/]+.cpp"
if [[ \$4 =~ \$REGEX ]];then
  ${CTIDY} \$@
fi
EOF
chmod +x ${REPO_DIR}/clang_tidy_wrapper
echo "clang_tidy_wrapper: "
cat ${REPO_DIR}/clang_tidy_wrapper
echo "_____________________________________________"

# Compile HiPACE++ using clang-tidy
echo
echo "*******************************************"
echo "* Compile HiPACE++ using clang-tidy       *"
echo "* Please ensure that all the dependencies *"
echo "* required to compile HiPACE++ are met    *"
echo "*******************************************"
echo

rm -rf ${REPO_DIR}/build_clang_tidy

cmake -S ${REPO_DIR} -B ${REPO_DIR}/build_clang_tidy \
  -DCMAKE_CXX_CLANG_TIDY="${REPO_DIR}/clang_tidy_wrapper;--system-headers=0;--config-file=${REPO_DIR}/.clang-tidy" \
  -DCMAKE_VERBOSE_MAKEFILE=ON  \
  -DHiPACE_MPI=ON              \
  -DHiPACE_COMPUTE=OMP         \
  -DHiPACE_OPENPMD=ON          \
  -DHiPACE_PRECISION=SINGLE

cmake --build ${REPO_DIR}/build_clang_tidy -j ${PARALLEL} 2> ${REPO_DIR}/build_clang_tidy/clang-tidy.log

if [ -s ${REPO_DIR}/build_clang_tidy/clang-tidy.log ]; then
    echo
    echo "clang-tidy found the following issues:"
    echo
    cat ${REPO_DIR}/build_clang_tidy/clang-tidy.log
    echo
    echo "============================================="
    exit 1
else
    echo
    echo "clang-tidy has not found any issue."
    echo
    echo "============================================="
    exit 0
fi
