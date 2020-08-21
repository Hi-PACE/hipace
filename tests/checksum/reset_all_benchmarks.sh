#!/usr/bin/env bash

set -e

# This file is part of the Hipace test suite.
# It resets all checksum benchmarks one by one,
# based on the current version of the code.

hipace_dir=$(echo $(cd ../.. && pwd))
build_dir=${hipace_dir}/build
checksum_dir=${hipace_dir}/tests/checksum

read -p "This will run \"rm -rf ${build_dir}\" and re-build. If unsure, check the script. proceed? y/n: " -n 1 -r
echo
if [[ $REPLY != "y" ]]
then
    echo "No, abort."
    exit 0
fi
echo "Yes, proceed."

### Compile code and reset benchmarks: serial ###
#################################################

rm -rf $build_dir
mkdir $build_dir
cd $build_dir
cmake .. -DHiPACE_MPI=OFF
make -j 4

# can_beam.Serial
cd $build_dir
ctest --output-on-failure -R can_beam.Serial \
    || echo "ctest command failed, maybe just because checksums are different. Keep going"
cd $checksum_dir
./checksumAPI.py --reset-benchmark \
                 --plotfile ${build_dir}/bin/plt00001 \
                 --test-name can_beam.Serial

# beam_in_vacuum.SI.Serial
cd $build_dir
ctest --output-on-failure -R beam_in_vacuum.SI.Serial \
    || echo "ctest command failed, maybe just because checksums are different. Keep going"
cd $checksum_dir
./checksumAPI.py --reset-benchmark \
                 --plotfile ${build_dir}/bin/plt00001 \
                 --test-name beam_in_vacuum.SI.Serial

# beam_in_vacuum.normalized.Serial
cd $build_dir
ctest --output-on-failure -R beam_in_vacuum.normalized.Serial \
    || echo "ctest command failed, maybe just because checksums are different. Keep going"
cd $checksum_dir
./checksumAPI.py --reset-benchmark \
                 --plotfile ${build_dir}/bin/plt00001 \
                 --test-name beam_in_vacuum.normalized.Serial

### Compile code and reset benchmarks: parallel ###
###################################################

rm -rf $build_dir
mkdir $build_dir
cd $build_dir
cmake .. -DHiPACE_MPI=ON
make -j 4

# can_beam.1Rank
cd $build_dir
ctest --output-on-failure -R can_beam.1Rank \
    || echo "ctest command failed, maybe just because checksums are different. Keep going"
cd $checksum_dir
./checksumAPI.py --reset-benchmark \
                 --plotfile ${build_dir}/bin/plt00001 \
                 --test-name can_beam.1Rank

# can_beam.2Rank
cd $build_dir
ctest --output-on-failure -R can_beam.2Rank \
    || echo "ctest command failed, maybe just because checksums are different. Keep going"
cd $checksum_dir
./checksumAPI.py --reset-benchmark \
                 --plotfile ${build_dir}/bin/plt00001 \
                 --test-name can_beam.2Rank
