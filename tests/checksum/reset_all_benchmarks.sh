#!/usr/bin/env bash

# Copyright 2020-2022
#
# This file is part of HiPACE++.
#
# Authors: AlexanderSinn, Axel Huebl, MaxThevenet, Severin Diederichs
#
# License: BSD-3-Clause-LBNL


# This file is part of the HiPACE++ test suite.
# It resets all checksum benchmarks one by one,
# based on the current version of the code.
#
# Usage:
#
# To reset all benchmarks, run:
# > ./reset_all_benchmarks.sh
#
# To reset only benchmark <benchmark_name>, run
# > /reset_all_benchmarks.sh <benchmark_name>

# abort on first encounted error
set -eu -o pipefail

# Check if the user wants to reset one benchmark or all of them
if [ "$#" -ne 1 ]
then
    all_tests=true
    one_test_name="no"
else
    all_tests=false
    one_test_name=$1
fi

# Depending on the user input, recompile serial and/or parallel
compile_serial=false
compile_parallel=false
if [[ $one_test_name == *"Serial" ]] || [[ $all_tests = true ]]
then
    compile_serial=true
fi
if [[ $one_test_name == *"Rank" ]] || [[ $all_tests = true ]]
then
    compile_parallel=true
fi

echo "Run all tests   : $all_tests"
echo "Run single test : $one_test_name"
echo "Compile serial  : $compile_serial"
echo "Compile parallel: $compile_parallel"

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

if [[ $compile_serial = true ]]
then
    rm -rf $build_dir
    mkdir $build_dir
    cd $build_dir
    cmake .. -DHiPACE_MPI=OFF
    make -j 4
fi

#blowout_wake.Serial
if [[ $all_tests = true ]] || [[ $one_test_name = "blowout_wake.Serial" ]]
then
   cd $build_dir
   ctest --output-on-failure -R blowout_wake.Serial \
       || echo "ctest command failed, maybe just because checksums are different. Keep going"
   cd $checksum_dir
   ./checksumAPI.py --reset-benchmark \
                    --file_name ${build_dir}/bin/blowout_wake.Serial \
                    --test-name blowout_wake.Serial
fi

#beam_in_vacuum.SI.Serial
if [[ $all_tests = true ]] || [[ $one_test_name = "beam_in_vacuum.SI.Serial" ]]
then
   cd $build_dir
   ctest --output-on-failure -R beam_in_vacuum.SI.Serial \
       || echo "ctest command failed, maybe just because checksums are different. Keep going"
   cd $checksum_dir
   ./checksumAPI.py --reset-benchmark \
                    --file_name ${build_dir}/bin/beam_in_vacuum.SI.Serial \
                    --test-name beam_in_vacuum.SI.Serial
fi

#beam_in_vacuum.normalized.Serial
if [[ $all_tests = true ]] || [[ $one_test_name = "beam_in_vacuum.normalized.Serial" ]]
then
   cd $build_dir
   ctest --output-on-failure -R beam_in_vacuum.normalized.Serial \
       || echo "ctest command failed, maybe just because checksums are different. Keep going"
   cd $checksum_dir
   ./checksumAPI.py --reset-benchmark \
                    --file_name ${build_dir}/bin/beam_in_vacuum.normalized.Serial \
                    --test-name beam_in_vacuum.normalized.Serial
fi

### Compile code and reset benchmarks: parallel ###
###################################################

if [[ $compile_parallel = true ]]
then
    rm -rf $build_dir
    mkdir $build_dir
    cd $build_dir
    cmake .. -DHiPACE_MPI=ON
    make -j 4
fi

#blowout_wake.2Rank
if [[ $all_tests = true ]] || [[ $one_test_name = "blowout_wake.2Rank" ]]
then
    cd $build_dir
    ctest --output-on-failure -R blowout_wake.2Rank \
        || echo "ctest command failed, maybe just because checksums are different. Keep going"
    cd $checksum_dir
    ./checksumAPI.py --reset-benchmark \
                     --file_name ${build_dir}/bin/normalized_data \
                     --test-name blowout_wake.2Rank
fi

##hosing.2Rank
#if [[ $all_tests = true ]] || [[ $one_test_name = "hosing.2Rank" ]]
#then
#    cd $build_dir
#    ctest --output-on-failure -R hosing.2Rank \
#        || echo "ctest command failed, maybe just because checksums are different. Keep going"
#    cd $checksum_dir
#    ./checksumAPI.py --reset-benchmark \
#                     --file_name ${build_dir}/bin/hosing_data \
#                     --test-name hosing.2Rank
#fi


#ion_motion.SI.1Rank
if [[ $all_tests = true ]] || [[ $one_test_name = "ion_motion.SI.1Rank" ]]
then
    cd $build_dir
    ctest --output-on-failure -R ion_motion.SI.1Rank \
        || echo "ctest command failed, maybe just because checksums are different. Keep going"
    cd $checksum_dir
    ./checksumAPI.py --reset-benchmark \
                     --file_name ${build_dir}/bin/ion_motion.SI.1Rank/e \
                     --test-name ion_motion.SI.1Rank
fi

#ionization.2Rank
if [[ $all_tests = true ]] || [[ $one_test_name = "ionization.2Rank" ]]
then
    cd $build_dir
    ctest --output-on-failure -R ionization.2Rank \
        || echo "ctest command failed, maybe just because checksums are different. Keep going"
    cd $checksum_dir
    ./checksumAPI.py --reset-benchmark \
                     --file_name ${build_dir}/bin/ionization.2Rank \
                     --test-name ionization.2Rank
fi

#blowout_wake_explicit.2Rank
if [[ $all_tests = true ]] || [[ $one_test_name = "blowout_wake_explicit.2Rank" ]]
then
    cd $build_dir
    ctest --output-on-failure -R blowout_wake_explicit.2Rank \
        || echo "ctest command failed, maybe just because checksums are different. Keep going"
    cd $checksum_dir
    ./checksumAPI.py --reset-benchmark \
                     --file_name ${build_dir}/bin/blowout_wake_explicit.2Rank/ \
                     --test-name blowout_wake_explicit.2Rank
fi

#laser_blowout_wake_explicit.1Rank
if [[ $all_tests = true ]] || [[ $one_test_name = "laser_blowout_wake_explicit.1Rank" ]]
then
    cd $build_dir
    ctest --output-on-failure -R laser_blowout_wake_explicit.1Rank \
        || echo "ctest command failed, maybe just because checksums are different. Keep going"
    cd $checksum_dir
    ./checksumAPI.py --reset-benchmark \
                     --file_name ${build_dir}/bin/laser_blowout_wake_explicit.1Rank/ \
                     --test-name laser_blowout_wake_explicit.1Rank \
                     --skip-particles
fi

#laser_blowout_wake_explicit.SI.1Rank
if [[ $all_tests = true ]] || [[ $one_test_name = "laser_blowout_wake_explicit.SI.1Rank" ]]
then
    cd $build_dir
    ctest --output-on-failure -R laser_blowout_wake_explicit.SI.1Rank \
        || echo "ctest command failed, maybe just because checksums are different. Keep going"
    cd $checksum_dir
    ./checksumAPI.py --reset-benchmark \
                     --file_name ${build_dir}/bin/laser_blowout_wake_explicit.SI.1Rank/ \
                     --test-name laser_blowout_wake_explicit.SI.1Rank \
                     --skip-particles
fi

# beam_evolution.1Rank
if [[ $all_tests = true ]] || [[ $one_test_name = "beam_evolution.1Rank" ]]
then
    cd $build_dir
    ctest --output-on-failure -R beam_evolution.1Rank \
        || echo "ctest command failed, maybe just because checksums are different. Keep going"
    cd $checksum_dir
    ./checksumAPI.py --reset-benchmark \
                     --file_name ${build_dir}/bin/beam_evolution.1Rank/ \
                     --test-name beam_evolution.1Rank
fi

# adaptive_time_step.1Rank
if [[ $all_tests = true ]] || [[ $one_test_name = "adaptive_time_step.1Rank" ]]
then
    cd $build_dir
    ctest --output-on-failure -R adaptive_time_step.1Rank \
        || echo "ctest command failed, maybe just because checksums are different. Keep going"
    cd $checksum_dir
    ./checksumAPI.py --reset-benchmark \
                     --file_name ${build_dir}/bin/adaptive_time_step.1Rank \
                     --test-name adaptive_time_step.1Rank
fi

# radiation_reaction.1Rank
if [[ $all_tests = true ]] || [[ $one_test_name = "radiation_reaction.1Rank" ]]
then
    cd $build_dir
    ctest --output-on-failure -R radiation_reaction.1Rank \
        || echo "ctest command failed, maybe just because checksums are different. Keep going"
    cd $checksum_dir
    ./checksumAPI.py --reset-benchmark \
                     --file_name ${build_dir}/bin/radiation_reaction.1Rank \
                     --test-name radiation_reaction.1Rank
fi

# grid_current.1Rank
if [[ $all_tests = true ]] || [[ $one_test_name = "grid_current.1Rank" ]]
then
    cd $build_dir
    ctest --output-on-failure -R grid_current.1Rank \
        || echo "ctest command failed, maybe just because checksums are different. Keep going"
    cd $checksum_dir
    ./checksumAPI.py --reset-benchmark \
                     --file_name ${build_dir}/bin/grid_current.1Rank \
                     --test-name grid_current.1Rank
fi

# reset.2Rank
if [[ $all_tests = true ]] || [[ $one_test_name = "reset.2Rank" ]]
then
    cd $build_dir
    ctest --output-on-failure -R reset.2Rank \
        || echo "ctest command failed, maybe just because checksums are different. Keep going"
    cd $checksum_dir
    ./checksumAPI.py --reset-benchmark \
                     --file_name ${build_dir}/bin/reset.2Rank \
                     --test-name reset.2Rank
fi

# linear_wake.normalized.1Rank
if [[ $all_tests = true ]] || [[ $one_test_name = "linear_wake.normalized.1Rank" ]]
then
    cd $build_dir
    ctest --output-on-failure -R linear_wake.normalized.1Rank \
        || echo "ctest command failed, maybe just because checksums are different. Keep going"
    cd $checksum_dir
    ./checksumAPI.py --reset-benchmark \
                     --file_name ${build_dir}/bin/linear_wake.normalized.1Rank \
                     --test-name linear_wake.normalized.1Rank
fi

# linear_wake.SI.1Rank
if [[ $all_tests = true ]] || [[ $one_test_name = "linear_wake.SI.1Rank" ]]
then
    cd $build_dir
    ctest --output-on-failure -R linear_wake.SI.1Rank \
        || echo "ctest command failed, maybe just because checksums are different. Keep going"
    cd $checksum_dir
    ./checksumAPI.py --reset-benchmark \
                     --file_name ${build_dir}/bin/linear_wake.SI.1Rank \
                     --test-name linear_wake.SI.1Rank
fi

# gaussian_linear_wake.normalized.1Rank
if [[ $all_tests = true ]] || [[ $one_test_name = "gaussian_linear_wake.normalized.1Rank" ]]
then
    cd $build_dir
    ctest --output-on-failure -R gaussian_linear_wake.normalized.1Rank \
        || echo "ctest command failed, maybe just because checksums are different. Keep going"
    cd $checksum_dir
    ./checksumAPI.py --reset-benchmark \
                     --file_name ${build_dir}/bin/gaussian_linear_wake.normalized.1Rank \
                     --test-name gaussian_linear_wake.normalized.1Rank
fi

# gaussian_linear_wake.SI.1Rank
if [[ $all_tests = true ]] || [[ $one_test_name = "gaussian_linear_wake.SI.1Rank" ]]
then
    cd $build_dir
    ctest --output-on-failure -R gaussian_linear_wake.SI.1Rank \
        || echo "ctest command failed, maybe just because checksums are different. Keep going"
    cd $checksum_dir
    ./checksumAPI.py --reset-benchmark \
                     --file_name ${build_dir}/bin/gaussian_linear_wake.SI.1Rank \
                     --test-name gaussian_linear_wake.SI.1Rank
fi

# beam_in_vacuum.SI.1Rank
if [[ $all_tests = true ]] || [[ $one_test_name = "beam_in_vacuum.SI.1Rank" ]]
then
    cd $build_dir
    ctest --output-on-failure -R beam_in_vacuum.SI.1Rank \
        || echo "ctest command failed, maybe just because checksums are different. Keep going"
    cd $checksum_dir
    ./checksumAPI.py --reset-benchmark \
                     --file_name ${build_dir}/bin/beam_in_vacuum.SI.1Rank \
                     --test-name beam_in_vacuum.SI.1Rank
fi

# beam_in_vacuum.normalized.1Rank
if [[ $all_tests = true ]] || [[ $one_test_name = "beam_in_vacuum.normalized.1Rank" ]]
then
    cd $build_dir
    ctest --output-on-failure -R beam_in_vacuum.normalized.1Rank \
        || echo "ctest command failed, maybe just because checksums are different. Keep going"
    cd $checksum_dir
    ./checksumAPI.py --reset-benchmark \
                     --file_name ${build_dir}/bin/beam_in_vacuum.normalized.1Rank \
                     --test-name beam_in_vacuum.normalized.1Rank
fi

# gaussian_weight.1Rank
if [[ $all_tests = true ]] || [[ $one_test_name = "gaussian_weight.1Rank" ]]
then
    cd $build_dir
    ctest --output-on-failure -R gaussian_weight.1Rank \
        || echo "ctest command failed, maybe just because checksums are different. Keep going"
    cd $checksum_dir
    ./checksumAPI.py --reset-benchmark \
                     --file_name ${build_dir}/bin/gaussian_weight.1Rank \
                     --test-name gaussian_weight.1Rank
fi

# collisions.SI.1Rank
if [[ $all_tests = true ]] || [[ $one_test_name = "collisions.SI.1Rank" ]]
then
    cd $build_dir
    ctest --output-on-failure -R collisions.SI.1Rank \
        || echo "ctest command failed, maybe just because checksums are different. Keep going"
    cd $checksum_dir
    ./checksumAPI.py --reset-benchmark \
                     --file_name ${build_dir}/bin/collisions.SI.1Rank \
                     --test-name collisions.SI.1Rank
fi

# collisions_beam.SI.1Rank
if [[ $all_tests = true ]] || [[ $one_test_name = "collisions_beam.SI.1Rank" ]]
then
    cd $build_dir
    ctest --output-on-failure -R collisions_beam.SI.1Rank \
        || echo "ctest command failed, maybe just because checksums are different. Keep going"
    cd $checksum_dir
    ./checksumAPI.py --reset-benchmark \
                     --file_name ${build_dir}/bin/collisions_beam.SI.1Rank \
                     --test-name collisions_beam.SI.1Rank
fi

# beam_in_vacuum_open_boundary.normalized.1Rank
if [[ $all_tests = true ]] || [[ $one_test_name = "beam_in_vacuum_open_boundary.normalized.1Rank" ]]
then
    cd $build_dir
    ctest --output-on-failure -R beam_in_vacuum_open_boundary.normalized.1Rank \
        || echo "ctest command failed, maybe just because checksums are different. Keep going"
    cd $checksum_dir
    ./checksumAPI.py --reset-benchmark \
                     --file_name ${build_dir}/bin/beam_in_vacuum_open_boundary.normalized.1Rank \
                     --test-name beam_in_vacuum_open_boundary.normalized.1Rank
fi

# laser_evolution.SI.2Rank
if [[ $all_tests = true ]] || [[ $one_test_name = "laser_evolution.SI.2Rank" ]]
then
    cd $build_dir
    ctest --output-on-failure -R laser_evolution.SI.2Rank \
        || echo "ctest command failed, maybe just because checksums are different. Keep going"
    cd $checksum_dir
    ./checksumAPI.py --reset-benchmark \
                     --file_name ${build_dir}/bin/laser_evolution.SI.2Rank \
                     --test-name laser_evolution.SI.2Rank \
                     --skip-particles
fi

# production.SI.2Rank
if [[ $all_tests = true ]] || [[ $one_test_name = "production.SI.2Rank" ]]
then
    cd $build_dir
    ctest --output-on-failure -R production.SI.2Rank \
        || echo "ctest command failed, maybe just because checksums are different. Keep going"
    cd $checksum_dir
    ./checksumAPI.py --reset-benchmark \
                     --file_name ${build_dir}/bin/production.SI.2Rank_pwfa \
                     --test-name production.SI.2Rank_pwfa
    ./checksumAPI.py --reset-benchmark \
                     --file_name ${build_dir}/bin/production.SI.2Rank_lwfa \
                     --test-name production.SI.2Rank_lwfa
fi

# transverse_benchmark.1Rank.sh
if [[ $all_tests = true ]] || [[ $one_test_name = "transverse_benchmark.1Rank.sh" ]]
then
    cd $build_dir
    ctest --output-on-failure -R transverse_benchmark.1Rank.sh \
        || echo "ctest command failed, maybe just because checksums are different. Keep going"
    cd $checksum_dir
    ./checksumAPI.py --reset-benchmark \
                     --file_name ${build_dir}/bin/transverse_benchmark.1Rank.sh \
                     --test-name transverse_benchmark.1Rank.sh
fi
