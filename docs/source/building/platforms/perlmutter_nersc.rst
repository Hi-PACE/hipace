Perlmutter @ NERSC
====================

This page only provides HiPACE++ specific instructions.
For more information please visit the `Perlmutter documentation <https://docs.nersc.gov/systems/perlmutter/>`__.

Log in with ``ssh <yourid>@perlmutter-p1.nersc.gov``.

Building for GPU
----------------

Create a file ``profile.hipace`` and ``source`` it whenever you log in and want to work with HiPACE++:

.. code-block:: bash

   # please set your project account
   export proj=<your project id>_g  # _g for GPU accounting

   # required dependencies
   module load cmake/3.24.3
   module load cray-hdf5-parallel/1.12.2.3

   # necessary to use CUDA-Aware MPI and run a job
   export CRAY_ACCEL_TARGET=nvidia80

   # optimize CUDA compilation for A100
   export AMREX_CUDA_ARCH=8.0

   # compiler environment hints
   export CC=cc
   export CXX=CC
   export FC=ftn
   export CUDACXX=$(which nvcc)
   export CUDAHOSTCXX=CC


Download HiPACE++ from GitHub (the first time, and whenever you want the latest version):

.. code-block:: bash

   git clone https://github.com/Hi-PACE/hipace.git $HOME/src/hipace # or any other path you prefer

Compile the code using CMake

.. code-block:: bash

   source profile.hipace # load the correct modules
   cd $HOME/src/hipace   # or where HiPACE++ is installed
   rm -rf build
   cmake -S . -B build -DHiPACE_COMPUTE=CUDA
   cmake --build build -j 16

You can get familiar with the HiPACE++ input file format in our :doc:`../../run/get_started` section, to prepare an input file that suits your needs.
You can then create your directory in your ``$PSCRATCH``, where you can put your input file and adapt the following submission script:

.. code-block:: bash

    #!/bin/bash -l

    #SBATCH -t 01:00:00
    #SBATCH -N 2
    #SBATCH -J HiPACE++
    #    note: <proj> must end on _g
    #SBATCH -A <proj>_g
    #SBATCH -q regular
    #SBATCH -C gpu
    #SBATCH -c 32
    #SBATCH --exclusive
    #SBATCH --gpu-bind=none
    #SBATCH --gpus-per-node=4
    #SBATCH -o hipace.o%j
    #SBATCH -e hipace.e%j

    # path to executable and input script
    EXE=$HOME/src/hipace/build/bin/hipace
    INPUTS=inputs

    # pin to closest NIC to GPU
    export MPICH_OFI_NIC_POLICY=GPU

    # for GPU-aware MPI use the first line
    #HIPACE_GPU_AWARE_MPI="comms_buffer.on_gpu=1"
    HIPACE_GPU_AWARE_MPI=""

    # CUDA visible devices are ordered inverse to local task IDs
    #   Reference: nvidia-smi topo -m
    srun --cpu-bind=cores bash -c "
        export CUDA_VISIBLE_DEVICES=\$((3-SLURM_LOCALID));
        ${EXE} ${INPUTS} ${HIPACE_GPU_AWARE_MPI}" \
      > output.txt


and use it to submit a simulation. Note, that this example simulation runs on 8 GPUs, since `-N = 2` yields 2 nodes with 4 GPUs each.

.. tip::
   Parallel simulations can be largely accelerated by using GPU-aware MPI.
   To utilize GPU-aware MPI, the input parameter ``comms_buffer.on_gpu = 1`` must be set (see the job script above).

   Note that using GPU-aware MPI may require more GPU memory.
