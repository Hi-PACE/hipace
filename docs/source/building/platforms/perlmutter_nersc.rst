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
   module load cmake/3.22.0
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
    #SBATCH --ntasks-per-gpu=1
    #SBATCH --gpus-per-node=4
    #SBATCH -o hipace.o%j
    #SBATCH -e hipace.e%j

    # GPU-aware MPI
    export MPICH_GPU_SUPPORT_ENABLED=1

    # expose one GPU per MPI rank
    export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID

    # path to executable and input script
    EXE=$HOME/src/hipace/build/bin/hipace
    INPUTS=inputs

    srun ${EXE} ${INPUTS}


and use it to submit a simulation. Note, that this example simulation runs on 8 GPUs, since `-N = 2` yields 2 nodes with 4 GPUs each.

.. tip::
   Parallel simulations can be largely accelerated by using GPU-aware MPI.
   To utilize GPU-aware MPI, the input parameter ``hipace.comms_buffer_on_gpu = 1`` must be set and  the following flag must be passed in the job script:

   .. code-block:: bash

      export MPICH_GPU_SUPPORT_ENABLED=1

   Note that using GPU-aware MPI may require more GPU memory.
