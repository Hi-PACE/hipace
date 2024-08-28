Juwels Booster @ JSC
====================

This page only provides HiPACE++ specific instructions.
For more information please visit the `JSC documentation <https://apps.fz-juelich.de/jsc/hps/juwels/index.html>`__.

Log in with ``<yourid>@juwels-booster.fz-juelich.de``.

Running on GPU
--------------

Create a file ``profile.hipace`` and ``source`` it whenever you log in and want to work with HiPACE++:

.. code-block:: bash

   # please set your project account
   export proj=<your project id>
   # required dependencies
   module load CMake
   module load GCC
   module load OpenMPI
   module load CUDA
   module load HDF5
   module load ccache # optional, accelerates recompilation
   # optimize CUDA compilation for A100
   export AMREX_CUDA_ARCH=8.0 # 8.0 for A100, 7.0 for V100

Install HiPACE++ (the first time, and whenever you want the latest version):

.. code-block:: bash

   source profile.hipace
   git clone https://github.com/Hi-PACE/hipace.git $HOME/src/hipace # only the first time
   cd $HOME/src/hipace
   rm -rf build
   cmake -S . -B build -DHiPACE_COMPUTE=CUDA
   cmake --build build -j 16

You can get familiar with the HiPACE++ input file format in our :doc:`../../run/get_started` section, to prepare an input file that suits your needs.
You can then create your directory in your ``$SCRATCH_<project id>``, where you can put your input file and adapt the following submission script:

.. code-block:: bash

   #!/bin/bash -l
   #SBATCH -A $proj
   #SBATCH --partition=booster
   #SBATCH --nodes=2
   #SBATCH --ntasks=8
   #SBATCH --ntasks-per-node=4
   #SBATCH --gres=gpu:4
   #SBATCH --time=00:05:00
   #SBATCH --job-name=hipace
   #SBATCH --output=hipace-%j-%N.txt
   #SBATCH --error=hipace-%j-%N.err
   export OMP_NUM_THREADS=1
   module load GCC
   module load OpenMPI
   module load CUDA
   module load HDF5
   # fix issue with MPI
   export UCX_CUDA_COPY_REG_WHOLE_ALLOC=on
   srun -n 8 --cpu_bind=sockets $HOME/src/hipace/build/bin/hipace.MPI.CUDA.DP.LF inputs

and use it to submit a simulation.

.. tip::
   Parallel simulations can be largely accelerated by using GPU-aware MPI.
   To utilize GPU-aware MPI, the input parameter ``comms_buffer.on_gpu = 1`` must be set.

   Note that using GPU-aware MPI may require more GPU memory.

Running on CPU
--------------

.. warning::
    The Juwels Booster is a GPU-accelerated supercomputer, and running on CPUs only is strongly discouraged.
    This section only illustrates how to efficiently run on CPU with OpenMP threading, which was tested on the Juwels Booster for practical reasons, but should apply to other supercomputers.
    In particular, the proposed values of OMP_PROC_BIND and OMP_PLACES give decent performance for both threaded FFTW and particle operations.

Create a file ``profile.hipace`` and ``source`` it whenever you log in and want to work with HiPACE++:

.. code-block:: bash

   # please set your project account
   export proj=<your project id>
   # required dependencies
   module load CMake
   module load GCC
   module load OpenMPI
   module load FFTW
   module load HDF5
   module load ccache # optional, accelerates recompilation

Install HiPACE++ (the first time, and whenever you want the latest version):

.. code-block:: bash

   source profile.hipace
   git clone https://github.com/Hi-PACE/hipace.git $HOME/src/hipace # only the first time
   cd $HOME/src/hipace
   rm -rf build
   cmake -S . -B build -DHiPACE_COMPUTE=OMP
   cmake --build build -j 16

You can get familiar with the HiPACE++ input file format in our :doc:`../../run/get_started` section, to prepare an input file that suits your needs.
You can then create your directory in your ``$SCRATCH_<project id>``, where you can put your input file and adapt the following submission script:

.. code-block:: bash

   #!/bin/bash -l
   #SBATCH -A $proj
   #SBATCH --partition=booster
   #SBATCH --nodes=1
   #SBATCH --ntasks=1
   #SBATCH --time=00:05:00
   #SBATCH --job-name=hipace
   #SBATCH --output=hipace-%j-%N.txt
   #SBATCH --error=hipace-%j-%N.err

   source $HOME/profile.hipace

   # These options give the best performance, in particular for the threaded FFTW
   export OMP_PROC_BIND=false # true false master close spread
   export OMP_PLACES=cores # threads cores sockets

   export OMP_NUM_THREADS=8 # Anything <= 16, depending on the problem size

   srun -n 8 --cpu_bind=sockets <path/to/executable> inputs

and use it to submit a simulation.
