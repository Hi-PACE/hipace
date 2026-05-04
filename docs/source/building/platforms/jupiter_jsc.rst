Jupiter Booster @ JSC
=====================

This page only provides HiPACE++ specific instructions.
For more information please visit the `JSC documentation <https://apps.fz-juelich.de/jsc/hps/jupiter/index.html>`__.

Log in with ``<yourid>@login.jupiter.fz-juelich.de``. Note that you first need to log into JuDoor
to be added to a compute project, set up an SSH key and set up 2FA.

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
   # optimize CUDA compilation for GH200
   export AMREX_CUDA_ARCH=9.0

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
   #SBATCH -A <your project id>
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
   # fix issues with MPI
   export UCX_CUDA_COPY_REG_WHOLE_ALLOC=on
   export UCX_CUDA_COPY_MAX_REG_RATIO=0
   export UCX_MEMTYPE_REG_WHOLE_ALLOC_TYPES=cuda
   export UCX_MEMTYPE_CACHE=n
   srun -n 8 --cpu_bind=sockets $HOME/src/hipace/build/bin/hipace inputs

and use it to submit a simulation.

.. tip::
   Parallel simulations can be largely accelerated by using GPU-aware MPI.
   To utilize GPU-aware MPI, the input parameter ``comms_buffer.on_gpu = 1`` must be set.
   Additionally ``comms_buffer.pre_register_memory = 1`` can help improve startup time.

   Note that using GPU-aware MPI may require more GPU memory.
