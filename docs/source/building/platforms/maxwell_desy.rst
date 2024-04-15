Maxwell cluster @ DESY
======================

This page only provides HiPACE++ specific instructions.
For more information please visit the
`Maxwell documentation <https://confluence.desy.de/display/MXW/Maxwell+Cluster>`__.

Create a file ``profile.hipace``, for instance in ``$HOME``, and ``source`` it whenever you log in and want to work with
HiPACE++:

.. code-block:: bash

   #!/usr/bin/env zsh # Shell is assumed to be zsh
   module purge
   module load maxwell gcc/9.3 openmpi/4
   module load maxwell cuda/11.8
   module load hdf5/1.10.6
   # pick correct GPU setting (this may differ for V100 nodes)
   export GPUS_PER_SOCKET=2
   export GPUS_PER_NODE=4
   # optimize CUDA compilation for A100
   export AMREX_CUDA_ARCH=8.0 # use 8.0 for A100 or 7.0 for V100

Install HiPACE++ (the first time, and whenever you want the latest version):

.. code-block:: bash

   source profile.hipace
   git clone https://github.com/Hi-PACE/hipace.git $HOME/src/hipace # only the first time
   cd $HOME/src/hipace
   rm -rf build
   cmake -S . -B build -DHiPACE_COMPUTE=CUDA
   cmake --build build -j 16

You can get familiar with the HiPACE++ input file format in our :doc:`../../run/get_started`
section, to prepare an input file that suits your needs. You can then create your directory on
BEEGFS ``/beegfs/desy/group/<your group>`` or ``/beegfs/desy/user/<your username>``,
where you can put your input file and adapt the following
submission script:

.. code-block:: bash

   #! /usr/bin/env zsh
   #SBATCH --partition=<partition> # mpa # maxgpu # allgpu
   #SBATCH --time=01:00:00
   #SBATCH --nodes=1
   #SBATCH --constraint=A100&GPUx4 # A100&GPUx1
   #SBATCH --job-name=HiPACE
   #SBATCH --output=hipace-%j-%N.out
   #SBATCH --error=hipace-%j-%N.err

   export OMP_NUM_THREADS=1
   source $HOME/profile.hipace # or correct path to your profile file

   mpiexec -n 4 -npernode 4 $HOME/src/hipace/build/bin/hipace inputs

The ``-npernode`` must be set to ``GPUS_PER_NODE``, otherwise not all GPUs are used correctly.
There are nodes with 4 GPUs and 1 GPU (see the `Maxwell documentation on compute infrastructure <https://confluence.desy.de/display/MXW/Compute+Infrastructure>`__.
for more details and the required constraints). Please set the value accordingly.

.. tip::
   If you encounter an error like ``module: command not found``, this can be fixed in most cases by adding the following piece of code before ``module purge`` in your ``profile.hipace`` file.

   .. code-block:: bash

      module ()
      {
          eval `modulecmd bash $*`
      }

.. tip::
   Parallel simulations can be largely accelerated by using GPU-aware MPI.
   To utilize GPU-aware MPI, the input parameter ``comms_buffer.on_gpu = 1`` must be set and the following flag must be passed in the job script:

   .. code-block:: bash

      export UCX_MEMTYPE_CACHE=n

   Note that using GPU-aware MPI may require more GPU memory.
