Maxwell cluster @ DESY
======================

This page only provides HiPACE++ specific instructions.
For more information please visit the
`Maxwell documentation <https://confluence.desy.de/display/MXW/Maxwell+Cluster>`__.

Create a file ``profile.hipace`` and ``source`` it whenever you log in and want to work with
HiPACE++:

.. code-block:: bash

   module purge
   module load maxwell gcc/9.3 openmpi/4
   module load maxwell cuda/11.3
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
   cmake3 -S . -B build -DHiPACE_COMPUTE=CUDA
   cmake3 --build build -j 16

You can get familiar with the HiPACE++ input file format in our :doc:`../../run/get_started`
section, to prepare an input file that suits your needs. You can then create your directory on
BEEGFS ``$SCRATCH_<project id>``, where you can put your input file and adapt the following
submission script:

.. code-block:: bash

   #! /usr/bin/env sh
   #SBATCH --partition=<partition> # mpa # maxgpu # allgpu
   #SBATCH --time=01:00:00
   #SBATCH --nodes=1
   #SBATCH --constraint=A100 # V100 # V100&GPUx2 # V100&GPUx4
   #SBATCH --job-name=HiPACE
   #SBATCH --output=hipace-%j-%N.out
   #SBATCH --error=hipace-%j-%N.err

   export OMP_NUM_THREADS=1
   module load maxwell gcc/9.3 openmpi/4 cuda/11.1

   mpiexec -n 4 -npernode 4 $HOME/src/hipace/build/bin/hipace inputs

The ``-npernode`` must be set to ``GPUS_PER_NODE``, otherwise not all GPUs are used correctly.
