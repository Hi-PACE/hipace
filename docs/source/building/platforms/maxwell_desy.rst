Maxwell cluster @ DESY
======================

This page only provides HiPACE++ specific instructions.
For more information please visit the
`Maxwell documentation <https://wiki.desy.de/maxwell/>`__.

Create a file ``profile.hipace``, for instance in ``$HOME``, and ``source`` it whenever you log in and want to work with
HiPACE++:

.. code-block:: bash

   #!/usr/bin/env zsh # Shell is assumed to be zsh
   module purge
   module load maxwell gcc/12.2 cuda/13.0 openmpi/4.1.8-cuda13.0 hdf5/1.12.1
   # optimize CUDA compilation for A100
   export CUDAARCHS=80 # use 70 for V100, 80 for A100 or 90 for H200
   # Fix issue with the Python 3.13 jhub environment
   export PATH="/usr/bin:$PATH"

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
DUST ``/data/dust/group/<your group>`` or ``/data/dust/user/<your username>``,
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

The ``-npernode`` must be set to the number of GPUs per node, otherwise not all GPUs are used correctly.
There are nodes with 4 GPUs and 1 GPU (see the `Maxwell documentation on compute infrastructure <https://wiki.desy.de/maxwell/infrastructure/compute/>`__.
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

Overallocating GPUs
-------------------

If the transverse domain size is small (:math:`<= 512^2`) then a single hipace process might not
expose enough parallelism to fully saturate a GPU. For these cases it can make sense to run
multiple hipace ranks on each GPU, which increases parallelism at the cost of higher memory
utilization as well as higher CPU usage. For good performance it is very important that all hipace
instances use the same CUDA context, which is achieved using the ``nvidia-cuda-mps-control`` tool.
Very small transverse problems (:math:`= 128^2`) show good scaling up to 8 ranks per GPU,
which gives up to four times better performance, provided that there are enough time steps.
The following submission script shows how to use 4 ranks per GPU:

.. code-block:: bash

   #! /usr/bin/env zsh
   #SBATCH --partition=<partition> # mpa # maxgpu # allgpu
   #SBATCH --time=01:00:00
   #SBATCH --nodes=2
   #SBATCH --constraint=A100&GPUx4
   #SBATCH --job-name=HiPACE
   #SBATCH --output=hipace-%j-%N.out
   #SBATCH --error=hipace-%j-%N.err

   export OMP_NUM_THREADS=1
   source $HOME/profile.hipace

   NODES=2
   GPUS_PER_NODE=4
   RANKS_PER_GPU=4
   RANKS_PER_NODE=$(( GPUS_PER_NODE*RANKS_PER_GPU ))
   RANKS=$(( GPUS_PER_NODE*RANKS_PER_GPU*NODES ))
   ALL_GPUS_PER_NODE=$(seq -s, 0 $((GPUS_PER_NODE - 1)))

   rm -rf ./select_gpu_start_mps
   cat << EOF > select_gpu_start_mps
   #!/bin/bash
   LOCAL_RANK=\$OMPI_COMM_WORLD_LOCAL_RANK
   export CUDA_VISIBLE_DEVICES=$ALL_GPUS_PER_NODE
   nvidia-cuda-mps-control -d > /dev/null
   export CUDA_VISIBLE_DEVICES=\$(( \$LOCAL_RANK / $RANKS_PER_GPU ))
   exec \$*
   EOF
   chmod +x ./select_gpu_start_mps

   mpiexec -n $RANKS -npernode $RANKS_PER_NODE ./select_gpu_start_mps \
   $HOME/src/hipace/build/bin/hipace inputs
