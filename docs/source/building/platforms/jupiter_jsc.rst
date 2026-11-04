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
   export CUDAARCHS=90

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

   #!/bin/bash -l
   #SBATCH -A <your project id>
   #SBATCH --partition=booster
   #SBATCH --nodes=2
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

   NODES=2
   GPUS_PER_NODE=4
   RANKS_PER_GPU=4
   RANKS_PER_NODE=$(( GPUS_PER_NODE*RANKS_PER_GPU ))
   RANKS=$(( GPUS_PER_NODE*RANKS_PER_GPU*NODES ))
   ALL_GPUS_PER_NODE=$(seq -s, 0 $((GPUS_PER_NODE - 1)))

   rm -rf ./select_gpu_start_mps
   cat << EOF > select_gpu_start_mps
   #!/bin/bash
   LOCAL_RANK=\$SLURM_LOCALID
   export CUDA_VISIBLE_DEVICES=$ALL_GPUS_PER_NODE
   nvidia-cuda-mps-control -d > /dev/null
   export CUDA_VISIBLE_DEVICES=\$(( \$LOCAL_RANK / $RANKS_PER_GPU ))
   exec \$*
   EOF
   chmod +x ./select_gpu_start_mps

   srun -n $RANKS --cpu_bind=sockets ./select_gpu_start_mps \
   $HOME/src/hipace/build/bin/hipace inputs
