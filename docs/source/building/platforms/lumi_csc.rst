LUMI @ CSC
==========

This page only provides HiPACE++ specific instructions.
For more information please visit the `LUMI documentation <https://docs.lumi-supercomputer.eu/>`__.

Log in with ``<yourid>@lumi.csc.fi``.

Running on GPU
--------------

Create a file ``profile.hipace`` and ``source`` it whenever you log in and want to work with HiPACE++:

.. code-block:: bash

   # please set your project account
   export proj=<your project id>
   # required dependencies
   module load LUMI
   module load partition/G
   module load PrgEnv-amd/8.3.3
   module load rocm/5.2.3
   module load buildtools/22.08
   module load cray-hdf5/1.12.1.5

   export MPICH_GPU_SUPPORT_ENABLED=1

   # optimize ROCm compilation for MI250X
   export AMREX_AMD_ARCH=gfx90a

   # compiler environment hints
   export CC=$(which cc)
   export CXX=$(which CC)
   export FC=$(which ftn)

Install HiPACE++ (the first time, and whenever you want the latest version):

.. code-block:: bash

   source profile.hipace
   mkdir -p $HOME/src # only the first time, create a src directory to put the codes sources
   git clone https://github.com/Hi-PACE/hipace.git $HOME/src/hipace # only the first time
   cd $HOME/src/hipace
   rm -rf build
   cmake -S . -B build -DHiPACE_COMPUTE=HIP -DHiPACE_openpmd_mpi=OFF
   cmake --build build -j 16

You can get familiar with the HiPACE++ input file format in our :doc:`../../run/get_started` section, to prepare an input file that suits your needs.
You can then create your directory in ``/project/project_<project id>``, where you can put your input file and adapt the following submission script:

.. code-block:: bash

    #!/bin/bash

    #SBATCH -A $proj
    #SBATCH -J hipace
    #SBATCH -o %x-%j.out
    #SBATCH -t 00:30:00
    #SBATCH --partition=standard-g
    #SBATCH --nodes=2
    #SBATCH --ntasks-per-node=8
    #SBATCH --gpus-per-node=8

    export MPICH_GPU_SUPPORT_ENABLED=1

    # note (12-12-22)
    # this environment setting is currently needed on LUMI to work-around a
    # known issue with Libfabric
    #export FI_MR_CACHE_MAX_COUNT=0  # libfabric disable caching
    # or, less invasive:
    export FI_MR_CACHE_MONITOR=memhooks  # alternative cache monitor
    # note (9-2-22, OLCFDEV-1079)
    # this environment setting is needed to avoid that rocFFT writes a cache in
    # the home directory, which does not scale.
    export ROCFFT_RTC_CACHE_PATH=/dev/null
    export OMP_NUM_THREADS=1

    # needed for high nz runs.
    # if too many mpi messages are send, the hardware counters can overflow, see
    # https://docs.nersc.gov/performance/network/
    # export FI_CXI_RX_MATCH_MODE=hybrid

    # setting correct CPU binding
    # (see https://docs.lumi-supercomputer.eu/runjobs/scheduled-jobs/lumig-job/)
    cat << EOF > select_gpu
    #!/bin/bash

    export ROCR_VISIBLE_DEVICES=\$SLURM_LOCALID
    exec \$*
    EOF

    chmod +x ./select_gpu

    CPU_BIND="map_cpu:49,57,17,25,1,9,33,41"

    srun --cpu-bind=${CPU_BIND} ./select_gpu $HOME/src/hipace/build/bin/hipace inputs
    rm -rf ./select_gpu


and use it to submit a simulation.

.. tip::
   Parallel simulations can be largely accelerated by using GPU-aware MPI.
   To utilize GPU-aware MPI, the input parameter ``comms_buffer.on_gpu = 1`` must be set and the following flag must be passed in the job script:

   .. code-block:: bash

      export FI_MR_CACHE_MAX_COUNT=0

   Note that using GPU-aware MPI may require more GPU memory.
