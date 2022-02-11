Spock @ OLCF
============

This page only provides HiPACE++ specific instructions.
For more information please visit the `OLCF documentation <https://docs.olcf.ornl.gov/systems/spock_quick_start_guide.html>`__.

Running on GPU
--------------

Create a file ``profile.hipace.spock`` and ``source`` it whenever you log in and want to work with HiPACE++:

.. code-block:: bash

   module load cmake/3.20.2 # cmake
   module load craype-accel-amd-gfx908
   module load rocm # rocm/4.5.0
   module load ccache
   module load ninja
   export AMREX_AMD_ARCH=gfx908
   export CC=$(which clang)
   export CXX=$(which hipcc)
   export LDFLAGS="-L${CRAYLIBS_X86_64} $(CC --cray-print-opts=libs) -lmpi"

Install HiPACE++ (the first time, and whenever you want the latest version):

.. code-block:: bash

   source profile.hipace
   git clone https://github.com/Hi-PACE/hipace.git $HOME/src/hipace # only the first time
   cd $HOME/src/hipace
   cmake -S . -B build -DHiPACE_COMPUTE=HIP -DAMReX_AMD_ARCH=gfx908 -DMPI_CXX_COMPILER=$(which CC) -DMPI_C_COMPILER=$(which cc) -DMPI_COMPILER_FLAGS="--cray-print-opts=all"
   cmake --build build -j 6

You can get familiar with the HiPACE++ input file format in our :doc:`../../run/get_started` section, to prepare an input file that suits your needs.
You can then create your directory in your ``$SCRATCH_<project id>``, where you can put your input file and adapt the following submission script:

.. code-block:: bash

   #!/bin/bash

   #SBATCH -A aph114
   #SBATCH -J hipace
   #SBATCH -o %x-%j.out
   #SBATCH -t 00:10:00
   #SBATCH -p ecp
   #SBATCH -N 1

   source $HOME/profile.hipace.spock

   export OMP_NUM_THREADS=1
   srun -n 1 -c 1 --ntasks-per-node=1 <path/to/executable>/hipace.MPI.HIP.DP inputs &> output.txt

and use it to submit a simulation.
