Maxwell cluster @ DESY
======================

This page only provides HiPACE++ specific instructions.
For more information please visit the `Maxwell documentation <https://confluence.desy.de/display/IS/Maxwell>`__.

Create a file ``profile.hipace`` and ``source`` it whenever you log in and want to work with HiPACE++:

.. code-block:: bash

   module purge
   module load maxwell gcc/9.3 openmpi/4
   module load maxwell cuda/11.1 # cuda/11.0
   module load hdf5/1.10.6

Install HiPACE++ (the first time, and whenever you want the latest version):

.. code-block:: bash

   source profile.hipace
   git clone https://github.com/Hi-PACE/hipace.git $HOME/src/hipace # only the first time
   cd $HOME/src/hipace
   rm -rf build
   mkdir build
   cd build
   cmake .. -DHiPACE_COMPUTE=CUDA
   make -j 16

You can get familiar with the HiPACE++ input file format in our :doc:`../../run/get_started` section, to prepare an input file that suits your needs.
You can then create your directory on BEEGFS ``$SCRATCH_<project id>``, where you can put your input file and adapt the following submission script:

.. code-block:: bash

   #! /usr/bin/env sh
   #SBATCH --partition=<partition> # maxgpu # allgpu
   #SBATCH --time=01:00:00
   #SBATCH --nodes=1
   #SBATCH --constraint=V100

   module ()
   {
     eval `modulecmd bash $*`
   }

   export OMP_NUM_THREADS=1
   module load maxwell gcc/9.3 openmpi/4 cuda/11.1

   mpiexec -n 2 $HOME/src/hipace/build/bin/hipace inputs
