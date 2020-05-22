# hipace
Highly efficient Plasma Accelerator Emulation, quasistatic particle-in-cell code

## Developers

### Install Dependencies

macOS/Linux:
```
brew update
brew install cmake
brew install open-mpi
```

or macOS/Linux:
```
spack env create hipace-dev
spack env activate hipace-dev
spack add cmake
spack add mpi
spack install
```
(in new terminals, re-activate the environment with `spack env activate hipace-dev` again)

Now, `cmake --version` should be at version 3.14.0 or newer.

### Configure your compiler

For example, using a GCC on macOS:
```
export CC=$(which gcc)
export CXX=$(which g++)
```

### Build & Test

From the HiPACE root directory, execute:
```
mkdir -p build
cd build

# find dependencies & configure
cmake ..

# build
make -j 4

# run tests
ctest --output-on-failure
```

Extensive documentation of `cmake .. -D<OPTION>=<VALUE>` settings: https://amrex-codes.github.io/amrex/docs_html/BuildingAMReX.html#customization-options

### Run a simulation and look at the results

After compiling HiPACE (see above), from the HiPACE root directory, execute
```
cd examples
../build/bin/HiPACE inputs # Run the simulation with fields and plasma and beam particles
./show.py # Plot results with yt, and save in img.png
```
