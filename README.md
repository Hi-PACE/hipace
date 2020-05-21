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

# build
cmake ..
make -j 4

# test
ctest
```

Extensive documentation of options: https://amrex-codes.github.io/amrex/docs_html/BuildingAMReX.html#customization-options
