# hipace
Highly efficient Plasma Accelerator Emulation, quasistatic particle-in-cell code

## Developers

### Install Dependencies

macOS/Linux:
```
brew update
brew install cmake
```

or macOS/Linux:
```
spack install cmake
spack load cmake
```

Now, `cmake --version` should be at version 3.14.0 or newer.

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
