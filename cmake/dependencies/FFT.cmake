# Helper Functions ############################################################
#
option(HiPACE_FFTW_IGNORE_OMP "Ignore FFTW3 OpenMP support, even if found" OFF)
mark_as_advanced(HiPACE_FFTW_IGNORE_OMP)

# Set the HIPACE_FFTW_OMP=1 define on HiPACE::thirdparty::FFT if TRUE and print
# a message
#
function(fftw_add_define HAS_FFTW_OMP_LIB)
    if(HAS_FFTW_OMP_LIB)
        message(STATUS "FFTW: Found OpenMP support")
        target_compile_definitions(HiPACE::thirdparty::FFT INTERFACE HIPACE_FFTW_OMP=1)
    else()
        message(STATUS "FFTW: Could NOT find OpenMP support")
    endif()
endfunction()

# Check if the found FFTW install location has an _omp library, e.g.,
# libfftw3(f)_omp.(a|so) shipped and if yes, set the HIPACE_FFTW_OMP=1 define.
#
function(fftw_check_omp library_paths fftw_precision_suffix)
    find_library(HAS_FFTW_OMP_LIB fftw3${fftw_precision_suffix}_omp
        PATHS ${library_paths}
        # this is intentional, so we don't mix different FFTW installs
        # and only check what is in the location hinted by the
        # "library_paths" variable
        NO_DEFAULT_PATH
        NO_PACKAGE_ROOT_PATH
        NO_CMAKE_PATH
        NO_CMAKE_ENVIRONMENT_PATH
        NO_SYSTEM_ENVIRONMENT_PATH
        NO_CMAKE_SYSTEM_PATH
        NO_CMAKE_FIND_ROOT_PATH
    )
    if(HAS_FFTW_OMP_LIB)
        # the .pc files here forget to link the _omp.a/so files
        # explicitly - we add those manually to avoid any trouble,
        # e.g., in static builds.
        target_link_libraries(HiPACE::thirdparty::FFT INTERFACE ${HAS_FFTW_OMP_LIB})
    endif()

    fftw_add_define("${HAS_FFTW_OMP_LIB}")
endfunction()


# Various FFT implementations that we want to use #############################
#

# cuFFT  (CUDA)
#   TODO: check if `find_package` search works

# rocFFT (HIP)
if(HiPACE_COMPUTE STREQUAL HIP)
    find_package(rocfft REQUIRED)

# FFTW   (NOACC, OMP, SYCL)
elseif(NOT HiPACE_COMPUTE STREQUAL CUDA)
    # On Windows, try searching for FFTW3(f)Config.cmake files first
    #   Installed .pc files wrongly and unconditionally add -lm
    #   https://github.com/FFTW/fftw3/issues/236

    # On Linux & macOS, note Autotools install bug:
    #   https://github.com/FFTW/fftw3/issues/235
    # Thus, rely on .pc files

    set(HiPACE_FFTW_SEARCH_VALUES PKGCONFIG CMAKE)
    set(HiPACE_FFTW_SEARCH_DEFAULT PKGCONFIG)
    if(WIN32)
        set(HiPACE_FFTW_SEARCH_DEFAULT CMAKE)
    endif()
    set(HiPACE_FFTW_SEARCH ${HiPACE_FFTW_SEARCH_DEFAULT}
        CACHE STRING "FFTW search method (PKGCONFIG/CMAKE)")
    set_property(CACHE HiPACE_FFTW_SEARCH PROPERTY STRINGS ${HiPACE_FFTW_SEARCH_VALUES})
    if(NOT HiPACE_FFTW_SEARCH IN_LIST HiPACE_FFTW_SEARCH_VALUES)
        message(FATAL_ERROR "HiPACE_FFTW_SEARCH (${HiPACE_FFTW_SEARCH}) must be one of ${HiPACE_FFTW_SEARCH_VALUES}")
    endif()
    mark_as_advanced(HiPACE_FFTW_SEARCH)

    # floating point precision suffixes: float, double and quad precision
    if(HiPACE_PRECISION STREQUAL "DOUBLE")
        set(HFFTWp "")
    else()
        set(HFFTWp "f")
    endif()

    if(HiPACE_FFTW_SEARCH STREQUAL CMAKE)
        find_package(FFTW3${HFFTWp} CONFIG REQUIRED)
        set(HiPACE_FFTW_LIBRARY_DIRS "${FFTW3${HFFTWp}_LIBRARY_DIRS}")
        message(STATUS "Found FFTW: ${FFTW3${HFFTWp}_DIR} (found version \"${FFTW3${HFFTWp}_VERSION}\")")
    else()
        find_package(PkgConfig REQUIRED QUIET)
        pkg_check_modules(fftw3${HFFTWp} REQUIRED IMPORTED_TARGET fftw3${HFFTWp})
        message(STATUS "Found FFTW: ${fftw3${HFFTWp}_PREFIX}")
        if(fftw3${HFFTWp}_LIBRARY_DIRS)
            set(HiPACE_FFTW_LIBRARY_DIRS "${fftw3${HFFTWp}_LIBRARY_DIRS}")
        else()
            set(HiPACE_FFTW_LIBRARY_DIRS "${fftw3${HFFTWp}_LIBDIR}")
        endif()
    endif()
endif()

# create an IMPORTED target: HiPACE::thirdparty::FFT
if(HiPACE_COMPUTE STREQUAL CUDA)
    # CUDA_ADD_CUFFT_TO_TARGET(HiPACE::thirdparty::FFT)
    make_third_party_includes_system(cufft FFT)
elseif(HiPACE_COMPUTE STREQUAL HIP)
    make_third_party_includes_system(roc::rocfft FFT)
else()
    if(HiPACE_FFTW_SEARCH STREQUAL CMAKE)
        make_third_party_includes_system(FFTW3::fftw3${HFFTWp} FFT)
    else()
        make_third_party_includes_system(PkgConfig::fftw3${HFFTWp} FFT)
    endif()
    if(HiPACE_COMPUTE STREQUAL OMP)
        if(HiPACE_FFTW_IGNORE_OMP)
            message(STATUS "FFTW: Requested to IGNORE OpenMP support")
        else()
            fftw_check_omp("${HiPACE_FFTW_LIBRARY_DIRS}" "${HFFTWp}")
        endif()
    else()
        message(STATUS "FFTW: Did NOT search for OpenMP support (HiPACE_COMPUTE!=OMP)")
    endif()
endif()
