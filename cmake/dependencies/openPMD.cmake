function(find_openpmd)
    if(Hipace_openpmd_src)
        message(STATUS "Compiling local openPMD-api ...")
        message(STATUS "openPMD-api source path: ${Hipace_openpmd_src}")
    elseif(Hipace_openpmd_internal)
        message(STATUS "Downloading openPMD-api ...")
        message(STATUS "openPMD-api repository: ${Hipace_openpmd_repo} (${Hipace_openpmd_branch})")
        include(FetchContent)
    endif()
    if(Hipace_openpmd_internal OR Hipace_openpmd_src)
        set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

        # see https://openpmd-api.readthedocs.io/en/0.13.1-alpha/dev/buildoptions.html
        set(openPMD_USE_MPI         ${Hipace_MPI} CACHE INTERNAL "")
        set(openPMD_USE_PYTHON      OFF           CACHE INTERNAL "")
        set(openPMD_BUILD_CLI_TOOLS OFF           CACHE INTERNAL "")
        set(openPMD_BUILD_EXAMPLES  OFF           CACHE INTERNAL "")
        set(openPMD_BUILD_TESTING   OFF           CACHE INTERNAL "")
        set(openPMD_BUILD_SHARED_LIBS OFF         CACHE INTERNAL "")

        if(Hipace_openpmd_src)
            add_subdirectory(${Hipace_openpmd_src} _deps/localopenpmd-build/)
        else()
            FetchContent_Declare(fetchedopenpmd
                GIT_REPOSITORY ${Hipace_openpmd_repo}
                GIT_TAG        ${Hipace_openpmd_branch}
                BUILD_IN_SOURCE 0
            )
            FetchContent_GetProperties(fetchedopenpmd)

            if(NOT fetchedopenpmd_POPULATED)
                FetchContent_Populate(fetchedopenpmd)
                add_subdirectory(${fetchedopenpmd_SOURCE_DIR} ${fetchedopenpmd_BINARY_DIR})
            endif()

            # advanced fetch options
            mark_as_advanced(FETCHCONTENT_BASE_DIR)
            mark_as_advanced(FETCHCONTENT_FULLY_DISCONNECTED)
            mark_as_advanced(FETCHCONTENT_QUIET)
            mark_as_advanced(FETCHCONTENT_SOURCE_DIR_FETCHEDOPENPMD)
            mark_as_advanced(FETCHCONTENT_UPDATES_DISCONNECTED)
            mark_as_advanced(FETCHCONTENT_UPDATES_DISCONNECTED_FETCHEDOPENPMD)
        endif()

        # openPMD options not relevant to Hipace users
        mark_as_advanced(openPMD_USE_INTERNAL_VARIANT)
        mark_as_advanced(openPMD_USE_INTERNAL_CATCH)
        mark_as_advanced(openPMD_USE_INTERNAL_PYBIND11)
        mark_as_advanced(openPMD_USE_INTERNAL_JSON)
        mark_as_advanced(openPMD_HAVE_PKGCONFIG)
        mark_as_advanced(openPMD_USE_INVASIVE_TESTS)
        mark_as_advanced(openPMD_USE_VERIFY)
        mark_as_advanced(ADIOS2_DIR)
        mark_as_advanced(ADIOS_CONFIG)
        mark_as_advanced(HDF5_DIR)
        mark_as_advanced(HDF5_C_LIBRARY_dl)
        mark_as_advanced(HDF5_C_LIBRARY_hdf5)
        mark_as_advanced(HDF5_C_LIBRARY_m)
        mark_as_advanced(HDF5_C_LIBRARY_z)
        mark_as_advanced(JSON_ImplicitConversions)
        mark_as_advanced(JSON_MultipleHeaders)
    else()
        if(Hipace_MPI)
            set(COMPONENT_WMPI MPI)
        else()
            set(COMPONENT_WMPI NOMPI)
        endif()
        find_package(openPMD 0.13.0 CONFIG REQUIRED COMPONENTS ${COMPONENT_WMPI})
        message(STATUS "openPMD-api: Found version '${openPMD_VERSION}'")
    endif()
endfunction()

if(Hipace_OPENPMD)
    # local source-tree
    set(Hipace_openpmd_src ""
        CACHE PATH
        "Local path to openPMD-api source directory (preferred if set)")

    # Git fetcher
    option(Hipace_openpmd_internal   "Download & build openPMD-api" ON)
    set(Hipace_openpmd_repo "https://github.com/openPMD/openPMD-api.git"
        CACHE STRING
        "Repository URI to pull and build openPMD-api from if(Hipace_openpmd_internal)")
    set(Hipace_openpmd_branch "ac083025ee662469b8cad1adf93eef48cde35f58"
        CACHE STRING
        "Repository branch for Hipace_openpmd_repo if(Hipace_openpmd_internal)")

    set(Hipace_HAVE_OPENPMD TRUE)
    find_openpmd()
endif()
