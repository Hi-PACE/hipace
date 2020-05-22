
# ...
function(make_third_party_includes_system imported_target propagated_name)
    add_library(HiPACE::thirdparty::${propagated_name} INTERFACE IMPORTED)
    target_link_libraries(HiPACE::thirdparty::${propagated_name} INTERFACE AMReX::amrex)
    get_target_property(ALL_INCLUDES ${imported_target} INCLUDE_DIRECTORIES)
    set_target_properties(HiPACE::thirdparty::${propagated_name} PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "")
    target_include_directories(HiPACE::thirdparty::${propagated_name} SYSTEM INTERFACE ${ALL_INCLUDES})
endfunction()


# ... set MPI_TEST_EXE variable ...
function(configure_mpiexec num_ranks)
    # OpenMPI root guard: https://github.com/open-mpi/ompi/issues/4451
    if("$ENV{USER}" STREQUAL "root")
        # calling even --help as root will abort and warn on stderr
        execute_process(COMMAND ${MPIEXEC_EXECUTABLE} --help
            ERROR_VARIABLE MPIEXEC_HELP_TEXT
            OUTPUT_STRIP_TRAILING_WHITESPACE)
            if(${MPIEXEC_HELP_TEXT} MATCHES "^.*allow-run-as-root.*$")
                set(MPI_ALLOW_ROOT --allow-run-as-root)
            endif()
    endif()
    set(MPI_TEST_EXE
        ${MPIEXEC_EXECUTABLE}
        ${MPI_ALLOW_ROOT}
        ${MPIEXEC_NUMPROC_FLAG} ${num_ranks}
    )
endfunction()


# ...
function(hipace_print_summary)
    message("")
    message("HiPACE build configuration:")
    message("  Version: ${HiPACE_VERSION}")
    message("  C++ Compiler: ${CMAKE_CXX_COMPILER_ID} "
                            "${CMAKE_CXX_COMPILER_VERSION} "
                            "${CMAKE_CXX_COMPILER_WRAPPER}")
    message("    ${CMAKE_CXX_COMPILER}")
    message("")
    message("  Installation prefix: ${CMAKE_INSTALL_PREFIX}")
    message("        bin: ${CMAKE_INSTALL_BINDIR}")
    message("        lib: ${CMAKE_INSTALL_LIBDIR}")
    message("    include: ${CMAKE_INSTALL_INCLUDEDIR}")
    message("      cmake: ${CMAKE_INSTALL_CMAKEDIR}")
    if(HiPACE_HAVE_PYTHON)
        message("     python: ${CMAKE_INSTALL_PYTHONDIR}")
    endif()
    message("")
    message("  Build Type: ${CMAKE_BUILD_TYPE}")
    #if(BUILD_SHARED_LIBS)
    #    message("  Library: shared")
    #else()
    #    message("  Library: static")
    #endif()
    message("  Testing: ${BUILD_TESTING}")
    message("  Build Options:")

    foreach(opt IN LISTS HiPACE_CONFIG_OPTIONS)
      if(${HiPACE_HAVE_${opt}})
        message("    ${opt}: ON")
      else()
        message("    ${opt}: OFF")
      endif()
    endforeach()
    message("")
endfunction()
