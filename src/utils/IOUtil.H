/* Copyright 2020-2021
 *
 * This file is part of HiPACE++.
 *
 * Authors: MaxThevenet, Severin Diederichs
 * License: BSD-3-Clause-LBNL
 */
#ifndef HIPACE_IOUTIL_H_
#define HIPACE_IOUTIL_H_

#include <AMReX_REAL.H>
#include <AMReX_IntVect.H>
#include <AMReX_RealVect.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Parser.H>

#include <cstdint>
#include <iostream>
#include <vector>

namespace utils
{
    /** Get the Relative Cell Position of Values in an MultiFab
     *
     * Translate the IndexType of a given MultiFab into a position relative to
     * the lower corner of a cell.
     *
     * @param[in] box the amrex::Box to get relative cell positions for
     * @return relative position to the lower corner, scaled to cell size [0.0:1.0)
     */
    std::vector< double >
    getRelativeCellPosition (amrex::Box const& box);

    /** \brief
     * Convert an IntVect to a std::vector<std::uint64_t>
     * and reverse the order of the elements
     * (used for compatibility with the openPMD API)
     */
    std::vector<std::uint64_t>
    getReversedVec ( const amrex::IntVect& v );

    /** \brief
     * Convert Real* pointer to a std::vector<double>,
     * and reverse the order of the elements
     * (used for compatibility with the openPMD API)
     */
    std::vector<double>
    getReversedVec ( const amrex::Real* v );

    struct DiagPeriod {
        std::string m_func_str;
        amrex::Parser m_parser;
        amrex::ParserExecutor<2> m_exe;

        DiagPeriod (const char * str) : m_func_str{str} {}

        void compile ();

        bool isNonZero () const { return m_func_str != "0"; }

        /** \brief
         * returns whether output should be writen to file
         * \param[in] output_step current step
         * \param[in] max_step maximum step of simulation
         * \param[in] output_time physical time of the current step
         * \param[in] max_time maximum simulation time
         */
        bool doDiagnostics (int output_step, int max_step,
                            amrex::Real output_time, amrex::Real max_time) const;
    };

    struct format_time {
        double seconds = 0;
    };
}

/** \brief print time in d-hh:mm:ss format
 * \param[in] os stream to print to
 * \param[in] ft time in seconds to print
 */
std::ostream& operator<< (std::ostream& os, utils::format_time ft);

#endif
