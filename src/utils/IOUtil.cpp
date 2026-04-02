/* Copyright 2020-2021
 *
 * This file is part of HiPACE++.
 *
 * Authors: MaxThevenet, Severin Diederichs
 * License: BSD-3-Clause-LBNL
 */
#include "IOUtil.H"
#include "Parser.H"

#include <AMReX_IndexType.H>
#include <AMReX_IOFormat.H>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>



std::vector< double >
utils::getRelativeCellPosition (amrex::Box const& box)
{
    amrex::IndexType const idx_type = box.ixType();
    std::vector< double > relative_position(AMREX_SPACEDIM, 0.0);
    // amrex::CellIndex::CELL means: 0.5 from lower corner for that index/direction
    // amrex::CellIndex::NODE means: at corner for that index/direction
    // WarpX::do_nodal means: all indices/directions on CellIndex::NODE
    for (int d = 0; d < AMREX_SPACEDIM; d++)
    {
        if (idx_type.cellCentered(d))
            relative_position.at(d) = 0.5;
    }
    // convert to C order
    std::reverse(relative_position.begin(), relative_position.end());
    return relative_position;
}

std::vector<std::uint64_t>
utils::getReversedVec ( const amrex::IntVect& v )
{
  // Convert the IntVect v to and std::vector u
  std::vector<std::uint64_t> u = {
    AMREX_D_DECL(
                 static_cast<std::uint64_t>(v[0]),
                 static_cast<std::uint64_t>(v[1]),
                 static_cast<std::uint64_t>(v[2])
                 )
  };
  // Reverse the order of elements, if v corresponds to the indices of a
  // Fortran-order array (like an AMReX FArrayBox)
  // but u is intended to be used with a C-order API (like openPMD)
  std::reverse( u.begin(), u.end() );
  return u;
}

std::vector<double>
utils::getReversedVec ( const amrex::Real* v )
{
  // Convert Real* v to and std::vector u
  std::vector<double> u = {
    AMREX_D_DECL(
                 static_cast<double>(v[0]),
                 static_cast<double>(v[1]),
                 static_cast<double>(v[2])
                 )
  };
  // Reverse the order of elements, if v corresponds to the indices of a
  // Fortran-order array (like an AMReX FArrayBox)
  // but u is intended to be used with a C-order API (like openPMD)
  std::reverse( u.begin(), u.end() );
  return u;
}

void
utils::DiagPeriod::compile () {
    m_exe = makeFunctionWithParser<2>(m_func_str, m_parser, {"current_step", "current_time"});
}

bool
utils::DiagPeriod::doDiagnostics (int output_step, amrex::Real output_time, bool is_last_step) const
{
    const auto output_period = static_cast<amrex::Long>(m_exe(output_step, output_time));
    return output_period > 0 && (
        is_last_step ||
        (output_step % output_period == 0) );
}

std::ostream& operator<<(std::ostream& os, utils::format_time ft) {
    long long seconds = static_cast<long long>(std::floor(ft.seconds));

    long long minutes = seconds / 60;
    seconds %= 60;

    long long hours = minutes / 60;
    minutes %= 60;

    long long days = hours / 24;
    hours %= 24;

    amrex::IOFormatSaver iofmtsaver(os);

    if (days > 0) {
        os << days << "-";
    }
    os  << std::setfill('0') << std::setw(2) << hours << ":"
        << std::setfill('0') << std::setw(2) << minutes << ":"
        << std::setfill('0') << std::setw(2) << seconds;

    return os;
}
