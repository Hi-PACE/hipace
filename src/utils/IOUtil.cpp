/* Copyright 2020-2021
 *
 * This file is part of HiPACE++.
 *
 * Authors: MaxThevenet, Severin Diederichs
 * License: BSD-3-Clause-LBNL
 */
#include "IOUtil.H"

#include <AMReX_IndexType.H>

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

bool
utils::doDiagnostics (int output_period, int output_step, int max_step,
               amrex::Real output_time, amrex::Real max_time)
{
    return output_period > 0 && (
        (output_time == max_time) ||
        (output_step == max_step) ||
        (output_step % output_period == 0) );
}

#ifdef HIPACE_USE_OPENPMD
std::pair< std::string, std::string >
utils::name2openPMD ( std::string const& fullName )
{
    std::string record_name = fullName;
    std::string component_name = openPMD::RecordComponent::SCALAR;
    std::size_t startComp = fullName.find_last_of("_");

    if( startComp != std::string::npos ) {  // non-scalar
        record_name = fullName.substr(0, startComp);
        component_name = fullName.substr(startComp + 1u);
    }
    return make_pair(record_name, component_name);
}

/** Get the openPMD physical dimensionality of a record
 *
 * @param record_name name of the openPMD record
 * @return map with base quantities and power scaling
 */
std::map< openPMD::UnitDimension, double >
utils::getUnitDimension ( std::string const & record_name )
{

    if( record_name == "position" ) return {
        {openPMD::UnitDimension::L,  1.}
    };
    else if( record_name == "positionOffset" ) return {
        {openPMD::UnitDimension::L,  1.}
    };
    else if( record_name == "momentum" ) return {
        {openPMD::UnitDimension::L,  1.},
        {openPMD::UnitDimension::M,  1.},
        {openPMD::UnitDimension::T, -1.}
    };
    else if( record_name == "charge" ) return {
        {openPMD::UnitDimension::T,  1.},
        {openPMD::UnitDimension::I,  1.}
    };
    else if( record_name == "mass" ) return {
        {openPMD::UnitDimension::M,  1.}
    };
    else if( record_name == "E" ) return {
        {openPMD::UnitDimension::L,  1.},
        {openPMD::UnitDimension::M,  1.},
        {openPMD::UnitDimension::T, -3.},
        {openPMD::UnitDimension::I, -1.},
    };
    else if( record_name == "B" ) return {
        {openPMD::UnitDimension::M,  1.},
        {openPMD::UnitDimension::I, -1.},
        {openPMD::UnitDimension::T, -2.}
    };
    else if( record_name == "spin" ) return {
        {openPMD::UnitDimension::L,  2.},
        {openPMD::UnitDimension::M,  1.},
        {openPMD::UnitDimension::T, -1.}
    };
    else return {};
}

std::ostream& operator<<(std::ostream& os, utils::format_time ft) {
    long long seconds = static_cast<long long>(std::floor(ft.seconds));

    long long minutes = seconds / 60;
    seconds %= 60;

    long long hours = minutes / 60;
    minutes %= 60;

    long long days = hours / 24;
    hours %= 24;

    if (days > 0) {
        os << days << "-";
    }
    os  << std::setfill('0') << std::setw(2) << hours << ":"
        << std::setfill('0') << std::setw(2) << minutes << ":"
        << std::setfill('0') << std::setw(2) << seconds;

    return os;
}
#endif
