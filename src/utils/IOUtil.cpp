#include "IOUtil.H"

#include <AMReX_IndexType.H>

#include <algorithm>



std::vector< double >
utils::getRelativeCellPosition(amrex::MultiFab const& mf)
{
    amrex::IndexType const idx_type = mf.ixType();
    std::vector< double > relative_position(AMREX_SPACEDIM, 0.0);
    // amrex::CellIndex::CELL means: 0.5 from lower corner for that index/direction
    // amrex::CellIndex::NODE means: at corner for that index/direction
    // WarpX::do_nodal means: all indices/directions on CellIndex::NODE
    for (int d = 0; d < AMREX_SPACEDIM; d++)
    {
        if (idx_type.cellCentered(d))
            relative_position.at(d) = 0.5;
    }
    return relative_position;
}

std::vector<std::uint64_t>
utils::getReversedVec( const amrex::IntVect& v )
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
utils::getReversedVec( const amrex::Real* v )
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
