#ifndef HIPACE_IOUTIL_H_
#define HIPACE_IOUTIL_H_

#include <AMReX_REAL.H>
#include <AMReX_IntVect.H>
#include <AMReX_RealVect.H>
#include <AMReX_MultiFab.H>

#include <cstdint>
#include <vector>


namespace utils
{
    /** Get the Relative Cell Position of Values in an MultiFab
     *
     * Translate the IndexType of a given MultiFab into a position relative to
     * the lower corner of a cell.
     *
     * @param[in] mf the amrex::MultiFab to get relative cell positions for
     * @return relative position to the lower corner, scaled to cell size [0.0:1.0)
     */
    std::vector< double >
    getRelativeCellPosition (amrex::MultiFab const& mf);

    /** \brief
     * Convert an IntVect to a std::vector<std::uint64_t>
     * and reverse the order of the elements
     * (used for compatibility with the openPMD API)
     */
    std::vector<std::uint64_t>
    getReversedVec( const amrex::IntVect& v );

    /** \brief
     * Convert Real* pointer to a std::vector<double>,
     * and reverse the order of the elements
     * (used for compatibility with the openPMD API)
     */
    std::vector<double>
    getReversedVec( const amrex::Real* v );
}

#endif
