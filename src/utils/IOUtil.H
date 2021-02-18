#ifndef HIPACE_IOUTIL_H_
#define HIPACE_IOUTIL_H_

#include <AMReX_REAL.H>
#include <AMReX_IntVect.H>
#include <AMReX_RealVect.H>
#include <AMReX_MultiFab.H>

#include <cstdint>
#include <vector>

#ifdef HIPACE_USE_OPENPMD
#include <openPMD/openPMD.hpp>
#endif

namespace utils
{
    /** Get the Relative Cell Position of Values in an MultiFab
     *
     * Translate the IndexType of a given MultiFab into a position relative to
     * the lower corner of a cell.
     *
     * @param[in] fab the amrex::FArrayBox to get relative cell positions for
     * @return relative position to the lower corner, scaled to cell size [0.0:1.0)
     */
    std::vector< double >
    getRelativeCellPosition (amrex::FArrayBox const& fab);

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

    /** A helper function to derive a globally unique particle ID
     * This function was copied from WarpX
     *
     * \param[in] id  AMReX particle ID (on local cpu/rank), AoS .id
     * \param[in] cpu AMReX particle CPU (rank) at creation of the particle, AoS .cpu
     * \return global particle ID that is unique and permanent in the whole simulation
     */
    constexpr uint64_t
    localIDtoGlobal(int const id, int const cpu)
    {
        static_assert(sizeof(int) * 2u <= sizeof(uint64_t),
                      "int size might cause collisions in global IDs");
        // implementation:
        //   - we cast both 32-bit (or smaller) ints to a 64bit unsigned int
        //   - this will leave half of the "upper range" bits in the 64bit unsigned int zeroed out
        //     because the corresponding (extended) value range was not part of the value range in
        //     the int representation
        //   - we bit-shift the cpu into the upper half of zero bits in the 64 bit unsigned int
        //     (imagine this step as "adding a per-cpu/rank offset to the local integers")
        //   - then we add this offset
        //     note: the add is expressed as bitwise OR (|) since this saves us from writing
        //           brackets for operator precedence between + and <<
        return uint64_t(id) | uint64_t(cpu) << 32u;
    }

#ifdef HIPACE_USE_OPENPMD
    std::pair< std::string, std::string >
    name2openPMD( std::string const& fullName );

    /** Get the openPMD physical dimensionality of a record
     *
     * @param record_name name of the openPMD record
     * @return map with base quantities and power scaling
     */
    std::map< openPMD::UnitDimension, double >
    getUnitDimension( std::string const & record_name );
#endif

}

#endif
