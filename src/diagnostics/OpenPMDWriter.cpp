/* Copyright 2020-2022
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn, MaxThevenet, Severin Diederichs
 * License: BSD-3-Clause-LBNL
 */
#include "diagnostics/OpenPMDWriter.H"
#include "diagnostics/Diagnostic.H"
#include "fields/Fields.H"
#include "utils/HipaceProfilerWrapper.H"
#include "utils/Constants.H"
#include "utils/IOUtil.H"
#include "Hipace.H"

#ifdef HIPACE_USE_OPENPMD

#include <openPMD/openPMD.hpp>

namespace utils {
    std::pair< std::string, std::string >
    name2openPMD ( std::string const& fullName )
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
    getUnitDimension ( std::string const & record_name )
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
}

void
OpenPMDWriter::ReadParameters ()
{
    amrex::ParmParse pp("hipace");
    queryWithParser(pp, "openpmd_backend", m_openpmd_backend);
    // pick first available backend if default is chosen
    if( m_openpmd_backend == "default" ) {
#if openPMD_HAVE_HDF5==1
        m_openpmd_backend = "h5";
#elif openPMD_HAVE_ADIOS2==1
        m_openpmd_backend = "bp";
#else
        m_openpmd_backend = "json";
#endif
    }

    // set default output path according to backend
    if (m_openpmd_backend == "h5") {
        m_file_prefix = "diags/hdf5";
    } else if (m_openpmd_backend == "bp") {
        m_file_prefix = "diags/adios2";
    } else if (m_openpmd_backend == "json") {
        m_file_prefix = "diags/json";
    }
    // overwrite output path by choice of the user
    queryWithParser(pp, "file_prefix", m_file_prefix);

    // temporary workaround until openPMD-viewer gets fixed
    amrex::ParmParse ppd("diagnostic");
    queryWithParser(ppd, "openpmd_viewer_u_workaround", m_openpmd_viewer_workaround);
}

OpenPMDWriter::OpenPMDWriter () {}

OpenPMDWriter::~OpenPMDWriter() {}

void
OpenPMDWriter::InitDiagnostics ()
{
    HIPACE_PROFILE("OpenPMDWriter::InitDiagnostics()");

    std::string filename = m_file_prefix + "/openpmd_%06T." + m_openpmd_backend;

    m_outputSeries = std::make_unique< openPMD::Series >(
        filename, openPMD::Access::CREATE);

    // TODO: meta-data: author, mesh path, extensions, software
}

void
OpenPMDWriter::WriteBeamDiagnostics (
    MultiBeam& a_multi_beam, const amrex::Real physical_time, const int output_step,
    const amrex::Vector< std::string > beamnames,
    amrex::Vector<amrex::Geometry> const& geom3D)
{
    openPMD::Iteration iteration = m_outputSeries->iterations[output_step];
    iteration.setTime(physical_time);

    WriteBeamParticleData(a_multi_beam, iteration, geom3D[0], beamnames);
}

void
OpenPMDWriter::WriteFieldDiagnostics (
    const amrex::Vector<FieldDiagnosticData>& field_diag,
    const MultiLaser& a_multi_laser, const amrex::Real physical_time, const int output_step)
{
    openPMD::Iteration iteration = m_outputSeries->iterations[output_step];
    iteration.setTime(physical_time);

    for (const auto& fd : field_diag) {
        if (fd.m_has_field) {
            WriteFieldData(fd, a_multi_laser, iteration);
        }
    }
}

void
OpenPMDWriter::WriteFieldData (
    const FieldDiagnosticData& fd, const MultiLaser& a_multi_laser, openPMD::Iteration& iteration)
{
    HIPACE_PROFILE("OpenPMDWriter::WriteFieldData()");

    // todo: periodicity/boundary, field solver, particle pusher, etc.
    auto meshes = iteration.meshes;

    // loop over field components
    for ( int icomp = 0; icomp < fd.m_nfields; ++icomp )
    {
        const bool is_laser_comp = fd.m_base_geom_type == FieldDiagnosticData::geom_type::laser;

        //                      "B"                "x" (todo)
        //                      "Bx"               ""  (just for now)
        openPMD::Mesh field = meshes[fd.m_comps_output[icomp]];
        openPMD::MeshRecordComponent field_comp = field[openPMD::MeshRecordComponent::SCALAR];

        // meta-data
        field.setDataOrder(openPMD::Mesh::DataOrder::C);

        const amrex::Geometry& geom = fd.m_geom_io;
        const amrex::Box data_box = is_laser_comp ? fd.m_F_laser.box() : fd.m_F.box();

        // node staggering, labels, spacing and offsets
        // convert AMReX Fortran index order to C order
        auto relative_cell_pos = utils::getRelativeCellPosition(data_box);
        std::vector< std::string > axisLabels {"z", "y", "x"};
        auto dCells = utils::getReversedVec(geom.CellSize()); // dz, dy, dx
        auto offWindow = utils::getReversedVec(geom.ProbLo());
        openPMD::Extent global_size = utils::getReversedVec(geom.Domain().size());
        const amrex::IntVect box_offset {0, 0, data_box.smallEnd(2) - geom.Domain().smallEnd(2)};
        openPMD::Offset chunk_offset = utils::getReversedVec(box_offset);
        openPMD::Extent chunk_size = utils::getReversedVec(data_box.size());
        if (fd.m_slice_dir >= 0) {
            const int remove_dir = 2 - fd.m_slice_dir;
            // User requested slice IO
            // remove the slicing direction in position, label, resolution, offset
            relative_cell_pos.erase(relative_cell_pos.begin() + remove_dir);
            axisLabels.erase(axisLabels.begin() + remove_dir);
            dCells.erase(dCells.begin() + remove_dir);
            offWindow.erase(offWindow.begin() + remove_dir);
            global_size.erase(global_size.begin() + remove_dir);
            chunk_offset.erase(chunk_offset.begin() + remove_dir);
            chunk_size.erase(chunk_size.begin() + remove_dir);
        }
        field_comp.setPosition(relative_cell_pos);
        field.setAxisLabels(axisLabels);
        field.setGridSpacing(dCells);
        field.setGridGlobalOffset(offWindow);

        openPMD::Datatype datatype = is_laser_comp ?
            openPMD::determineDatatype< std::complex<amrex::Real> >() :
            openPMD::determineDatatype< amrex::Real >();
        // set data type and global size of the simulation
        openPMD::Dataset dataset(datatype, global_size);
        field_comp.resetDataset(dataset);

        if (is_laser_comp) {
            // set laser attributes and store laser
            field.setAttribute("envelopeField", "normalized_vector_potential");
            field.setAttribute("angularFrequency",
                double(2.) * MathConst::pi * PhysConstSI::c / a_multi_laser.GetLambda0());
            std::vector< std::complex<double> > polarization {{1., 0.}, {0., 0.}};
            field.setAttribute("polarization", polarization);
            field_comp.storeChunkRaw(
                reinterpret_cast<const std::complex<amrex::Real>*>(fd.m_F_laser.dataPtr()),
                chunk_offset, chunk_size);
        } else {
            field_comp.storeChunkRaw(fd.m_F.dataPtr(icomp), chunk_offset, chunk_size);
        }
    }
}

void
OpenPMDWriter::InitBeamData (MultiBeam& beams, const amrex::Vector< std::string > beamnames)
{
    HIPACE_PROFILE("OpenPMDWriter::InitBeamData()");

    const int nbeams = beams.get_nbeams();
    m_offset.resize(nbeams);
    m_uint64_beam_data.resize(nbeams);
    m_real_beam_data.resize(nbeams);
    for (int ibeam = 0; ibeam < nbeams; ibeam++) {

        std::string name = beams.get_name(ibeam);
        if(std::find(beamnames.begin(), beamnames.end(), name) ==  beamnames.end() ) continue;

        auto& beam = beams.getBeam(ibeam);

        // initialize beam IO on first slice
        uint64_t np_total = beam.getTotalNumParticles();

        if (beam.m_output_ratio > 1) {
            np_total = (np_total + beam.m_output_ratio - 1) / beam.m_output_ratio;
        }

        m_uint64_beam_data[ibeam].resize(m_int_names.size());

        for (std::size_t idx=0; idx<m_uint64_beam_data[ibeam].size(); idx++) {
            m_uint64_beam_data[ibeam][idx].resize(np_total);
        }

        if (beam.m_do_spin_tracking) {
            m_real_beam_data[ibeam].resize(m_real_names.size() + m_real_names_spin.size());
        } else {
            m_real_beam_data[ibeam].resize(m_real_names.size());
        }

        for (std::size_t idx=0; idx<m_real_beam_data[ibeam].size(); idx++) {
            m_real_beam_data[ibeam][idx].resize(np_total);
        }

        // if first slice of loop over slices, reset offset
        m_offset[ibeam] = 0;
    }
}

void
OpenPMDWriter::WriteBeamParticleData (MultiBeam& beams, openPMD::Iteration& iteration,
                                      const amrex::Geometry& geom,
                                      const amrex::Vector< std::string > beamnames)
{
    HIPACE_PROFILE("OpenPMDWriter::WriteBeamParticleData()");

    // sync GPU to get ids
    amrex::Gpu::streamSynchronize();

    const int nbeams = beams.get_nbeams();
    for (int ibeam = 0; ibeam < nbeams; ibeam++) {

        std::string name = beams.get_name(ibeam);
        if(std::find(beamnames.begin(), beamnames.end(), name) ==  beamnames.end() ) continue;

        openPMD::ParticleSpecies beam_species = iteration.particles[name];

        auto& beam = beams.getBeam(ibeam);

        amrex::Vector<std::string> real_names = m_real_names;
        if (beam.m_do_spin_tracking) {
            real_names.insert(real_names.end(), m_real_names_spin.begin(), m_real_names_spin.end());
        }

        // initialize beam IO on first slice
        const uint64_t np_total = m_offset[ibeam];

        SetupPos(beam_species, beam, np_total, geom);
        SetupRealProperties(beam_species, real_names, np_total);

        if (np_total == 0) {
            amrex::ErrorStream() << "WARNING: Beam '" << name
                                 << "' has no particles! No output will be written.\n";
            continue;
        }

        for (std::size_t idx=0; idx<m_uint64_beam_data[ibeam].size(); idx++) {
            uint64_t * const uint64_data = m_uint64_beam_data[ibeam][idx].data();

            for (uint64_t i=0; i<np_total; ++i) {
                uint64_t id = uint64_data[i];
                // in the amrex format valid idcpus start with 1 and invalid with 0
                amrex::ParticleIDWrapper{id}.make_invalid();
                uint64_data[i] = id;
            }

            // handle scalar and non-scalar records by name
            auto [record_name, component_name] = utils::name2openPMD(m_int_names[idx]);
            auto& currRecord = beam_species[record_name];
            auto& currRecordComp = currRecord[component_name];
            // not read until the data is flushed
            currRecordComp.storeChunkRaw(m_uint64_beam_data[ibeam][idx].data(), {0ull}, {np_total});
        }

        for (std::size_t idx=0; idx<m_real_beam_data[ibeam].size(); idx++) {
            // handle scalar and non-scalar records by name
            auto [record_name, component_name] = utils::name2openPMD(real_names[idx]);
            auto& currRecord = beam_species[record_name];
            auto& currRecordComp = currRecord[component_name];
            // not read until the data is flushed
            currRecordComp.storeChunkRaw(m_real_beam_data[ibeam][idx].data(), {0ull}, {np_total});
        }
    }
}

void
OpenPMDWriter::CopyBeams (MultiBeam& beams, const amrex::Vector< std::string > beamnames)
{
    HIPACE_PROFILE("OpenPMDWriter::CopyBeams()");

    const int nbeams = beams.get_nbeams();
    for (int ibeam = 0; ibeam < nbeams; ibeam++) {

        std::string name = beams.get_name(ibeam);
        if(std::find(beamnames.begin(), beamnames.end(), name) ==  beamnames.end() ) continue;

        auto& beam = beams.getBeam(ibeam);

        uint64_t np = beam.getNumParticles(WhichBeamSlice::This);

        const int output_ratio = beam.m_output_ratio;

        if (output_ratio > 1) {
            np = amrex::partitionParticles(beam.getBeamSlice(WhichBeamSlice::This),
                [=] AMREX_GPU_DEVICE (auto& ptd, int i) {
                    return i < int(np) && ptd.idcpu(i) % output_ratio == 0;
                }
            );
        }

        if (np != 0) {
            // copy data from GPU to IO buffer
            auto& soa = beam.getBeamSlice(WhichBeamSlice::This).GetStructOfArrays();

            for (std::size_t idx=0; idx<m_uint64_beam_data[ibeam].size(); idx++) {
                const auto old_size = m_uint64_beam_data[ibeam][idx].size();
                if (old_size < m_offset[ibeam] + np) {
                    m_uint64_beam_data[ibeam][idx].resize(
                        std::max<uint64_t>(old_size+old_size/4, m_offset[ibeam] + np)
                    );
                }
                amrex::Gpu::copyAsync(amrex::Gpu::deviceToHost,
                    soa.GetIdCPUData().begin(),
                    soa.GetIdCPUData().begin() + np,
                    m_uint64_beam_data[ibeam][idx].data() + m_offset[ibeam]);
            }

            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                int(m_real_beam_data[ibeam].size()) == soa.NumRealComps(),
                "List of real names in openPMD Writer class does not match the beam");

            for (std::size_t idx=0; idx<m_real_beam_data[ibeam].size(); idx++) {
                const auto old_size = m_real_beam_data[ibeam][idx].size();
                if (old_size < m_offset[ibeam] + np) {
                    m_real_beam_data[ibeam][idx].resize(
                        std::max<uint64_t>(old_size+old_size/4, m_offset[ibeam] + np)
                    );
                }
                amrex::Gpu::copyAsync(amrex::Gpu::deviceToHost,
                    soa.GetRealData(idx).begin(),
                    soa.GetRealData(idx).begin() + np,
                    m_real_beam_data[ibeam][idx].data() + m_offset[ibeam]);
            }
        }

        m_offset[ibeam] += np;
    }
}

void
OpenPMDWriter::SetupPos (openPMD::ParticleSpecies& currSpecies, BeamParticleContainer& beam,
                         const unsigned long long& np, const amrex::Geometry& geom)
{
    const PhysConst phys_const_SI = make_constants_SI();
    auto const realType = openPMD::Dataset(openPMD::determineDatatype<amrex::ParticleReal>(), {np});
    auto const idType = openPMD::Dataset(openPMD::determineDatatype< uint64_t >(), {np});

    std::vector< std::string > const positionComponents{"x", "y", "z"};
    for( auto const& comp : positionComponents ) {
        currSpecies["positionOffset"][comp].resetDataset( realType );
        currSpecies["positionOffset"][comp].makeConstant( 0. );
    }

    auto const scalar = openPMD::RecordComponent::SCALAR;
    currSpecies["id"][scalar].resetDataset( idType );
    currSpecies["charge"][scalar].resetDataset( realType );
    currSpecies["charge"][scalar].makeConstant( beam.m_charge );
    currSpecies["mass"][scalar].resetDataset( realType );
    currSpecies["mass"][scalar].makeConstant( beam.m_mass );

    // meta data
    currSpecies["positionOffset"].setUnitDimension( utils::getUnitDimension("positionOffset") );
    currSpecies["charge"].setUnitDimension( utils::getUnitDimension("charge") );
    currSpecies["mass"].setUnitDimension( utils::getUnitDimension("mass") );

    // calculate the multiplier to convert from Hipace to SI units
    double hipace_to_SI_pos = 1.;
    double hipace_to_SI_weight = 1.;
    double hipace_to_SI_momentum = beam.m_mass;
    double hipace_to_unitSI_momentum = beam.m_mass;
    double hipace_to_SI_charge = 1.;
    double hipace_to_SI_mass = 1.;

    if(Hipace::m_normalized_units) {
        const auto dx = geom.CellSizeArray();
        const double n_0 = 1.;
        currSpecies.setAttribute("HiPACE++_Plasma_Density", n_0);
        const double omega_p = (double)phys_const_SI.q_e * sqrt( (double)n_0 /
                                      ( (double)phys_const_SI.ep0 * (double)phys_const_SI.m_e ) );
        const double kp_inv = (double)phys_const_SI.c / omega_p;
        hipace_to_SI_pos = kp_inv;
        hipace_to_SI_weight = n_0 * dx[0] * dx[1] * dx[2] * kp_inv * kp_inv * kp_inv;
        hipace_to_SI_momentum = beam.m_mass * phys_const_SI.m_e * phys_const_SI.c;
        hipace_to_SI_charge = phys_const_SI.q_e;
        hipace_to_SI_mass = phys_const_SI.m_e;
    }

    // temporary workaround until openPMD-viewer does not autonormalize momentum
    if(m_openpmd_viewer_workaround) {
        if(Hipace::m_normalized_units) {
            hipace_to_unitSI_momentum = beam.m_mass * phys_const_SI.c;
        }
    }

    // write SI conversion
    currSpecies.setAttribute("HiPACE++_use_reference_unitSI", true);
    const std::string attr = "HiPACE++_reference_unitSI";
    for( auto const& comp : positionComponents ) {
        currSpecies["position"][comp].setAttribute( attr, hipace_to_SI_pos );
        //posOffset allways 0
        currSpecies["positionOffset"][comp].setAttribute( attr, hipace_to_SI_pos );
        currSpecies["momentum"][comp].setAttribute( attr, hipace_to_SI_momentum );
        currSpecies["momentum"][comp].setUnitSI( hipace_to_unitSI_momentum );
    }
    currSpecies["weighting"][scalar].setAttribute( attr, hipace_to_SI_weight );
    currSpecies["charge"][scalar].setAttribute( attr, hipace_to_SI_charge );
    currSpecies["mass"][scalar].setAttribute( attr, hipace_to_SI_mass );
}

void
OpenPMDWriter::SetupRealProperties (openPMD::ParticleSpecies& currSpecies,
                                    const amrex::Vector<std::string>& real_comp_names,
                                    const unsigned long long np)
{
    auto particlesLineup = openPMD::Dataset(openPMD::determineDatatype<amrex::ParticleReal>(),{np});

    /* we have 7 or 10 SoA real attributes: x, y, z, weight, ux, uy, uz, (sx, sy, sz) */
    int const NumSoARealAttributes = real_comp_names.size();
    std::set< std::string > addedRecords; // add meta-data per record only once

    for (int i = 0; i < NumSoARealAttributes; ++i)
    {
        // handle scalar and non-scalar records by name
        std::string record_name, component_name;
        std::tie(record_name, component_name) = utils::name2openPMD(real_comp_names[i]);

        auto particleVarComp = currSpecies[record_name][component_name];
        particleVarComp.resetDataset(particlesLineup);

        auto currRecord = currSpecies[record_name];

        // meta data for ED-PIC extension
        bool newRecord = false;
        std::tie(std::ignore, newRecord) = addedRecords.insert(record_name);
        if( newRecord ) {
            currRecord.setUnitDimension( utils::getUnitDimension(record_name) );

            if( record_name == "weighting") {
                currRecord.setAttribute( "macroWeighted", 1u );
            } else {
                currRecord.setAttribute( "macroWeighted", 0u );
            }

            if( record_name == "weighting" || record_name == "momentum" || record_name == "spin") {
                currRecord.setAttribute( "weightingPower", 1.0 );
            } else {
                currRecord.setAttribute( "weightingPower", 0.0 );
            }
        } // end if newRecord
    } // end for NumSoARealAttributes
}

void OpenPMDWriter::flush ()
{
    amrex::Gpu::streamSynchronize();
    if (m_outputSeries) {
        HIPACE_PROFILE("OpenPMDWriter::flush()");
        m_outputSeries->flush();
    }
    // need to keep these alive until after the flush
    m_uint64_beam_data.resize(0);
    m_real_beam_data.resize(0);
    m_outputSeries.reset();
}

#endif // HIPACE_USE_OPENPMD
