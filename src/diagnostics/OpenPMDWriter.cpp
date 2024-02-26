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

OpenPMDWriter::OpenPMDWriter ()
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
OpenPMDWriter::WriteDiagnostics (
    const amrex::Vector<FieldDiagnosticData>& field_diag, MultiBeam& a_multi_beam,
    const MultiLaser& a_multi_laser, const amrex::Real physical_time, const int output_step,
    const amrex::Vector< std::string > beamnames,
    amrex::Vector<amrex::Geometry> const& geom3D,
    const OpenPMDWriterCallType call_type)
{
    openPMD::Iteration iteration = m_outputSeries->iterations[output_step];
    iteration.setTime(physical_time);

    if (call_type == OpenPMDWriterCallType::beams ) {
        WriteBeamParticleData(a_multi_beam, iteration, geom3D[0], beamnames);
    } else if (call_type == OpenPMDWriterCallType::fields) {
        for (const auto& fd : field_diag) {
            if (fd.m_has_field) {
                WriteFieldData(fd, a_multi_laser, iteration);
            }
        }
    }
}

void
OpenPMDWriter::WriteFieldData (
    const FieldDiagnosticData& fd, const MultiLaser& a_multi_laser, openPMD::Iteration iteration)
{
    HIPACE_PROFILE("OpenPMDWriter::WriteFieldData()");

    // todo: periodicity/boundary, field solver, particle pusher, etc.
    auto meshes = iteration.meshes;

    amrex::Vector<std::string> varnames = fd.m_comps_output;

    if (fd.m_do_laser) {
        // laser must be at the end of varnames
        varnames.push_back(fd.m_laser_io_name);
    }

    // loop over field components
    for ( int icomp = 0; icomp < varnames.size(); ++icomp )
    {
        const bool is_laser_comp = varnames[icomp].find("laserEnvelope") == 0;

        //                      "B"                "x" (todo)
        //                      "Bx"               ""  (just for now)
        openPMD::Mesh field = meshes[varnames[icomp]];
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

        // initialize beam IO on first slice
        const uint64_t np_total = beams.getBeam(ibeam).getTotalNumParticles();

        m_uint64_beam_data[ibeam].resize(m_int_names.size());

        for (std::size_t idx=0; idx<m_uint64_beam_data[ibeam].size(); idx++) {
            m_uint64_beam_data[ibeam][idx].reset(
                reinterpret_cast<uint64_t*>(
                    amrex::The_Pinned_Arena()->alloc(sizeof(uint64_t)*np_total)
                ),
                [](uint64_t *p){
                    amrex::The_Pinned_Arena()->free(reinterpret_cast<void*>(p));
                });
        }

        if (beams.getBeam(ibeam).m_do_spin_tracking) {
            m_real_beam_data[ibeam].resize(m_real_names.size() + m_real_names_spin.size());
        } else {
            m_real_beam_data[ibeam].resize(m_real_names.size());
        }

        for (std::size_t idx=0; idx<m_real_beam_data[ibeam].size(); idx++) {
            m_real_beam_data[ibeam][idx].reset(
                reinterpret_cast<amrex::ParticleReal*>(
                    amrex::The_Pinned_Arena()->alloc(sizeof(amrex::ParticleReal)*np_total)
                ),
                [](amrex::ParticleReal *p){
                    amrex::The_Pinned_Arena()->free(reinterpret_cast<void*>(p));
                });
        }

        // if first slice of loop over slices, reset offset
        m_offset[ibeam] = 0;
    }
}

void
OpenPMDWriter::WriteBeamParticleData (MultiBeam& beams, openPMD::Iteration iteration,
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
        AMREX_ALWAYS_ASSERT(m_offset[ibeam] <= beam.getTotalNumParticles());
        const uint64_t np_total = m_offset[ibeam];

        SetupPos(beam_species, beam, np_total, geom);
        SetupRealProperties(beam_species, real_names, np_total);

        for (std::size_t idx=0; idx<m_uint64_beam_data[ibeam].size(); idx++) {
            uint64_t * const uint64_data = m_uint64_beam_data[ibeam][idx].get();

            for (uint64_t i=0; i<np_total; ++i) {
                amrex::Long id = amrex::ConstParticleIDWrapper(uint64_data[i]);
                uint64_data[i] = id;
            }

            // handle scalar and non-scalar records by name
            auto [record_name, component_name] = utils::name2openPMD(m_int_names[idx]);
            auto& currRecord = beam_species[record_name];
            auto& currRecordComp = currRecord[component_name];
            // not read until the data is flushed
            currRecordComp.storeChunk(m_uint64_beam_data[ibeam][idx], {0ull}, {np_total});
        }

        for (std::size_t idx=0; idx<m_real_beam_data[ibeam].size(); idx++) {
            // handle scalar and non-scalar records by name
            auto [record_name, component_name] = utils::name2openPMD(real_names[idx]);
            auto& currRecord = beam_species[record_name];
            auto& currRecordComp = currRecord[component_name];
            // not read until the data is flushed
            currRecordComp.storeChunk(m_real_beam_data[ibeam][idx], {0ull}, {np_total});
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

        const uint64_t np = beam.getNumParticles(WhichBeamSlice::This);

        if (np != 0) {
            // copy data from GPU to IO buffer
            auto& soa = beam.getBeamSlice(WhichBeamSlice::This).GetStructOfArrays();

            for (std::size_t idx=0; idx<m_uint64_beam_data[ibeam].size(); idx++) {
                amrex::Gpu::copyAsync(amrex::Gpu::deviceToHost,
                    soa.GetIdCPUData().begin(),
                    soa.GetIdCPUData().begin() + np,
                    m_uint64_beam_data[ibeam][idx].get() + m_offset[ibeam]);
            }

            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_real_beam_data[ibeam].size() == soa.NumRealComps(),
                "List of real names in openPMD Writer class does not match the beam");

            for (std::size_t idx=0; idx<m_real_beam_data[ibeam].size(); idx++) {
                amrex::Gpu::copyAsync(amrex::Gpu::deviceToHost,
                    soa.GetRealData(idx).begin(),
                    soa.GetRealData(idx).begin() + np,
                    m_real_beam_data[ibeam][idx].get() + m_offset[ibeam]);
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
    m_uint64_beam_data.resize(0);
    m_real_beam_data.resize(0);
    if (m_outputSeries) {
        HIPACE_PROFILE("OpenPMDWriter::flush()");
        m_outputSeries->flush();
    }
    m_outputSeries.reset();
}

#endif // HIPACE_USE_OPENPMD
