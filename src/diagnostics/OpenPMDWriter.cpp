#include "diagnostics/OpenPMDWriter.H"
#include "fields/Fields.H"
#include "utils/HipaceProfilerWrapper.H"
#include "utils/Constants.H"
#include "utils/IOUtil.H"

#ifdef HIPACE_USE_OPENPMD

OpenPMDWriter::OpenPMDWriter ()
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_real_names.size() == BeamIdx::nattribs,
        "List of real names in openPMD Writer class do not match BeamIdx::nattribs");
}

void
OpenPMDWriter::InitDiagnostics ()
{
    HIPACE_PROFILE("OpenPMDWriter::InitDiagnostics()");


    std::string filename = "diags/h5/openpmd.h5"; // bp or h5
#ifdef AMREX_USE_MPI
    m_outputSeries = std::make_unique< openPMD::Series >(
        filename, openPMD::Access::CREATE, amrex::ParallelDescriptor::Communicator());
#else
    m_outputSeries = std::make_unique< openPMD::Series >(
        filename, openPMD::Access::CREATE);
#endif

    // open files early and collectively, so later flush calls are non-collective
    m_outputSeries->setIterationEncoding( openPMD::IterationEncoding::groupBased );
    m_outputSeries->flush();

    // TODO: meta-data: author, mesh path, extensions, software
}

void
OpenPMDWriter::WriteDiagnostics (
    amrex::Vector<amrex::MultiFab> const& a_mf, MultiBeam& a_multi_beam,
    amrex::Vector<amrex::Geometry> const& geom,
    const amrex::Real physical_time, const int output_step, const int lev,
    const int slice_dir, const amrex::Vector< std::string > varnames,
    const amrex::Geometry& geom3D)
{
    openPMD::Iteration iteration = m_outputSeries->iterations[output_step];
    iteration.setTime(physical_time);

    WriteFieldData(a_mf[lev], geom[lev], slice_dir, varnames, iteration);

    WriteBeamParticleData(a_multi_beam, iteration, geom3D);

    m_outputSeries->flush();

    // back conversion after the flush, to not change the data to be written to file
}

void
OpenPMDWriter::WriteFieldData (
    amrex::MultiFab const& mf, amrex::Geometry const& geom,
    const int slice_dir, const amrex::Vector< std::string > varnames,
    openPMD::Iteration iteration)
{
    // todo: periodicity/boundary, field solver, particle pusher, etc.
    auto meshes = iteration.meshes;

    // loop over field components
    for (int icomp = 0; icomp < Comps[WhichSlice::This]["N"]; ++icomp)
    {
        std::string fieldname = varnames[icomp];
        //                      "B"                "x" (todo)
        //                      "Bx"               ""  (just for now)
        openPMD::Mesh field = meshes[fieldname];
        openPMD::MeshRecordComponent field_comp = field[openPMD::MeshRecordComponent::SCALAR];

        // meta-data
        field.setDataOrder(openPMD::Mesh::DataOrder::C);
        //   node staggering
        auto relative_cell_pos = utils::getRelativeCellPosition(mf);      // AMReX Fortran index order
        std::reverse(relative_cell_pos.begin(), relative_cell_pos.end()); // now in C order
        //   labels, spacing and offsets
        std::vector< std::string > axisLabels {"z", "y", "x"};
        auto dCells = utils::getReversedVec(geom.CellSize()); // dx, dy, dz
        auto offWindow = utils::getReversedVec(geom.ProbLo()); // start of moving window
        if (slice_dir >= 0) {
            // User requested slice IO
            // remove the slicing direction in position, label, resolution, offset
            relative_cell_pos.erase(relative_cell_pos.begin() + 2-slice_dir);
            axisLabels.erase(axisLabels.begin() + 2-slice_dir);
            dCells.erase(dCells.begin() + 2-slice_dir);
            offWindow.erase(offWindow.begin() + 2-slice_dir);
        }
        field_comp.setPosition(relative_cell_pos);
        field.setAxisLabels(axisLabels);
        field.setGridSpacing(dCells);
        field.setGridGlobalOffset(offWindow);

        // data type and global size of the simulation
        openPMD::Datatype datatype = openPMD::determineDatatype< amrex::Real >();
        openPMD::Extent global_size = utils::getReversedVec(geom.Domain().size());
        // If slicing requested, remove number of points for the slicing direction
        if (slice_dir >= 0) global_size.erase(global_size.begin() + 2-slice_dir);

        openPMD::Dataset dataset(datatype, global_size);
        field_comp.resetDataset(dataset);

        // Loop over longitudinal boxes on this rank, from head to tail:
        // Loop through the multifab and store each box as a chunk with openpmd
        for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi)
        {
            amrex::FArrayBox const& fab = mf[mfi]; // note: this might include guards
            amrex::Box const data_box = mfi.validbox();  // w/o guards in all cases
            std::shared_ptr< amrex::Real const > data;
            if (mfi.validbox() == fab.box() ) {
                data = openPMD::shareRaw( fab.dataPtr( icomp ) ); // non-owning view until flush()
            } else {
                // copy data to cut away guards
                amrex::FArrayBox io_fab(mfi.validbox(), 1, amrex::The_Pinned_Arena());
                io_fab.copy< amrex::RunOn::Host >(fab, fab.box(), icomp, mfi.validbox(), 0, 1);
                // move ownership into a shared pointer: openPMD-api will free the copy in flush()
                data = std::move(io_fab.release()); // note: a move honors the custom array-deleter
            }

            // Determine the offset and size of this data chunk in the global output
            amrex::IntVect const box_offset = data_box.smallEnd();
            openPMD::Offset chunk_offset = utils::getReversedVec(box_offset);
            openPMD::Extent chunk_size = utils::getReversedVec(data_box.size());
            if (slice_dir >= 0) { // remove Ny components
                chunk_offset.erase(chunk_offset.begin() + 2-slice_dir);
                chunk_size.erase(chunk_size.begin() + 2-slice_dir);
            }

            field_comp.storeChunk(data, chunk_offset, chunk_size);
        }
    }
}

void
OpenPMDWriter::WriteBeamParticleData (
    MultiBeam& beams, openPMD::Iteration iteration, const amrex::Geometry& geom)
{
    HIPACE_PROFILE("WriteBeamParticleData()");

    for (int ibeam = 0; ibeam < beams.get_nbeams(); ibeam++) {

        std::string name = beams.get_name(ibeam);
        openPMD::ParticleSpecies beam_species = iteration.particles[name];

        const unsigned long long np = beams.get_total_num_particles(ibeam);
        SetupPos(beam_species, np, geom);
        SetupRealProperties(beam_species, m_real_names, np);

        const int lev = 0; // we only have 1 level for now

        uint64_t offset = static_cast<uint64_t>( beams.get_upstream_n_part(ibeam) );
        // Loop over particle boxes NOTE: Only 1 particle box allowed at the moment
        for (BeamParticleIterator pti(beams.getBeam(ibeam), lev); pti.isValid(); ++pti)
        {
            auto const numParticleOnTile = pti.numParticles();
            uint64_t const numParticleOnTile64 = static_cast<uint64_t>( numParticleOnTile );
            // get position and particle ID from aos
            // note: this implementation iterates the AoS 4x...
            // if we flush late as we do now, we can also copy out the data in one go
            const auto& aos = pti.GetArrayOfStructs();  // size =  numParticlesOnTile
            const auto& pos_structs = aos.begin();
            {
                // Save positions
                std::vector< std::string > const positionComponents{"x", "y", "z"};

                for (auto currDim = 0; currDim < AMREX_SPACEDIM; currDim++)
                {
                    std::shared_ptr< amrex::ParticleReal > curr(
                        new amrex::ParticleReal[numParticleOnTile],
                        [](amrex::ParticleReal const *p){ delete[] p; } );

                    for (auto i=0; i<numParticleOnTile; i++) {
                        curr.get()[i] = pos_structs[i].pos(currDim);
                    }
                    std::string const positionComponent = positionComponents[currDim];
                    beam_species["position"][positionComponent].storeChunk(curr, {offset},
                                                                           {numParticleOnTile64});
                }

                // save particle ID after converting it to a globally unique ID
                std::shared_ptr< uint64_t > ids( new uint64_t[numParticleOnTile],
                    [](uint64_t const *p){ delete[] p; } );
                for (auto i=0; i<numParticleOnTile; i++) {
                    ids.get()[i] = utils::localIDtoGlobal( aos[i].id(), aos[i].cpu() );
                }
                auto const scalar = openPMD::RecordComponent::SCALAR;
                beam_species["id"][scalar].storeChunk(ids, {offset}, {numParticleOnTile64});
            }
            //  save "extra" particle properties in SoA (momenta and weight)
            SaveRealProperty(pti, beam_species, offset, m_real_names);
        }  // end for (BeamParticleIterator pti(beams.getBeam(ibeam), lev); pti.isValid(); ++pti)
    }
}

void
OpenPMDWriter::SetupPos(openPMD::ParticleSpecies& currSpecies,
         const unsigned long long& np, const amrex::Geometry& geom)
{
    const PhysConst phys_const_SI = make_constants_SI();
    auto const realType = openPMD::Dataset(openPMD::determineDatatype<amrex::ParticleReal>(), {np});
    auto const idType = openPMD::Dataset(openPMD::determineDatatype< uint64_t >(), {np});

    std::vector< std::string > const positionComponents{"x", "y", "z"};
    for( auto const& comp : positionComponents ) {
        currSpecies["positionOffset"][comp].resetDataset( realType );
        currSpecies["positionOffset"][comp].makeConstant( 0. );
        currSpecies["position"][comp].resetDataset( realType );
    }

    auto const scalar = openPMD::RecordComponent::SCALAR;
    currSpecies["id"][scalar].resetDataset( idType );
    currSpecies["charge"][scalar].resetDataset( realType );
    currSpecies["charge"][scalar].makeConstant( phys_const_SI.q_e );
    currSpecies["mass"][scalar].resetDataset( realType );
    currSpecies["mass"][scalar].makeConstant( phys_const_SI.m_e );

    // meta data
    currSpecies["position"].setUnitDimension( utils::getUnitDimension("position") );
    currSpecies["positionOffset"].setUnitDimension( utils::getUnitDimension("positionOffset") );
    currSpecies["charge"].setUnitDimension( utils::getUnitDimension("charge") );
    currSpecies["mass"].setUnitDimension( utils::getUnitDimension("mass") );

    // calculate the multiplier to convert from Hipace to SI units
    double hipace_to_SI_pos = 1.;
    double hipace_to_SI_weight = 1.;
    double hipace_to_SI_momentum = phys_const_SI.m_e;

    if(Hipace::m_normalized_units) {
        const auto dx = geom.CellSizeArray();
        const double n_0 = 1.;
        currSpecies.setAttribute("Hipace++_Plasma_Density", n_0);
        const double omega_p = (double)phys_const_SI.q_e * sqrt( (double)n_0 /
                                      ( (double)phys_const_SI.ep0 * (double)phys_const_SI.m_e ) );
        const double kp_inv = (double)phys_const_SI.c / omega_p;
        hipace_to_SI_pos = kp_inv;
        hipace_to_SI_weight = n_0 * dx[0] * dx[1] * dx[2] * kp_inv * kp_inv * kp_inv;
        hipace_to_SI_momentum *= phys_const_SI.c;
    }

    // write SI conversion
    currSpecies["position"]["x"].setUnitSI(hipace_to_SI_pos);
    currSpecies["position"]["y"].setUnitSI(hipace_to_SI_pos);
    currSpecies["position"]["z"].setUnitSI(hipace_to_SI_pos);
    currSpecies["positionOffset"]["x"].setUnitSI(hipace_to_SI_pos); //posOffset allways 0
    currSpecies["positionOffset"]["y"].setUnitSI(hipace_to_SI_pos);
    currSpecies["positionOffset"]["z"].setUnitSI(hipace_to_SI_pos);
    currSpecies["momentum"]["x"].setUnitSI(hipace_to_SI_momentum );
    currSpecies["momentum"]["y"].setUnitSI(hipace_to_SI_momentum );
    currSpecies["momentum"]["z"].setUnitSI(hipace_to_SI_momentum );
    currSpecies["weighting"][scalar].setUnitSI(hipace_to_SI_weight);
    currSpecies["charge"][scalar].setUnitSI(1.);
    currSpecies["mass"][scalar].setUnitSI(1.);
}

void
OpenPMDWriter::SetupRealProperties(openPMD::ParticleSpecies& currSpecies,
                      const amrex::Vector<std::string>& real_comp_names,
                      const unsigned long long np)
{
    auto particlesLineup = openPMD::Dataset(openPMD::determineDatatype<amrex::ParticleReal>(), {np});

    /* we have 4 SoA real attributes: weight, ux, uy, uz */
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
            currRecord.setAttribute( "macroWeighted", 0u );
        if( record_name == "momentum" )
            currRecord.setAttribute( "weightingPower", 1.0 );
        else
            currRecord.setAttribute( "weightingPower", 0.0 );
        } // end if newRecord
    } // end for NumSoARealAttributes
}

void
OpenPMDWriter::SaveRealProperty(BeamParticleIterator& pti,
                       openPMD::ParticleSpecies& currSpecies,
                       unsigned long long const offset,
                        amrex::Vector<std::string> const& real_comp_names)

{
    /* we have 4 SoA real attributes: weight, ux, uy, uz */
    int const NumSoARealAttributes = real_comp_names.size();

    auto const numParticleOnTile = pti.numParticles();
    uint64_t const numParticleOnTile64 = static_cast<uint64_t>( numParticleOnTile );
    auto const& soa = pti.GetStructOfArrays();
    {
        for (int idx=0; idx<NumSoARealAttributes; idx++) {

            // handle scalar and non-scalar records by name
            std::string record_name, component_name;
            std::tie(record_name, component_name) = utils::name2openPMD(real_comp_names[idx]);
            auto& currRecord = currSpecies[record_name];
            auto& currRecordComp = currRecord[component_name];

            currRecordComp.storeChunk(openPMD::shareRaw(soa.GetRealData(idx)),
                {offset}, {numParticleOnTile64});
        } // end for NumSoARealAttributes
    }
}

#endif // HIPACE_USE_OPENPMD
