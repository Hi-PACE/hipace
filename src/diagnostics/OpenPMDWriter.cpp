#include "diagnostics/OpenPMDWriter.H"

// #include "Hipace.H"
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
OpenPMDWriter::SetupPos(openPMD::ParticleSpecies& currSpecies,
         const unsigned long long& np)
{
    PhysConst const phys_const = get_phys_const();
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
    // currSpecies["charge"][scalar].makeConstant( phys_const.q_e ); // if this is set, the charge is wrong
    currSpecies["mass"][scalar].resetDataset( realType );

    // meta data
    currSpecies["position"].setUnitDimension( utils::getUnitDimension("position") );
    currSpecies["positionOffset"].setUnitDimension( utils::getUnitDimension("positionOffset") );
    currSpecies["charge"].setUnitDimension( utils::getUnitDimension("charge") );
    currSpecies["mass"].setUnitDimension( utils::getUnitDimension("mass") );

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

void
OpenPMDWriter::WriteBeamParticleData (MultiBeam& beams, openPMD::Iteration iteration)
{
    HIPACE_PROFILE("WriteBeamParticleData()");

    for (int ibeam = 0; ibeam < beams.get_nbeams(); ibeam++) {

        std::string name = beams.get_name(ibeam);
        openPMD::ParticleSpecies beam_species = iteration.particles[name];

        const unsigned long long np = beams.get_total_num_particles(ibeam);
        SetupPos(beam_species, np);
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

#endif // HIPACE_USE_OPENPMD
