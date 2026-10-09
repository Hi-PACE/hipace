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
        std::size_t startComp = fullName.find_last_of("/");

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

        if( (record_name == "position") ||
            (record_name == "positionOffset")) return {
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
        m_file_prefix = Hipace::m_output_folder + "/hdf5";
    } else if (m_openpmd_backend == "bp") {
        m_file_prefix = Hipace::m_output_folder + "/adios2";
    } else if (m_openpmd_backend == "json") {
        m_file_prefix = Hipace::m_output_folder + "/json";
    }
    // overwrite output path by choice of the user
    const bool set_file_prefix = queryWithParser(pp, "file_prefix", m_file_prefix);
    if (set_file_prefix) {
        amrex::Print() <<
            "It is recommended to use hipace.output_folder instead of hipace.file_prefix\n";
    }
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

    m_outputSeries->setSoftware("HiPACE++", Hipace::Version());

    // TODO: meta-data: author, mesh path, extensions
}

void
OpenPMDWriter::WriteDiagnostics (
    amrex::Vector<DiagnosticData>& diag_data,
    const MultiLaser& multi_laser, const MultiBeam& beams, const MultiPlasma& plasmas,
    const amrex::Geometry& geom, const amrex::Real physical_time, const int output_step
)
{
    HIPACE_PROFILE("OpenPMDWriter::WriteDiagnostics()");

    {
        openPMD::Iteration iteration = m_outputSeries->iterations[output_step];
        iteration.setTime(physical_time);

        for (auto& fd : diag_data) {
            if (fd.m_has_output) {
                switch (fd.m_base_diag_type) {
                    case DiagnosticData::diag_type::field:
                    case DiagnosticData::diag_type::laser:
                    case DiagnosticData::diag_type::histogram:
                        WriteFieldData(fd, multi_laser, iteration);
                        break;
                    case DiagnosticData::diag_type::beam:
                    case DiagnosticData::diag_type::plasma_slice:
                    case DiagnosticData::diag_type::particle_boundary:
                        WriteParticleData(fd, iteration, beams, plasmas, geom);
                        break;
                }
            }
        }
    }

    amrex::Gpu::streamSynchronize();
    if (m_outputSeries) {
        m_outputSeries->flush();
    }
    m_outputSeries.reset();
}

void
OpenPMDWriter::WriteFieldData (
    const DiagnosticData& fd, const MultiLaser& multi_laser, openPMD::Iteration& iteration)
{
    // todo: periodicity/boundary, field solver, particle pusher, etc.
    auto meshes = iteration.meshes;

    // loop over field components
    for ( int icomp = 0; icomp < fd.m_nfields; ++icomp )
    {
        //                      "B"                "x" (todo)
        //                      "Bx"               ""  (just for now)
        openPMD::Mesh field = meshes[fd.m_comps_output[icomp]];
        openPMD::MeshRecordComponent field_comp = field[openPMD::MeshRecordComponent::SCALAR];

        // meta-data
        field.setDataOrder(openPMD::Mesh::DataOrder::C);

        const amrex::Geometry& geom = fd.m_geom_io;

        // node staggering, labels, spacing and offsets
        // convert AMReX Fortran index order to C order
        auto relative_cell_pos = utils::getRelativeCellPosition(geom.Domain());
        auto dCells = utils::getReversedVec(geom.CellSize()); // dz, dy, dx
        auto offWindow = utils::getReversedVec(geom.ProbLo());
        openPMD::Extent global_size = utils::getReversedVec(geom.Domain().size());
        const amrex::IntVect box_offset {0, 0, 0};
        openPMD::Offset chunk_offset = utils::getReversedVec(box_offset);
        openPMD::Extent chunk_size = utils::getReversedVec(geom.Domain().size());

        for (int i=0; i<3; ++i) {
            if (fd.m_remove_axis[i]) {
                const int remove_dir = 2 - i;
                // User requested slice IO
                // remove the slicing direction in position, label, resolution, offset
                // Remove entries starting from the back of the vectors
                relative_cell_pos.erase(relative_cell_pos.begin() + remove_dir);
                dCells.erase(dCells.begin() + remove_dir);
                offWindow.erase(offWindow.begin() + remove_dir);
                global_size.erase(global_size.begin() + remove_dir);
                chunk_offset.erase(chunk_offset.begin() + remove_dir);
                chunk_size.erase(chunk_size.begin() + remove_dir);
            }
        }

        std::vector<std::string> axisLabels;
        for (int i=fd.m_axis_labels.size()-1; i>=0; --i) {
            axisLabels.push_back(fd.m_axis_labels[i]);
        }

        field_comp.setPosition(relative_cell_pos);
        field.setAxisLabels(axisLabels);
        field.setGridSpacing(dCells);
        field.setGridGlobalOffset(offWindow);

        openPMD::Datatype datatype = fd.m_base_diag_type == DiagnosticData::diag_type::laser ?
            openPMD::determineDatatype< std::complex<amrex::Real> >() :
            openPMD::determineDatatype< amrex::Real >();
        // set data type and global size of the simulation
        openPMD::Dataset dataset(datatype, global_size);
        field_comp.resetDataset(dataset);

        switch (fd.m_base_diag_type) {
            case DiagnosticData::diag_type::field:
            case DiagnosticData::diag_type::histogram:
                field_comp.storeChunkRaw(fd.m_F_real.dataPtr(icomp), chunk_offset, chunk_size);
                break;
            case DiagnosticData::diag_type::laser:
                // set laser attributes and store laser
                if (fd.m_comps_output[icomp] == "laserEnvelope") {
                    field.setAttribute("envelopeField", "normalized_vector_potential");
                    field.setAttribute("angularFrequency",
                        double(2.) * MathConst::pi * PhysConstSI::c / multi_laser.GetLambda0());
                    std::vector< std::complex<double> > polarization {{1., 0.}, {0., 0.}};
                    field.setAttribute("polarization", polarization);
                }
                field_comp.storeChunkRaw(
                    reinterpret_cast<const std::complex<amrex::Real>*>(
                        fd.m_F_complex.dataPtr(icomp)),
                    chunk_offset, chunk_size);
                break;
            default:
                break;
        }
    }
}

void
OpenPMDWriter::WriteParticleData (DiagnosticData& fd, openPMD::Iteration& iteration,
    const MultiBeam& beams, const MultiPlasma& plasmas, const amrex::Geometry& geom)
{
    for (amrex::Long i = 0; i < fd.m_species_names.size(); ++i) {
        const std::string& species_name = fd.m_species_names[i];

        openPMD::ParticleSpecies particle_species = iteration.particles[fd.m_comps_output[i]];
        std::size_t np_total = static_cast<std::size_t>(fd.m_species_data[i].numParticles());

        SetupAttributes(species_name, particle_species, np_total, beams, plasmas, geom,
            fd.m_base_diag_type == DiagnosticData::diag_type::plasma_slice);

        std::set<std::string> addedRecords;

        auto dataset_idcpu = openPMD::Dataset(openPMD::determineDatatype<uint64_t>(), {np_total});
        if (fd.m_idcpu_name[i] != "") {
            uint64_t * const uint64_data = fd.m_species_data[i].GetIdCPUData().data();

            for (uint64_t j=0; j<np_total; ++j) {
                uint64_t id = uint64_data[j];
                // in the amrex format valid idcpus start with 1 and invalid with 0
                amrex::ParticleIDWrapper{id}.make_invalid();
                uint64_data[j] = id;
            }

            // handle scalar and non-scalar records by name
            auto [record_name, component_name] = utils::name2openPMD(fd.m_idcpu_name[i]);
            auto& currRecord = particle_species[record_name];
            SetupRecord(currRecord, record_name, addedRecords);
            auto& currRecordComp = currRecord[component_name];
            // not read until the data is flushed
            currRecordComp.resetDataset(dataset_idcpu);
            if (np_total != 0) {
                currRecordComp.storeChunkRaw(uint64_data, {0ull}, {np_total});
            }
        }

        auto dataset_real = openPMD::Dataset(openPMD::determineDatatype<amrex::Real>(), {np_total});
        for (std::size_t idx=0; idx<fd.m_real_names[i].size(); idx++) {
            // handle scalar and non-scalar records by name
            auto [record_name, component_name] = utils::name2openPMD(fd.m_real_names[i][idx]);
            auto& currRecord = particle_species[record_name];
            SetupRecord(currRecord, record_name, addedRecords);
            auto& currRecordComp = currRecord[component_name];
            // not read until the data is flushed
            currRecordComp.resetDataset(dataset_real);
            if (np_total != 0) {
                currRecordComp.storeChunkRaw(
                    fd.m_species_data[i].GetRealData(idx).data(), {0ull}, {np_total});
            }
        }

        auto dataset_int = openPMD::Dataset(openPMD::determineDatatype<int>(), {np_total});
        for (std::size_t idx=0; idx<fd.m_int_names[i].size(); idx++) {
            // handle scalar and non-scalar records by name
            auto [record_name, component_name] = utils::name2openPMD(fd.m_int_names[i][idx]);
            auto& currRecord = particle_species[record_name];
            SetupRecord(currRecord, record_name, addedRecords);
            auto& currRecordComp = currRecord[component_name];
            // not read until the data is flushed
            currRecordComp.resetDataset(dataset_int);
            if (np_total != 0) {
                currRecordComp.storeChunkRaw(
                    fd.m_species_data[i].GetIntData(idx).data(), {0ull}, {np_total});
            }
        }
    }
}

void
OpenPMDWriter::SetupAttributes (
    const std::string& species_name, openPMD::ParticleSpecies& particle_species,
    std::size_t np_total, const MultiBeam& beams, const MultiPlasma& plasmas,
    const amrex::Geometry& geom, bool only_xy)
{
    amrex::Real charge = 0;
    amrex::Real mass = 0;
    if (plasmas.HasPlasma(species_name)) {
        auto& plasma = plasmas.GetPlasma(species_name);
        charge = plasma.m_charge;
        mass = plasma.m_mass;
    } else {
        auto& beam = beams.getBeam(species_name);
        charge = beam.m_charge;
        mass = beam.m_mass;
    }

    const PhysConst phys_const_SI = make_constants_SI();
    auto const realType = openPMD::Dataset(openPMD::determineDatatype<amrex::Real>(), {np_total});

    std::vector<std::string> positionComponents;
    if (only_xy) {
        positionComponents = {"x", "y"};
    } else {
        positionComponents = {"x", "y", "z"};
    }
    for( auto const& comp : positionComponents ) {
        particle_species["positionOffset"][comp].resetDataset( realType );
        particle_species["positionOffset"][comp].makeConstant( 0. );
    }

    auto const scalar = openPMD::RecordComponent::SCALAR;
    particle_species["charge"][scalar].resetDataset( realType );
    particle_species["charge"][scalar].makeConstant( charge );
    particle_species["mass"][scalar].resetDataset( realType );
    particle_species["mass"][scalar].makeConstant( mass );

    // meta data
    particle_species["positionOffset"].setUnitDimension( utils::getUnitDimension("positionOffset") );
    particle_species["charge"].setUnitDimension( utils::getUnitDimension("charge") );
    particle_species["mass"].setUnitDimension( utils::getUnitDimension("mass") );

    // calculate the multiplier to convert from Hipace to SI units
    double hipace_to_SI_pos = 1.;
    double hipace_to_SI_weight = 1.;
    double hipace_to_SI_momentum = mass * phys_const_SI.c;
    double hipace_to_unitSI_momentum = mass * phys_const_SI.c;
    double hipace_to_SI_charge = 1.;
    double hipace_to_SI_mass = 1.;

    if(Hipace::m_normalized_units) {
        const auto dx = geom.CellSizeArray();
        const double n_0 = 1.;
        particle_species.setAttribute("HiPACE++_Plasma_Density", n_0);
        const double omega_p = (double)phys_const_SI.q_e * sqrt( (double)n_0 /
                                      ( (double)phys_const_SI.ep0 * (double)phys_const_SI.m_e ) );
        const double kp_inv = (double)phys_const_SI.c / omega_p;
        hipace_to_SI_pos = kp_inv;
        hipace_to_SI_weight = n_0 * dx[0] * dx[1] * dx[2] * kp_inv * kp_inv * kp_inv;
        hipace_to_SI_momentum = mass * phys_const_SI.m_e * phys_const_SI.c;
        hipace_to_SI_charge = phys_const_SI.q_e;
        hipace_to_SI_mass = phys_const_SI.m_e;
    }

    // write SI conversion
    particle_species.setAttribute("HiPACE++_use_reference_unitSI", true);
    const std::string attr = "HiPACE++_reference_unitSI";
    for( auto const& comp : positionComponents ) {
        particle_species["position"][comp].setAttribute( attr, hipace_to_SI_pos );
        //posOffset allways 0
        particle_species["positionOffset"][comp].setAttribute( attr, hipace_to_SI_pos );
        particle_species["momentum"][comp].setAttribute( attr, hipace_to_SI_momentum );
        particle_species["momentum"][comp].setUnitSI( hipace_to_unitSI_momentum );
    }
    particle_species["weighting"][scalar].setAttribute( attr, hipace_to_SI_weight );
    particle_species["charge"][scalar].setAttribute( attr, hipace_to_SI_charge );
    particle_species["mass"][scalar].setAttribute( attr, hipace_to_SI_mass );
}

void
OpenPMDWriter::SetupRecord (
    openPMD::Record particle_record, const std::string& currRecord,
    std::set<std::string>& addedRecords)
{
    if (addedRecords.count(currRecord) == 0) {

        particle_record.setUnitDimension( utils::getUnitDimension(currRecord) );

        if( currRecord == "weighting") {
            particle_record.setAttribute( "macroWeighted", 1u );
        } else {
            particle_record.setAttribute( "macroWeighted", 0u );
        }

        if( currRecord == "weighting" || currRecord == "momentum" || currRecord == "spin") {
            particle_record.setAttribute( "weightingPower", 1.0 );
        } else {
            particle_record.setAttribute( "weightingPower", 0.0 );
        }

        addedRecords.insert(currRecord);
    }
}

#endif // HIPACE_USE_OPENPMD
