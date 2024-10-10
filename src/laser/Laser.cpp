/* Copyright 2022
 *
 * This file is part of HiPACE++.
 *
 * Authors: MaxThevenet, AlexanderSinn
 * Severin Diederichs, atmyers, Angel Ferran Pousa
 * License: BSD-3-Clause-LBNL
 */

#include "Laser.H"
#include "utils/Parser.H"
#include "Hipace.H"

#include <AMReX_Vector.H>
#include <AMReX_ParmParse.H>
#include "particles/particles_utils/ShapeFactors.H"

Laser::Laser (std::string name,  amrex::Geometry laser_geom_3D)
{
    m_name = name;
    amrex::ParmParse pp(m_name);
    queryWithParser(pp, "init_type", m_laser_init_type);
    if (m_laser_init_type == "from_file") {
        queryWithParser(pp, "input_file", m_input_file_path);
        queryWithParser(pp, "openPMD_laser_name", m_file_envelope_name);
        queryWithParser(pp, "iteration", m_file_num_iteration);
        if (Hipace::HeadRank()) {
            m_F_input_file.resize(laser_geom_3D.Domain(), 2, amrex::The_Pinned_Arena());
            GetEnvelopeFromFileHelper(laser_geom_3D);
        }
        return;
    }
    else if (m_laser_init_type == "gaussian") {
        queryWithParser(pp, "a0", m_a0);
        queryWithParser(pp, "w0", m_w0);
        queryWithParser(pp, "CEP", m_CEP);
        queryWithParser(pp, "propagation_angle_yz", m_propagation_angle_yz);
        queryWithParser(pp, "PFT_yz", m_PFT_yz);
        bool length_is_specified = queryWithParser(pp, "L0", m_L0);
        bool duration_is_specified = queryWithParser(pp, "tau", m_tau);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE( length_is_specified + duration_is_specified == 1,
        "Please specify exlusively either the pulse length L0 or the duration tau of gaussian lasers");
        if (duration_is_specified) m_L0 = m_tau*get_phys_const().c;
        queryWithParser(pp, "focal_distance", m_focal_distance);
        queryWithParser(pp, "position_mean",  m_position_mean);
        return;
    }
    else if (m_laser_init_type == "parser") {
        std::string profile_real_str = "";
        std::string profile_imag_str = "";
        getWithParser(pp, "laser_real(x,y,z)", profile_real_str);
        getWithParser(pp, "laser_imag(x,y,z)", profile_imag_str);
        m_profile_real = makeFunctionWithParser<3>( profile_real_str, m_parser_lr, {"x", "y", "z"});
        m_profile_imag = makeFunctionWithParser<3>( profile_imag_str, m_parser_li, {"x", "y", "z"});
        return;
    }
    else {
        amrex::Abort("Illegal init type specified for laser. Must be one of: gaussian, from_file, parser.");
    }
}

void
Laser::GetEnvelopeFromFileHelper (amrex::Geometry laser_geom_3D) {

    HIPACE_PROFILE("MultiLaser::GetEnvelopeFromFileHelper()");
#ifdef HIPACE_USE_OPENPMD
    openPMD::Datatype input_type = openPMD::Datatype::INT;
    {
        // Check what kind of Datatype is used in the Laser file
        auto series = openPMD::Series( m_input_file_path , openPMD::Access::READ_ONLY );

        if(!series.iterations.contains(m_file_num_iteration)) {
            amrex::Abort("Could not find iteration " + std::to_string(m_file_num_iteration) +
                         " in file " + m_input_file_path + "\n");
        }

        auto iteration = series.iterations[m_file_num_iteration];

        if(!iteration.meshes.contains(m_file_envelope_name)) {
            amrex::Abort("Could not find mesh '" + m_file_envelope_name + "' in file "
                + m_input_file_path + "\n");
        }

        auto mesh = iteration.meshes[m_file_envelope_name];

        if (!mesh.containsAttribute("angularFrequency")) {
            amrex::Abort("Could not find Attribute 'angularFrequency' of iteration "
                + std::to_string(m_file_num_iteration) + " in file "
                + m_input_file_path + "\n");
        }

        m_lambda0_from_file = 2.*MathConst::pi*PhysConstSI::c
            / mesh.getAttribute("angularFrequency").get<double>();

        if(!mesh.contains(openPMD::RecordComponent::SCALAR)) {
            amrex::Abort("Could not find component '" +
                std::string(openPMD::RecordComponent::SCALAR) +
                "' in file " + m_input_file_path + "\n");
        }

        input_type = mesh[openPMD::RecordComponent::SCALAR].getDatatype();
    }

    if (input_type == openPMD::Datatype::CFLOAT) {
        GetEnvelopeFromFile<std::complex<float>>(laser_geom_3D);
    } else if (input_type == openPMD::Datatype::CDOUBLE) {
        GetEnvelopeFromFile<std::complex<double>>(laser_geom_3D);
    } else {
        amrex::Abort("Unknown Datatype used in Laser input file. Must use CDOUBLE or CFLOAT\n");
    }
#else
    amrex::Abort("loading a laser envelope from an external file requires openPMD support: "
                 "Add HiPACE_OPENPMD=ON when compiling HiPACE++.\n");
#endif // HIPACE_USE_OPENPMD
}

template<typename input_type>
void
Laser::GetEnvelopeFromFile (amrex::Geometry laser_geom_3D) {

    using namespace amrex::literals;

    HIPACE_PROFILE("MultiLaser::GetEnvelopeFromFile()");
#ifdef HIPACE_USE_OPENPMD
    const PhysConst phc = get_phys_const();
    const amrex::Real clight = phc.c;

    const amrex::Box& domain = laser_geom_3D.Domain();

    auto series = openPMD::Series( m_input_file_path , openPMD::Access::READ_ONLY );
    auto laser = series.iterations[m_file_num_iteration].meshes[m_file_envelope_name];
    auto laser_comp = laser[openPMD::RecordComponent::SCALAR];

    const std::vector<std::string> axis_labels = laser.axisLabels();
    if (axis_labels[0] == "t" && axis_labels[1] == "y" && axis_labels[2] == "x") {
        m_file_geometry = "xyt";
    } else if (axis_labels[0] == "z" && axis_labels[1] == "y" && axis_labels[2] == "x") {
        m_file_geometry = "xyz";
    } else if (axis_labels[0] == "t" && axis_labels[1] == "r") {
        m_file_geometry = "rt";
    } else {
        amrex::Abort("Incorrect axis labels in laser file, must be either tyx, zyx or tr");
    }

    const std::shared_ptr<input_type> data = laser_comp.loadChunk<input_type>();
    auto extent = laser_comp.getExtent();
    double unitSI = laser_comp.unitSI();

    // Extract grid offset and grid spacing from laser file
    std::vector<double> offset = laser.gridGlobalOffset();
    std::vector<double> position = laser_comp.position<double>();
    std::vector<double> spacing = laser.gridSpacing<double>();

    //lasy: tyx in C order, tr in C order
    amrex::Dim3 arr_begin = {0, 0, 0};
    amrex::Dim3 arr_end = {static_cast<int>(extent[2]), static_cast<int>(extent[1]),
                            static_cast<int>(extent[0])};
    amrex::Array4<input_type> input_file_arr(data.get(), arr_begin, arr_end, 1);

    //hipace: xyt in Fortran order
    amrex::Array4<amrex::Real> laser_arr = m_F_input_file.array();

    series.flush();

    constexpr int interp_order_xy = 1;
    const amrex::Real dx = laser_geom_3D.CellSize(Direction::x);
    const amrex::Real dy = laser_geom_3D.CellSize(Direction::y);
    const amrex::Real dz = laser_geom_3D.CellSize(Direction::z);
    const amrex::Real xmin = laser_geom_3D.ProbLo(Direction::x)+dx/2;
    const amrex::Real ymin = laser_geom_3D.ProbLo(Direction::y)+dy/2;
    const amrex::Real zmin = laser_geom_3D.ProbLo(Direction::z)+dz/2;
    const amrex::Real zmax = laser_geom_3D.ProbHi(Direction::z)-dz/2;
    const int imin = domain.smallEnd(0);
    const int jmin = domain.smallEnd(1);
    const int kmin = domain.smallEnd(2);

    if (m_file_geometry == "xyt") {
        // Calculate the min and max of the grid from laser file
        amrex::Real ymin_laser = offset[1] + position[1]*spacing[1];
        amrex::Real xmin_laser = offset[2] + position[2]*spacing[2];
        AMREX_ALWAYS_ASSERT(position[0] == 0 && position[1] == 0 && position[2] == 0);


        for (int k = kmin; k <= domain.bigEnd(2); ++k) {
            for (int j = jmin; j <= domain.bigEnd(1); ++j) {
                for (int i = imin; i <= domain.bigEnd(0); ++i) {

                    const amrex::Real x = (i-imin)*dx + xmin;
                    const amrex::Real xmid = (x - xmin_laser)/spacing[2];
                    amrex::Real sx_cell[interp_order_xy+1];
                    const int i_cell = compute_shape_factor<interp_order_xy>(sx_cell, xmid);

                    const amrex::Real y = (j-jmin)*dy + ymin;
                    const amrex::Real ymid = (y - ymin_laser)/spacing[1];
                    amrex::Real sy_cell[interp_order_xy+1];
                    const int j_cell = compute_shape_factor<interp_order_xy>(sy_cell, ymid);

                    const amrex::Real z = (k-kmin)*dz + zmin;
                    const amrex::Real tmid = (zmax-z)/clight/spacing[0];
                    amrex::Real st_cell[interp_order_xy+1];
                    const int k_cell = compute_shape_factor<interp_order_xy>(st_cell, tmid);

                    laser_arr(i, j, k, 0) = 0._rt;
                    laser_arr(i, j, k, 1) = 0._rt;
                    for (int it=0; it<=interp_order_xy; it++){
                        for (int iy=0; iy<=interp_order_xy; iy++){
                            for (int ix=0; ix<=interp_order_xy; ix++){
                                if (i_cell+ix >= 0 && i_cell+ix < static_cast<int>(extent[2]) &&
                                    j_cell+iy >= 0 && j_cell+iy < static_cast<int>(extent[1]) &&
                                    k_cell+it >= 0 && k_cell+it < static_cast<int>(extent[0])) {
                                    laser_arr(i, j, k, 0) += sx_cell[ix] * sy_cell[iy] * st_cell[it] *
                                        static_cast<amrex::Real>(
                                            input_file_arr(i_cell+ix, j_cell+iy, k_cell+it).real() * unitSI
                                        );
                                    laser_arr(i, j, k, 1) += sx_cell[ix] * sy_cell[iy] * st_cell[it] *
                                        static_cast<amrex::Real>(
                                            input_file_arr(i_cell+ix, j_cell+iy, k_cell+it).imag() * unitSI
                                        );
                                }
                            }
                        }
                    } // End of 3 loops (1 per dimension) over laser array from file
                }
            }
        } // End of 3 loops (1 per dimension) over laser array from simulation
    } else if (m_file_geometry == "xyz") {
        // Calculate the min and max of the grid from laser file
        amrex::Real zmin_laser = offset[0] + position[0]*spacing[0];
        amrex::Real ymin_laser = offset[1] + position[1]*spacing[1];
        amrex::Real xmin_laser = offset[2] + position[2]*spacing[2];

        for (int k = kmin; k <= domain.bigEnd(2); ++k) {
            for (int j = jmin; j <= domain.bigEnd(1); ++j) {
                for (int i = imin; i <= domain.bigEnd(0); ++i) {

                    const amrex::Real x = (i-imin)*dx + xmin;
                    const amrex::Real xmid = (x - xmin_laser)/spacing[2];
                    amrex::Real sx_cell[interp_order_xy+1];
                    const int i_cell = compute_shape_factor<interp_order_xy>(sx_cell, xmid);

                    const amrex::Real y = (j-jmin)*dy + ymin;
                    const amrex::Real ymid = (y - ymin_laser)/spacing[1];
                    amrex::Real sy_cell[interp_order_xy+1];
                    const int j_cell = compute_shape_factor<interp_order_xy>(sy_cell, ymid);

                    const amrex::Real z = (k-kmin)*dz + zmin;
                    const amrex::Real zmid = (z - zmin_laser)/spacing[0];
                    amrex::Real sz_cell[interp_order_xy+1];
                    const int k_cell = compute_shape_factor<interp_order_xy>(sz_cell, zmid);

                    laser_arr(i, j, k, 0) = 0._rt;
                    laser_arr(i, j, k, 1) = 0._rt;
                    for (int iz=0; iz<=interp_order_xy; iz++){
                        for (int iy=0; iy<=interp_order_xy; iy++){
                            for (int ix=0; ix<=interp_order_xy; ix++){
                                if (i_cell+ix >= 0 && i_cell+ix < static_cast<int>(extent[2]) &&
                                    j_cell+iy >= 0 && j_cell+iy < static_cast<int>(extent[1]) &&
                                    k_cell+iz >= 0 && k_cell+iz < static_cast<int>(extent[0])) {
                                    laser_arr(i, j, k, 0) += sx_cell[ix] * sy_cell[iy] * sz_cell[iz] *
                                        static_cast<amrex::Real>(
                                            input_file_arr(i_cell+ix, j_cell+iy, k_cell+iz).real() * unitSI
                                        );
                                    laser_arr(i, j, k, 1) += sx_cell[ix] * sy_cell[iy] * sz_cell[iz] *
                                        static_cast<amrex::Real>(
                                            input_file_arr(i_cell+ix, j_cell+iy, k_cell+iz).imag() * unitSI
                                        );
                                }
                            }
                        }
                    } // End of 3 loops (1 per dimension) over laser array from file
                }
            }
        } // End of 3 loops (1 per dimension) over laser array from simulation
    } else if (m_file_geometry == "rt") {

        // extent = {nmodes, nt, nr}

        // Calculate the min and max of the grid from laser file
        amrex::Real rmin_laser = offset[1] + position[1]*spacing[1];
        AMREX_ALWAYS_ASSERT(position[0] == 0 && position[1] == 0);

        for (int k = kmin; k <= domain.bigEnd(2); ++k) {
            for (int j = jmin; j <= domain.bigEnd(1); ++j) {
                for (int i = imin; i <= domain.bigEnd(0); ++i) {

                    const amrex::Real x = (i-imin)*dx + xmin;
                    const amrex::Real y = (j-jmin)*dy + ymin;
                    const amrex::Real r = std::sqrt(x*x + y*y);
                    const amrex::Real theta = std::atan2(y, x);
                    const amrex::Real rmid = (r - rmin_laser)/spacing[1];
                    amrex::Real sr_cell[interp_order_xy+1];
                    const int i_cell = compute_shape_factor<interp_order_xy>(sr_cell, rmid);

                    const amrex::Real z = (k-kmin)*dz + zmin;
                    const amrex::Real tmid = (zmax-z)/clight/spacing[0];
                    amrex::Real st_cell[interp_order_xy+1];
                    const int k_cell = compute_shape_factor<interp_order_xy>(st_cell, tmid);

                    laser_arr(i, j, k, 0) = 0._rt;
                    laser_arr(i, j, k, 1) = 0._rt;
                    for (int it=0; it<=interp_order_xy; it++){
                        for (int ir=0; ir<=interp_order_xy; ir++){
                            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(i_cell+ir >= 0,
                                "Touching a r<0 cell in laser file reader. Is staggering correct?");
                            if (i_cell+ir < static_cast<int>(extent[2]) &&
                                k_cell+it >= 0 && k_cell+it < static_cast<int>(extent[1])) {
                                // mode 0
                                laser_arr(i, j, k, 0) += sr_cell[ir] * st_cell[it] *
                                    static_cast<amrex::Real>(
                                    input_file_arr(i_cell+ir, k_cell+it, 0).real() * unitSI);
                                laser_arr(i, j, k, 1) += sr_cell[ir] * st_cell[it] *
                                    static_cast<amrex::Real>(
                                    input_file_arr(i_cell+ir, k_cell+it, 0).imag() * unitSI);
                                for (int im=1; im<=static_cast<int>(extent[0])/2; im++) {
                                    // cos(m*theta) part of the mode
                                    laser_arr(i, j, k, 0) += sr_cell[ir] * st_cell[it] *
                                        std::cos(im*theta) * static_cast<amrex::Real>(
                                        input_file_arr(i_cell+ir, k_cell+it, 2*im-1).real() * unitSI);
                                    laser_arr(i, j, k, 1) += sr_cell[ir] * st_cell[it] *
                                        std::cos(im*theta) * static_cast<amrex::Real>(
                                        input_file_arr(i_cell+ir, k_cell+it, 2*im-1).imag() * unitSI);
                                    // sin(m*theta) part of the mode
                                    laser_arr(i, j, k, 0) += sr_cell[ir] * st_cell[it] *
                                        std::sin(im*theta) * static_cast<amrex::Real>(
                                        input_file_arr(i_cell+ir, k_cell+it, 2*im).real() * unitSI);
                                    laser_arr(i, j, k, 1) += sr_cell[ir] * st_cell[it] *
                                        std::sin(im*theta) * static_cast<amrex::Real>(
                                        input_file_arr(i_cell+ir, k_cell+it, 2*im).imag() * unitSI);
                                } // End of loop over modes of laser array from file
                            }
                        }
                    } // End of 2 loops (1 per RT dimension) over laser array from file
                }
            }
        } // End of 3 loops (1 per dimension) over laser array from simulation
    } // End if statement over file laser geometry (rt or xyt)
#else
    amrex::Abort("loading a laser envelope from an external file requires openPMD support: "
                 "Add HiPACE_OPENPMD=ON when compiling HiPACE++.\n");
#endif // HIPACE_USE_OPENPMD
}
