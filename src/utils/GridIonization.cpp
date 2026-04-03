/* Copyright 2026
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn
 * License: BSD-3-Clause-LBNL
 */
#include "GridIonization.H"
#include "Hipace.H"
#include "HipaceProfilerWrapper.H"
#include "GPUUtil.H"
#include "Constants.H"
#include "particles/particles_utils/FieldGather.H"

void
GridIonization::ReadParameters ()
{
    amrex::ParmParse ppg("grid_ionization");

    queryWithParser(ppg, "plasma_names", m_names);

    m_use_grid_ionization = m_names.size() > 0 && m_names[0] != "no_gridplasma";
}

std::vector<std::string>
GridIonization::GetFieldComponents (const MultiPlasma& multi_plasma)
{
    std::vector<std::string> ret{};

    if (!m_use_grid_ionization) {
        return ret;
    }

    ret.push_back("grid_ionization_w_elec");
    ret.push_back("grid_ionization_ux^2_elec");
    ret.push_back("grid_ionization_uy^2_elec");
    ret.push_back("grid_ionization_uz_elec");
    ret.push_back("grid_ionization_uz^2_elec");

    for (auto& plasma_name : m_names) {
        const auto& plasma = multi_plasma.GetPlasma(plasma_name);
        for (int ionlev=0; ionlev <= plasma.m_max_ion_lev; ++ionlev) {
            ret.push_back("grid_ionization_w_" + plasma_name + "_" + std::to_string(ionlev));
        }
    }

    return ret;
}

void
GridIonization::InitData (Fields& fields, const MultiPlasma& multi_plasma,
                          const amrex::Geometry& geom, int const lev)
{
    if (!m_use_grid_ionization) {
        return;
    }

    HIPACE_PROFILE("GridIonization::InitData()");

    for (auto& plasma_name : m_names) {
        const auto& plasma = multi_plasma.GetPlasma(plasma_name);

        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            plasma.m_can_laser_ionize,
            "Plasma '" + plasma_name +
            "' used by grid ionization must have laser ionization enabled (can_laser_ionize = 1)"
        );

        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            plasma.m_ppc[0] == 0 && plasma.m_ppc[1] == 0 &&
            plasma.m_ppc_fine[0] == 0 && plasma.m_ppc_fine[1] == 0 &&
            plasma.m_ppc_fine2[0] == 0 && plasma.m_ppc_fine2[1] == 0,
            "Plasma '" + plasma_name +
            "' used by grid ionization must have 0 0 particles per cell (ppc = 0 0)"
        );

        const int weight_comp = Comps[WhichSlice::This][
            "grid_ionization_w_" + plasma_name + "_" + std::to_string(plasma.m_init_ion_lev)
        ];

        amrex::MultiFab& S = fields.getSlices(lev);

        for (amrex::MFIter mfi(S, DfltMfi); mfi.isValid(); ++mfi ){
            const amrex::Box& bx = mfi.tilebox();
            const Array2<amrex::Real> weight_arr = S.array(mfi, weight_comp);

            const amrex::Real dx = geom.CellSize(0);
            const amrex::Real dy = geom.CellSize(1);

            const amrex::Real poff_x = GetPosOffset(0, geom, geom.Domain());
            const amrex::Real poff_y = GetPosOffset(1, geom, geom.Domain());

            auto density_func = plasma.m_density_func;
            const amrex::Real c_t = get_phys_const().c * Hipace::m_physical_time;

            const amrex::Real radius_sq =
                plasma.m_radius == std::numeric_limits<amrex::Real>::max() ?
                std::numeric_limits<amrex::Real>::max() : plasma.m_radius * plasma.m_radius;
            const amrex::Real hollow_core_radius = plasma.m_hollow_core_radius;
            const amrex::Real min_density = plasma.m_min_density;

            amrex::ParallelFor(to2D(bx),
                [=] AMREX_GPU_DEVICE (int i, int j) {
                    const amrex::Real x = i * dx + poff_x;
                    const amrex::Real y = j * dy + poff_y;

                    const amrex::Real rsq = x * x + y * y;

                    amrex::Real density = density_func(x, y, c_t);

                    if (rsq > radius_sq ||
                        rsq < hollow_core_radius * hollow_core_radius ||
                        density <= min_density)
                    {
                        density = 0;
                    }

                    weight_arr(i, j) = density;
                }
            );
        }
    }
}

void
GridIonization::IonizeGrid (Fields& fields, const MultiPlasma& multi_plasma,
                            const MultiLaser& laser,
                            const amrex::Geometry& geom, int const lev,
                            const amrex::Geometry& laser_geom)
{
    if (!m_use_grid_ionization) {
        return;
    }

    HIPACE_PROFILE("GridIonization::IonizeGrid()");

    using namespace amrex::literals;
    using Complex = amrex::GpuComplex<amrex::Real>;
    constexpr Complex I(0.,1.);
    const PhysConst phys_const = get_phys_const();

    amrex::MultiFab& S = fields.getSlices(lev);

    const amrex::GpuArray<int, 6> comps {
        Comps[WhichSlice::This]["chi"],
        Comps[WhichSlice::This]["grid_ionization_w_elec"],
        Comps[WhichSlice::This]["grid_ionization_ux^2_elec"],
        Comps[WhichSlice::This]["grid_ionization_uy^2_elec"],
        Comps[WhichSlice::This]["grid_ionization_uz_elec"],
        Comps[WhichSlice::This]["grid_ionization_uz^2_elec"]
    };

    for (auto& plasma_name : m_names) {
        const auto& plasma = multi_plasma.GetPlasma(plasma_name);

        const int ion_weight_comp =
            Comps[WhichSlice::This]["grid_ionization_w_" + plasma_name + "_0"];

        for (amrex::MFIter mfi(S, DfltMfi); mfi.isValid(); ++mfi ){
            const amrex::Box& bx = mfi.tilebox();
            const Array3<amrex::Real> arr = S.array(mfi);

            const amrex::Real dx = geom.CellSize(0);
            const amrex::Real dy = geom.CellSize(1);

            const amrex::Real poff_x = GetPosOffset(0, geom, geom.Domain());
            const amrex::Real poff_y = GetPosOffset(1, geom, geom.Domain());

            // Calcuation of E0 in SI units for denormalization
            const amrex::Real wp = std::sqrt(static_cast<double>(Hipace::m_background_density_SI) *
                                            PhysConstSI::q_e*PhysConstSI::q_e /
                                            (PhysConstSI::ep0 * PhysConstSI::m_e) );
            const amrex::Real E0 = Hipace::m_normalized_units ?
                                wp * PhysConstSI::m_e * PhysConstSI::c / PhysConstSI::q_e : 1;
            const amrex::Real lambda0 = laser.GetLambda0();
            const amrex::Real omega0 = 2.0 * MathConst::pi * phys_const.c / lambda0;
            const bool linear_polarization = laser.LinearPolarization();

            const amrex::Real* adk_prefactor = plasma.m_adk_prefactor.data();
            const amrex::Real* adk_exp_prefactor = plasma.m_adk_exp_prefactor.data();
            const amrex::Real* adk_power = plasma.m_adk_power.data();
            const amrex::Real* laser_adk_prefactor = plasma.m_laser_adk_prefactor.data();
            const amrex::Real* laser_dp_prefactor = plasma.m_laser_dp_prefactor.data();
            const int max_ion_lev = plasma.m_max_ion_lev;

            // Extract laser array
            Array3<const amrex::Real> const laser_arr = laser.getSlices().const_array(mfi);

            const CheckDomainBounds laser_bounds {laser_geom};

            // Extract properties associated with physical size of the box
            const amrex::Real dx_inv = laser_geom.InvCellSize(0);
            const amrex::Real dy_inv = laser_geom.InvCellSize(1);
            const amrex::Real dzeta_inv = laser_geom.InvCellSize(2);

            // Offset for converting positions to indexes
            amrex::Real const x_pos_offset = GetPosOffset(0, laser_geom, laser_geom.Domain());
            amrex::Real const y_pos_offset = GetPosOffset(1, laser_geom, laser_geom.Domain());

            const amrex::Real chi_factor_elec =
                phys_const.q_e * (phys_const.q_e * phys_const.mu0 / phys_const.m_e);

            const amrex::Real chi_factor_ion =
                plasma.m_charge * (plasma.m_charge * phys_const.mu0 / plasma.m_mass);

            amrex::ParallelFor(to2D(bx),
                [=] AMREX_GPU_DEVICE (int i, int j) {
                    const amrex::Real x = i * dx + poff_x;
                    const amrex::Real y = j * dy + poff_y;

                    Complex A = 0;
                    Complex A_dx = 0;
                    Complex A_dzeta = 0;

                    if (laser_bounds.contains(x, y)) {
                        doLaserGatherShapeN<1>(x, y, A, A_dx, A_dzeta, laser_arr,
                            dx_inv, dy_inv, dzeta_inv, x_pos_offset, y_pos_offset);
                    }

                    // Convert from vector potential to electric field. Units are fixed later.
                    const Complex Et = I * A * omega0 + A_dzeta * phys_const.c; // transverse component
                    const Complex El = - A_dx * phys_const.c; // longitudinal component

                    // Get amplitude of the electric field envelope and normalize to correct SI unit.
                    amrex::Real Ep = std::sqrt( amrex::abs(Et*Et) + amrex::abs(El*El) );
                    Ep *= phys_const.m_e * phys_const.c / phys_const.q_e * E0;

                    amrex::Real chi = 0;

                    for (int ion_lev = 0; ion_lev < max_ion_lev; ++ion_lev) {

                        // ion has no momentum
                        amrex::Real p = 0;
                        if (Ep > 1e-30_rt) {
                            const amrex::Real w_dtau_dc = adk_prefactor[ion_lev] *
                                std::pow(Ep, adk_power[ion_lev]) *
                                std::exp( adk_exp_prefactor[ion_lev] / Ep );

                            const amrex::Real w_dtau_ac = w_dtau_dc *
                                (linear_polarization ?
                                    std::sqrt(Ep * laser_adk_prefactor[ion_lev]) : 1._rt);

                            p = 1._rt - std::exp( - w_dtau_ac );
                        }

                        const amrex::Real old_weight = arr(i, j, ion_weight_comp + ion_lev);
                        const amrex::Real transferred_weight = old_weight * p;
                        const amrex::Real new_weight = old_weight - transferred_weight;
                        chi += new_weight * chi_factor_ion * ion_lev * ion_lev;

                        arr(i, j, ion_weight_comp + ion_lev) = new_weight;
                        arr(i, j, ion_weight_comp + ion_lev + 1) += transferred_weight;
                        // w
                        arr(i, j, comps[1]) += transferred_weight;

                        if (linear_polarization) {
                            const amrex::Real delta = std::sqrt(Ep) * laser_dp_prefactor[ion_lev];
                            const amrex::Real delta2 = delta * delta;
                            const amrex::Real delta4 = delta2 * delta2;
                            const amrex::Real alpha = -adk_power[ion_lev];
                            const amrex::Real s1 = - (7._rt/4._rt) + alpha / 2._rt;
                            const amrex::Real s2 = (1._rt/16._rt) * ( 8._rt * (alpha*alpha)
                                - 68._rt*alpha + 131._rt );
                            const amrex::Real width_p = amrex::abs(A) * delta *
                                (1._rt + s1*delta2 + s2*delta4);
                            const amrex::Real a_fact = amrex::abs(A * A) * 0.25_rt;
                            // ux^2
                            arr(i, j, comps[2]) += transferred_weight * width_p * width_p;
                            // uz
                            arr(i, j, comps[4]) += transferred_weight *
                                (a_fact + width_p * width_p * 0.5_rt);
                            // uz^2
                            arr(i, j, comps[5]) += transferred_weight * (
                                a_fact * a_fact +
                                a_fact * width_p * width_p +
                                0.75_rt * width_p * width_p * width_p * width_p
                            );
                        } else {
                            // A_t = A (e_x +/- i e_y) in circular polarization.
                            // ux and uy differ from Massimo PRE 2020 by a factor of sqrt(2) due
                            // to different convention for linear vs. circular polarization.
                            // uz differs from Massimo PRE 2020 by a factor of 2 due to different
                            // convention for linear vs. circular polarization.
                            amrex::Real a_fact = amrex::abs(A*A);
                            // ux^2
                            arr(i, j, comps[2]) += transferred_weight * 0.5_rt * a_fact;
                            // uy^2
                            arr(i, j, comps[3]) += transferred_weight * 0.5_rt * a_fact;
                            // uz
                            arr(i, j, comps[4]) += transferred_weight * a_fact;
                            // uz^2
                            arr(i, j, comps[5]) += transferred_weight * a_fact * a_fact;
                        }
                    }

                    // chi
                    // chi of new electrons does not depend on the laser field strength
                    // or the the new random momentum
                    chi += arr(i, j, comps[1]) * chi_factor_elec;

                    // last ion level
                    chi += arr(i, j, ion_weight_comp + max_ion_lev) *
                        chi_factor_ion * max_ion_lev * max_ion_lev;

                    arr(i, j, comps[0]) += chi;
                }
            );
        }
    }
}
