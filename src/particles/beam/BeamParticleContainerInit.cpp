/* Copyright 2020-2021
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn, Andrew Myers, Axel Huebl, MaxThevenet
 * Severin Diederichs, Weiqun Zhang
 * License: BSD-3-Clause-LBNL
 */
#include "BeamParticleContainer.H"
#include "utils/Constants.H"
#include "particles/particles_utils/ParticleUtil.H"
#include "particles/pusher/GetAndSetPosition.H"
#include "Hipace.H"
#include "utils/HipaceProfilerWrapper.H"
#include <AMReX_REAL.H>

#ifdef HIPACE_USE_OPENPMD
#include <openPMD/openPMD.hpp>
#include <iostream> // std::cout
#include <memory>   // std::shared_ptr
#endif  // HIPACE_USE_OPENPMD



namespace
{
#ifdef HIPACE_USE_OPENPMD
    /** \brief Adds a single beam particle
     *
     * \param[in,out] ptd real and int beam data
     * \param[in] x position in x
     * \param[in] y position in y
     * \param[in] z position in z
     * \param[in] ux gamma * beta_x
     * \param[in] uy gamma * beta_y
     * \param[in] uz gamma * beta_z
     * \param[in] weight weight of the single particle
     * \param[in] pid particle ID to be assigned to the particle
     * \param[in] ip index of the particle
     * \param[in] speed_of_light speed of light in the current units
     * \param[in] enforceBC functor to enforce the boundary condition
     */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void AddOneBeamParticle (
        const BeamTileInit::ParticleTileDataType& ptd, const amrex::Real& x,
        const amrex::Real& y, const amrex::Real& z, const amrex::Real& ux, const amrex::Real& uy,
        const amrex::Real& uz, const amrex::Real& weight, const amrex::Long pid,
        const amrex::Long ip, const amrex::Real& speed_of_light, const EnforceBC& enforceBC) noexcept
    {
        amrex::Real xp = x;
        amrex::Real yp = y;
        amrex::Real uxp = ux * speed_of_light;
        amrex::Real uyp = uy * speed_of_light;
        if (enforceBC(ptd, ip, xp, yp, uxp, uyp, BeamIdx::w)) return;

        ptd.rdata(BeamIdx::x  )[ip] = xp;
        ptd.rdata(BeamIdx::y  )[ip] = yp;
        ptd.rdata(BeamIdx::z  )[ip] = z;
        ptd.rdata(BeamIdx::ux )[ip] = uxp;
        ptd.rdata(BeamIdx::uy )[ip] = uyp;
        ptd.rdata(BeamIdx::uz )[ip] = uz * speed_of_light;
        ptd.rdata(BeamIdx::w  )[ip] = std::abs(weight);

        ptd.idcpu(ip) = pid + ip;
        ptd.id(ip).make_valid();
    }
#endif // HIPACE_USE_OPENPMD

    /** \brief Adds a single beam particle into the per-slice BeamTile
     *
     * \param[in,out] ptd real and int beam data
     * \param[in] x position in x
     * \param[in] y position in y
     * \param[in] z position in z
     * \param[in] ux gamma * beta_x
     * \param[in] uy gamma * beta_y
     * \param[in] uz gamma * beta_z
     * \param[in] weight weight of the single particle
     * \param[in] pid particle ID to be assigned to the particle at index 0
     * \param[in] ip index of the particle
     * \param[in] speed_of_light speed of light in the current units
     * \param[in] enforceBC functor to enforce the boundary condition
     * \param[in] is_valid if the particle is valid
     */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void AddOneBeamParticleSlice (
        const BeamTile::ParticleTileDataType& ptd, const amrex::Real x,
        const amrex::Real y, const amrex::Real z, const amrex::Real ux, const amrex::Real uy,
        const amrex::Real uz, const amrex::Real weight, const amrex::Long pid,
        const amrex::Long ip, const amrex::Real speed_of_light, const EnforceBC& enforceBC,
        const bool is_valid=true) noexcept
    {
        amrex::Real xp = x;
        amrex::Real yp = y;
        amrex::Real uxp = ux * speed_of_light;
        amrex::Real uyp = uy * speed_of_light;
        if (enforceBC(ptd, ip, xp, yp, uxp, uyp, BeamIdx::w)) return;

        ptd.rdata(BeamIdx::x  )[ip] = xp;
        ptd.rdata(BeamIdx::y  )[ip] = yp;
        ptd.rdata(BeamIdx::z  )[ip] = z;
        ptd.rdata(BeamIdx::ux )[ip] = uxp;
        ptd.rdata(BeamIdx::uy )[ip] = uyp;
        ptd.rdata(BeamIdx::uz )[ip] = uz * speed_of_light;
        ptd.rdata(BeamIdx::w  )[ip] = std::abs(weight);

        ptd.idata(BeamIdx::nsubcycles)[ip] = 0;
        ptd.idata(BeamIdx::mr_level)[ip] = 0;

        ptd.idcpu(ip) = pid + ip;
        if (is_valid) {
            ptd.id(ip).make_valid(); // ensure id is valid
        } else {
            ptd.id(ip).make_invalid();
        }
    }
}

void
BeamParticleContainer::
InitBeamFixedPPC3D ()
{
    HIPACE_PROFILE("BeamParticleContainer::InitBeamFixedPPC3D()");

    if (!Hipace::HeadRank()) { return; }

    const amrex::Geometry geom_3D = Hipace::GetInstance().m_3D_geom[0];
    const amrex::Box domain_box = geom_3D.Domain();
    const auto dx = geom_3D.CellSizeArray();
    const auto plo = geom_3D.ProbLoArray();

    const amrex::IntVect ppc = m_ppc;
    const int num_ppc = ppc[0] * ppc[1] * ppc[2];

    const amrex::Real x_mean = m_position_mean[0];
    const amrex::Real y_mean = m_position_mean[1];
    const amrex::Real z_min = m_zmin;
    const amrex::Real z_max = m_zmax;
    const amrex::Real radius = m_radius;
    const amrex::Real min_density = m_min_density;
    const amrex::GpuArray<int, 3> rand_ppc {m_random_ppc[0], m_random_ppc[1], m_random_ppc[2]};

    const GetInitialDensity get_density = m_get_density;

    amrex::ReduceOps<amrex::ReduceOpSum> reduce_op;
    amrex::ReduceData<uint64_t> reduce_data(reduce_op);
    using ReduceTuple = typename decltype(reduce_data)::Type;
    reduce_op.eval(
        domain_box.numPts(), reduce_data,
        [=] AMREX_GPU_DEVICE (amrex::Long idx) -> ReduceTuple
        {
            auto [i, j, k] = domain_box.atOffset3d(idx).arr;

            uint64_t count = 0;

            for (int i_part=0; i_part<num_ppc;i_part++)
            {
                amrex::Real r[3];

                ParticleUtil::get_position_unit_cell(r, ppc, i_part);

                amrex::Real x = plo[0] + (i + r[0])*dx[0];
                amrex::Real y = plo[1] + (j + r[1])*dx[1];
                amrex::Real z = plo[2] + (k + r[2])*dx[2];

                if (rand_ppc[0] + rand_ppc[1] + rand_ppc[2] == false ) {
                    // If particles are evenly spaced, discard particles
                    // individually if they are out of bounds
                    if (z >= z_max || z < z_min ||
                        ((x-x_mean)*(x-x_mean)+(y-y_mean)*(y-y_mean)) > radius*radius) {
                            continue;
                        }
                } else {
                    // If particles are randomly spaced, discard particles
                    // if the cell is outside the domain
                    amrex::Real xc = plo[0]+i*dx[0];
                    amrex::Real yc = plo[1]+j*dx[1];
                    amrex::Real zc = plo[2]+k*dx[2];
                    if (zc >= z_max || zc < z_min ||
                        ((xc-x_mean)*(xc-x_mean)+(yc-y_mean)*(yc-y_mean)) > radius*radius) {
                            continue;
                        }
                }

                const amrex::Real density = get_density(x, y, z);
                if (density <= min_density) continue;

                ++count;
            }

            return {count};
        });

    ReduceTuple a = reduce_data.value();
    m_total_num_particles = amrex::get<0>(a);
}

void
BeamParticleContainer::
InitBeamFixedPPCSlice (const int islice, const int which_beam_slice)
{
    HIPACE_PROFILE("BeamParticleContainer::InitBeamFixedPPCSlice()");

    if (!Hipace::HeadRank()) { return; }

    const amrex::Geometry geom_3D = Hipace::GetInstance().m_3D_geom[0];
    const amrex::Geometry slice_geom = Hipace::GetInstance().m_slice_geom[0];
    const amrex::Box slice_box = slice_geom.Domain();
    const auto dx = geom_3D.CellSizeArray();
    const auto plo = geom_3D.ProbLoArray();

    const amrex::IntVect ppc = m_ppc;
    const int num_ppc = ppc[0] * ppc[1] * ppc[2];

    const amrex::Real scale_fac = Hipace::m_normalized_units ?
        1./num_ppc : dx[0]*dx[1]*dx[2]/num_ppc;

    const amrex::Real x_mean = m_position_mean[0];
    const amrex::Real y_mean = m_position_mean[1];
    const amrex::Real z_min = m_zmin;
    const amrex::Real z_max = m_zmax;
    const amrex::Real radius = m_radius;
    const amrex::Real min_density = m_min_density;
    const amrex::GpuArray<int, 3> rand_ppc {m_random_ppc[0], m_random_ppc[1], m_random_ppc[2]};

    const GetInitialDensity get_density = m_get_density;
    const GetInitialMomentum get_momentum = m_get_momentum;

    // First: loop over all cells, and count the particles effectively injected.

    amrex::Gpu::DeviceVector<int> counts(slice_box.numPts(), 0);
    const Array2<int> count_arr {{counts.dataPtr(), amrex::begin(slice_box),
                                  amrex::end(slice_box), 1}};

    amrex::Gpu::DeviceVector<int> offsets(slice_box.numPts());
    const Array2<int> offset_arr {{offsets.dataPtr(), amrex::begin(slice_box),
                                   amrex::end(slice_box), 1}};

    amrex::ParallelForRNG(to2D(slice_box),
        [=] AMREX_GPU_DEVICE (int i, int j, const amrex::RandomEngine& engine) noexcept
        {
            int count = 0;

            for (int i_part=0; i_part<num_ppc;i_part++)
            {
                amrex::Real r[3];

                ParticleUtil::get_position_unit_cell(r, ppc, i_part, engine, rand_ppc);

                amrex::Real x = plo[0] + (i + r[0])*dx[0];
                amrex::Real y = plo[1] + (j + r[1])*dx[1];
                amrex::Real z = plo[2] + (islice + r[2])*dx[2];

                if (rand_ppc[0] + rand_ppc[1] + rand_ppc[2] == false ) {
                    // If particles are evenly spaced, discard particles
                    // individually if they are out of bounds
                    if (z >= z_max || z < z_min ||
                        ((x-x_mean)*(x-x_mean)+(y-y_mean)*(y-y_mean)) > radius*radius) {
                            continue;
                        }
                } else {
                    // If particles are randomly spaced, discard particles
                    // if the cell is outside the domain
                    amrex::Real xc = plo[0]+i*dx[0];
                    amrex::Real yc = plo[1]+j*dx[1];
                    amrex::Real zc = plo[2]+islice*dx[2];
                    if (zc >= z_max || zc < z_min ||
                        ((xc-x_mean)*(xc-x_mean)+(yc-y_mean)*(yc-y_mean)) > radius*radius) {
                            continue;
                        }
                }

                const amrex::Real density = get_density(x, y, z);
                if (density <= min_density) continue;

                ++count;
            }

            count_arr(i, j) = count;
        });

    int num_to_add = amrex::Scan::ExclusiveSum(counts.size(), counts.data(), offsets.data());

    // Second: allocate the memory for these particles
    resize(which_beam_slice, num_to_add, 0);
    if (num_to_add == 0) return;
    auto& particle_tile = getBeamSlice(which_beam_slice);

    const auto ptd = particle_tile.getParticleTileData();

    const uint64_t pid = m_id64;
    m_id64 += num_to_add;

    const amrex::Real speed_of_light = get_phys_const().c;

    const auto enforceBC = EnforceBC();

    amrex::ParallelForRNG(to2D(slice_box),
        [=] AMREX_GPU_DEVICE (int i, int j, const amrex::RandomEngine& engine) noexcept
        {
            int pidx = offset_arr(i, j);

            for (int i_part=0; i_part<num_ppc;i_part++)
            {
                amrex::Real r[3] = {0.,0.,0.};

                ParticleUtil::get_position_unit_cell(r, ppc, i_part, engine, rand_ppc);

                amrex::Real x = plo[0] + (i + r[0])*dx[0];
                amrex::Real y = plo[1] + (j + r[1])*dx[1];
                amrex::Real z = plo[2] + (islice + r[2])*dx[2];

                if (rand_ppc[0] + rand_ppc[1] + rand_ppc[2] == false) {
                    // If particles are evenly spaced, discard particles
                    // individually if they are out of bounds
                    if (z >= z_max || z < z_min ||
                        ((x-x_mean)*(x-x_mean)+(y-y_mean)*(y-y_mean)) > radius*radius) {
                            continue;
                        }
                } else {
                    // If particles are randomly spaced, discard particles
                    // if the cell is outside the domain
                    amrex::Real xc = plo[0]+i*dx[0];
                    amrex::Real yc = plo[1]+j*dx[1];
                    amrex::Real zc = plo[2]+islice*dx[2];
                    if (zc >= z_max || zc < z_min ||
                        ((xc-x_mean)*(xc-x_mean)+(yc-y_mean)*(yc-y_mean)) > radius*radius) {
                            continue;
                        }
                }

                const amrex::Real density = get_density(x, y, z);
                if (density <= min_density) continue;

                amrex::Real u[3] = {0.,0.,0.};
                get_momentum(u[0], u[1], u[2], engine);

                const amrex::Real weight = density * scale_fac;

                AddOneBeamParticleSlice(ptd, x, y, z, u[0], u[1], u[2], weight,
                                        pid, pidx, speed_of_light, enforceBC, true);

                ++pidx;
            }
        });
}

void
BeamParticleContainer::
InitBeamFixedWeight3D ()
{
    HIPACE_PROFILE("BeamParticleContainer::InitBeamFixedWeight3D()");
    using namespace amrex::literals;

    if (!Hipace::HeadRank() || m_num_particles == 0) { return; }

    amrex::Long num_to_add = m_num_particles;
    if (m_do_symmetrize) num_to_add /= 4;

    m_z_array.setArena(m_initialize_on_cpu ? amrex::The_Pinned_Arena() : amrex::The_Arena());
    m_z_array.resize(num_to_add);
    amrex::Real * const pos_z = m_z_array.dataPtr();

    const bool can = m_can_profile;
    const amrex::Real z_min = m_zmin;
    const amrex::Real z_max = m_zmax;
    const amrex::Real z_mean = m_pos_mean_z;
    const amrex::Real z_std = m_position_std[2];

    amrex::ParallelForRNG(
        num_to_add,
        [=] AMREX_GPU_DEVICE (amrex::Long i, const amrex::RandomEngine& engine) noexcept
        {
            pos_z[i] = can
                ? amrex::Random(engine) * (z_max - z_min) + z_min
                : amrex::RandomNormal(z_mean, z_std, engine);
        });

    return;
}

void
BeamParticleContainer::
InitBeamFixedWeightSlice (int slice, int which_slice)
{
    HIPACE_PROFILE("BeamParticleContainer::InitBeamFixedWeightSlice()");
    using namespace amrex::literals;

    if (!Hipace::HeadRank() || m_num_particles == 0) { return; }

    const int num_to_add = m_init_sorter.m_box_counts_cpu[slice];
    if (m_do_symmetrize) {
        resize(which_slice, 4*num_to_add, 0);
    } else {
        resize(which_slice, num_to_add, 0);
    }

    if (num_to_add == 0) return;

    const amrex::Real clight = get_phys_const().c;

    auto& particle_tile = getBeamSlice(which_slice);

    // Access particles' SoA
    const auto ptd = particle_tile.getParticleTileData();

    const amrex::Long slice_offset = m_init_sorter.m_box_offsets_cpu[slice];
    const auto permutations = m_init_sorter.m_box_permutations.dataPtr();
    amrex::Real * const pos_z = m_z_array.dataPtr();

    const uint64_t pid = m_id64;
    m_id64 += m_do_symmetrize ? 4*num_to_add : num_to_add;

    const bool can = m_can_profile;
    const bool do_symmetrize = m_do_symmetrize;
    const amrex::Real duz_per_uz0_dzeta = m_duz_per_uz0_dzeta;
    const amrex::Real z_min = m_zmin;
    const amrex::Real z_max = m_zmax;
    const amrex::Real z_mean = can ? 0.5_rt * (z_min + z_max) : m_pos_mean_z;
    const amrex::RealVect pos_std = m_position_std;
    const amrex::Real z_foc = m_z_foc;
    const amrex::Real radius = m_radius;
    auto pos_mean_x = m_pos_mean_x_func;
    auto pos_mean_y = m_pos_mean_y_func;
    const amrex::Real weight = m_total_charge / (m_num_particles * m_charge);
    const GetInitialMomentum get_momentum = m_get_momentum;
    const auto enforceBC = EnforceBC();

    amrex::ParallelForRNG(
        num_to_add,
        [=] AMREX_GPU_DEVICE (amrex::Long i, const amrex::RandomEngine& engine) noexcept
        {
            const amrex::Real z_central = pos_z[permutations[slice_offset + i]];
            amrex::Real x = amrex::RandomNormal(0, pos_std[0], engine);
            amrex::Real y = amrex::RandomNormal(0, pos_std[1], engine);

            amrex::Real u[3] = {0.,0.,0.};
            get_momentum(u[0], u[1], u[2], engine, z_central - z_mean, duz_per_uz0_dzeta);

            bool is_valid = true;
            if (z_central < z_min || z_central > z_max || x*x + y*y > radius*radius) {
                is_valid = false;
            }

            // Propagate each electron ballistically for z_foc
            x -= z_foc*u[0]/u[2];
            y -= z_foc*u[1]/u[2];

            const amrex::Real cental_x_pos = pos_mean_x(z_central);
            const amrex::Real cental_y_pos = pos_mean_y(z_central);

            if (!do_symmetrize)
            {
                AddOneBeamParticleSlice(ptd, cental_x_pos+x, cental_y_pos+y,
                                        z_central, u[0], u[1], u[2], weight,
                                        pid, i, clight, enforceBC, is_valid);

            } else {
                AddOneBeamParticleSlice(ptd, cental_x_pos+x, cental_y_pos+y,
                                        z_central, u[0], u[1], u[2], weight,
                                        pid, 4*i, clight, enforceBC, is_valid);
                AddOneBeamParticleSlice(ptd, cental_x_pos-x, cental_y_pos+y,
                                        z_central, -u[0], u[1], u[2], weight,
                                        pid, 4*i+1, clight, enforceBC, is_valid);
                AddOneBeamParticleSlice(ptd, cental_x_pos+x, cental_y_pos-y,
                                        z_central, u[0], -u[1], u[2], weight,
                                        pid, 4*i+2, clight, enforceBC, is_valid);
                AddOneBeamParticleSlice(ptd, cental_x_pos-x, cental_y_pos-y,
                                        z_central, -u[0], -u[1], u[2], weight,
                                        pid, 4*i+3, clight, enforceBC, is_valid);
            }
        });

    return;
}

void
BeamParticleContainer::
InitBeamFixedWeightPDF3D ()
{
    HIPACE_PROFILE("BeamParticleContainer::InitBeamFixedWeightPDF3D()");
    using namespace amrex::literals;

    if (!Hipace::HeadRank() || m_num_particles == 0) { return; }

    amrex::Long num_to_add = m_num_particles;
    if (m_do_symmetrize) num_to_add /= 4;

    const amrex::Geometry geom = Hipace::GetInstance().m_3D_geom[0];
    const amrex::Box domain = geom.Domain();

    m_num_particles_slice.resize(domain.length(2) * m_pdf_ref_ratio);

    const amrex::Real zoffset = geom.ProbLo(2);
    const amrex::Real zscale = geom.CellSize(2) / m_pdf_ref_ratio;

    amrex::Real integral = 0._rt;
    amrex::Real max_density = 0._rt;
    amrex::Real avg_uz = 0._rt;
    amrex::Real avg_uz_sq = 0._rt;

    for (int slice=domain.length(2)*m_pdf_ref_ratio-1; slice>=0; --slice) {
        const amrex::Real zmin = zoffset + slice*zscale;
        const amrex::Real zmax = zoffset + (slice+1)*zscale;
        const amrex::Real zmid = 0.5_rt*(zmin + zmax);

        const amrex::Real pdf_zmin = m_pdf_func(zmin);
        const amrex::Real pdf_zmax = m_pdf_func(zmax);
        const amrex::Real local_weight = 0.5_rt*(pdf_zmin+pdf_zmax);

        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(pdf_zmin >= 0._rt &&  pdf_zmax >= 0._rt,
            "PDF must be >= 0 everywhere");

        if (m_peak_density_is_specified) {
            max_density = std::max(max_density, local_weight
                / (zscale * m_pdf_pos_func[2](zmid) * m_pdf_pos_func[3](zmid) * 2._rt*MathConst::pi)
            );
        }

        // calculate uz and uz_std for AdaptiveTimeStep
        amrex::Real uz_mean_local = m_pdf_u_func[2](zmid);
        amrex::Real uz_std_local = m_pdf_u_func[5](zmid);
        avg_uz += local_weight * uz_mean_local;
        avg_uz_sq += local_weight * (uz_mean_local*uz_mean_local + uz_std_local*uz_std_local);

        integral += local_weight;
        m_num_particles_slice[slice] = 0;
    }

    if (m_peak_density_is_specified) {
        m_total_weight = m_density * integral / max_density;
    } else {
        m_total_weight = m_total_charge / m_charge;
    }

    // calculate uz and uz_std for AdaptiveTimeStep
    m_get_momentum.m_u_mean[2] = avg_uz/integral;
    m_get_momentum.m_u_std[2] = std::sqrt(avg_uz_sq/integral - (avg_uz/integral)*(avg_uz/integral));

    if (Hipace::m_normalized_units) {
        m_total_weight *= geom.InvCellSize(0)*geom.InvCellSize(1)*geom.InvCellSize(2);
    }

    amrex::Long num_added = 0;

    while (num_added != num_to_add) {

        // It is very unlikely that the correct amount of particles will be initialized in the first
        // iteration. Another iteration is done with remaining particles (or subtracting extra
        // particles). This converges quickly as the expected deviation is sqrt(num_to_add_now),
        // resulting in a time complexity of O(num_slices*log(log(num_to_add)))
        const amrex::Long num_to_add_now = num_to_add - num_added;

        for (int slice=domain.length(2)*m_pdf_ref_ratio-1; slice >=0; --slice) {
            const amrex::Real zmin = zoffset + slice*zscale;
            const amrex::Real zmax = zoffset + (slice+1)*zscale;

            const amrex::Real pdf_zmin = m_pdf_func(zmin);
            const amrex::Real pdf_zmax = m_pdf_func(zmax);
            const amrex::Real local_weight = 0.5_rt*(pdf_zmin+pdf_zmax);

            const amrex::Real mean_particles = num_to_add_now*local_weight/integral;

            if (mean_particles >= 0) {
                // use a Poisson distribution to mimic how many independent particles would be
                // initialized according to the full PDF
                const unsigned int n = amrex::RandomPoisson(mean_particles);
                m_num_particles_slice[slice] += n;
                num_added += n;
            } else {
                // if there were too many particles initialized in an earlier iteration we need to
                // remove some but also avoid having less than zero particles per slice
                const unsigned int n = std::min(amrex::RandomPoisson(-mean_particles),
                                                m_num_particles_slice[slice]);
                m_num_particles_slice[slice] -= n;
                num_added -= n;
            }
        }
    }
}

void
BeamParticleContainer::
InitBeamFixedWeightPDFSlice (int slice, int which_slice)
{
    HIPACE_PROFILE("BeamParticleContainer::InitBeamFixedWeightPDFSlice()");
    using namespace amrex::literals;

    if (!Hipace::HeadRank() || m_num_particles == 0) { return; }

    unsigned int num_to_add_full = 0;
    for (int r=m_pdf_ref_ratio-1; r>=0; --r) {
        num_to_add_full += m_num_particles_slice[slice*m_pdf_ref_ratio+r];
    }
    if (m_do_symmetrize) {
        resize(which_slice, 4*num_to_add_full, 0);
    } else {
        resize(which_slice, num_to_add_full, 0);
    }

    unsigned int loc_index = 0;
    for (int r=m_pdf_ref_ratio-1; r>=0; --r) {
        const unsigned int num_to_add = m_num_particles_slice[slice*m_pdf_ref_ratio+r];
        if (num_to_add == 0) continue;

        auto& particle_tile = getBeamSlice(which_slice);
        // Access particles' SoA
        const auto ptd = particle_tile.getParticleTileData();

        const uint64_t pid = m_id64;
        m_id64 += m_do_symmetrize ? 4*num_to_add : num_to_add;

        const amrex::Real clight = get_phys_const().c;
        const bool do_symmetrize = m_do_symmetrize;
        const bool peak_density_is_specified = m_peak_density_is_specified;
        const amrex::Real z_foc = m_z_foc;
        const amrex::Real radius = m_radius;
        const amrex::Real weight = m_total_weight / m_num_particles;
        const auto pos_func = m_pdf_pos_func;
        const auto u_func = m_pdf_u_func;
        const amrex::Geometry& geom = Hipace::GetInstance().m_3D_geom[0];
        const amrex::Real dz = geom.CellSize(2) / m_pdf_ref_ratio;
        const amrex::Real zmin = geom.ProbLo(2) + (slice*m_pdf_ref_ratio+r)*dz;
        const amrex::Real zmax = geom.ProbLo(2) + (slice*m_pdf_ref_ratio+r+1)*dz;
        const amrex::Real lo_weight = m_pdf_func(zmin);
        const amrex::Real hi_weight = m_pdf_func(zmax);
        AMREX_ALWAYS_ASSERT(lo_weight + hi_weight > 0._rt);
        // the proper formula is not defined for hi_weight == lo_weight and may
        // have precision issues around that point, so we use a Taylor expansion instead
        // if the hi_weight and lo_weight are within 10% of each other.
        const bool use_taylor = std::min(lo_weight, hi_weight)*1.1 > std::max(lo_weight, hi_weight);
        const amrex::Real lo_hi_weight_inv = use_taylor ?
            1._rt/(hi_weight+lo_weight) : 1._rt/(hi_weight-lo_weight);
        const auto enforceBC = EnforceBC();

        amrex::ParallelForRNG(
            num_to_add,
            [=] AMREX_GPU_DEVICE (unsigned int i, const amrex::RandomEngine& engine) noexcept
            {
                // if m_pdf_ref_ratio is greater than one, a single slice of beam particles
                // needs to be initialized by multiple kernels so we need to keep track of the
                // local index offset for each kernel
                i += loc_index;

                const amrex::Real w = amrex::Random(engine);
                amrex::Real z = zmin;
                if (use_taylor) {
                    z += dz*(w - w*(w-1._rt)*(hi_weight-lo_weight)*lo_hi_weight_inv);
                } else {
                    z += dz*((std::sqrt(lo_weight*lo_weight
                        +w*(hi_weight*hi_weight-lo_weight*lo_weight))-lo_weight)*lo_hi_weight_inv);
                }

                const amrex::Real x_mean = pos_func[0](z);
                const amrex::Real y_mean = pos_func[1](z);
                const amrex::Real x_std = pos_func[2](z);
                const amrex::Real y_std = pos_func[3](z);

                amrex::Real x = 0._rt;
                amrex::Real y = 0._rt;
                bool is_valid = false;
                do {
                    x = amrex::RandomNormal(0, x_std, engine);
                    y = amrex::RandomNormal(0, y_std, engine);
                    is_valid = x*x + y*y <= radius*radius;
                } while (!peak_density_is_specified && !is_valid);

                const amrex::Real ux = amrex::RandomNormal(u_func[0](z), u_func[3](z), engine);
                const amrex::Real uy = amrex::RandomNormal(u_func[1](z), u_func[4](z), engine);
                const amrex::Real uz = amrex::RandomNormal(u_func[2](z), u_func[5](z), engine);

                // Propagate each electron ballistically for z_foc
                x -= z_foc*ux/uz;
                y -= z_foc*uy/uz;

                if (!do_symmetrize)
                {
                    AddOneBeamParticleSlice(ptd, x_mean+x, y_mean+y,
                                            z, ux, uy, uz, weight, pid, i, clight, enforceBC, is_valid);

                } else {
                    AddOneBeamParticleSlice(ptd, x_mean+x, y_mean+y,
                                            z, ux, uy, uz, weight, pid, 4*i, clight, enforceBC, is_valid);
                    AddOneBeamParticleSlice(ptd, x_mean-x, y_mean+y,
                                            z, -ux, uy, uz, weight, pid, 4*i+1, clight, enforceBC, is_valid);
                    AddOneBeamParticleSlice(ptd, x_mean+x, y_mean-y,
                                            z, ux, -uy, uz, weight, pid, 4*i+2, clight, enforceBC, is_valid);
                    AddOneBeamParticleSlice(ptd, x_mean-x, y_mean-y,
                                            z, -ux, -uy, uz, weight, pid, 4*i+3, clight, enforceBC, is_valid);
                }
            });

        loc_index += num_to_add;
    }
}

#ifdef HIPACE_USE_OPENPMD
amrex::Real
BeamParticleContainer::
InitBeamFromFileHelper (const std::string input_file,
                        const bool coordinates_specified,
                        const amrex::Array<std::string, AMREX_SPACEDIM> file_coordinates_xyz,
                        const amrex::Geometry& geom,
                        amrex::Real n_0,
                        const int num_iteration,
                        const std::string species_name,
                        const bool species_specified)
{
    HIPACE_PROFILE("BeamParticleContainer::InitParticles()");

    openPMD::Datatype input_type = openPMD::Datatype::INT;
    bool species_known;
    {
        // Check what kind of Datatype is used in beam file
        auto series = openPMD::Series( input_file , openPMD::Access::READ_ONLY );

        if(!series.iterations.contains(num_iteration)) {
            amrex::Abort("Could not find iteration " + std::to_string(num_iteration) +
                                                        " in file " + input_file + "\n");
        }
        species_known = series.iterations[num_iteration].particles.contains(species_name);

        for( auto const& particle_type : series.iterations[num_iteration].particles ) {
            if( (!species_known) || (particle_type.first == species_name) ) {
                for( auto const& physical_quantity : particle_type.second ) {
                    if( physical_quantity.first != "id") {
                        for( auto const& axes_direction : physical_quantity.second ) {
                            input_type = axes_direction.second.getDatatype();
                        }
                    }
                }
            }
        }

        if( input_type == openPMD::Datatype::INT || (species_specified && !species_known)) {
            std::string err = "Error, the particle species name " + species_name +
                  " was not found or does not contain any data. The input file contains the" +
                  " following particle species names:\n";
            for( auto const& species_type : series.iterations[num_iteration].particles ) {
                err += species_type.first + "\n";
            }
            if( !species_specified ) {
                err += "Use beam.openPMD_species_name NAME to specify a paricle species\n";
            }
            amrex::Abort(err);
        }

    }

    amrex::Real ptime {0.};
    if( input_type == openPMD::Datatype::FLOAT ) {
        ptime = InitBeamFromFile<float>(input_file, coordinates_specified, file_coordinates_xyz,
                                        geom, n_0, num_iteration, species_name, species_known);
    }
    else if( input_type == openPMD::Datatype::DOUBLE ) {
        ptime = InitBeamFromFile<double>(input_file, coordinates_specified, file_coordinates_xyz,
                                         geom, n_0, num_iteration, species_name, species_known);
    }
    else{
        amrex::Abort("Unknown Datatype used in Beam Input file. Must use double or float\n");
    }
    return ptime;
}

template <typename input_type>
amrex::Real
BeamParticleContainer::
InitBeamFromFile (const std::string input_file,
                  const bool coordinates_specified,
                  const amrex::Array<std::string, AMREX_SPACEDIM> file_coordinates_xyz,
                  const amrex::Geometry& geom,
                  amrex::Real n_0,
                  const int num_iteration,
                  const std::string species_name,
                  const bool species_specified)
{
    HIPACE_PROFILE("BeamParticleContainer::InitParticles()");

    amrex::Real physical_time {0.};

    auto series = openPMD::Series( input_file , openPMD::Access::READ_ONLY );

    if( series.iterations[num_iteration].containsAttribute("time") ) {
        physical_time = series.iterations[num_iteration].time<input_type>();
    }

    if (!Hipace::HeadRank()) return physical_time;

    // Initialize variables to translate between names from the file and names in Hipace
    std::string name_particle ="";
    std::string name_r ="", name_rx ="", name_ry ="", name_rz ="";
    std::string name_u ="", name_ux ="", name_uy ="", name_uz ="";
    std::string name_m ="", name_mm ="";
    std::string name_q ="", name_qq ="";
    std::string name_g ="", name_gg ="";
    bool u_is_momentum = false;

    // Iterate through all matadata in file, search for unit combination for Distance, Velocity,
    // Charge, Mass. Auto detect position, weighting and coordinates if named x y z or X Y Z etc.
    for( auto const& particle_type : series.iterations[num_iteration].particles ) {
        if( (!species_specified) || (particle_type.first == species_name) ) {
            name_particle = particle_type.first;
            for( auto const& physical_quantity : particle_type.second ) {

                std::array<double,7> units = physical_quantity.second.unitDimension();

                if(units == std::array<double,7> {1., 0., 0., 0., 0., 0., 0.}) {
                    if( (!particle_type.second.contains("position")) ||
                                                      (physical_quantity.first == "position")) {
                        name_r = physical_quantity.first;
                        for( auto const& axes_direction : physical_quantity.second ) {
                            if(axes_direction.first == "x" || axes_direction.first == "X") {
                                name_rx = axes_direction.first;
                            }
                            if(axes_direction.first == "y" || axes_direction.first == "Y") {
                                name_ry = axes_direction.first;
                            }
                            if(axes_direction.first == "z" || axes_direction.first == "Z") {
                                name_rz = axes_direction.first;
                            }
                        }
                    }
                }
                else if(units == std::array<double,7> {1., 0., -1., 0., 0., 0., 0.}) {
                    // proper velocity = gamma * v
                    name_u = physical_quantity.first;
                    u_is_momentum = false;
                    for( auto const& axes_direction : physical_quantity.second ) {
                        if(axes_direction.first == "x" || axes_direction.first == "X") {
                            name_ux = axes_direction.first;
                        }
                        if(axes_direction.first == "y" || axes_direction.first == "Y") {
                            name_uy = axes_direction.first;
                        }
                        if(axes_direction.first == "z" || axes_direction.first == "Z") {
                            name_uz = axes_direction.first;
                        }
                    }
                }
                else if(units == std::array<double,7> {1., 1., -1., 0., 0., 0., 0.}) {
                    // momentum = gamma * m * v
                    name_u = physical_quantity.first;
                    u_is_momentum = true;
                    for( auto const& axes_direction : physical_quantity.second ) {
                        if(axes_direction.first == "x" || axes_direction.first == "X") {
                            name_ux = axes_direction.first;
                        }
                        if(axes_direction.first == "y" || axes_direction.first == "Y") {
                            name_uy = axes_direction.first;
                        }
                        if(axes_direction.first == "z" || axes_direction.first == "Z") {
                            name_uz = axes_direction.first;
                        }
                    }
                }
                else if(units == std::array<double,7> {0., 1., 0., 0., 0., 0., 0.}) {
                    name_m = physical_quantity.first;
                    for( auto const& axes_direction : physical_quantity.second ) {
                        name_mm = axes_direction.first;
                    }
                }
                else if(units == std::array<double,7> {0., 0., 1., 1., 0., 0., 0.}) {
                    name_q = physical_quantity.first;
                    for( auto const& axes_direction : physical_quantity.second ) {
                        name_qq = axes_direction.first;
                    }
                }
                else if(units == std::array<double,7> {0., 0., 0., 0., 0., 0., 0.}) {
                    if(physical_quantity.first == "weighting") {
                        name_g = physical_quantity.first;
                        for( auto const& axes_direction : physical_quantity.second ) {
                            name_gg = axes_direction.first;
                        }
                    }
                }
            }
        }
    }

    // Overide coordinate names with those from file_coordinates_xyz argument
    if(coordinates_specified) {
        name_rx = name_ux = file_coordinates_xyz[0];
        name_ry = name_uy = file_coordinates_xyz[1];
        name_rz = name_uz = file_coordinates_xyz[2];
    }

    // Determine whether to use momentum or normalized momentum as well as weight, charge or mass
    // set conversion factor appropriately
    const PhysConst phys_const_SI = make_constants_SI();
    const PhysConst phys_const = get_phys_const();

    std::string name_w = "", name_ww = "";
    std::string weighting_type = "";
    std::string momentum_type = "Proper velocity";

    input_type si_to_norm_pos = 1.;
    input_type si_to_norm_momentum = phys_const_SI.c;
    input_type si_to_norm_weight = 1.;

    if(u_is_momentum) {
        si_to_norm_momentum = m_mass * (phys_const_SI.m_e / phys_const.m_e) * phys_const_SI.c;
        momentum_type = "Momentum";
    }

    if(name_gg != "") {
        name_w = name_g;
        name_ww = name_gg;
        weighting_type = "Weighting";
    }
    else if(name_qq != "") {
        name_w = name_q;
        name_ww = name_qq;
        si_to_norm_weight = m_charge * (phys_const_SI.q_e / phys_const.q_e);
        weighting_type = "Charge";
    }
    else if(name_mm != "") {
        name_w = name_m;
        name_ww = name_mm;
        si_to_norm_weight = m_mass * (phys_const_SI.m_e / phys_const.m_e);
        weighting_type = "Mass";
    }
    else {
        amrex::Abort("Could not find Charge of dimension I * T in file\n");
    }

    // Abort if file necessary information couldn't be found in file
    if(name_r == "") {
        amrex::Abort("Could not find Position of dimension L in file\n");
    }
    if(name_u == "") {
        amrex::Abort("Could not find u or Momentum of dimension L / T or M * L / T in file\n");
    }
    if(name_rx == "" || name_ux == "") {
        amrex::Abort("Coud not find x coordinate in file. Use file_coordinates_xyz x1 x2 x3\n");
    }
    if(name_ry == "" || name_uy == "") {
        amrex::Abort("Coud not find y coordinate in file. Use file_coordinates_xyz x1 x2 x3\n");
    }
    if(name_rz == "" || name_uz == "") {
        amrex::Abort("Coud not find z coordinate in file. Use file_coordinates_xyz x1 x2 x3\n");
    }

    for(std::string name_r_c : {name_rx, name_ry, name_rz}) {
        if(!series.iterations[num_iteration].particles[name_particle][name_r].contains(name_r_c)) {
            amrex::Abort("Beam input file does not contain " + name_r_c + " coordinate in " +
            name_r + " (position)\n");
        }
    }
    for(std::string name_u_c : {name_ux, name_uy, name_uz}) {
        if(!series.iterations[num_iteration].particles[name_particle][name_u].contains(name_u_c)) {
            amrex::Abort("Beam input file does not contain " + name_u_c + " coordinate in " +
            name_u + " (momentum)\n");
        }
    }

    // print the names of the arrays in the file that are used for the simulation
    if(Hipace::m_verbose >= 3){
       amrex::Print() << "Beam Input File '" << input_file << "' in Iteration '" << num_iteration
          << "' and Paricle '" << name_particle
          << "' imported with:\nPositon '" << name_r << "' (coordinates '" << name_rx << "', '"
          << name_ry << "', '" << name_rz << "')\n"
          << momentum_type << " '" << name_u << "' (coordinates '" << name_ux
          << "', '" << name_uy << "', '" << name_uz << "')\n"
          << weighting_type << " '" << name_w << "' (in '" << name_ww << "')\n";
    }

    auto electrons = series.iterations[num_iteration].particles[name_particle];

    const auto num_to_add = electrons[name_r][name_rx].getExtent()[0];

    if(num_to_add >= 2147483647) {
        amrex::Abort("Beam from file can't have more than 2'147'483'646 Particles\n");
    }

    auto del = [](input_type *p){ amrex::The_Pinned_Arena()->free(reinterpret_cast<void*>(p)); };

    // copy Data to pinned memory
    std::shared_ptr<input_type> r_x_data{ reinterpret_cast<input_type*>(
        amrex::The_Pinned_Arena()->alloc(sizeof(input_type)*num_to_add) ), del};
    std::shared_ptr<input_type> r_y_data{ reinterpret_cast<input_type*>(
        amrex::The_Pinned_Arena()->alloc(sizeof(input_type)*num_to_add) ), del};
    std::shared_ptr<input_type> r_z_data{ reinterpret_cast<input_type*>(
        amrex::The_Pinned_Arena()->alloc(sizeof(input_type)*num_to_add) ), del};
    std::shared_ptr<input_type> u_x_data{ reinterpret_cast<input_type*>(
        amrex::The_Pinned_Arena()->alloc(sizeof(input_type)*num_to_add) ), del};
    std::shared_ptr<input_type> u_y_data{ reinterpret_cast<input_type*>(
        amrex::The_Pinned_Arena()->alloc(sizeof(input_type)*num_to_add) ), del};
    std::shared_ptr<input_type> u_z_data{ reinterpret_cast<input_type*>(
        amrex::The_Pinned_Arena()->alloc(sizeof(input_type)*num_to_add) ), del};
    std::shared_ptr<input_type> w_w_data{ reinterpret_cast<input_type*>(
        amrex::The_Pinned_Arena()->alloc(sizeof(input_type)*num_to_add) ), del};

    electrons[name_r][name_rx].loadChunk<input_type>(r_x_data, {0u}, {num_to_add});
    electrons[name_r][name_ry].loadChunk<input_type>(r_y_data, {0u}, {num_to_add});
    electrons[name_r][name_rz].loadChunk<input_type>(r_z_data, {0u}, {num_to_add});
    electrons[name_u][name_ux].loadChunk<input_type>(u_x_data, {0u}, {num_to_add});
    electrons[name_u][name_uy].loadChunk<input_type>(u_y_data, {0u}, {num_to_add});
    electrons[name_u][name_uz].loadChunk<input_type>(u_z_data, {0u}, {num_to_add});
    electrons[name_w][name_ww].loadChunk<input_type>(w_w_data, {0u}, {num_to_add});

    series.flush();

    // calculate the multiplier to convert to Hipace units
    if(Hipace::m_normalized_units) {
        if(n_0 == 0) {
            if(electrons.containsAttribute("HiPACE++_Plasma_Density")) {
                n_0 = electrons.getAttribute("HiPACE++_Plasma_Density").get<double>();
            }
            else {
                amrex::Abort("Please specify the plasma density of the external beam "
                             "to use it with normalized units with beam.plasma_density");
            }
        }
        const auto dx = geom.CellSizeArray();
        const double omega_p = (double)phys_const_SI.q_e * sqrt( (double)n_0 /
                                      ( (double)phys_const_SI.ep0 * (double)phys_const_SI.m_e ) );
        const double kp_inv = (double)phys_const_SI.c / omega_p;
        si_to_norm_pos = (input_type)kp_inv;
        si_to_norm_weight *= (input_type)( n_0 * dx[0] * dx[1] * dx[2] * kp_inv * kp_inv * kp_inv );
    }

    input_type unit_rx, unit_ry, unit_rz, unit_ux, unit_uy, unit_uz, unit_ww;
    bool hipace_restart = false;
    const std::string attr = "HiPACE++_reference_unitSI";
    if(electrons.containsAttribute("HiPACE++_use_reference_unitSI")) {
        if(electrons.getAttribute("HiPACE++_use_reference_unitSI").get<bool>() == true) {
            hipace_restart = true;
        }
    }

    if(hipace_restart) {
        unit_rx = electrons[name_r][name_rx].getAttribute(attr).get<double>() / si_to_norm_pos;
        unit_ry = electrons[name_r][name_ry].getAttribute(attr).get<double>() / si_to_norm_pos;
        unit_rz = electrons[name_r][name_rz].getAttribute(attr).get<double>() / si_to_norm_pos;
        unit_ux = electrons[name_u][name_ux].getAttribute(attr).get<double>() / si_to_norm_momentum;
        unit_uy = electrons[name_u][name_uy].getAttribute(attr).get<double>() / si_to_norm_momentum;
        unit_uz = electrons[name_u][name_uz].getAttribute(attr).get<double>() / si_to_norm_momentum;
        unit_ww = electrons[name_w][name_ww].getAttribute(attr).get<double>() / si_to_norm_weight;
    }
    else {
        unit_rx = electrons[name_r][name_rx].unitSI() / si_to_norm_pos;
        unit_ry = electrons[name_r][name_ry].unitSI() / si_to_norm_pos;
        unit_rz = electrons[name_r][name_rz].unitSI() / si_to_norm_pos;
        unit_ux = electrons[name_u][name_ux].unitSI() / si_to_norm_momentum;
        unit_uy = electrons[name_u][name_uy].unitSI() / si_to_norm_momentum;
        unit_uz = electrons[name_u][name_uz].unitSI() / si_to_norm_momentum;
        unit_ww = electrons[name_w][name_ww].unitSI() / si_to_norm_weight;
    }

    // input data using AddOneBeamParticle function, make necessary variables and arrays
    auto& particle_tile = getBeamInitSlice();
    auto old_size = particle_tile.size();
    auto new_size = old_size + num_to_add;
    particle_tile.resize(new_size);
    const auto ptd = particle_tile.getParticleTileData();
    const auto enforceBC = EnforceBC();

    const uint64_t pid = m_id64;
    m_id64 += num_to_add;

    const input_type * const r_x_ptr = r_x_data.get();
    const input_type * const r_y_ptr = r_y_data.get();
    const input_type * const r_z_ptr = r_z_data.get();
    const input_type * const u_x_ptr = u_x_data.get();
    const input_type * const u_y_ptr = u_y_data.get();
    const input_type * const u_z_ptr = u_z_data.get();
    const input_type * const w_w_ptr = w_w_data.get();

    amrex::ParallelFor(amrex::Long(num_to_add),
        [=] AMREX_GPU_DEVICE (const amrex::Long i) {
            AddOneBeamParticle(ptd,
                static_cast<amrex::Real>(r_x_ptr[i] * unit_rx),
                static_cast<amrex::Real>(r_y_ptr[i] * unit_ry),
                static_cast<amrex::Real>(r_z_ptr[i] * unit_rz),
                static_cast<amrex::Real>(u_x_ptr[i] * unit_ux), // = gamma * beta
                static_cast<amrex::Real>(u_y_ptr[i] * unit_uy),
                static_cast<amrex::Real>(u_z_ptr[i] * unit_uz),
                static_cast<amrex::Real>(w_w_ptr[i] * unit_ww),
                pid, i, phys_const.c, enforceBC);
        });

    amrex::Gpu::streamSynchronize();

    return physical_time;
}
#endif // HIPACE_USE_OPENPMD
