#include "BeamParticleContainer.H"
#include "utils/Constants.H"
#include "ParticleUtil.H"
#include "Hipace.H"
#include "utils/HipaceProfilerWrapper.H"

#include <AMReX_REAL.H>

namespace
{
    /** \brief Adds a single beam particle
     *
     * \param[in,out] pstruct array with AoS beam data
     * \param[in,out] arrdata array with SoA beam data
     * \param[in] x position in x
     * \param[in] y position in y
     * \param[in] z position in z
     * \param[in] ux momentum in x
     * \param[in] uy momentum in y
     * \param[in] uz momentum in z
     * \param[in] weight weight of the single particle
     * \param[in] pid particle ID to be assigned to the particle
     * \param[in] procID processor ID to be assigned to the particle
     * \param[in] ip index of the particle
     * \param[in] speed_of_light speed of light in SI units
     */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void AddOneBeamParticle (
        BeamParticleContainer::ParticleType* pstruct,
        amrex::GpuArray<amrex::ParticleReal*, BeamIdx::nattribs> arrdata, const amrex::Real& x, const amrex::Real& y, const amrex::Real& z,
        const amrex::Real& ux, const amrex::Real& uy, const amrex::Real& uz, const amrex::Real& weight,
        const int& pid, const int& procID, const int& ip, const amrex::Real& speed_of_light) noexcept
    {
        BeamParticleContainer::ParticleType& p = pstruct[ip];
        // Set particle AoS
        p.id()   = pid + ip;
        p.cpu()  = procID;
        p.pos(0) = x;
        p.pos(1) = y;
        p.pos(2) = z;

        // Set particle SoA
        arrdata[BeamIdx::ux  ][ip] = ux * speed_of_light;
        arrdata[BeamIdx::uy  ][ip] = uy * speed_of_light;
        arrdata[BeamIdx::uz  ][ip] = uz * speed_of_light;
        arrdata[BeamIdx::w][ip] = weight;
    }
}

void
BeamParticleContainer::
InitBeamFixedPPC (const amrex::IntVect& a_num_particles_per_cell,
                  const GetInitialDensity& get_density,
                  const GetInitialMomentum& get_momentum,
                  const amrex::Geometry& a_geom,
                  const amrex::Real a_zmin,
                  const amrex::Real a_zmax,
                  const amrex::Real a_radius)
{
    using namespace amrex::literals;

    HIPACE_PROFILE("BeamParticleContainer::InitParticles");

    constexpr int lev = 0;

    // Since each box is allows to be very large, its number of cells may exceed the largest
    // int (~2.e9). To avoid this, we use a coarsened box (the coarsening ratio is cr, see below)
    // to inject particles. This is just a trick to have fewer cells, it injects the same
    // by using fewer larger cells and more particles per cell.
    amrex::IntVect cr {Hipace::m_beam_injection_cr,Hipace::m_beam_injection_cr,1};
    AMREX_ALWAYS_ASSERT(cr[AMREX_SPACEDIM-1] == 1);
    auto dx = a_geom.CellSizeArray();
    for (int i=0; i<AMREX_SPACEDIM; i++) dx[i] *= cr[i];
    const auto plo = a_geom.ProbLoArray();

    amrex::IntVect ppc_cr = a_num_particles_per_cell;
    for (int i=0; i<AMREX_SPACEDIM; i++) ppc_cr[i] *= cr[i];

    const int num_ppc = AMREX_D_TERM( ppc_cr[0],
                                      *ppc_cr[1],
                                      *ppc_cr[2]);

    const amrex::Real scale_fac = Hipace::m_normalized_units ?
        1._rt/num_ppc*cr[0]*cr[1]*cr[2] :
        dx[0]*dx[1]*dx[2]/num_ppc;

    for(amrex::MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi)
    {
        // First: loop over all cells, and count the particles effectively injected.
        amrex::Box tile_box  = mfi.tilebox();
        tile_box.coarsen(cr);
        const auto lo = amrex::lbound(tile_box);
        const auto hi = amrex::ubound(tile_box);

        amrex::Gpu::DeviceVector<unsigned int> counts(tile_box.numPts(), 0);
        unsigned int* pcount = counts.dataPtr();

        amrex::Gpu::DeviceVector<unsigned int> offsets(tile_box.numPts());
        unsigned int* poffset = offsets.dataPtr();

        amrex::ParallelFor(tile_box,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            for (int i_part=0; i_part<num_ppc;i_part++)
            {
                amrex::Real r[3];

                ParticleUtil::get_position_unit_cell(r, ppc_cr, i_part);

                amrex::Real x = plo[0] + (i + r[0])*dx[0];
                amrex::Real y = plo[1] + (j + r[1])*dx[1];
                amrex::Real z = plo[2] + (k + r[2])*dx[2];

                if (z >= a_zmax || z < a_zmin ||
                    (x*x+y*y) > a_radius*a_radius) continue;

                int ix = i - lo.x;
                int iy = j - lo.y;
                int iz = k - lo.z;
                int nx = hi.x-lo.x+1;
                int ny = hi.y-lo.y+1;
                int nz = hi.z-lo.z+1;
                unsigned int uix = amrex::min(nx-1,amrex::max(0,ix));
                unsigned int uiy = amrex::min(ny-1,amrex::max(0,iy));
                unsigned int uiz = amrex::min(nz-1,amrex::max(0,iz));
                unsigned int cellid = (uix * ny + uiy) * nz + uiz;
                pcount[cellid] += 1;
            }
        });

        int num_to_add = amrex::Scan::ExclusiveSum(counts.size(), counts.data(), offsets.data());

        // Second: allocate the memory for these particles
        auto& particles = GetParticles(lev);
        auto& particle_tile = particles[std::make_pair(mfi.index(), mfi.LocalTileIndex())];

        auto old_size = particle_tile.GetArrayOfStructs().size();
        auto new_size = old_size + num_to_add;
        particle_tile.resize(new_size);

        if (num_to_add == 0) continue;

        // Third: Actually initialize the particles at the right locations
        ParticleType* pstruct = particle_tile.GetArrayOfStructs()().data();

        amrex::GpuArray<amrex::ParticleReal*, BeamIdx::nattribs> arrdata =
            particle_tile.GetStructOfArrays().realarray();

        int procID = amrex::ParallelDescriptor::MyProc();
        int pid = ParticleType::NextID();
        ParticleType::NextID(pid + num_to_add);

        PhysConst phys_const = get_phys_const();

        amrex::ParallelFor(tile_box,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            int ix = i - lo.x;
            int iy = j - lo.y;
            int iz = k - lo.z;
            int nx = hi.x-lo.x+1;
            int ny = hi.y-lo.y+1;
            int nz = hi.z-lo.z+1;
            unsigned int uix = amrex::min(nx-1,amrex::max(0,ix));
            unsigned int uiy = amrex::min(ny-1,amrex::max(0,iy));
            unsigned int uiz = amrex::min(nz-1,amrex::max(0,iz));
            unsigned int cellid = (uix * ny + uiy) * nz + uiz;

            int pidx = int(poffset[cellid] - poffset[0]);

            for (int i_part=0; i_part<num_ppc;i_part++)
            {
                amrex::Real r[3] = {0.,0.,0.};

                ParticleUtil::get_position_unit_cell(r, ppc_cr, i_part);

                amrex::Real x = plo[0] + (i + r[0])*dx[0];
                amrex::Real y = plo[1] + (j + r[1])*dx[1];
                amrex::Real z = plo[2] + (k + r[2])*dx[2];

                if (z >= a_zmax || z < a_zmin ||
                    (x*x+y*y) > a_radius*a_radius) continue;

                amrex::Real u[3] = {0.,0.,0.};
                get_momentum(u[0],u[1],u[2]);

                const amrex::Real weight = get_density(x, y, z) * scale_fac;
                AddOneBeamParticle(pstruct, arrdata, x, y, z, u[0], u[1], u[2], weight,
                                   pid, procID, pidx, phys_const.c);

                ++pidx;
            }
        });
    }
    AMREX_ASSERT(OK());
}

void
BeamParticleContainer::
InitBeamFixedWeight (int num_to_add,
                     const GetInitialMomentum& get_momentum,
                     const amrex::RealVect pos_mean,
                     const amrex::RealVect pos_std,
                     const amrex::Real total_charge,
                     const bool do_symmetrize,
                     const amrex::Real dx_per_dzeta,
                     const amrex::Real dy_per_dzeta)
{
    HIPACE_PROFILE("BeamParticleContainer::InitParticles");

    constexpr int lev = 0;

    if (num_to_add == 0) return;
    if (do_symmetrize) num_to_add /=4;

    PhysConst phys_const = get_phys_const();

    if (amrex::ParallelDescriptor::IOProcessor()) {

        // WARNING Implemented for 1 box per MPI rank.
        for(amrex::MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi)
        {
            // Allocate the memory for these particles
            auto& particles = GetParticles(lev);
            auto& particle_tile = particles[std::make_pair(mfi.index(), mfi.LocalTileIndex())];
            auto old_size = particle_tile.GetArrayOfStructs().size();
            auto new_size = do_symmetrize? old_size + 4*num_to_add : old_size + num_to_add;
            particle_tile.resize(new_size);

            // Access particles' AoS and SoA
            ParticleType* pstruct = particle_tile.GetArrayOfStructs()().data();
            amrex::GpuArray<amrex::ParticleReal*, BeamIdx::nattribs> arrdata =
                particle_tile.GetStructOfArrays().realarray();

            const int procID = amrex::ParallelDescriptor::MyProc();
            const int pid = ParticleType::NextID();
            ParticleType::NextID(pid + num_to_add);

            amrex::ParallelFor(
                num_to_add,
                [=] AMREX_GPU_DEVICE (int i) noexcept
                {
                    const amrex::Real x = amrex::RandomNormal(0, pos_std[0]);
                    const amrex::Real y = amrex::RandomNormal(0, pos_std[1]);
                    const amrex::Real z = amrex::RandomNormal(0, pos_std[2]);
                    amrex::Real u[3] = {0.,0.,0.};
                    get_momentum(u[0],u[1],u[2]);

                    const amrex::Real cental_x_pos = pos_mean[0] + z*dx_per_dzeta;
                    const amrex::Real cental_y_pos = pos_mean[1] + z*dy_per_dzeta;

                    amrex::Real weight = total_charge / num_to_add / phys_const.q_e;
                    if (!do_symmetrize)
                    {
                        AddOneBeamParticle(pstruct, arrdata, cental_x_pos+x, cental_y_pos+y,
                                           pos_mean[2]+z, u[0], u[1], u[2], weight,
                                           pid, procID, i, phys_const.c);
                    } else {
                        weight /= 4;
                        AddOneBeamParticle(pstruct, arrdata, cental_x_pos+x, cental_y_pos+y,
                                           pos_mean[2]+z, u[0], u[1], u[2], weight,
                                           pid, procID, 4*i, phys_const.c);
                        AddOneBeamParticle(pstruct, arrdata, cental_x_pos-x, cental_y_pos+y,
                                           pos_mean[2]+z, -u[0], u[1], u[2], weight,
                                           pid, procID, 4*i+1, phys_const.c);
                        AddOneBeamParticle(pstruct, arrdata, cental_x_pos+x, cental_y_pos-y,
                                           pos_mean[2]+z, u[0], -u[1], u[2], weight,
                                           pid, procID, 4*i+2, phys_const.c);
                        AddOneBeamParticle(pstruct, arrdata, cental_x_pos-x, cental_y_pos-y,
                                           pos_mean[2]+z, -u[0], -u[1], u[2], weight,
                                           pid, procID, 4*i+3, phys_const.c);
                    }
                });
        }
    }
    Redistribute();
    AMREX_ASSERT(OK());
    return;
}
