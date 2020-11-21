#include "BeamParticleContainer.H"
#include "Constants.H"
#include "ParticleUtil.H"
#include "Hipace.H"
#include "HipaceProfilerWrapper.H"

#include <AMReX_REAL.H>

using namespace amrex;

void
BeamParticleContainer::
InitBeamFixedPPC (const IntVect& a_num_particles_per_cell,
                  const GetInitialDensity& get_density,
                  const GetInitialMomentum& get_momentum,
                  const Geometry& a_geom,
                  const amrex::Real a_zmin,
                  const amrex::Real a_zmax,
                  const amrex::Real a_radius)
{
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

    const Real scale_fac = Hipace::m_normalized_units ?
        1._rt/num_ppc*cr[0]*cr[1]*cr[2] :
        dx[0]*dx[1]*dx[2]/num_ppc;

    for(MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi)
    {
        // First: loop over all cells, and count the particles effectively injected.
        Box tile_box  = mfi.tilebox();
        tile_box.coarsen(cr);
        const auto lo = amrex::lbound(tile_box);
        const auto hi = amrex::ubound(tile_box);

        Gpu::DeviceVector<unsigned int> counts(tile_box.numPts(), 0);
        unsigned int* pcount = counts.dataPtr();

        Gpu::DeviceVector<unsigned int> offsets(tile_box.numPts());
        unsigned int* poffset = offsets.dataPtr();

        amrex::ParallelFor(tile_box,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            for (int i_part=0; i_part<num_ppc;i_part++)
            {
                Real r[3];

                ParticleUtil::get_position_unit_cell(r, ppc_cr, i_part);

                Real x = plo[0] + (i + r[0])*dx[0];
                Real y = plo[1] + (j + r[1])*dx[1];
                Real z = plo[2] + (k + r[2])*dx[2];

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

        int num_to_add = Scan::ExclusiveSum(counts.size(), counts.data(), offsets.data());

        // Second: allocate the memory for these particles
        auto& particles = GetParticles(lev);
        auto& particle_tile = particles[std::make_pair(mfi.index(), mfi.LocalTileIndex())];

        auto old_size = particle_tile.GetArrayOfStructs().size();
        auto new_size = old_size + num_to_add;
        particle_tile.resize(new_size);

        if (num_to_add == 0) continue;

        // Third: Actually initialize the particles at the right locations
        ParticleType* pstruct = particle_tile.GetArrayOfStructs()().data();

        auto arrdata = particle_tile.GetStructOfArrays().realarray();

        int procID = ParallelDescriptor::MyProc();
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

                Real x = plo[0] + (i + r[0])*dx[0];
                Real y = plo[1] + (j + r[1])*dx[1];
                Real z = plo[2] + (k + r[2])*dx[2];

                if (z >= a_zmax || z < a_zmin ||
                    (x*x+y*y) > a_radius*a_radius) continue;

                amrex::Real u[3] = {0.,0.,0.};
                get_momentum(u[0],u[1],u[2]);
                ParticleType& p = pstruct[pidx];
                p.id()   = pid + pidx;
                p.cpu()  = procID;
                p.pos(0) = x;
                p.pos(1) = y;
                p.pos(2) = z;

                arrdata[BeamIdx::ux  ][pidx] = u[0] * phys_const.c;
                arrdata[BeamIdx::uy  ][pidx] = u[1] * phys_const.c;
                arrdata[BeamIdx::uz  ][pidx] = u[2] * phys_const.c;
                arrdata[BeamIdx::w][pidx] = get_density(x, y, z);
                arrdata[BeamIdx::w][pidx]  *= scale_fac;
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
                     const bool do_symmetrize)
{
    HIPACE_PROFILE("BeamParticleContainer::InitParticles");

    constexpr int lev = 0;

    if (num_to_add == 0) return;
    if (do_symmetrize) num_to_add /=4;

    PhysConst phys_const = get_phys_const();

    if (ParallelDescriptor::IOProcessor()) {

        // WARNING Implemented for 1 box per MPI rank.
        for(MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi)
        {
            // Allocate the memory for these particles
            auto& particles = GetParticles(lev);
            auto& particle_tile = particles[std::make_pair(mfi.index(), mfi.LocalTileIndex())];
            auto old_size = particle_tile.GetArrayOfStructs().size();
            auto new_size = do_symmetrize? old_size + 4*num_to_add : old_size + num_to_add;
            particle_tile.resize(new_size);

            // Access particles' AoS and SoA
            ParticleType* pstruct = particle_tile.GetArrayOfStructs()().data();
            auto arrdata = particle_tile.GetStructOfArrays().realarray();

            const int procID = ParallelDescriptor::MyProc();
            const int pid = ParticleType::NextID();
            ParticleType::NextID(pid + num_to_add);

            if (!do_symmetrize)
            {
                amrex::ParallelFor(
                    num_to_add,
                    [=] AMREX_GPU_DEVICE (int i) noexcept
                    {
                        const Real x = amrex::RandomNormal(pos_mean[0], pos_std[0]);
                        const Real y = amrex::RandomNormal(pos_mean[1], pos_std[1]);
                        const Real z = amrex::RandomNormal(pos_mean[2], pos_std[2]);
                        amrex::Real u[3] = {0.,0.,0.};
                        get_momentum(u[0],u[1],u[2]);
                        ParticleType& p = pstruct[i];

                        // Set particle AoS
                        p.id()   = pid + i;
                        p.cpu()  = procID;
                        p.pos(0) = x;
                        p.pos(1) = y;
                        p.pos(2) = z;

                        // Set particle SoA
                        arrdata[BeamIdx::ux  ][i] = u[0] * phys_const.c;
                        arrdata[BeamIdx::uy  ][i] = u[1] * phys_const.c;
                        arrdata[BeamIdx::uz  ][i] = u[2] * phys_const.c;
                        arrdata[BeamIdx::w][i] = total_charge / num_to_add / phys_const.q_e;
                    });
            }
            else // do_symmetrize = True
            {
                /* for each particle (x, y, ux, uy) three more particles are generated with
                 * (-x, y, -ux, uy), (x, -y, ux, -uy), and (-x, -y, -ux, -uy)
                 */
                const amrex::Real weight = total_charge / num_to_add /4  / phys_const.q_e;

                amrex::ParallelFor(
                    num_to_add,
                    [=] AMREX_GPU_DEVICE (int i) noexcept
                    {
                        const Real x = amrex::RandomNormal(0, pos_std[0]);
                        const Real y = amrex::RandomNormal(0, pos_std[1]);
                        const Real z = amrex::RandomNormal(0, pos_std[2]);
                        amrex::Real u[3] = {0.,0.,0.};
                        get_momentum(u[0],u[1],u[2]);

                        // first particle
                        ParticleType& p1 = pstruct[4*i];
                        // Set particle AoS
                        p1.id()   = pid + 4*i;
                        p1.cpu()  = procID;
                        p1.pos(0) = pos_mean[0]+x;
                        p1.pos(1) = pos_mean[1]+y;
                        p1.pos(2) = pos_mean[2]+z;
                        // Set particle SoA
                        arrdata[BeamIdx::ux  ][4*i] = u[0] * phys_const.c;
                        arrdata[BeamIdx::uy  ][4*i] = u[1] * phys_const.c;
                        arrdata[BeamIdx::uz  ][4*i] = u[2] * phys_const.c;
                        arrdata[BeamIdx::w][4*i] = weight;

                        // second particle
                        ParticleType& p2 = pstruct[4*i+1];
                        // Set particle AoS
                        p2.id()   = pid + 4*i+1;
                        p2.cpu()  = procID;
                        p2.pos(0) = pos_mean[0]-x;
                        p2.pos(1) = pos_mean[1]+y;
                        p2.pos(2) = pos_mean[2]+z;
                        // Set particle SoA
                        arrdata[BeamIdx::ux  ][4*i+1] = -u[0] * phys_const.c;
                        arrdata[BeamIdx::uy  ][4*i+1] = u[1] * phys_const.c;
                        arrdata[BeamIdx::uz  ][4*i+1] = u[2] * phys_const.c;
                        arrdata[BeamIdx::w][4*i+1] = weight;

                        //third particle
                        ParticleType& p3 = pstruct[4*i+2];
                        // Set particle AoS
                        p3.id()   = pid + 4*i+2;
                        p3.cpu()  = procID;
                        p3.pos(0) = pos_mean[0]+x;
                        p3.pos(1) = pos_mean[1]-y;
                        p3.pos(2) = pos_mean[2]+z;
                        // Set particle SoA
                        arrdata[BeamIdx::ux  ][4*i+2] = u[0] * phys_const.c;
                        arrdata[BeamIdx::uy  ][4*i+2] = -u[1] * phys_const.c;
                        arrdata[BeamIdx::uz  ][4*i+2] = u[2] * phys_const.c;
                        arrdata[BeamIdx::w][4*i+2] = weight;

                        // fourth particle
                        ParticleType& p4 = pstruct[4*i+3];
                        // Set particle AoS
                        p4.id()   = pid + 4*i+3;
                        p4.cpu()  = procID;
                        p4.pos(0) = pos_mean[0]-x;
                        p4.pos(1) = pos_mean[1]-y;
                        p4.pos(2) = pos_mean[2]+z;
                        // Set particle SoA
                        arrdata[BeamIdx::ux  ][4*i+3] = -u[0] * phys_const.c;
                        arrdata[BeamIdx::uy  ][4*i+3] = -u[1] * phys_const.c;
                        arrdata[BeamIdx::uz  ][4*i+3] = u[2] * phys_const.c;
                        arrdata[BeamIdx::w][4*i+3] = weight;
                    });
            } // end if do_symmetrize = True
        }
    }
    Redistribute();
    AMREX_ASSERT(OK());
    return;
}
