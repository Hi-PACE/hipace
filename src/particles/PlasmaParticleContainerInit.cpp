#include "PlasmaParticleContainer.H"
#include "Constant.H"

using namespace amrex;

namespace
{
    AMREX_GPU_HOST_DEVICE void get_position_unit_cell(Real* r, const IntVect& nppc, int i_part)
    {
        int nx = nppc[0];
        int ny = nppc[1];
        int nz = nppc[2];

        int ix_part = i_part/(ny * nz);
        int iy_part = (i_part % (ny * nz)) % ny;
        int iz_part = (i_part % (ny * nz)) / ny;

        r[0] = (0.5+ix_part)/nx;
        r[1] = (0.5+iy_part)/ny;
        r[2] = (0.5+iz_part)/nz;
    }

    AMREX_GPU_HOST_DEVICE void get_gaussian_random_momentum(Real* u, Real u_mean, Real u_std) {
        Real ux_th = amrex::RandomNormal(0.0, u_std);
        Real uy_th = amrex::RandomNormal(0.0, u_std);
        Real uz_th = amrex::RandomNormal(0.0, u_std);

        u[0] = u_mean + ux_th;
        u[1] = u_mean + uy_th;
        u[2] = u_mean + uz_th;
    }
}

void
PlasmaParticleContainer::
InitParticles(const IntVect& a_num_particles_per_cell,
              const Real     a_thermal_momentum_std,
              const Real     a_thermal_momentum_mean,
              const Real     a_density,
              const Geometry& geom,
              /*const RealBox& a_bounds,*/
              const int      a_problem)
{
    RealBox a_bounds = geom.ProbDomain();

    BL_PROFILE("PlasmaParticleContainer::InitParticles");

    const int lev = 0;
    const auto dx = geom.CellSizeArray();
    const auto plo = geom.ProbLoArray();

    const int num_ppc = AMREX_D_TERM( a_num_particles_per_cell[0],
                                      *a_num_particles_per_cell[1],
                                      *a_num_particles_per_cell[2]);
    const Real scale_fac = dx[0]*dx[1]*dx[2]/num_ppc;

    for(MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi)
    {
        const Box& tile_box  = mfi.tilebox();

        const auto lo = amrex::lbound(tile_box);
        const auto hi = amrex::ubound(tile_box);

        Gpu::ManagedVector<unsigned int> counts(tile_box.numPts(), 0);
        unsigned int* pcount = counts.dataPtr();

        Gpu::ManagedVector<unsigned int> offsets(tile_box.numPts());
        unsigned int* poffset = offsets.dataPtr();

        amrex::ParallelFor(tile_box,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            for (int i_part=0; i_part<num_ppc;i_part++)
            {
                Real r[3];

                get_position_unit_cell(r, a_num_particles_per_cell, i_part);

                Real x = plo[0] + (i + r[0])*dx[0];
                Real y = plo[1] + (j + r[1])*dx[1];
                Real z = plo[2] + (k + r[2])*dx[2];

                if (x >= a_bounds.hi(0) || x < a_bounds.lo(0) ||
                    y >= a_bounds.hi(1) || y < a_bounds.lo(1) ||
                    z >= a_bounds.hi(2) || z < a_bounds.lo(2) ) continue;

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

        Gpu::inclusive_scan(counts.begin(), counts.end(), offsets.begin());

        int num_to_add = offsets[tile_box.numPts()-1];

        auto& particles = GetParticles(lev);
        auto& particle_tile = particles[std::make_pair(mfi.index(), mfi.LocalTileIndex())];

        auto old_size = particle_tile.GetArrayOfStructs().size();
        auto new_size = old_size + num_to_add;
        particle_tile.resize(new_size);

        if (num_to_add == 0) continue;

        ParticleType* pstruct = particle_tile.GetArrayOfStructs()().data();

        auto arrdata = particle_tile.GetStructOfArrays().realarray();

        int procID = ParallelDescriptor::MyProc();
        int pid = ParticleType::NextID();
        ParticleType::NextID(pid + num_to_add);

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
                Real r[3] = {0.,0.,0.};
                Real u[3] = {0.,0.,0.};

                get_position_unit_cell(r, a_num_particles_per_cell, i_part);

                Real x = plo[0] + (i + r[0])*dx[0];
                Real y = plo[1] + (j + r[1])*dx[1];
                Real z = plo[2] + (k + r[2])*dx[2];

                if (a_problem == 0) {
                    get_gaussian_random_momentum(u, a_thermal_momentum_mean,
                                                 a_thermal_momentum_std);
                }
                else if (a_problem == 1 ) {
                    u[0] = 0.01;
                    u[1] = 0.0;
                    u[2] = 0.0;
                } else {
                    amrex::Abort("problem type not valid");
                }

                if (x >= a_bounds.hi(0) || x < a_bounds.lo(0) ||
                    y >= a_bounds.hi(1) || y < a_bounds.lo(1) ||
                    z >= a_bounds.hi(2) || z < a_bounds.lo(2) ) continue;

                ParticleType& p = pstruct[pidx];
                p.id()   = pid + pidx;
                p.cpu()  = procID;
                p.pos(0) = x;
                p.pos(1) = y;
                p.pos(2) = z;

                arrdata[PlasmaIdx::w    ][pidx] = a_density * scale_fac;
                arrdata[PlasmaIdx::ux   ][pidx] = u[0] * PhysConst::c;
                arrdata[PlasmaIdx::uy   ][pidx] = u[1] * PhysConst::c;
                arrdata[PlasmaIdx::phi  ][pidx] = 0.;
                arrdata[PlasmaIdx::Fx1  ][pidx] = 1.;
                arrdata[PlasmaIdx::Fx2  ][pidx] = 2.;
                arrdata[PlasmaIdx::Fx3  ][pidx] = 3.;
                arrdata[PlasmaIdx::Fx4  ][pidx] = 4.;
                arrdata[PlasmaIdx::Fx5  ][pidx] = 5.;
                arrdata[PlasmaIdx::Fy1  ][pidx] = 11.;
                arrdata[PlasmaIdx::Fy2  ][pidx] = 12.;
                arrdata[PlasmaIdx::Fy3  ][pidx] = 13.;
                arrdata[PlasmaIdx::Fy4  ][pidx] = 14.;
                arrdata[PlasmaIdx::Fy5  ][pidx] = 15.;
                arrdata[PlasmaIdx::Fux1 ][pidx] = 21.;
                arrdata[PlasmaIdx::Fux2 ][pidx] = 22.;
                arrdata[PlasmaIdx::Fux3 ][pidx] = 23.;
                arrdata[PlasmaIdx::Fux4 ][pidx] = 24.;
                arrdata[PlasmaIdx::Fux5 ][pidx] = 25.;
                arrdata[PlasmaIdx::Fuy1 ][pidx] = 31.;
                arrdata[PlasmaIdx::Fuy2 ][pidx] = 32.;
                arrdata[PlasmaIdx::Fuy3 ][pidx] = 33.;
                arrdata[PlasmaIdx::Fuy4 ][pidx] = 34.;
                arrdata[PlasmaIdx::Fuy5 ][pidx] = 35.;
                arrdata[PlasmaIdx::Fphi1][pidx] = 41.;
                arrdata[PlasmaIdx::Fphi2][pidx] = 42.;
                arrdata[PlasmaIdx::Fphi3][pidx] = 43.;
                arrdata[PlasmaIdx::Fphi4][pidx] = 44.;
                arrdata[PlasmaIdx::Fphi5][pidx] = 45.;
                ++pidx;
            }
        });
    }
}
