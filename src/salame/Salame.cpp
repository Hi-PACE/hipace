/* Copyright 2022
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn
 * License: BSD-3-Clause-LBNL
 */
#include "Salame.H"
#include "particles/particles_utils/FieldGather.H"
#include "utils/GPUUtil.H"
#include "utils/HipaceProfilerWrapper.H"

void
SalameModule (Hipace* hipace, const int n_iter, const bool do_advance, int& last_islice,
              bool& overloaded, const int lev, const int step, const int islice,
              const int islice_local, const amrex::Vector<BeamBins>& beam_bin, const int ibox)
{
    HIPACE_PROFILE("SalameModule()");

    // always use the Ez field from before SALAME has started to avoid buildup of small errors
    if (islice + 1 != last_islice) {
        hipace->m_fields.duplicate(lev, WhichSlice::Salame, {"Ez_target"},
                                        WhichSlice::This, {"Ez"});
        last_islice = islice;
        overloaded = false;
    }

    for (int iter=0; iter<n_iter; ++iter) {

        // STEP 1: Calculate what Ez would be with the initial SALAME beam weight

        // advance plasma to the temp slice, only shift once
        hipace->m_multi_plasma.AdvanceParticles(hipace->m_fields, hipace->m_multi_laser, hipace->Geom(lev),
                                                true, true, true, iter==0, lev);

        hipace->m_fields.duplicate(lev, WhichSlice::Salame, {"jx", "jy"},
                                        WhichSlice::Next, {"jx_beam", "jy_beam"});

        // deposit plasma jx and jy on the next temp slice, to the SALANE slice
        hipace->m_multi_plasma.DepositCurrent(hipace->m_fields, hipace->m_multi_laser,
                WhichSlice::Salame, true, true, false, false, false, hipace->Geom(lev), lev);

        // use an initial guess of zero for Bx and By in MG solver to reduce relative error
        hipace->m_fields.setVal(0., lev, WhichSlice::Salame, "Ez", "jz_beam", "Sy", "Sx", "Bx", "By");

        hipace->m_fields.SolvePoissonEz(hipace->Geom(), lev, islice, WhichSlice::Salame);

        hipace->m_fields.duplicate(lev, WhichSlice::Salame, {"Ez_no_salame"},
                                        WhichSlice::Salame, {"Ez"});

        // STEP 2: Calculate the contribution to Ez from only the SALAME beam

        // deposit SALAME beam jz
        hipace->m_multi_beam.DepositCurrentSlice(hipace->m_fields, hipace->Geom(), lev, step,
            islice_local, beam_bin, hipace->m_box_sorters, ibox, false, true, false, WhichSlice::Salame);

        SalameInitializeSxSyWithBeam(hipace, lev);

        hipace->ExplicitMGSolveBxBy(lev, WhichSlice::Salame);

        hipace->m_fields.setVal(0., lev, WhichSlice::Salame, "Ez", "jx", "jy");

        // get jx jy (SALAME only) on the next slice using Bx By (SALAME only) on this slice
        if (do_advance) {
            SalameOnlyAdvancePlasma(hipace, lev);

            hipace->m_multi_plasma.DepositCurrent(hipace->m_fields, hipace->m_multi_laser,
                    WhichSlice::Salame, true, true, false, false, false, hipace->Geom(lev), lev);
        } else {
            SalameGetJxJyFromBxBy(hipace, lev);
        }

        // necessary after push to temp slice
        hipace->m_multi_plasma.ResetParticles(lev);

        hipace->m_fields.SolvePoissonEz(hipace->Geom(), lev, islice, WhichSlice::Salame);

        // STEP 3: find ideal weighting factor of the SALAME beam using the computed Ez fields,
        // and update the beam with it

        // W = sum(jz * (Ez_target - Ez_no_salame) / Ez_only_salame) / sum(jz) + 1
        // + 1 because Ez_only_salame already includes the SALAME beam with a weight of 1
        // W_total = W * sum(jz)
        auto [W, W_total] = SalameGetW(hipace, lev);

        if (W < 0 || overloaded) {
            W = 0;
            W_total = 0;
            amrex::Print() << "Salame beam is overloaded, setting weight to zero\n";
            iter = n_iter-1; // this is the last iteration
            overloaded = true;
        }

        amrex::Print() << "Salame weight factor on slice " << islice << " is " << W
                       << " Total weight is " << W_total << '\n';

        SalameMultiplyBeamWeight(W, hipace, islice_local, beam_bin, ibox);

        // STEP 4: recompute Bx and By with the new SALAME beam weight.
        // This is done a bit overkill by depositing again. A linear combination of the available
        // B-fields would be sufficient but might have some numerical differences.

        hipace->m_fields.setVal(0., lev, WhichSlice::This, "jz_beam", "Sy", "Sx");

        // deposit beam jz
        hipace->m_multi_beam.DepositCurrentSlice(hipace->m_fields, hipace->Geom(), lev, step,
            islice_local, beam_bin, hipace->m_box_sorters, ibox, false, true, false, WhichSlice::This);

        hipace->m_grid_current.DepositCurrentSlice(hipace->m_fields, hipace->Geom(lev), lev, islice);

        hipace->InitializeSxSyWithBeam(lev);

        hipace->m_multi_plasma.ExplicitDeposition(hipace->m_fields, hipace->m_multi_laser,
                                                  hipace->Geom(lev), lev);

        hipace->ExplicitMGSolveBxBy(lev, WhichSlice::This);
    }
}


void
SalameInitializeSxSyWithBeam (Hipace* hipace, const int lev)
{
    using namespace amrex::literals;

    amrex::MultiFab& slicemf = hipace->m_fields.getSlices(lev, WhichSlice::Salame);

    const amrex::Real dx = hipace->Geom(lev).CellSize(Direction::x);
    const amrex::Real dy = hipace->Geom(lev).CellSize(Direction::y);

    for ( amrex::MFIter mfi(slicemf, DfltMfiTlng); mfi.isValid(); ++mfi ){

        Array3<amrex::Real> const arr = slicemf.array(mfi);

        const int Sx = Comps[WhichSlice::Salame]["Sx"];
        const int Sy = Comps[WhichSlice::Salame]["Sy"];
        const int jzb = Comps[WhichSlice::Salame]["jz_beam"];

        const amrex::Real mu0 = hipace->m_phys_const.mu0;

        amrex::ParallelFor(mfi.tilebox(),
            [=] AMREX_GPU_DEVICE (int i, int j, int) noexcept
            {
                const amrex::Real dx_jzb = (arr(i+1,j,jzb)-arr(i-1,j,jzb))/(2._rt*dx);
                const amrex::Real dy_jzb = (arr(i,j+1,jzb)-arr(i,j-1,jzb))/(2._rt*dy);

                // same Hipace::InitializeSxSyWithBeam just with only the salame beam
                // and without transverse currents

                arr(i,j,Sy) =   mu0 * ( - dy_jzb);
                arr(i,j,Sx) = - mu0 * ( - dx_jzb);
            });
    }
}


void
SalameGetJxJyFromBxBy (Hipace* hipace, const int lev)
{
    using namespace amrex::literals;

    amrex::MultiFab& salame_slicemf = hipace->m_fields.getSlices(lev, WhichSlice::Salame);
    amrex::MultiFab& slicemf = hipace->m_fields.getSlices(lev, WhichSlice::This);

    // first 5. order adams bashforth coefficient, same as in plasma pusher
    // Analytically the coefficient should be 2 but 1901/720 gives much better results
    // due to the plasma pusher used in Hipace
    const amrex::Real a1_times_dz = ( 1901._rt / 720._rt ) * hipace->Geom(lev).CellSize(Direction::z);

    for ( amrex::MFIter mfi(salame_slicemf, DfltMfiTlng); mfi.isValid(); ++mfi ){

        Array3<amrex::Real> const salame_arr = salame_slicemf.array(mfi);
        Array2<const amrex::Real> const chi_arr = slicemf.const_array(mfi, Comps[WhichSlice::This]["chi"]);

        const int Bx = Comps[WhichSlice::Salame]["Bx"];
        const int By = Comps[WhichSlice::Salame]["By"];
        const int jx = Comps[WhichSlice::Salame]["jx"];
        const int jy = Comps[WhichSlice::Salame]["jy"];

        const amrex::Real mu0 = hipace->m_phys_const.mu0;

        amrex::ParallelFor(mfi.tilebox(),
            [=] AMREX_GPU_DEVICE (int i, int j, int) noexcept
            {
                salame_arr(i,j,jx) =  a1_times_dz * chi_arr(i,j) * salame_arr(i,j,By) / mu0;
                salame_arr(i,j,jy) = -a1_times_dz * chi_arr(i,j) * salame_arr(i,j,Bx) / mu0;
            });
    }
}

void
SalameOnlyAdvancePlasma (Hipace* hipace, const int lev)
{
    using namespace amrex::literals;

    for (int i=0; i<hipace->m_multi_plasma.GetNPlasmas(); i++) {
        auto& plasma = hipace->m_multi_plasma.m_all_plasmas[i];
        auto& bins = hipace->m_multi_plasma.m_all_bins[i];

        if (plasma.m_level != lev) return;

        const auto gm = hipace->Geom(lev);
        const bool do_tiling = Hipace::m_do_tiling;

        amrex::Real const * AMREX_RESTRICT dx = gm.CellSize();

        for (PlasmaParticleIterator pti(plasma, lev); pti.isValid(); ++pti)
        {
            const amrex::FArrayBox& slice_fab = hipace->m_fields.getSlices(lev, WhichSlice::Salame)[pti];
            Array3<const amrex::Real> const slice_arr = slice_fab.const_array();
            const int bx_comp = Comps[WhichSlice::Salame]["Bx"];
            const int by_comp = Comps[WhichSlice::Salame]["By"];

            const amrex::Real dx_inv = 1._rt/dx[0];
            const amrex::Real dy_inv = 1._rt/dx[1];
            const amrex::Real dz = dx[2];

            // Offset for converting positions to indexes
            amrex::Real const x_pos_offset = GetPosOffset(0, gm, slice_fab.box());
            const amrex::Real y_pos_offset = GetPosOffset(1, gm, slice_fab.box());

            auto& soa = pti.GetStructOfArrays();

            amrex::Real * const x_prev = soa.GetRealData(PlasmaIdx::x_prev).data();
            amrex::Real * const y_prev = soa.GetRealData(PlasmaIdx::y_prev).data();
            amrex::Real * const ux_temp = soa.GetRealData(PlasmaIdx::ux_temp).data();
            amrex::Real * const uy_temp = soa.GetRealData(PlasmaIdx::uy_temp).data();
            int * const ion_lev = soa.GetIntData(PlasmaIdx::ion_lev).data();

            const amrex::Real charge_mass_ratio = plasma.m_charge / plasma.m_mass;
            const bool can_ionize = plasma.m_can_ionize;

            const int ntiles = do_tiling ? bins.numBins() : 1;

#ifdef AMREX_USE_OMP
#pragma omp parallel for if (amrex::Gpu::notInLaunchRegion())
#endif
            for (int itile=0; itile<ntiles; itile++){
                BeamBins::index_type const * const indices =
                    do_tiling ? bins.permutationPtr() : nullptr;
                BeamBins::index_type const * const offsets =
                    do_tiling ? bins.offsetsPtr() : nullptr;
                int const num_particles =
                    do_tiling ? offsets[itile+1]-offsets[itile] : pti.numParticles();
                amrex::ParallelFor(
                    amrex::TypeList<amrex::CompileTimeOptions<0, 1, 2, 3>>{},
                    {Hipace::m_depos_order_xy},
                    num_particles,
                    [=] AMREX_GPU_DEVICE (long idx, auto depos_order) {
                        const int ip = do_tiling ? indices[offsets[itile]+idx] : idx;
                        const amrex::Real xp = x_prev[ip];
                        const amrex::Real yp = y_prev[ip];

                        amrex::Real Bxp = 0._rt;
                        amrex::Real Byp = 0._rt;

                        // Gather Bx and By along with 4 dummy fields to use this funciton
                        doBxByGatherShapeN<depos_order.value>(xp, yp, Bxp, Byp, slice_arr,
                            bx_comp, by_comp, dx_inv, dy_inv, x_pos_offset, y_pos_offset);

                        const amrex::Real q_mass_ratio = can_ionize ?
                            ion_lev[ip] * charge_mass_ratio : charge_mass_ratio;

                        const amrex::Real a1_times_dz = ( 1901._rt / 720._rt ) * dz;

                        ux_temp[ip] =  a1_times_dz * q_mass_ratio * Byp;
                        uy_temp[ip] = -a1_times_dz * q_mass_ratio * Bxp;
                    });
            }
        }
    }
}

std::pair<amrex::Real, amrex::Real>
SalameGetW (Hipace* hipace, const int lev)
{
    using namespace amrex::literals;

    amrex::Real sum_W = 0._rt;
    amrex::Real sum_jz = 0._rt;

    amrex::MultiFab& slicemf = hipace->m_fields.getSlices(lev, WhichSlice::Salame);

    for ( amrex::MFIter mfi(slicemf, DfltMfiTlng); mfi.isValid(); ++mfi ){
        amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum> reduce_op;
        amrex::ReduceData<amrex::Real, amrex::Real> reduce_data(reduce_op);
        using ReduceTuple = typename decltype(reduce_data)::Type;
        Array3<amrex::Real> const arr = slicemf.array(mfi);

        const int Ez = Comps[WhichSlice::Salame]["Ez"];
        const int Ez_target = Comps[WhichSlice::Salame]["Ez_target"];
        const int Ez_no_salame = Comps[WhichSlice::Salame]["Ez_no_salame"];
        const int jz = Comps[WhichSlice::Salame]["jz_beam"];

        reduce_op.eval(mfi.tilebox(), reduce_data,
            [=] AMREX_GPU_DEVICE (int i, int j, int) noexcept -> ReduceTuple
            {
                // avoid NaN
                if (arr(i,j,Ez) == 0._rt) return {0._rt, 0._rt};
                return {
                    arr(i,j,jz) * (arr(i,j,Ez_target) - arr(i,j,Ez_no_salame)) / arr(i,j,Ez),
                    arr(i,j,jz)
                };
            });
        auto res = reduce_data.value(reduce_op);
        sum_W += amrex::get<0>(res);
        sum_jz += amrex::get<1>(res);
    }
    // first return = sum(jz * (Ez_target - Ez_no_salame) / Ez_only_salame) / sum(jz) + 1
    // + 1 because Ez_only_salame already includes the SALAME beam with a weight of 1
    // second return = first return * sum(jz)
    return {(sum_W / sum_jz) + 1._rt,  sum_W + sum_jz};
}

void
SalameMultiplyBeamWeight (const amrex::Real W, Hipace* hipace, const int islice,
                          const amrex::Vector<BeamBins>& bins, const int ibox)
{
    for (int i=0; i<(hipace->m_multi_beam.get_nbeams()); i++) {
        auto& beam = hipace->m_multi_beam.getBeam(i);

        if (!beam.m_do_salame) continue;

        const int box_offset = hipace->m_box_sorters[i].boxOffsetsPtr()[ibox];

        auto& aos = beam.GetArrayOfStructs(); // For id
        auto pos_structs = aos.begin() + box_offset;
        auto& soa = beam.GetStructOfArrays(); // For momenta and weights
        amrex::Real * const wp = soa.GetRealData(BeamIdx::w).data() + box_offset;

        BeamBins::index_type const * const indices = bins[i].permutationPtr();
        BeamBins::index_type const * const offsets = bins[i].offsetsPtrCpu();

        BeamBins::index_type cell_start = offsets[islice];
        BeamBins::index_type cell_stop = offsets[islice+1];

        int const num_particles = cell_stop-cell_start;

        amrex::ParallelFor(
            num_particles,
            [=] AMREX_GPU_DEVICE (long idx) {
                // Particles in the same slice must be accessed through the bin sorter
                const int ip = indices[cell_start+idx];
                // Skip invalid particles and ghost particles not in the last slice
                if (pos_structs[ip].id() < 0) return;

                // invalidate particles with a weight of zero
                if (W == 0) {
                    pos_structs[ip].id() = -pos_structs[ip].id();
                    return;
                }

                // Multiply SALAME beam particles on this slice with W
                wp[ip] *= W;
            });
    }
}
