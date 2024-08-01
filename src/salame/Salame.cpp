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
              bool& overloaded, const int current_N_level, const int step, const int islice,
              const amrex::Real relative_tolerance)
{
    HIPACE_PROFILE("SalameModule()");

    // always use the Ez field from before SALAME has started to avoid buildup of small errors
    if (islice + 1 != last_islice) {
        for (int lev=0; lev<current_N_level; ++lev) {
            hipace->m_fields.duplicate(lev, WhichSlice::Salame, {"Ez_target"},
                                            WhichSlice::This, {"Ez"});
        }
        overloaded = false;
        hipace->m_salame_zeta_initial = islice * hipace->m_3D_geom[0].CellSize(2) +
            GetPosOffset(2, hipace->m_3D_geom[0], hipace->m_3D_geom[0].Domain());
    }
    last_islice = islice;

    for (int lev=0; lev<current_N_level; ++lev) {
        hipace->m_fields.setVal(0., lev, WhichSlice::This, "Sy", "Sx");
        hipace->m_multi_plasma.ExplicitDeposition(hipace->m_fields, hipace->m_3D_geom, lev);

        // Back up Sx and Sy from the plasma only. This can only be done before the plasma push
        hipace->m_fields.duplicate(lev, WhichSlice::Salame, {"Sy_back", "Sx_back"},
                                        WhichSlice::This, {"Sy", "Sx"});
    }

    for (int iter=0; iter<n_iter; ++iter) {

        // STEP 1: Calculate what Ez would be with the initial SALAME beam weight

        for (int lev=0; lev<current_N_level; ++lev) {
            // advance plasma to the temp slice
            hipace->m_multi_plasma.AdvanceParticles(hipace->m_fields, hipace->m_3D_geom, true, lev);

            hipace->m_fields.duplicate(lev, WhichSlice::Salame, {"jx", "jy"},
                                            WhichSlice::Next, {"jx_beam", "jy_beam"});
        }

        if (hipace->m_N_level > 1) {
            // tag to temp slice for deposition
            hipace->m_multi_plasma.TagByLevel(current_N_level, hipace->m_3D_geom);
        }

        for (int lev=0; lev<current_N_level; ++lev) {
            if (hipace->m_do_tiling) {
                hipace->m_multi_plasma.TileSort(
                    hipace->m_slice_geom[lev].Domain(), hipace->m_slice_geom[lev]);
            }

            // deposit plasma jx and jy on the next temp slice, to the SALAME slice
            hipace->m_multi_plasma.DepositCurrent(hipace->m_fields,
                    WhichSlice::Salame, true, false, false, false, false, hipace->m_3D_geom, lev);

            // use an initial guess of zero for Bx and By in MG solver to reduce relative error
            hipace->m_fields.setVal(0., lev, WhichSlice::Salame,
                "Ez", "jz_beam", "Sy", "Sx", "Bx", "By");
        }

        hipace->m_fields.SolvePoissonEz(hipace->m_3D_geom, current_N_level, WhichSlice::Salame);

        for (int lev=0; lev<current_N_level; ++lev) {
            hipace->m_fields.duplicate(lev, WhichSlice::Salame, {"Ez_no_salame"},
                                            WhichSlice::Salame, {"Ez"});
        }

        // STEP 2: Calculate the contribution to Ez from only the SALAME beam

        for (int lev=0; lev<current_N_level; ++lev) {
            // deposit SALAME beam jz
            hipace->m_multi_beam.DepositCurrentSlice(hipace->m_fields, hipace->m_3D_geom, lev, step,
                false, true, false, WhichSlice::Salame, WhichBeamSlice::This);
        }

        for (int lev=0; lev<current_N_level; ++lev) {
            SalameInitializeSxSyWithBeam(hipace, lev);
        }

        for (int lev=0; lev<current_N_level; ++lev) {
            hipace->ExplicitMGSolveBxBy(lev, WhichSlice::Salame);
        }

        for (int lev=0; lev<current_N_level; ++lev) {
            hipace->m_fields.setVal(0., lev, WhichSlice::Salame, "Ez", "jx", "jy");
        }

        // get jx jy (SALAME only) on the next slice using Bx By (SALAME only) on this slice
        if (do_advance) {
            if (hipace->m_N_level > 1) {
                // tag to prev slice for ux uy push
                hipace->m_multi_plasma.TagByLevel(current_N_level, hipace->m_3D_geom, true);
            }

            for (int lev=0; lev<current_N_level; ++lev) {
                SalameOnlyAdvancePlasma(hipace, lev);
            }

            if (hipace->m_N_level > 1) {
                // tag to temp slice for deposition
                hipace->m_multi_plasma.TagByLevel(current_N_level, hipace->m_3D_geom);
            }

            for (int lev=0; lev<current_N_level; ++lev) {
                if (hipace->m_do_tiling) {
                    hipace->m_multi_plasma.TileSort(
                        hipace->m_slice_geom[lev].Domain(), hipace->m_slice_geom[lev]);
                }

                hipace->m_multi_plasma.DepositCurrent(hipace->m_fields,
                    WhichSlice::Salame, true, false, false, false, false, hipace->m_3D_geom, lev);
            }
        } else {
            for (int lev=0; lev<current_N_level; ++lev) {
                SalameGetJxJyFromBxBy(hipace, lev);
            }
        }

        hipace->m_fields.SolvePoissonEz(hipace->m_3D_geom, current_N_level, WhichSlice::Salame);

        // STEP 3: find ideal weighting factor of the SALAME beam using the computed Ez fields,
        // and update the beam with it

        for (int lev=0; lev<current_N_level; ++lev) {
            hipace->m_fields.setVal(0., lev, WhichSlice::Salame, "jz_beam");
            // deposit SALAME beam jz only on the highest level of each particle for SalameGetW,
            // since the most accurate field (on the highest level) is supposed to be flattened
            hipace->m_multi_beam.DepositCurrentSlice(hipace->m_fields, hipace->m_3D_geom, lev, step,
                false, true, false, WhichSlice::Salame, WhichBeamSlice::This, true);
        }

        // W = (Ez_target - Ez_no_salame) / Ez_only_salame + 1
        // + 1 because Ez_no_salame already includes the SALAME beam with a weight of 1
        // W_total = W * sum(jz)
        auto [W, W_total] = SalameGetW(hipace, current_N_level, islice);

        if (W < 0 || overloaded) {
            W = 0;
            W_total = 0;
            amrex::Print() << "Salame beam is overloaded, setting weight to zero\n";
            iter = n_iter-1; // this is the last iteration
            overloaded = true;
        }

        amrex::Print() << "Salame weight factor on slice " << islice << " is " << W
                       << " Total weight is " << W_total;

        if (!overloaded && iter >= 1 && std::abs(W - 1.) < relative_tolerance) {
            // SALAME is converged
            iter = n_iter-1; // this is the last iteration
            amrex::Print() << " (converged)";
        }

        amrex::Print() << '\n';

        SalameMultiplyBeamWeight(W, hipace);

        // STEP 4: recompute Bx and By with the new SALAME beam weight.
        // This is done a bit overkill by depositing again. A linear combination of the available
        // B-fields would be sufficient but might have some numerical differences.

        for (int lev=0; lev<current_N_level; ++lev) {
            hipace->m_fields.setVal(0., lev, WhichSlice::This, "jz_beam", "Sy", "Sx");

            // deposit beam jz
            hipace->m_multi_beam.DepositCurrentSlice(hipace->m_fields, hipace->m_3D_geom, lev, step,
                false, true, false, WhichSlice::This, WhichBeamSlice::This);

            hipace->m_grid_current.DepositCurrentSlice(hipace->m_fields, hipace->m_3D_geom[lev], lev, islice);
        }

        for (int lev=0; lev<current_N_level; ++lev) {
            hipace->InitializeSxSyWithBeam(lev);

            // add result of explicit deposition
            hipace->m_fields.add(lev, WhichSlice::This, {"Sy", "Sx"},
                                    WhichSlice::Salame, {"Sy_back", "Sx_back"});

            hipace->ExplicitMGSolveBxBy(lev, WhichSlice::This);
        }
    }

    if (hipace->m_N_level > 1) {
        // tag to prev slice for push
        hipace->m_multi_plasma.TagByLevel(current_N_level, hipace->m_3D_geom, true);
    }
}


void
SalameInitializeSxSyWithBeam (Hipace* hipace, const int lev)
{
    using namespace amrex::literals;

    amrex::MultiFab& slicemf = hipace->m_fields.getSlices(lev);

    const amrex::Real dxih = 0.5_rt*hipace->m_3D_geom[lev].InvCellSize(Direction::x);
    const amrex::Real dyih = 0.5_rt*hipace->m_3D_geom[lev].InvCellSize(Direction::y);

    for ( amrex::MFIter mfi(slicemf, DfltMfiTlng); mfi.isValid(); ++mfi ){

        Array3<amrex::Real> const arr = slicemf.array(mfi);

        const int Sx = Comps[WhichSlice::Salame]["Sx"];
        const int Sy = Comps[WhichSlice::Salame]["Sy"];
        const int jzb = Comps[WhichSlice::Salame]["jz_beam"];

        const amrex::Real mu0 = hipace->m_phys_const.mu0;

        amrex::ParallelFor(mfi.tilebox(),
            [=] AMREX_GPU_DEVICE (int i, int j, int) noexcept
            {
                const amrex::Real dx_jzb = (arr(i+1,j,jzb)-arr(i-1,j,jzb))*dxih;
                const amrex::Real dy_jzb = (arr(i,j+1,jzb)-arr(i,j-1,jzb))*dyih;

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

    amrex::MultiFab& slicemf = hipace->m_fields.getSlices(lev);

#ifdef HIPACE_USE_AB5_PUSH
    const amrex::Real dz = ( 1901._rt / 720._rt ) * hipace->m_3D_geom[lev].CellSize(Direction::z);
#else
    const amrex::Real dz = 1.5_rt * hipace->m_3D_geom[lev].CellSize(Direction::z);
#endif

    for ( amrex::MFIter mfi(slicemf, DfltMfiTlng); mfi.isValid(); ++mfi ){

        Array3<amrex::Real> const arr = slicemf.array(mfi);

        const int Bx = Comps[WhichSlice::Salame]["Bx"];
        const int By = Comps[WhichSlice::Salame]["By"];
        const int jx = Comps[WhichSlice::Salame]["jx"];
        const int jy = Comps[WhichSlice::Salame]["jy"];
        const int chi = Comps[WhichSlice::This]["chi"];

        const amrex::Real mu0_inv = 1._rt / hipace->m_phys_const.mu0;

        amrex::ParallelFor(mfi.tilebox(),
            [=] AMREX_GPU_DEVICE (int i, int j, int) noexcept
            {
                arr(i,j,jx) =  dz * arr(i,j,chi) * arr(i,j,By) * mu0_inv;
                arr(i,j,jy) = -dz * arr(i,j,chi) * arr(i,j,Bx) * mu0_inv;
            });
    }
}

void
SalameOnlyAdvancePlasma (Hipace* hipace, const int lev)
{
    using namespace amrex::literals;

    for (int i=0; i<hipace->m_multi_plasma.GetNPlasmas(); i++) {
        auto& plasma = hipace->m_multi_plasma.m_all_plasmas[i];

        const auto gm = hipace->m_3D_geom[lev];

        for (PlasmaParticleIterator pti(plasma); pti.isValid(); ++pti)
        {
            const amrex::FArrayBox& slice_fab = hipace->m_fields.getSlices(lev)[pti];
            Array3<const amrex::Real> const slice_arr = slice_fab.const_array();
            const int bx_comp = Comps[WhichSlice::Salame]["Bx"];
            const int by_comp = Comps[WhichSlice::Salame]["By"];

            const amrex::Real dx_inv = gm.InvCellSize(0);
            const amrex::Real dy_inv = gm.InvCellSize(1);
            const amrex::Real dz = gm.CellSize(2);

            // Offset for converting positions to indexes
            amrex::Real const x_pos_offset = GetPosOffset(0, gm, slice_fab.box());
            const amrex::Real y_pos_offset = GetPosOffset(1, gm, slice_fab.box());

            const auto ptd = pti.GetParticleTile().getParticleTileData();

            const amrex::Real charge_mass_ratio = plasma.m_charge / plasma.m_mass;
            const bool can_ionize = plasma.m_can_ionize;

#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
            {
                amrex::Long const num_particles = pti.numParticles();
#ifdef AMREX_USE_OMP
                amrex::Long const idx_begin = (num_particles * omp_get_thread_num()) / omp_get_num_threads();
                amrex::Long const idx_end = (num_particles * (omp_get_thread_num()+1)) / omp_get_num_threads();
#else
                amrex::Long constexpr idx_begin = 0;
                amrex::Long const idx_end = num_particles;
#endif

                amrex::ParallelFor(
                    amrex::TypeList<amrex::CompileTimeOptions<0, 1, 2, 3>>{},
                    {Hipace::m_depos_order_xy},
                    int(idx_end - idx_begin),
                    [=] AMREX_GPU_DEVICE (long idx, auto depos_order) {
                        const int ip = idx + idx_begin;
                        // only push plasma particles on their according MR level
                        if (!ptd.id(ip).is_valid() || ptd.cpu(ip) != lev) return;

                        const amrex::Real xp = ptd.rdata(PlasmaIdx::x_prev)[ip];
                        const amrex::Real yp = ptd.rdata(PlasmaIdx::y_prev)[ip];

                        amrex::Real Bxp = 0._rt;
                        amrex::Real Byp = 0._rt;

                        // Gather Bx and By
                        doBxByGatherShapeN<depos_order.value>(xp, yp, Bxp, Byp, slice_arr,
                            bx_comp, by_comp, dx_inv, dy_inv, x_pos_offset, y_pos_offset);

                        const amrex::Real q_mass_ratio = can_ionize ?
                            ptd.idata(PlasmaIdx::ion_lev)[ip] * charge_mass_ratio
                            : charge_mass_ratio;

#ifdef HIPACE_USE_AB5_PUSH
                        ptd.rdata(PlasmaIdx::ux)[ip] =  ( 1901._rt / 720._rt )*dz * q_mass_ratio * Byp;
                        ptd.rdata(PlasmaIdx::uy)[ip] = -( 1901._rt / 720._rt )*dz * q_mass_ratio * Bxp;
#else
                        ptd.rdata(PlasmaIdx::ux)[ip] =  1.5_rt*dz * q_mass_ratio * Byp;
                        ptd.rdata(PlasmaIdx::uy)[ip] = -1.5_rt*dz * q_mass_ratio * Bxp;
#endif
                    });
            }
        }
    }
}

std::pair<amrex::Real, amrex::Real>
SalameGetW (Hipace* hipace, const int current_N_level, const int islice)
{
    using namespace amrex::literals;

    amrex::Real sum_Ez_target = 0._rt;
    amrex::Real sum_Ez_no_salame = 0._rt;
    amrex::Real sum_Ez_only_salame = 0._rt;
    amrex::Real sum_jz = 0._rt;

    for (int lev=0; lev<current_N_level; ++lev) {

        amrex::MultiFab& slicemf = hipace->m_fields.getSlices(lev);

        for ( amrex::MFIter mfi(slicemf, DfltMfiTlng); mfi.isValid(); ++mfi ){
            amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum,
                            amrex::ReduceOpSum, amrex::ReduceOpSum> reduce_op;
            amrex::ReduceData<amrex::Real, amrex::Real,
                            amrex::Real, amrex::Real> reduce_data(reduce_op);
            using ReduceTuple = typename decltype(reduce_data)::Type;
            Array3<amrex::Real> const arr = slicemf.array(mfi);

            const int Ez = Comps[WhichSlice::Salame]["Ez"];
            const int Ez_target = Comps[WhichSlice::Salame]["Ez_target"];
            const int Ez_no_salame = Comps[WhichSlice::Salame]["Ez_no_salame"];
            const int jz = Comps[WhichSlice::Salame]["jz_beam"];

            // factor to account for different cell size with MR
            const amrex::Real factor =
                hipace->m_3D_geom[lev].CellSize(0) * hipace->m_3D_geom[lev].CellSize(1)
                / (hipace->m_3D_geom[0].CellSize(0) * hipace->m_3D_geom[0].CellSize(1));

            reduce_op.eval(mfi.tilebox(), reduce_data,
                [=] AMREX_GPU_DEVICE (int i, int j, int) noexcept -> ReduceTuple
                {
                    return {
                        factor * arr(i,j,jz) * arr(i,j,Ez_target),
                        factor * arr(i,j,jz) * arr(i,j,Ez_no_salame),
                        factor * arr(i,j,jz) * arr(i,j,Ez),
                        factor * arr(i,j,jz)
                    };
                });
            auto res = reduce_data.value(reduce_op);
            sum_Ez_target += amrex::get<0>(res);
            sum_Ez_no_salame += amrex::get<1>(res);
            sum_Ez_only_salame += amrex::get<2>(res);
            sum_jz += amrex::get<3>(res);
        }
    }

    sum_Ez_target /= sum_jz;
    sum_Ez_no_salame /= sum_jz;
    sum_Ez_only_salame /= sum_jz;

    // - 1 because this is for the Ez field of the next slice
    const amrex::Real zeta = (islice-1) * hipace->m_3D_geom[0].CellSize(2) +
                             GetPosOffset(2, hipace->m_3D_geom[0], hipace->m_3D_geom[0].Domain());
    // update target with user function
    sum_Ez_target = hipace->m_salame_target_func(
                        zeta,  hipace->m_salame_zeta_initial, sum_Ez_target);

    // + 1 because sum_Ez_no_salame already includes the SALAME beam with a weight of 1
    amrex::Real W = (sum_Ez_target - sum_Ez_no_salame)/sum_Ez_only_salame + 1._rt;
    return {W,  W * sum_jz};
}

void
SalameMultiplyBeamWeight (const amrex::Real W, Hipace* hipace)
{
    for (int i=0; i<(hipace->m_multi_beam.get_nbeams()); i++) {
        auto& beam = hipace->m_multi_beam.getBeam(i);

        if (!beam.m_do_salame) continue;

        // For id and weights
        auto& soa = beam.getBeamSlice(WhichBeamSlice::This).GetStructOfArrays();
        amrex::Real * const wp = soa.GetRealData(BeamIdx::w).data();
        auto * const idcpup = soa.GetIdCPUData().data();

        amrex::ParallelFor(
            beam.getNumParticles(WhichBeamSlice::This),
            [=] AMREX_GPU_DEVICE (long ip) {
                // Skip invalid particles and ghost particles not in the last slice
                auto id = amrex::ParticleIDWrapper(idcpup[ip]);
                if (!id.is_valid()) return;

                // invalidate particles with a weight of zero
                if (W == 0) {
                    id.make_invalid();
                    return;
                }

                // Multiply SALAME beam particles on this slice with W
                wp[ip] *= W;
            });
    }
}
