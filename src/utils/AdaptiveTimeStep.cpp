/* Copyright 2020-2021
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn, MaxThevenet, Severin Diederichs
 * License: BSD-3-Clause-LBNL
 */
#include "AdaptiveTimeStep.H"
#include "utils/DeprecatedInput.H"
#include "particles/pusher/GetAndSetPosition.H"
#include "particles/particles_utils/FieldGather.H"
#include "Hipace.H"
#include "HipaceProfilerWrapper.H"
#include "Constants.H"

/** \brief describes which double is used for the adaptive time step */
struct WhichDouble {
    enum Comp { MinUz=0, MinAcc, SumWeights, SumWeightsTimesUz, SumWeightsTimesUzSquared, N };
};

AdaptiveTimeStep::AdaptiveTimeStep (const int nbeams)
{
    amrex::ParmParse ppa("hipace");
    std::string str_dt = "";
    queryWithParser(ppa, "dt", str_dt);
    if (str_dt == "adaptive"){
        m_do_adaptive_time_step = true;
        queryWithParser(ppa, "nt_per_betatron", m_nt_per_betatron);
        queryWithParser(ppa, "dt_max", m_dt_max);
        queryWithParser(ppa, "adaptive_threshold_uz", m_threshold_uz);
        queryWithParser(ppa, "adaptive_phase_tolerance", m_adaptive_phase_tolerance);
        queryWithParser(ppa, "adaptive_predict_step", m_adaptive_predict_step);
        queryWithParser(ppa, "adaptive_control_phase_advance", m_adaptive_control_phase_advance);
        queryWithParser(ppa, "adaptive_phase_substeps", m_adaptive_phase_substeps);
        queryWithParser(ppa, "adaptive_gather_ez", m_adaptive_gather_ez);
    }
    DeprecatedInput("hipace", "do_adaptive_time_step", "dt = adaptive");

    if (m_adaptive_gather_ez) {
        amrex::Print()<<"WARNING: hipace.adaptive_gather_ez = 1 is buggy and NOT recommended";
    }

    // create time step data container per beam
    m_timestep_data.resize(nbeams);
    for (int ibeam = 0; ibeam < nbeams; ibeam++) {
        m_timestep_data[ibeam].resize(WhichDouble::N);
        m_timestep_data[ibeam][WhichDouble::MinUz] = 1e30;
        m_timestep_data[ibeam][WhichDouble::MinAcc] = 0.;
        m_timestep_data[ibeam][WhichDouble::SumWeights] = 0.;
        m_timestep_data[ibeam][WhichDouble::SumWeightsTimesUz] = 0.;
        m_timestep_data[ibeam][WhichDouble::SumWeightsTimesUzSquared] = 0.;
    }

    m_nbeams = nbeams;
}


void
AdaptiveTimeStep::BroadcastTimeStep (amrex::Real& dt)
{
#ifdef AMREX_USE_MPI
    if (!m_do_adaptive_time_step) return;

    // Broadcast time step
    MPI_Bcast(&dt,
              1,
              amrex::ParallelDescriptor::Mpi_typemap<amrex::Real>::type(),
              Hipace::HeadRankID(),
              amrex::ParallelDescriptor::Communicator());

    // Broadcast minimum uz
    MPI_Bcast(&m_min_uz_mq,
              1,
              amrex::ParallelDescriptor::Mpi_typemap<amrex::Real>::type(),
              Hipace::HeadRankID(),
              amrex::ParallelDescriptor::Communicator());
#else
    amrex::ignore_unused(dt);
#endif
}

void
AdaptiveTimeStep::GatherMinUzSlice (MultiBeam& beams, const bool initial)
{
    using namespace amrex::literals;

    if (!m_do_adaptive_time_step) return;

    HIPACE_PROFILE("AdaptiveTimeStep::GatherMinUzSlice()");

    const PhysConst phys_const = get_phys_const();
    const amrex::Real clightinv = 1._rt/phys_const.c;

    const int nbeams = beams.get_nbeams();

    for (int ibeam = 0; ibeam < nbeams; ibeam++) {
        auto& beam = beams.getBeam(ibeam);

        if (initial && beam.m_injection_type != "from_file") {
            // estimate values before the beam is initialized
            m_timestep_data[ibeam][WhichDouble::SumWeights] = 1._rt;
            m_timestep_data[ibeam][WhichDouble::SumWeightsTimesUz] =
                beam.m_get_momentum.m_u_mean[2];
            m_timestep_data[ibeam][WhichDouble::SumWeightsTimesUzSquared] =
                beam.m_get_momentum.m_u_mean[2] * beam.m_get_momentum.m_u_mean[2] +
                beam.m_get_momentum.m_u_std[2] * beam.m_get_momentum.m_u_std[2];
            m_timestep_data[ibeam][WhichDouble::MinUz] =
                beam.m_get_momentum.m_u_mean[2] - 4._rt * beam.m_get_momentum.m_u_std[2];
            continue;
        }

        unsigned long long num_particles = 0;
        const amrex::Real * uzp = nullptr;
        const amrex::Real * wp = nullptr;
        const std::uint64_t * idcpup = nullptr;

        // Extract particle properties
        // For momenta and weights
        if (initial) {
            const auto& soa = beam.getBeamInitSlice().GetStructOfArrays();
            num_particles = beam.getBeamInitSlice().size();
            uzp = soa.GetRealData(BeamIdx::uz).data();
            wp = soa.GetRealData(BeamIdx::w).data();
            idcpup = soa.GetIdCPUData().data();
        } else {
            const auto& soa = beam.getBeamSlice(WhichBeamSlice::This).GetStructOfArrays();
            num_particles = beam.getNumParticles(WhichBeamSlice::This);
            uzp = soa.GetRealData(BeamIdx::uz).data();
            wp = soa.GetRealData(BeamIdx::w).data();
            idcpup = soa.GetIdCPUData().data();
        }

        amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum,
                         amrex::ReduceOpSum, amrex::ReduceOpMin> reduce_op;
        amrex::ReduceData<amrex::Real, amrex::Real, amrex::Real, amrex::Real>
                        reduce_data(reduce_op);
        using ReduceTuple = typename decltype(reduce_data)::Type;

        reduce_op.eval(num_particles, reduce_data,
            [=] AMREX_GPU_DEVICE (unsigned long long ip) noexcept -> ReduceTuple
            {
                if (amrex::ConstParticleIDWrapper(idcpup[ip]) < 0) return {
                    0._rt, 0._rt, 0._rt, std::numeric_limits<amrex::Real>::infinity()
                };
                return {
                    wp[ip],
                    wp[ip] * uzp[ip] * clightinv,
                    wp[ip] * uzp[ip] * uzp[ip] * clightinv * clightinv,
                    uzp[ip] * clightinv
                };
            });

        auto res = reduce_data.value(reduce_op);
        m_timestep_data[ibeam][WhichDouble::SumWeights] += amrex::get<0>(res);
        m_timestep_data[ibeam][WhichDouble::SumWeightsTimesUz] += amrex::get<1>(res);
        m_timestep_data[ibeam][WhichDouble::SumWeightsTimesUzSquared] += amrex::get<2>(res);
        m_timestep_data[ibeam][WhichDouble::MinUz] =
            std::min(m_timestep_data[ibeam][WhichDouble::MinUz], amrex::get<3>(res));
    }
}

void
AdaptiveTimeStep::CalculateFromMinUz (
    amrex::Real t, amrex::Real& dt, MultiBeam& beams, MultiPlasma& plasmas)
{
    using namespace amrex::literals;

    if (!m_do_adaptive_time_step) return;

    HIPACE_PROFILE("AdaptiveTimeStep::CalculateFromMinUz()");

    const PhysConst phys_const = get_phys_const();
    const amrex::Real c = phys_const.c;
    const amrex::Real ep0 = phys_const.ep0;

    // Extract properties associated with physical size of the box
    const int nbeams = beams.get_nbeams();
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(nbeams >= 1,
        "Must have at least one beam to use adaptive time step");
    const int numprocs = Hipace::m_numprocs;

    amrex::Vector<amrex::Real> new_dts;
    new_dts.resize(nbeams);
    amrex::Vector<amrex::Real> beams_min_uz_mq;
    beams_min_uz_mq.resize(nbeams, std::numeric_limits<amrex::Real>::max());

    for (int ibeam = 0; ibeam < nbeams; ibeam++) {
        new_dts[ibeam] = dt;

        const auto& beam = beams.getBeam(ibeam);
        if (beam.m_charge == 0.) { continue; }
        const amrex::Real mass_charge_ratio = beam.m_mass / beam.m_charge;

        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_timestep_data[ibeam][WhichDouble::SumWeights] != 0,
            "The sum of all weights is 0! Probably no beam particles are initialized\n");
        const amrex::Real mean_uz = m_timestep_data[ibeam][WhichDouble::SumWeightsTimesUz]
            /m_timestep_data[ibeam][WhichDouble::SumWeights];
        const amrex::Real sigma_uz =
            std::sqrt(std::abs(m_timestep_data[ibeam][WhichDouble::SumWeightsTimesUzSquared]
                                /m_timestep_data[ibeam][WhichDouble::SumWeights]
                                - mean_uz*mean_uz));
        const amrex::Real sigma_uz_dev = mean_uz - 4.*sigma_uz;
        const amrex::Real max_supported_uz = 1.e30;
        amrex::Real chosen_min_uz =
            std::min(std::max(sigma_uz_dev, m_timestep_data[ibeam][WhichDouble::MinUz]),
                        max_supported_uz);

        if (Hipace::m_verbose >=2 ){
            amrex::Print()<<"Minimum gamma of beam " << ibeam <<
                " to calculate new time step: " << chosen_min_uz << "\n";
        }

        if (chosen_min_uz < m_threshold_uz) {
            amrex::Print()<<"WARNING: beam particles of beam "<< ibeam <<
                " have non-relativistic velocities!\n";
        }
        chosen_min_uz = std::max(chosen_min_uz, m_threshold_uz);
        beams_min_uz_mq[ibeam] = std::abs(chosen_min_uz * mass_charge_ratio);

        /*
            Calculate the time step for this beam used in the next time iteration
            of the current rank, to resolve the betatron period with m_nt_per_betatron
            points per period, assuming full blowout regime. The z-dependence of the
            plasma profile is considered. As this time step will start in numprocs
            time steps, so the beam may have propagated a significant distance.
            If m_adaptive_predict_step is true, we estimate this distance and use it
            for a more accurate time step estimation.
        */
        amrex::Real new_dt = dt;
        amrex::Real new_time = t;
        amrex::Real min_uz = chosen_min_uz;
        const int niter = m_adaptive_predict_step ? numprocs : 1;
        for (int i = 0; i < niter; i++)
        {
            amrex::Real plasma_charge_density = plasmas.maxChargeDensity(c * new_time);
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE( plasma_charge_density > 0.,
                "A >0 plasma density must be specified to use an adaptive time step.");
            if (m_adaptive_gather_ez) {
                min_uz += m_timestep_data[ibeam][WhichDouble::MinAcc] * new_dt;
            }
            // Just make sure min_uz is >0, to avoid nans below.
            min_uz = std::max(min_uz, 0.001_rt*m_threshold_uz);
            amrex::Real omega_b = std::sqrt(plasma_charge_density /
                                            (2. * std::abs(min_uz * mass_charge_ratio) * ep0));
            new_dt = 2. * MathConst::pi / omega_b / m_nt_per_betatron;
            new_time += new_dt;
            if (min_uz > m_threshold_uz) {
                new_dts[ibeam] = new_dt;
            }
        }
    }
    // Store min uz across beams, used in the phase advance method
    m_min_uz_mq = *std::min_element(beams_min_uz_mq.begin(), beams_min_uz_mq.end());
    /* set the new time step */
    dt = *std::min_element(new_dts.begin(), new_dts.end());
    // Make sure the new time step is smaller than the upper bound
    dt = std::min(dt, m_dt_max);
}

void
AdaptiveTimeStep::GatherMinAccSlice (MultiBeam& beams, const amrex::Geometry& geom,
                                     const Fields& fields)
{
    using namespace amrex::literals;

    if (!m_do_adaptive_time_step) return;
    if (!m_adaptive_gather_ez) return;

    HIPACE_PROFILE("AdaptiveTimeStep::GatherMinAccSlice()");

    const PhysConst phys_const = get_phys_const();
    const amrex::Real clightinv = 1._rt/phys_const.c;

    constexpr int lev = 0;

    // Extract properties associated with physical size of the box
    const int nbeams = beams.get_nbeams();

    for (int ibeam = 0; ibeam < nbeams; ibeam++) {
        const auto& beam = beams.getBeam(ibeam);
        const amrex::Real charge_mass_ratio = beam.m_charge / beam.m_mass;

        amrex::ReduceOps<amrex::ReduceOpMin> reduce_op;
        amrex::ReduceData<amrex::Real> reduce_data(reduce_op);
        using ReduceTuple = typename decltype(reduce_data)::Type;

        // Data required to gather the Ez field
        const amrex::FArrayBox& slice_fab = fields.getSlices(lev)[0];
        Array3<const amrex::Real> const slice_arr = slice_fab.const_array();
        const int ez_comp = Comps[WhichSlice::This]["Ez"];
        const amrex::Real dx_inv = geom.InvCellSize(0);
        const amrex::Real dy_inv = geom.InvCellSize(1);
        amrex::Real const x_pos_offset = GetPosOffset(0, geom, slice_fab.box());
        const amrex::Real y_pos_offset = GetPosOffset(1, geom, slice_fab.box());

        auto const& soa = beam.getBeamSlice(WhichBeamSlice::This).GetStructOfArrays();
        const auto pos_x = soa.GetRealData(BeamIdx::x).data();
        const auto pos_y = soa.GetRealData(BeamIdx::y).data();
        const auto idcpup = soa.GetIdCPUData().data();

        reduce_op.eval(beam.getNumParticles(WhichBeamSlice::This), reduce_data,
            [=] AMREX_GPU_DEVICE (long ip) noexcept -> ReduceTuple
            {
                if (amrex::ConstParticleIDWrapper(idcpup[ip]) < 0) return { 0._rt };
                const amrex::Real xp = pos_x[ip];
                const amrex::Real yp = pos_y[ip];

                amrex::Real Ezp = 0._rt;
                doGatherEz(xp, yp, Ezp, slice_arr, ez_comp,
                           dx_inv, dy_inv, x_pos_offset, y_pos_offset);
                return {
                    charge_mass_ratio * Ezp * clightinv
                };
            });

        auto res = reduce_data.value(reduce_op);
        m_timestep_data[ibeam][WhichDouble::MinAcc] =
            std::min(m_timestep_data[ibeam][WhichDouble::MinAcc], amrex::get<0>(res));
    }
}

void
AdaptiveTimeStep::CalculateFromDensity (amrex::Real t, amrex::Real& dt, MultiPlasma& plasmas)
{
    using namespace amrex::literals;

    if (!m_do_adaptive_time_step) return;

    for (int ibeam = 0; ibeam < m_nbeams; ibeam++) {
        m_timestep_data[ibeam][WhichDouble::MinUz] = 1e30;
        m_timestep_data[ibeam][WhichDouble::MinAcc] = 0.;
        m_timestep_data[ibeam][WhichDouble::SumWeights] = 0.;
        m_timestep_data[ibeam][WhichDouble::SumWeightsTimesUz] = 0.;
        m_timestep_data[ibeam][WhichDouble::SumWeightsTimesUzSquared] = 0.;
    }

    if (!m_adaptive_control_phase_advance) return;

    HIPACE_PROFILE("AdaptiveTimeStep::CalculateFromDensity()");

    const PhysConst pc = get_phys_const();

    amrex::Real dt_sub = dt / m_adaptive_phase_substeps;
    amrex::Real phase_advance = 0.;
    amrex::Real phase_advance0 = 0.;

    // Get plasma density at beginning of step
    const amrex::Real plasma_charge_density0 = plasmas.maxChargeDensity(pc.c * t);
    const amrex::Real omgb0 = std::sqrt(plasma_charge_density0 / (2. * m_min_uz_mq * pc.ep0));

    // Numerically integrate the phase advance from t to t+dt. The time step is reduced such that
    // the expected phase advance equals that of a uniform plasma up to a tolerance level.
    for (int i = 0; i < m_adaptive_phase_substeps; i++)
    {
        const amrex::Real plasma_charge_density = plasmas.maxChargeDensity(pc.c * (t+i*dt_sub));
        const amrex::Real omgb = std::sqrt(plasma_charge_density / (2. * m_min_uz_mq * pc.ep0));
        phase_advance += omgb * dt_sub;
        phase_advance0 += omgb0 * dt_sub;
        if(std::abs(phase_advance - phase_advance0) >
           2.*MathConst::pi*m_adaptive_phase_tolerance/m_nt_per_betatron)
        {
            if (i==0) amrex::AllPrint()<<"WARNING: adaptive time step exits at first substep."<<
                                         " Consider increasing hipace.adaptive_phase_substeps!\n";
            dt = i*dt_sub;
            return;
        }
    }
}
