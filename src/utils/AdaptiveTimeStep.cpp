/* Copyright 2020-2021
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn, MaxThevenet, Severin Diederichs
 * License: BSD-3-Clause-LBNL
 */
#include "AdaptiveTimeStep.H"
#include "utils/DeprecatedInput.H"
#include "particles/particles_utils/FieldGather.H"
#include "Hipace.H"
#include "HipaceProfilerWrapper.H"
#include "Constants.H"

#ifdef AMREX_USE_MPI
namespace {
    constexpr int comm_z_tag = 2000;
}
#endif

/** \brief describes which double is used for the adaptive time step */
struct WhichDouble {
    enum Comp { Dt=0, MinUz, MinAcc, SumWeights, SumWeightsTimesUz, SumWeightsTimesUzSquared, N };
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
        queryWithParser(ppa, "adaptive_phase_tolerance", m_adaptive_phase_tolerance);
    }
    DeprecatedInput("hipace", "do_adaptive_time_step", "dt = adaptive");

    // create time step data container per beam
    for (int ibeam = 0; ibeam < nbeams; ibeam++) {
        amrex::Vector<amrex::Real> ts_data;
        ts_data.resize(6, 0.);
        ts_data[1] = 1e30; // max possible uz be taken into account
        m_timestep_data.emplace_back(ts_data);
    }

}

#ifdef AMREX_USE_MPI
void
AdaptiveTimeStep::NotifyTimeStep (amrex::Real dt, MPI_Comm a_comm_z)
{
    if (m_do_adaptive_time_step == 0) return;
    const int my_rank_z = amrex::ParallelDescriptor::MyProc();
    if (my_rank_z >= 1)
    {
        MPI_Send(&dt, 1, amrex::ParallelDescriptor::Mpi_typemap<amrex::Real>::type(),
                 my_rank_z-1, comm_z_tag, a_comm_z);
    }
}

void
AdaptiveTimeStep::WaitTimeStep (amrex::Real& dt, MPI_Comm a_comm_z)
{
    if (m_do_adaptive_time_step == 0) return;
    const int my_rank_z = amrex::ParallelDescriptor::MyProc();
    if (!Hipace::HeadRank())
    {
        MPI_Status status;
        MPI_Recv(&dt, 1, amrex::ParallelDescriptor::Mpi_typemap<amrex::Real>::type(),
                 my_rank_z+1, comm_z_tag, a_comm_z, &status);
    }
}
#endif

void
AdaptiveTimeStep::Calculate (
    amrex::Real t, amrex::Real& dt, MultiBeam& beams, MultiPlasma& plasmas,
    const amrex::Geometry& geom, const Fields& fields,
    const int it, const amrex::Vector<BoxSorter>& a_box_sorter_vec,
    const bool initial)
{
    HIPACE_PROFILE("AdaptiveTimeStep::Calculate()");
    using namespace amrex::literals;

    if (m_do_adaptive_time_step == 0) return;
    if (!Hipace::HeadRank() && initial) return;

    const PhysConst phys_const = get_phys_const();
    const amrex::Real c = phys_const.c;
    const amrex::Real q_e = phys_const.q_e;
    const amrex::Real m_e = phys_const.m_e;
    const amrex::Real ep0 = phys_const.ep0;
    const amrex::Real clightinv = 1._rt/c;
    amrex::Real plasma_density = plasmas.maxDensity(c * t);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE( plasma_density > 0.,
        "A >0 plasma density must be specified to use an adaptive time step.");

    constexpr int lev = 0;

    // Extract properties associated with physical size of the box
    const int nbeams = beams.get_nbeams();
    const int numprocs_z = Hipace::GetInstance().m_numprocs_z;

    amrex::Vector<amrex::Real> new_dts;
    new_dts.resize(nbeams);

    for (int ibeam = 0; ibeam < nbeams; ibeam++) {
        const auto& beam = beams.getBeam(ibeam);
        const amrex::Real charge_mass_ratio = beam.m_charge / beam.m_mass;

        // first box resets time step data
        if (it == amrex::ParallelDescriptor::NProcs()-1) {
            m_timestep_data[ibeam][WhichDouble::SumWeights] = 0.;
            m_timestep_data[ibeam][WhichDouble::SumWeightsTimesUz] = 0.;
            m_timestep_data[ibeam][WhichDouble::SumWeightsTimesUzSquared] = 0.;
            m_timestep_data[ibeam][WhichDouble::MinUz] = 1e30;
        }

        const uint64_t box_offset = initial ? 0 : a_box_sorter_vec[ibeam].boxOffsetsPtr()[it];
        const uint64_t numParticleOnTile = initial ? beam.numParticles()
                                                   : a_box_sorter_vec[ibeam].boxCountsPtr()[it];


        // Extract particle properties
        const auto& aos = beam.GetArrayOfStructs(); // For positions
        const auto& pos_structs = aos.begin() + box_offset;
        const auto& soa = beam.GetStructOfArrays(); // For momenta and weights
        const auto uzp = soa.GetRealData(BeamIdx::uz).data() + box_offset;
        const auto wp = soa.GetRealData(BeamIdx::w).data() + box_offset;

        amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum,
                         amrex::ReduceOpSum, amrex::ReduceOpMin,
                         amrex::ReduceOpMin> reduce_op;
        amrex::ReduceData<amrex::Real, amrex::Real, amrex::Real,
                          amrex::Real, amrex::Real> reduce_data(reduce_op);
        using ReduceTuple = typename decltype(reduce_data)::Type;

        const amrex::FArrayBox& slice_fab = fields.getSlices(lev)[0];
        Array3<const amrex::Real> const slice_arr = slice_fab.const_array();
        const int ez_comp = Comps[WhichSlice::This]["Ez"];
        amrex::Real const * AMREX_RESTRICT dx = geom.CellSize();
        const amrex::Real dx_inv = 1._rt/dx[0];
        const amrex::Real dy_inv = 1._rt/dx[1];
        // Offset for converting positions to indexes
        amrex::Real const x_pos_offset = GetPosOffset(0, geom, slice_fab.box());
        const amrex::Real y_pos_offset = GetPosOffset(1, geom, slice_fab.box());

        reduce_op.eval(numParticleOnTile, reduce_data,
            [=] AMREX_GPU_DEVICE (long ip) noexcept -> ReduceTuple
            {
                if (pos_structs[ip].id() < 0) return {
                    0._rt, 0._rt, 0._rt, std::numeric_limits<amrex::Real>::infinity(), 0._rt
                };
                amrex::Real Ezp = 0._rt;
                doGatherEz(pos_structs[ip].pos(0), pos_structs[ip].pos(1), Ezp, slice_arr, ez_comp,
                           dx_inv, dy_inv, x_pos_offset, y_pos_offset);
                const amrex::Real uz = uzp[ip] + dt * charge_mass_ratio * Ezp;
                return {
                    wp[ip],
                    wp[ip] * uz * clightinv,
                    wp[ip] * uz * uz * clightinv * clightinv,
                    uz * clightinv,
                    charge_mass_ratio * Ezp * clightinv
                };
            });

        auto res = reduce_data.value(reduce_op);
        m_timestep_data[ibeam][WhichDouble::SumWeights] += amrex::get<0>(res);
        m_timestep_data[ibeam][WhichDouble::SumWeightsTimesUz] += amrex::get<1>(res);
        m_timestep_data[ibeam][WhichDouble::SumWeightsTimesUzSquared] += amrex::get<2>(res);
        m_timestep_data[ibeam][WhichDouble::MinUz] =
            std::min(m_timestep_data[ibeam][WhichDouble::MinUz], amrex::get<3>(res));
        m_timestep_data[ibeam][WhichDouble::MinAcc] =
            std::min(m_timestep_data[ibeam][WhichDouble::MinAcc], amrex::get<4>(res));
    }

    // only the last box or at initialization the adaptive time step is calculated
    // from the full beam information
    if (it == 0 || initial)
    {
        for (int ibeam = 0; ibeam < nbeams; ibeam++) {

            const auto& beam = beams.getBeam(ibeam);

            AMREX_ALWAYS_ASSERT_WITH_MESSAGE( m_timestep_data[ibeam][WhichDouble::SumWeights] != 0,
                "The sum of all weights is 0! Probably no beam particles are initialized");
            const amrex::Real mean_uz = m_timestep_data[ibeam][WhichDouble::SumWeightsTimesUz]
                                           /m_timestep_data[ibeam][WhichDouble::SumWeights];
            const amrex::Real sigma_uz = std::sqrt(std::abs(m_timestep_data[ibeam][WhichDouble::SumWeightsTimesUzSquared]
                                              /m_timestep_data[ibeam][WhichDouble::SumWeights]
                                              - mean_uz*mean_uz));
            const amrex::Real sigma_uz_dev = mean_uz - 4.*sigma_uz;
            const amrex::Real max_supported_uz = 1e30;
            amrex::Real chosen_min_uz = std::min(amrex::max(sigma_uz_dev,
                                                m_timestep_data[ibeam][WhichDouble::MinUz]),
                                                max_supported_uz);
            m_min_uz = std::min(m_min_uz,
                chosen_min_uz*beam.m_mass*beam.m_mass/m_e/m_e);
            m_min_uz = amrex::max(m_min_uz, 1.);

            if (Hipace::m_verbose >=2 ){
                amrex::Print()<<"Minimum gamma of beam " << ibeam << " to calculate new time step: "
                              << chosen_min_uz << "\n";
            }

            if (chosen_min_uz < 1) {
                amrex::Print()<<"WARNING: beam particles of beam "<< ibeam <<
                                " have non-relativistic velocities!";
            }

            new_dts[ibeam] = dt;

            amrex::Real new_dt = dt;
            amrex::Real new_time = t;
            for (int i = 0; i < numprocs_z; i++)
            {
                plasma_density = plasmas.maxDensity(c * new_time);
                chosen_min_uz += m_timestep_data[ibeam][WhichDouble::MinAcc] * new_dt;
                const amrex::Real omega_p = std::sqrt(plasma_density * q_e * q_e / (ep0 * m_e));
                amrex::Real omega_b = omega_p / std::sqrt(2. * chosen_min_uz) * m_e / beam.m_mass;
                new_dt = 2. * MathConst::pi / omega_b / m_nt_per_betatron;
                new_time += new_dt;
                if (chosen_min_uz > 1) new_dts[ibeam] = new_dt;
            }
        }
        /* set the new time step */
        dt = *std::min_element(new_dts.begin(), new_dts.end());
        // Make sure the new time step is smaller than the upper bound
        dt = std::min(dt, m_dt_max);
    }
}

void
AdaptiveTimeStep::CalculateFromDensity (amrex::Real t, amrex::Real& dt, MultiPlasma& plasmas)
{
    HIPACE_PROFILE("AdaptiveTimeStep::Calculate()");

    using namespace amrex::literals;

    if (m_do_adaptive_time_step == 0) return;

    const PhysConst pc = get_phys_const();

    constexpr int nsub = 100;
    amrex::Real dt_sub = dt / nsub;
    amrex::Real phase_advance = 0.;
    amrex::Real phase_advance0 = 0.;

    // Get plasma density at beginning of step
    const amrex::Real plasma_density = plasmas.maxDensity(pc.c * t);
    const amrex::Real omega_p = std::sqrt(plasma_density * pc.q_e * pc.q_e / (pc.ep0 * pc.m_e));
    amrex::Real omgb0 = omega_p / std::sqrt(2. *m_min_uz);

    for (int i = 0; i < nsub; i++)
    {
        const amrex::Real plasma_density = plasmas.maxDensity(pc.c * (t+i*dt_sub));
        const amrex::Real omega_p = std::sqrt(plasma_density * pc.q_e * pc.q_e / (pc.ep0 * pc.m_e));
        amrex::Real omgb = omega_p / std::sqrt(2. *m_min_uz);
        phase_advance += omgb * dt_sub;
        phase_advance0 += omgb0 * dt_sub;
        if(std::abs(phase_advance - phase_advance0) >
           2.*MathConst::pi/m_nt_per_betatron*m_adaptive_phase_tolerance)
        {
            dt = i*dt_sub;
            return;
        }
    }
}
