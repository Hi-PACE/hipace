/* Copyright 2020-2021
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn, MaxThevenet
 * License: BSD-3-Clause-LBNL
 */
/* Copyright 2020 Axel Huebl, Maxence Thevenet
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef HIPACE_PROFILERWRAPPER_H_
#define HIPACE_PROFILERWRAPPER_H_

#include <AMReX_GpuDevice.H>
#include <AMReX_BLProfiler.H>
#include <AMReX_Algorithm.H>
#include <map>
#include <string>
#include <vector>
#include <array>


/** Whether to call amrex::Gpu::streamSynchronize() around all profiler region */
inline int DO_DEVICE_SYNCHRONIZE = 0;




struct StreamProfilerNode {

    using Event_t = cudaEvent_t;

    StreamProfilerNode* m_parent;
    std::map<std::string, StreamProfilerNode> m_children{};

    unsigned long long m_ncalls = 0;
    static constexpr int m_nevents = 8;

    Event_t m_start[m_nevents];
    Event_t m_stop[m_nevents];

    double m_time_inc = 0;
    double m_time_exc = 0;
    double m_time_inc_sq = 0;

    static StreamProfilerNode* m_root;
    static StreamProfilerNode* m_stack_location;


    StreamProfilerNode () : m_parent{m_stack_location} {
        for (int i=0; i<m_nevents; ++i) {
            AMREX_CUDA_SAFE_CALL(cudaEventCreate(&m_start[i]));
            AMREX_CUDA_SAFE_CALL(cudaEventCreate(&m_stop[i]));
        }
    }

    ~StreamProfilerNode () {
        for (int i=0; i<m_nevents; ++i) {
            AMREX_CUDA_SAFE_CALL(cudaEventDestroy(m_start[i]));
            AMREX_CUDA_SAFE_CALL(cudaEventDestroy(m_stop[i]));
        }
    }

    void start () {
        ++m_ncalls;
        const int ievent = (m_ncalls-1) % m_nevents;

        if (m_ncalls > m_nevents) {
            float time = 0.;
            if (cudaEventQuery(m_stop[ievent]) != cudaSuccess) {
                AMREX_CUDA_SAFE_CALL(cudaEventSynchronize(m_stop[ievent]));
            }
            AMREX_CUDA_SAFE_CALL(cudaEventElapsedTime(&time, m_start[ievent], m_stop[ievent]));
            m_time_inc += time/1000.;
            m_time_inc_sq += time/1000.*time/1000.;
        }

        AMREX_CUDA_SAFE_CALL(cudaEventRecord(m_start[ievent], amrex::Gpu::gpuStream()));

        m_stack_location = this;
    }

    void stop () {
        const int ievent = (m_ncalls-1) % m_nevents;
        AMREX_CUDA_SAFE_CALL(cudaEventRecord(m_stop[ievent], amrex::Gpu::gpuStream()));
        m_stack_location = m_parent;
    }

    static void Initialize ();
    static void Finalize ();

private:

    void GetExcData ();
    void FinishEvents ();
    void PrintTime (std::vector<std::array<std::string,6>>&, const std::string&,
                                                             const std::string&, double);
};


struct StreamProfiler {
    StreamProfilerNode* m_node;

    StreamProfiler (const std::string& fname) {
        m_node = &(StreamProfilerNode::m_stack_location->m_children[fname]);
        //std::cout << fname << " " << m_node->m_ncalls << std::endl;
        m_node->start();
    }

    ~StreamProfiler () {
        m_node->stop();
    }
};



#define HIPACE_PROFILE(fname) StreamProfiler BL_PROFILE_PASTE(PROF_SCOPE_, __COUNTER__){fname}
//#define HIPACE_DETAIL_PROFILE(fname) HIPACE_PROFILE(fname)
#define HIPACE_DETAIL_PROFILE(fname)

/*
template<int detail_level>
AMREX_FORCE_INLINE
void doStreamSynchronize ()
{
    if ( DO_DEVICE_SYNCHRONIZE >= detail_level )
        amrex::Gpu::streamSynchronize();
}

// Note that objects are destructed in the reverse order of declaration
template<int detail_level>
struct synchronizeOnDestruct {
    AMREX_FORCE_INLINE
    ~synchronizeOnDestruct () {
        doStreamSynchronize<detail_level>();
    }
};

#define HIPACE_PROFILE(fname) doStreamSynchronize<1>(); BL_PROFILE(fname); synchronizeOnDestruct<1> BL_PROFILE_PASTE(SYNC_SCOPE_, __COUNTER__){}
#define HIPACE_PROFILE_VAR(fname, vname) doStreamSynchronize<1>(); BL_PROFILE_VAR(fname, vname); synchronizeOnDestruct<1> SYNC_V_##vname{}
#define HIPACE_PROFILE_VAR_NS(fname, vname) BL_PROFILE_VAR_NS(fname, vname); synchronizeOnDestruct<1> SYNC_V_##vname{}
#define HIPACE_PROFILE_VAR_START(vname) doStreamSynchronize<1>(); BL_PROFILE_VAR_START(vname)
#define HIPACE_PROFILE_VAR_STOP(vname) doStreamSynchronize<1>(); BL_PROFILE_VAR_STOP(vname)
#define HIPACE_PROFILE_REGION(rname) doStreamSynchronize<1>(); BL_PROFILE_REGION(rname); synchronizeOnDestruct<1> BL_PROFILE_PASTE(SYNC_R_, __COUNTER__){}

#define HIPACE_DETAIL_PROFILE(fname) doStreamSynchronize<2>(); BL_PROFILE(fname); synchronizeOnDestruct<2> BL_PROFILE_PASTE(SYNC_SCOPE_, __COUNTER__){}
#define HIPACE_DETAIL_PROFILE_VAR(fname, vname) doStreamSynchronize<2>(); BL_PROFILE_VAR(fname, vname); synchronizeOnDestruct<2> SYNC_V_##vname{}
#define HIPACE_DETAIL_PROFILE_VAR_NS(fname, vname) BL_PROFILE_VAR_NS(fname, vname); synchronizeOnDestruct<2> SYNC_V_##vname{}
#define HIPACE_DETAIL_PROFILE_VAR_START(vname) doStreamSynchronize<2>(); BL_PROFILE_VAR_START(vname)
#define HIPACE_DETAIL_PROFILE_VAR_STOP(vname) doStreamSynchronize<2>(); BL_PROFILE_VAR_STOP(vname)
#define HIPACE_DETAIL_PROFILE_REGION(rname) doStreamSynchronize<2>(); BL_PROFILE_REGION(rname); synchronizeOnDestruct<2> BL_PROFILE_PASTE(SYNC_R_, __COUNTER__){}
*/

#endif // HIPACE_PROFILERWRAPPER_H_
