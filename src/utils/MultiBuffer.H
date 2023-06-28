/* Copyright 2023
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn
 * License: BSD-3-Clause-LBNL
 */
#ifndef HIPACE_MultiBuffer_H_
#define HIPACE_MultiBuffer_H_

#include <AMReX_AmrCore.H>
#include "particles/beam/MultiBeam.H"

#ifndef AMREX_USE_MPI
using MPI_Request = int;
using MPI_Status = int;
using MPI_Comm = int;
constexpr MPI_Request MPI_REQUEST_NULL = 0;
constexpr MPI_Comm MPI_COMM_WORLD = 0;
#endif


class MultiBuffer
{

public:

    void initialize (int nslices, int nbeams, int rank_num, int n_ranks, bool buffer_on_host);

    void get_data (int slice, MultiBeam& beams, int beam_slice);

    void put_data (int slice, MultiBeam& beams, int beam_slice, bool is_last_time_step);

    ~MultiBuffer();

private:

    enum struct memory_location {
        nowhere,
        pinned,
        device
    };

    enum struct comm_progress {
        uninitialized,
        ready_to_define,
        in_use,
        ready_to_send,
        send_started,
        sent,
        receive_started,
        received,
        sim_completed
    };

    struct DataNode {
        char* m_buffer = nullptr;
        std::size_t m_buffer_size = 0;
        memory_location m_location = memory_location::nowhere;
        comm_progress m_progress = comm_progress::uninitialized;
        MPI_Request m_request = MPI_REQUEST_NULL;
        comm_progress m_metadata_progress = comm_progress::uninitialized;
        MPI_Request m_metadata_request = MPI_REQUEST_NULL;
    };

    using storage_type = amrex::ParallelDescriptor::lull_t;

    static constexpr std::size_t buffer_size_roundup = std::lcm(alignof(amrex::Real), alignof(int));

    amrex::Gpu::PinnedVector<std::size_t> m_metadata {};
    amrex::Vector<DataNode> m_datanodes {};

    int m_rank_send_to = 0;
    int m_rank_receive_from =0;
    int m_tag_buffer_start = 0;
    int m_tag_metadata_start = 0;
    MPI_Comm m_comm = MPI_COMM_WORLD;

    bool m_buffer_on_host = true;

    int m_nslices = 0;
    int m_nbeams = 0;
    bool m_is_head_rank = false;

    std::size_t get_metadata_size ();

    std::size_t* get_metadata_location (int slice);

    void allocate_buffer (int slice);

    void free_buffer (int slice);

    void make_progress (int slice, bool is_blocking);

    void write_metadata (int slice, MultiBeam& beams, int beam_slice);

    std::size_t get_buffer_offset_real (int slice, int ibeam, int rcomp);

    std::size_t get_buffer_offset_int (int slice, int ibeam, int icomp);

    void pack_data (int slice, MultiBeam& beams, int beam_slice);

public:

    void unpack_data (int slice, MultiBeam& beams, int beam_slice);

};

#endif
