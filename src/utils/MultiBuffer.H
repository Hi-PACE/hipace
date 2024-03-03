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
#include "laser/MultiLaser.H"

class MultiBuffer
{

public:

    // initialize MultiBuffer and open initial receive requests
    void initialize (int nslices, int nbeams, bool buffer_on_host, bool use_laser,
                     amrex::Box laser_box, int max_leading_slices, int max_trailing_slices);

    // receive data from previous rank and unpack it into MultiBeam and MultiLaser
    void get_data (int slice, MultiBeam& beams, MultiLaser& laser, int beam_slice);

    // pack data from MultiBeam and MultiLaser into buffer and send it to the next rank
    void put_data (int slice, MultiBeam& beams, MultiLaser& laser, int beam_slice,
                   bool is_last_time_step);

    // receive physical time from previous rank
    amrex::Real get_time ();

    // send physical time to next rank
    void put_time (amrex::Real time);

    // destructor to clean up all open MPI requests
    ~MultiBuffer();

private:

    // to keep track of per-slice buffer location
    enum struct memory_location {
        nowhere,
        pinned,
        device
    };

    // to keep track of per-slice MPI communication status
    enum comm_progress : int {
        uninitialized,
        sim_completed,
        async_progress_begin,
        ready_to_send,
        send_started,
        sent,
        receive_started,
        received,
        async_progress_end,
        ready_to_define,
        in_use,
        nprogress,
    };

    // to specify the input for get_buffer_offset
    enum struct offset_type {
        beam_idcpu,
        beam_real,
        beam_int,
        laser,
        total
    };

    // struct to store per-slice data
    struct DataNode {
        char* m_buffer = nullptr;
        std::size_t m_buffer_size = 0;
        memory_location m_location = memory_location::nowhere;
        comm_progress m_progress = comm_progress::uninitialized;
        MPI_Request m_request = MPI_REQUEST_NULL;
        comm_progress m_metadata_progress = comm_progress::uninitialized;
        MPI_Request m_metadata_request = MPI_REQUEST_NULL;
    };

#ifdef AMREX_USE_MPI
    using storage_type = amrex::ParallelDescriptor::lull_t;
#else
    using storage_type = unsigned long long[8];
#endif

    // round up the capacity of a buffer so there are never any alignment problems
    static constexpr std::size_t buffer_size_roundup = alignof(amrex::Real) / alignof(int);

    // 2D array for all metadata
    amrex::Gpu::PinnedVector<std::size_t> m_metadata {};
    // per-slice data
    amrex::Vector<DataNode> m_datanodes {};

    // MPI parameters
    bool m_is_head_rank = false;
    bool m_is_serial = true;
    int m_rank_send_to = 0;
    int m_rank_receive_from = 0;
    int m_tag_time_start = 0;
    int m_tag_buffer_start = 0;
    int m_tag_metadata_start = 0;
    MPI_Comm m_comm = MPI_COMM_NULL;

    // general parameters
    bool m_buffer_on_host = true;
    int m_nslices = 0;
    int m_nbeams = 0;
    bool m_use_laser = false;
    int m_laser_ncomp = 4;
    amrex::Box m_laser_slice_box {};
    int m_max_leading_slices = std::numeric_limits<int>::max();
    int m_max_trailing_slices = std::numeric_limits<int>::max();

    // parameters to send physical time
    amrex::Real m_time_send_buffer = 0.;
    MPI_Request m_time_send_request = MPI_REQUEST_NULL;
    bool m_time_send_started = false;

    // slice index of where to continue making async progress
    std::array<int, comm_progress::nprogress> m_async_metadata_slice {};
    std::array<int, comm_progress::nprogress> m_async_data_slice {};

    // helper functions to read 2D metadata array
    std::size_t get_metadata_size ();
    std::size_t* get_metadata_location (int slice);

    // helper functions to allocate and free buffers using the correct arena
    void allocate_buffer (int slice);
    void free_buffer (int slice);

    // function containing main progress loop to deal with asynchronous MPI requests
    void make_progress (int slice, bool is_blocking, int current_slice);

    // write MultiBeam sizes into the metadata array
    void write_metadata (int slice, MultiBeam& beams, int beam_slice);

    // helper function to get location of individual arrays inside a buffer
    std::size_t get_buffer_offset (int slice, offset_type type, MultiBeam& beams,
                                   int ibeam, int comp);

    // copy gpu array into buffer at buffer_offset, either dtoh or dtod
    void memcpy_to_buffer (int slice, std::size_t buffer_offset,
                           const void* src_ptr, std::size_t num_bytes);

    // copy buffer array at buffer_offset into gpu array, either htod or dtod
    void memcpy_from_buffer (int slice, std::size_t buffer_offset,
                             void* dst_ptr, std::size_t num_bytes);

    // pack MultiBeam and MultiLaser into buffer
    void pack_data (int slice, MultiBeam& beams, MultiLaser& laser, int beam_slice);

public: // needed for ParallelFor

    // unpack MultiBeam and MultiLaser from buffer
    void unpack_data (int slice, MultiBeam& beams, MultiLaser& laser, int beam_slice);

};

#endif
