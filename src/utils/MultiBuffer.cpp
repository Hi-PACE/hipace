/* Copyright 2023
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn
 * License: BSD-3-Clause-LBNL
 */
#include "MultiBuffer.H"
#include "Hipace.H"
#include "HipaceProfilerWrapper.H"
#include "Parser.H"


std::size_t MultiBuffer::get_metadata_size () {
    // 0: buffer size
    // 1: number of particles for beam 0
    // 2: number of particles for beam 1
    // ...
    return 1 + m_nbeams;
}

std::size_t* MultiBuffer::get_metadata_location (int slice) {
    return m_metadata.dataPtr() + slice*get_metadata_size();
}

void MultiBuffer::allocate_buffer (int slice) {
    AMREX_ALWAYS_ASSERT(m_datanodes[slice].m_location == memory_location::nowhere);
    if (!m_buffer_on_gpu) {
        m_datanodes[slice].m_buffer = reinterpret_cast<char*>(amrex::The_Pinned_Arena()->alloc(
            m_datanodes[slice].m_buffer_size * sizeof(storage_type)
        ));
        m_datanodes[slice].m_location = memory_location::pinned;
    } else {
        m_datanodes[slice].m_buffer = reinterpret_cast<char*>(amrex::The_Device_Arena()->alloc(
            m_datanodes[slice].m_buffer_size * sizeof(storage_type)
        ));
        m_datanodes[slice].m_location = memory_location::device;
    }
    m_current_buffer_size += m_datanodes[slice].m_buffer_size * sizeof(storage_type);
}

void MultiBuffer::free_buffer (int slice) {
    AMREX_ALWAYS_ASSERT(m_datanodes[slice].m_location != memory_location::nowhere);
    if (m_datanodes[slice].m_location == memory_location::pinned) {
        amrex::The_Pinned_Arena()->free(m_datanodes[slice].m_buffer);
    } else {
        amrex::The_Device_Arena()->free(m_datanodes[slice].m_buffer);
    }
    m_current_buffer_size -= m_datanodes[slice].m_buffer_size * sizeof(storage_type);
    m_datanodes[slice].m_location = memory_location::nowhere;
    m_datanodes[slice].m_buffer = nullptr;
    m_datanodes[slice].m_buffer_size = 0;
}

void MultiBuffer::initialize (int nslices, MultiBeam& beams, MultiLaser& laser) {

    amrex::ParmParse pp("comms_buffer");

    m_comm = amrex::ParallelDescriptor::Communicator();
    const int rank_id = amrex::ParallelDescriptor::MyProc();
    const int n_ranks = amrex::ParallelDescriptor::NProcs();

    m_nslices = nslices;
    m_nbeams = beams.get_nbeams();

    m_rank_send_to = (rank_id + 1) % n_ranks;
    m_rank_receive_from = (rank_id - 1 + n_ranks) % n_ranks;

    m_is_head_rank = Hipace::HeadRank();
    m_is_serial = n_ranks == 1;

    m_tag_time_start = 0;
    m_tag_buffer_start = 1;
    m_tag_metadata_start = m_tag_buffer_start + m_nslices;

    queryWithParser(pp, "on_gpu", m_buffer_on_gpu);
    queryWithParser(pp, "max_leading_slices", m_max_leading_slices);
    queryWithParser(pp, "max_trailing_slices", m_max_trailing_slices);
#ifdef AMREX_USE_GPU
    queryWithParser(pp, "async_memcpy", m_async_memcpy);
    if (m_buffer_on_gpu)
#endif
    {
        m_async_memcpy = false;
    }

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        ((double(m_max_trailing_slices) * n_ranks) > nslices)
        || (Hipace::m_max_step < amrex::ParallelDescriptor::NProcs()),
        "comms_buffer.max_trailing_slices must be large enough"
        " to distribute all slices between all ranks if there are more timesteps than ranks");

    double max_size_GiB = -1.;
    queryWithParser(pp, "max_size_GiB", max_size_GiB);
    if(max_size_GiB >= 0.) {
        m_max_buffer_size = static_cast<std::size_t>(max_size_GiB*1024*1024*1024);

        double size_estimate = 0.;
        for (int b = 0; b < m_nbeams; ++b) {
            auto& beam = beams.getBeam(b);
            const double num_particles = static_cast<double>(beam.getTotalNumParticles());

            if (beam.communicateIdCpuComponent()) {
                size_estimate += num_particles * sizeof(std::uint64_t);
            }

            for (int rcomp = 0; rcomp < beam.numRealComponents(); ++rcomp) {
                if (beam.communicateRealComponent(rcomp)) {
                    size_estimate += num_particles * sizeof(amrex::Real);
                }
            }

            for (int icomp = 0; icomp < beam.numIntComponents(); ++icomp) {
                if (beam.communicateIntComponent(icomp)) {
                    size_estimate += num_particles * sizeof(int);
                }
            }
        }

        if (laser.UseLaser()) {
            size_estimate += laser.GetLaserGeom().Domain().numPts()
                * m_laser_ncomp * sizeof(amrex::Real);
        }

        size_estimate /= 1024*1024*1024;
        if (!((1.05*size_estimate < max_size_GiB*n_ranks)
            || (Hipace::m_max_step < amrex::ParallelDescriptor::NProcs()))) {
            amrex::Abort("comms_buffer.max_size_GiB must be large enough to fit "
                         "all the data needed for all beams and the laser "
                         "between all ranks if there are more timesteps than ranks!\n"
                         "Data needed: " + std::to_string(1.05*size_estimate) + " GiB\n"
                         "Space available: " + std::to_string(max_size_GiB*n_ranks) + " GiB\n");
        }
    }

    bool do_pre_register = false;
    queryWithParser(pp, "pre_register_memory", do_pre_register);

    if (do_pre_register) {
        pre_register_memory();
    }

    for (int p = 0; p < comm_progress::nprogress; ++p) {
        m_async_metadata_slice[p] = m_nslices - 1;
        m_async_data_slice[p] = m_nslices - 1;
    }

    m_metadata.resize(get_metadata_size() * m_nslices);
    m_datanodes.resize(m_nslices);

    if (m_is_head_rank) {
        // head rank needs to initialize the beam
        for (int i = m_nslices-1; i >= 0; --i) {
            m_datanodes[i].m_progress = comm_progress::ready_to_define;
            m_datanodes[i].m_metadata_progress = comm_progress::ready_to_define;
        }
    } else {
        // other ranks receive beam, set progress to receive_started - 1
        for (int i = m_nslices-1; i >= 0; --i) {
            m_datanodes[i].m_progress = comm_progress::sent;
            m_datanodes[i].m_metadata_progress = comm_progress::sent;
        }
    }

    // open initial receives
    for (int i = m_nslices-1; i >= 0; --i) {
        make_progress(i, false, m_nslices-1);
    }
}

void MultiBuffer::pre_register_memory () {
#ifdef AMREX_USE_MPI
    HIPACE_PROFILE("MultiBuffer::pre_register_memory()");
    // On some platforms, such as JUWELS booster, the memory passed into MPI needs to be
    // registered to the network card, which can take a long time. In this function, all ranks
    // can do this all at once in initialization instead of one after another
    // as part of the communication pipeline.
    void* send_buffer = nullptr;
    void* recv_buffer = nullptr;
    const int count = 1024;
    MPI_Request send_request = MPI_REQUEST_NULL;
    MPI_Request recv_request = MPI_REQUEST_NULL;
    if (!m_buffer_on_gpu) {
        send_buffer = amrex::The_Pinned_Arena()->alloc(count * sizeof(storage_type));
        recv_buffer = amrex::The_Pinned_Arena()->alloc(count * sizeof(storage_type));
    } else {
        send_buffer = amrex::The_Device_Arena()->alloc(count * sizeof(storage_type));
        recv_buffer = amrex::The_Device_Arena()->alloc(count * sizeof(storage_type));
    }
    // send and receive dummy message
    // use the same MPI functions and arguments as in the real communication
    MPI_Isend(
        send_buffer,
        count,
        amrex::ParallelDescriptor::Mpi_typemap<storage_type>::type(),
        m_rank_send_to,
        m_tag_metadata_start + m_nslices,
        m_comm,
        &send_request);
    MPI_Irecv(
        recv_buffer,
        count,
        amrex::ParallelDescriptor::Mpi_typemap<storage_type>::type(),
        m_rank_receive_from,
        m_tag_metadata_start + m_nslices,
        m_comm,
        &recv_request);
    MPI_Wait(&send_request, MPI_STATUS_IGNORE);
    MPI_Wait(&recv_request, MPI_STATUS_IGNORE);
    if (!m_buffer_on_gpu) {
        amrex::The_Pinned_Arena()->free(send_buffer);
        amrex::The_Pinned_Arena()->free(recv_buffer);
    } else {
        amrex::The_Device_Arena()->free(send_buffer);
        amrex::The_Device_Arena()->free(recv_buffer);
    }
#endif
}

MultiBuffer::~MultiBuffer () {
#ifdef AMREX_USE_MPI
    // wait for sends to complete and cancel receives
    for (int slice = m_nslices-1; slice >= 0; --slice) {
        if (m_datanodes[slice].m_metadata_progress == comm_progress::ready_to_send) {
            MPI_Isend(
                get_metadata_location(slice),
                get_metadata_size(),
                amrex::ParallelDescriptor::Mpi_typemap<std::size_t>::type(),
                m_rank_send_to,
                m_tag_metadata_start + slice,
                m_comm,
                &(m_datanodes[slice].m_metadata_request));
            m_datanodes[slice].m_metadata_progress = comm_progress::send_started;
        }

        if (m_datanodes[slice].m_progress == comm_progress::ready_to_send) {
            if (m_datanodes[slice].m_buffer_size == 0) {
                m_datanodes[slice].m_progress = comm_progress::sent;
            } else {
                MPI_Isend(
                    m_datanodes[slice].m_buffer,
                    m_datanodes[slice].m_buffer_size,
                    amrex::ParallelDescriptor::Mpi_typemap<storage_type>::type(),
                    m_rank_send_to,
                    m_tag_buffer_start + slice,
                    m_comm,
                    &(m_datanodes[slice].m_request));
                m_datanodes[slice].m_progress = comm_progress::send_started;
            }
        }

        if (m_datanodes[slice].m_metadata_progress == comm_progress::send_started) {
            MPI_Wait(&(m_datanodes[slice].m_metadata_request), MPI_STATUS_IGNORE);
            m_datanodes[slice].m_metadata_progress = comm_progress::sent;
        }

        if (m_datanodes[slice].m_progress == comm_progress::send_started) {
            MPI_Wait(&(m_datanodes[slice].m_request), MPI_STATUS_IGNORE);
            free_buffer(slice);
            m_datanodes[slice].m_progress = comm_progress::sent;
        }

        if (m_datanodes[slice].m_metadata_progress == comm_progress::receive_started) {
            MPI_Cancel(&(m_datanodes[slice].m_metadata_request));
            MPI_Wait(&(m_datanodes[slice].m_metadata_request), MPI_STATUS_IGNORE);
            m_datanodes[slice].m_metadata_progress = comm_progress::sim_completed;
        }

        if (m_datanodes[slice].m_metadata_progress == comm_progress::sent) {
            m_datanodes[slice].m_metadata_progress = comm_progress::sim_completed;
        }

        if (m_datanodes[slice].m_progress == comm_progress::sent) {
            m_datanodes[slice].m_progress = comm_progress::sim_completed;
        }

        AMREX_ALWAYS_ASSERT(m_datanodes[slice].m_metadata_progress == comm_progress::sim_completed);
        AMREX_ALWAYS_ASSERT(m_datanodes[slice].m_progress == comm_progress::sim_completed);
    }
    if (m_time_send_started) {
        MPI_Wait(&m_time_send_request, MPI_STATUS_IGNORE);
        m_time_send_started = false;
    }
#endif
}

void MultiBuffer::make_progress (int slice, bool is_blocking, int current_slice) {
    const bool is_first_slice_with_recv_data =
        m_async_data_slice[comm_progress::receive_started] == slice;
    const bool is_last_slice_with_send_data =
        m_async_data_slice[comm_progress::sent] == slice;
    const bool is_blocking_send = is_blocking ||
        ((m_nslices + slice - current_slice) % m_nslices > m_max_trailing_slices) ||
        (is_last_slice_with_send_data && (m_current_buffer_size > m_max_buffer_size));
    const bool is_blocking_recv = is_blocking;
    const bool skip_recv = !is_blocking_recv && (slice == current_slice ||
        (m_nslices - slice + current_slice) % m_nslices > m_max_leading_slices);

    if (m_is_serial) {
        if (is_blocking) {
            // send buffer to myself
            AMREX_ALWAYS_ASSERT(m_datanodes[slice].m_metadata_progress == comm_progress::ready_to_send);
            AMREX_ALWAYS_ASSERT(m_datanodes[slice].m_progress == comm_progress::ready_to_send);
            m_datanodes[slice].m_metadata_progress = comm_progress::received;
            m_datanodes[slice].m_progress = comm_progress::received;
        }
        return;
    }

#ifdef AMREX_USE_MPI

    if (m_datanodes[slice].m_metadata_progress == comm_progress::ready_to_send) {
        MPI_Isend(
            get_metadata_location(slice),
            get_metadata_size(),
            amrex::ParallelDescriptor::Mpi_typemap<std::size_t>::type(),
            m_rank_send_to,
            m_tag_metadata_start + slice,
            m_comm,
            &(m_datanodes[slice].m_metadata_request));
        m_datanodes[slice].m_metadata_progress = comm_progress::send_started;
    }

    if (m_datanodes[slice].m_progress == comm_progress::ready_to_send) {
        if (m_datanodes[slice].m_buffer_size == 0) {
            // don't send empty buffer
            m_datanodes[slice].m_progress = comm_progress::sent;
        } else {
            MPI_Isend(
                m_datanodes[slice].m_buffer,
                m_datanodes[slice].m_buffer_size,
                amrex::ParallelDescriptor::Mpi_typemap<storage_type>::type(),
                m_rank_send_to,
                m_tag_buffer_start + slice,
                m_comm,
                &(m_datanodes[slice].m_request));
            m_datanodes[slice].m_progress = comm_progress::send_started;
        }
    }

    if (m_datanodes[slice].m_metadata_progress == comm_progress::send_started) {
        if (is_blocking_send) {
            MPI_Wait(&(m_datanodes[slice].m_metadata_request), MPI_STATUS_IGNORE);
            m_datanodes[slice].m_metadata_progress = comm_progress::sent;
        } else {
            int is_complete = false;
            MPI_Test(&(m_datanodes[slice].m_metadata_request), &is_complete, MPI_STATUS_IGNORE);
            if (is_complete) {
                m_datanodes[slice].m_metadata_progress = comm_progress::sent;
            }
        }
    }

    if (m_datanodes[slice].m_metadata_progress == comm_progress::sent && !skip_recv) {
        MPI_Irecv(
            get_metadata_location(slice),
            get_metadata_size(),
            amrex::ParallelDescriptor::Mpi_typemap<std::size_t>::type(),
            m_rank_receive_from,
            m_tag_metadata_start + slice,
            m_comm,
            &(m_datanodes[slice].m_metadata_request));
        m_datanodes[slice].m_metadata_progress = comm_progress::receive_started;
    }

    if (m_datanodes[slice].m_metadata_progress == comm_progress::receive_started) {
        if (is_blocking_recv) {
            MPI_Wait(&(m_datanodes[slice].m_metadata_request), MPI_STATUS_IGNORE);
            m_datanodes[slice].m_metadata_progress = comm_progress::received;
        } else {
            int is_complete = false;
            MPI_Test(&(m_datanodes[slice].m_metadata_request), &is_complete, MPI_STATUS_IGNORE);
            if (is_complete) {
                m_datanodes[slice].m_metadata_progress = comm_progress::received;
            }
        }
    }

    if (m_datanodes[slice].m_progress == comm_progress::send_started) {
        if (is_blocking_send) {
            MPI_Wait(&(m_datanodes[slice].m_request), MPI_STATUS_IGNORE);
            free_buffer(slice);
            m_datanodes[slice].m_progress = comm_progress::sent;
        } else {
            int is_complete = false;
            MPI_Test(&(m_datanodes[slice].m_request), &is_complete, MPI_STATUS_IGNORE);
            if (is_complete) {
                free_buffer(slice);
                m_datanodes[slice].m_progress = comm_progress::sent;
            }
        }
    }

    if (m_datanodes[slice].m_progress == comm_progress::sent &&
        m_datanodes[slice].m_metadata_progress == comm_progress::received) {

        AMREX_ALWAYS_ASSERT(m_datanodes[slice].m_location == memory_location::nowhere);

        m_datanodes[slice].m_buffer_size = get_metadata_location(slice)[0];

        if (m_datanodes[slice].m_buffer_size == 0) {
            // don't receive empty buffer
            m_datanodes[slice].m_progress = comm_progress::received;
        } else {
            // enforce that slices are received in order
            if (is_blocking_recv || (is_first_slice_with_recv_data &&
                (m_current_buffer_size + m_datanodes[slice].m_buffer_size * sizeof(storage_type)
                <= m_max_buffer_size))) {
                allocate_buffer(slice);
                MPI_Irecv(
                    m_datanodes[slice].m_buffer,
                    m_datanodes[slice].m_buffer_size,
                    amrex::ParallelDescriptor::Mpi_typemap<storage_type>::type(),
                    m_rank_receive_from,
                    m_tag_buffer_start + slice,
                    m_comm,
                    &(m_datanodes[slice].m_request));
                m_datanodes[slice].m_progress = comm_progress::receive_started;
            }
        }
    }

    if (m_datanodes[slice].m_progress == comm_progress::receive_started) {
        if (is_blocking_recv) {
            MPI_Wait(&(m_datanodes[slice].m_request), MPI_STATUS_IGNORE);
            m_datanodes[slice].m_progress = comm_progress::received;
        } else {
            int is_complete = false;
            MPI_Test(&(m_datanodes[slice].m_request), &is_complete, MPI_STATUS_IGNORE);
            if (is_complete) {
                m_datanodes[slice].m_progress = comm_progress::received;
            }
        }
    }

    if (is_blocking_recv) {
        AMREX_ALWAYS_ASSERT(m_datanodes[slice].m_metadata_progress == comm_progress::received);
        AMREX_ALWAYS_ASSERT(m_datanodes[slice].m_progress == comm_progress::received);
    }

#endif
}

void MultiBuffer::get_data (int slice, MultiBeam& beams, MultiLaser& laser, int beam_slice) {
    HIPACE_PROFILE("MultiBuffer::get_data()");
    if (m_datanodes[slice].m_progress == comm_progress::ready_to_define) {
        // initialize MultiBeam and MultiLaser per slice on the first timestep
        for (int b = 0; b < m_nbeams; ++b) {
            beams.getBeam(b).initializeSlice(slice, beam_slice);
        }
        if (laser.UseLaser(slice)) {
            using namespace WhichLaserSlice;
            const int laser_comp = (beam_slice == WhichBeamSlice::Next) ? n00jp2_r : n00j00_r;
            laser.InitSliceEnvelope(slice, laser_comp);
        }
    } else {
        // receive and unpack buffer
        if (m_async_memcpy) {
            if (slice == m_nslices - 1) {
                // receive fist slice
                make_progress(slice, true, slice);
                if (m_datanodes[slice].m_buffer_size != 0) {
                    async_memcpy_from_buffer(slice);
                }
            }

            if (m_datanodes[slice].m_buffer_size != 0) {
                async_memcpy_from_buffer_finish();
                unpack_data(slice, beams, laser, beam_slice);
                free_buffer(slice);
            }

            if (slice > 0) {
                // receive next slice and start async memcpy
                make_progress(slice-1, true, slice);
                if (m_datanodes[slice-1].m_buffer_size != 0) {
                    async_memcpy_from_buffer(slice-1);
                }
            }
        } else {
            make_progress(slice, true, slice);
            if (m_datanodes[slice].m_buffer_size != 0) {
                unpack_data(slice, beams, laser, beam_slice);
                free_buffer(slice);
            }
        }
    }
    m_datanodes[slice].m_progress = comm_progress::in_use;
    m_datanodes[slice].m_metadata_progress = comm_progress::in_use;
}

void MultiBuffer::put_data (int slice, MultiBeam& beams, MultiLaser& laser, int beam_slice,
                            bool is_last_time_step) {
    HIPACE_PROFILE("MultiBuffer::put_data()");
    if (is_last_time_step) {
        // don't send buffer on the last step
        m_datanodes[slice].m_progress = comm_progress::sim_completed;
        m_datanodes[slice].m_metadata_progress = comm_progress::sim_completed;
    } else {
        // pack and asynchronously send buffer
        write_metadata(slice, beams, laser, beam_slice);
        m_datanodes[slice].m_metadata_progress = comm_progress::ready_to_send;
        if (m_async_memcpy) {
            if (slice < m_nslices - 1) {
                // finish memcpy of previous slice
                if (m_datanodes[slice+1].m_buffer_size != 0) {
                    async_memcpy_to_buffer_finish();
                }
                m_datanodes[slice+1].m_progress = comm_progress::ready_to_send;
            }

            if (m_datanodes[slice].m_buffer_size != 0) {
                allocate_buffer(slice);
                m_trailing_gpu_buffer.resize(0);
                m_trailing_gpu_buffer.resize(m_datanodes[slice].m_buffer_size*sizeof(storage_type));
                pack_data(slice, beams, laser, beam_slice);
                async_memcpy_to_buffer(slice);
            }

            if (slice == 0) {
                // finish memcpy of last slice
                if (m_datanodes[slice].m_buffer_size != 0) {
                    async_memcpy_to_buffer_finish();
                }
                m_datanodes[slice].m_progress = comm_progress::ready_to_send;
            }
        } else {
            if (m_datanodes[slice].m_buffer_size != 0) {
                allocate_buffer(slice);
                pack_data(slice, beams, laser, beam_slice);
            }
            m_datanodes[slice].m_progress = comm_progress::ready_to_send;
        }
    }

    make_progress(slice, false, slice);

    // make asynchronous progress for metadata
    // only check slices that have a chance of making progress
    for (int p=comm_progress::async_progress_end-1; p>comm_progress::async_progress_begin; --p) {
        if (p == comm_progress::async_progress_end-1) {
            // first progress type starts at slice-1 or where it last stopped
            if (m_async_metadata_slice[p] == slice) {
                if (slice == 0) {
                    m_async_metadata_slice[p] = m_nslices - 1;
                } else {
                    --m_async_metadata_slice[p];
                }
            }
        } else {
            // all other progress types start at the minimum of where they or
            // the previous progress type last stopped
            if ((m_async_metadata_slice[p+1] < slice) == (m_async_metadata_slice[p] <= slice)) {
                if (m_async_metadata_slice[p+1] < m_async_metadata_slice[p]) {
                    m_async_metadata_slice[p] = m_async_metadata_slice[p+1];
                }
            } else if (m_async_metadata_slice[p+1] > slice && m_async_metadata_slice[p] <= slice) {
                m_async_metadata_slice[p] = m_async_metadata_slice[p+1];
            }
        }

        // start at slice-1 (next slice), iterate backwards, loop around, stop at slice+1
        for (int i = m_async_metadata_slice[p]; i!=slice; (i==0) ? i=m_nslices-1 : --i) {
            m_async_metadata_slice[p] = i;
            if (m_datanodes[i].m_metadata_progress < p) {
                make_progress(i, false, slice);
            }
            if (m_datanodes[i].m_metadata_progress < p) {
                break;
            }
        }
    }

    // make asynchronous progress for data
    // only check slices that have a chance of making progress
    for (int p=comm_progress::async_progress_end-1; p>comm_progress::async_progress_begin; --p) {
        if (p == comm_progress::async_progress_end-1) {
            // first progress type starts at slice-1 or where it last stopped
            if (m_async_data_slice[p] == slice) {
                if (slice == 0) {
                    m_async_data_slice[p] = m_nslices - 1;
                } else {
                    --m_async_data_slice[p];
                }
            }
        } else {
            // all other progress types start at the minimum of where they or
            // the previous progress type last stopped
            if ((m_async_data_slice[p+1] < slice) == (m_async_data_slice[p] <= slice)) {
                if (m_async_data_slice[p+1] < m_async_data_slice[p]) {
                    m_async_data_slice[p] = m_async_data_slice[p+1];
                }
            } else if (m_async_data_slice[p+1] > slice && m_async_data_slice[p] <= slice) {
                m_async_data_slice[p] = m_async_data_slice[p+1];
            }
        }

        // start at slice-1 (next slice), iterate backwards, loop around, stop at slice+1
        for (int i = m_async_data_slice[p]; i!=slice; (i==0) ? i=m_nslices-1 : --i) {
            m_async_data_slice[p] = i;
            if (m_datanodes[i].m_progress < p) {
                make_progress(i, false, slice);
            }
            if (m_datanodes[i].m_progress < p) {
                break;
            }
        }
    }
}

amrex::Real MultiBuffer::get_time () {
    if (m_is_serial) {
        return m_time_send_buffer;
    }

#ifdef AMREX_USE_MPI
    amrex::Real time_buffer = 0.;
    MPI_Recv(
        &time_buffer,
        1,
        amrex::ParallelDescriptor::Mpi_typemap<amrex::Real>::type(),
        m_rank_receive_from,
        m_tag_time_start,
        m_comm,
        MPI_STATUS_IGNORE);
    return time_buffer;
#else
    return m_time_send_buffer;
#endif
}

void MultiBuffer::put_time (amrex::Real time) {
    if (m_is_serial) {
        m_time_send_buffer = time;
        return;
    }

#ifdef AMREX_USE_MPI
    if (m_time_send_started) {
        MPI_Wait(&m_time_send_request, MPI_STATUS_IGNORE);
        m_time_send_started = false;
    }
    m_time_send_buffer = time;
    MPI_Isend(
        &m_time_send_buffer,
        1,
        amrex::ParallelDescriptor::Mpi_typemap<amrex::Real>::type(),
        m_rank_send_to,
        m_tag_time_start,
        m_comm,
        &m_time_send_request);
    m_time_send_started = true;
#endif
}

void MultiBuffer::write_metadata (int slice, MultiBeam& beams, MultiLaser& laser, int beam_slice) {
    for (int b = 0; b < m_nbeams; ++b) {
        // write number of beam particles (per beam)
        get_metadata_location(slice)[b + 1] = beams.getBeam(b).getNumParticles(beam_slice);
    }
    std::size_t offset = get_buffer_offset(slice, offset_type::total, beams, laser, 0, 0);
    // write total buffer size
    get_metadata_location(slice)[0] = (offset+sizeof(storage_type)-1) / sizeof(storage_type);
    m_datanodes[slice].m_buffer_size = get_metadata_location(slice)[0];
    // MPI uses int as index type so check it wont overflow,
    // we use a 64 byte storage type for this reason
    AMREX_ALWAYS_ASSERT(get_metadata_location(slice)[0] < std::numeric_limits<int>::max());
}

std::size_t MultiBuffer::get_buffer_offset (int slice, offset_type type, MultiBeam& beams,
                                            MultiLaser& laser, int ibeam, int comp) {
    // calculate offset for each chunk of data in one place
    // to ensure consistency between packing and unpacking
    std::size_t offset = 0;

    for (int b = 0; b < m_nbeams; ++b) {
        auto& beam = beams.getBeam(b);
        // Roundup the number of particles to a value that ensures proper alignment between types
        const int num_particles_round_up = (get_metadata_location(slice)[b + 1]
            + buffer_size_roundup - 1) / buffer_size_roundup * buffer_size_roundup;

        // add offset for idcpu, if used
        if (beam.communicateIdCpuComponent()) {
            if (type == offset_type::beam_idcpu && ibeam == b) {
                return offset;
            }
            offset += num_particles_round_up * sizeof(std::uint64_t);
        }

        // add offset for real components, if used
        for (int rcomp = 0; rcomp < beam.numRealComponents(); ++rcomp) {
            if (beam.communicateRealComponent(rcomp)) {
                if (type == offset_type::beam_real && ibeam == b && rcomp == comp) {
                    return offset;
                }
                offset += num_particles_round_up * sizeof(amrex::Real);
            }
        }

        // add offset for int components, if used
        for (int icomp = 0; icomp < beam.numIntComponents(); ++icomp) {
            if (beam.communicateIntComponent(icomp)) {
                if (type == offset_type::beam_int && ibeam == b && icomp == comp) {
                    return offset;
                }
                offset += num_particles_round_up * sizeof(int);
            }
        }
    }

    // add offset for laser, if used
    if (laser.UseLaser(slice)) {
        for (int lcomp = 0; lcomp < m_laser_ncomp; ++lcomp) {
            if (type == offset_type::laser && lcomp == comp) {
                return offset;
            }
            offset += laser.getSlices()[0].box().numPts() * sizeof(amrex::Real);
        }
    }

    if (type == offset_type::total) {
        return offset;
    }

    // requested component is not supposed to be communicated, abort
    amrex::Abort("MultiBuffer::get_buffer_offset invalid argument");
    return 0;
}

void MultiBuffer::memcpy_to_buffer (int slice, std::size_t buffer_offset,
                                    const void* src_ptr, std::size_t num_bytes) {
#ifdef AMREX_USE_GPU
    if (m_async_memcpy) {
        amrex::Gpu::dtod_memcpy_async(
            m_trailing_gpu_buffer.dataPtr() + buffer_offset, src_ptr, num_bytes);
    } else if (m_datanodes[slice].m_location == memory_location::pinned) {
        amrex::Gpu::dtoh_memcpy_async(
            m_datanodes[slice].m_buffer + buffer_offset, src_ptr, num_bytes);
    } else {
        amrex::Gpu::dtod_memcpy_async(
            m_datanodes[slice].m_buffer + buffer_offset, src_ptr, num_bytes);
    }
#else
    std::memcpy(m_datanodes[slice].m_buffer + buffer_offset, src_ptr, num_bytes);
#endif
}

void MultiBuffer::memcpy_from_buffer (int slice, std::size_t buffer_offset,
                                      void* dst_ptr, std::size_t num_bytes) {
#ifdef AMREX_USE_GPU
    if (m_async_memcpy) {
        amrex::Gpu::dtod_memcpy_async(
            dst_ptr, m_leading_gpu_buffer.dataPtr() + buffer_offset, num_bytes);
    } else if (m_datanodes[slice].m_location == memory_location::pinned) {
        amrex::Gpu::htod_memcpy_async(
            dst_ptr, m_datanodes[slice].m_buffer + buffer_offset, num_bytes);
    } else {
        amrex::Gpu::dtod_memcpy_async(
            dst_ptr, m_datanodes[slice].m_buffer + buffer_offset, num_bytes);
    }
#else
    std::memcpy(dst_ptr, m_datanodes[slice].m_buffer + buffer_offset, num_bytes);
#endif
}

void MultiBuffer::async_memcpy_to_buffer (int slice) {
    std::size_t num_bytes = m_datanodes[slice].m_buffer_size * sizeof(storage_type);

    amrex::Gpu::Device::setStreamIndex(1);
#ifdef AMREX_USE_GPU
    amrex::Gpu::dtoh_memcpy_async(
        m_datanodes[slice].m_buffer, m_trailing_gpu_buffer.dataPtr(), num_bytes);
#else
    std::memcpy(m_datanodes[slice].m_buffer, m_trailing_gpu_buffer.dataPtr(), num_bytes);
#endif
    amrex::Gpu::Device::resetStreamIndex();
}

void MultiBuffer::async_memcpy_from_buffer (int slice) {
    std::size_t num_bytes = m_datanodes[slice].m_buffer_size * sizeof(storage_type);
    m_leading_gpu_buffer.resize(0);
    m_leading_gpu_buffer.resize(num_bytes);

    amrex::Gpu::Device::setStreamIndex(2);
#ifdef AMREX_USE_GPU
    amrex::Gpu::htod_memcpy_async(
        m_leading_gpu_buffer.dataPtr(), m_datanodes[slice].m_buffer, num_bytes);
#else
    std::memcpy(m_leading_gpu_buffer.dataPtr(), m_datanodes[slice].m_buffer, num_bytes);
#endif
    amrex::Gpu::Device::resetStreamIndex();
}

void MultiBuffer::async_memcpy_to_buffer_finish () {
    amrex::Gpu::Device::setStreamIndex(1);
    amrex::Gpu::streamSynchronize();
    amrex::Gpu::Device::resetStreamIndex();
}

void MultiBuffer::async_memcpy_from_buffer_finish () {
    amrex::Gpu::Device::setStreamIndex(2);
    amrex::Gpu::streamSynchronize();
    amrex::Gpu::Device::resetStreamIndex();
}

void MultiBuffer::pack_data (int slice, MultiBeam& beams, MultiLaser& laser, int beam_slice) {
    for (int b = 0; b < m_nbeams; ++b) {
        auto& beam = beams.getBeam(b);
        const int num_particles = beam.getNumParticles(beam_slice);
        auto& soa = beam.getBeamSlice(beam_slice).GetStructOfArrays();

        if (beam.communicateIdCpuComponent()) {
            // only pack idcpu component if it should be communicated
            memcpy_to_buffer(slice, get_buffer_offset(slice, offset_type::beam_idcpu,
                                                      beams, laser, b, 0),
                             soa.GetIdCPUData().dataPtr(),
                             num_particles * sizeof(std::uint64_t));
        }

        for (int rcomp = 0; rcomp < beam.numRealComponents(); ++rcomp) {
            // only pack real component if it should be communicated
            if (beam.communicateRealComponent(rcomp)) {
                memcpy_to_buffer(slice, get_buffer_offset(slice, offset_type::beam_real,
                                                          beams, laser, b, rcomp),
                                 soa.GetRealData(rcomp).dataPtr(),
                                 num_particles * sizeof(amrex::Real));
            }
        }

        for (int icomp = 0; icomp < beam.numIntComponents(); ++icomp) {
            // only pack int component if it should be communicated
            if (beam.communicateIntComponent(icomp)) {
                memcpy_to_buffer(slice, get_buffer_offset(slice, offset_type::beam_int,
                                                          beams, laser, b, icomp),
                                 soa.GetIntData(icomp).dataPtr(),
                                 num_particles * sizeof(int));
            }
        }
    }
    if (laser.UseLaser(slice)) {
        using namespace WhichLaserSlice;
        const int laser_comp_0_1 = (beam_slice == WhichBeamSlice::Next) ? np1jp2_r : np1j00_r;
        const int laser_comp_2_3 = (beam_slice == WhichBeamSlice::Next) ? n00jp2_r : n00j00_r;
        // copy real and imag components in one operation
        memcpy_to_buffer(slice, get_buffer_offset(slice, offset_type::laser, beams, laser, 0, 0),
                         laser.getSlices()[0].dataPtr(laser_comp_0_1),
                         2 * laser.getSlices()[0].box().numPts() * sizeof(amrex::Real));
        memcpy_to_buffer(slice, get_buffer_offset(slice, offset_type::laser, beams, laser, 0, 2),
                         laser.getSlices()[0].dataPtr(laser_comp_2_3),
                         2 * laser.getSlices()[0].box().numPts() * sizeof(amrex::Real));
    }
    amrex::Gpu::streamSynchronize();
    for (int b = 0; b < m_nbeams; ++b) {
        // remove all beam particles
        beams.getBeam(b).resize(beam_slice, 0, 0);
    }
}

void MultiBuffer::unpack_data (int slice, MultiBeam& beams, MultiLaser& laser, int beam_slice) {
    for (int b = 0; b < m_nbeams; ++b) {
        auto& beam = beams.getBeam(b);
        const int num_particles = get_metadata_location(slice)[b + 1];
        beam.resize(beam_slice, num_particles, 0);
        auto& soa = beam.getBeamSlice(beam_slice).GetStructOfArrays();

        if (beam.communicateIdCpuComponent()) {
            // only undpack idcpu component if it should be communicated
            memcpy_from_buffer(slice, get_buffer_offset(slice, offset_type::beam_idcpu,
                                                        beams, laser, b, 0),
                               soa.GetIdCPUData().dataPtr(),
                               num_particles * sizeof(std::uint64_t));
        } else {
            // if idcpu is not communicated, then we need to initialize it here
            std::uint64_t* data_ptr = soa.GetIdCPUData().dataPtr();
            amrex::ParallelFor(num_particles, [=] AMREX_GPU_DEVICE (int i) {
                amrex::ParticleIDWrapper{data_ptr[i]} = 1;
                amrex::ParticleCPUWrapper{data_ptr[i]} = 0;
            });
        }

        for (int rcomp = 0; rcomp < beam.numRealComponents(); ++rcomp) {
            if (beam.communicateRealComponent(rcomp)) {
                // only unpack real component if it should be communicated
                memcpy_from_buffer(slice, get_buffer_offset(slice, offset_type::beam_real,
                                                            beams, laser, b, rcomp),
                                   soa.GetRealData(rcomp).dataPtr(),
                                   num_particles * sizeof(amrex::Real));
            } else {
                // initialize per-slice-only real components to zero
                amrex::Real* data_ptr = soa.GetRealData(rcomp).dataPtr();
                amrex::ParallelFor(num_particles, [=] AMREX_GPU_DEVICE (int i) {
                    data_ptr[i] = amrex::Real(0.);
                });
            }
        }

        for (int icomp = 0; icomp < beam.numIntComponents(); ++icomp) {
            if (beam.communicateIntComponent(icomp)) {
                // only unpack int component if it should be communicated
                memcpy_from_buffer(slice, get_buffer_offset(slice, offset_type::beam_int,
                                                            beams, laser, b, icomp),
                                   soa.GetIntData(icomp).dataPtr(),
                                   num_particles * sizeof(int));
            } else {
                // initialize per-slice-only int components to zero
                int* data_ptr = soa.GetIntData(icomp).dataPtr();
                amrex::ParallelFor(num_particles, [=] AMREX_GPU_DEVICE (int i) {
                    data_ptr[i] = 0;
                });
            }
        }
    }
    if (laser.UseLaser(slice)) {
        using namespace WhichLaserSlice;
        const int laser_comp_0_1 = (beam_slice == WhichBeamSlice::Next) ? n00jp2_r : n00j00_r;
        const int laser_comp_2_3 = (beam_slice == WhichBeamSlice::Next) ? nm1jp2_r : nm1j00_r;
        // copy real and imag components in one operation
        memcpy_from_buffer(slice, get_buffer_offset(slice, offset_type::laser, beams, laser, 0, 0),
                           laser.getSlices()[0].dataPtr(laser_comp_0_1),
                           2 * laser.getSlices()[0].box().numPts() * sizeof(amrex::Real));
        memcpy_from_buffer(slice, get_buffer_offset(slice, offset_type::laser, beams, laser, 0, 2),
                           laser.getSlices()[0].dataPtr(laser_comp_2_3),
                           2 * laser.getSlices()[0].box().numPts() * sizeof(amrex::Real));
    }
    amrex::Gpu::streamSynchronize();
}
