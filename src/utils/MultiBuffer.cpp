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


std::size_t MultiBuffer::get_metadata_size () {
    return 1 + m_nbeams;
}

std::size_t* MultiBuffer::get_metadata_location (int slice) {
    return m_metadata.dataPtr() + slice*get_metadata_size();
}

void MultiBuffer::allocate_buffer (int slice) {
    AMREX_ALWAYS_ASSERT(m_datanodes[slice].m_location == memory_location::nowhere);
    if (m_buffer_on_host) {
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
}

void MultiBuffer::free_buffer (int slice) {
    AMREX_ALWAYS_ASSERT(m_datanodes[slice].m_location != memory_location::nowhere);
    if (m_datanodes[slice].m_location == memory_location::pinned) {
        amrex::The_Pinned_Arena()->free(m_datanodes[slice].m_buffer);
    } else {
        amrex::The_Device_Arena()->free(m_datanodes[slice].m_buffer);
    }
    m_datanodes[slice].m_location = memory_location::nowhere;
    m_datanodes[slice].m_buffer = nullptr;
    m_datanodes[slice].m_buffer_size = 0;
}

void MultiBuffer::initialize (int nslices, int nbeams, bool buffer_on_host, bool use_laser,
                              amrex::Box laser_box) {

    m_comm = amrex::ParallelDescriptor::Communicator();
    const int rank_id = amrex::ParallelDescriptor::MyProc();
    const int n_ranks = amrex::ParallelDescriptor::NProcs();

    m_nslices = nslices;
    m_nbeams = nbeams;
    m_buffer_on_host = buffer_on_host;
    m_use_laser = use_laser;
    m_laser_slice_box = laser_box;

    m_rank_send_to = (rank_id - 1 + n_ranks) % n_ranks;
    m_rank_receive_from = (rank_id + 1) % n_ranks;

    m_is_head_rank = rank_id + 1 == n_ranks;
    m_is_serial = n_ranks == 1;

    m_tag_time_start = 0;
    m_tag_buffer_start = 1;
    m_tag_metadata_start = m_tag_buffer_start + m_nslices;

    for (int p = 0; p < comm_progress::nprogress; ++p) {
        m_async_metadata_slice[p] = m_nslices - 1;
        m_async_data_slice[p] = m_nslices - 1;
    }

    m_metadata.resize(get_metadata_size() * m_nslices);
    m_datanodes.resize(m_nslices);

    if (m_is_head_rank) {
        for (int i = m_nslices-1; i >= 0; --i) {
            m_datanodes[i].m_progress = comm_progress::ready_to_define;
            m_datanodes[i].m_metadata_progress = comm_progress::ready_to_define;
        }
    } else {
        for (int i = m_nslices-1; i >= 0; --i) {
            m_datanodes[i].m_progress = comm_progress::sent;
            m_datanodes[i].m_metadata_progress = comm_progress::sent;
        }
    }

    for (int i = m_nslices-1; i >= 0; --i) {
        make_progress(i, false);
    }
}


MultiBuffer::~MultiBuffer() {
#ifdef AMREX_USE_MPI
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

void MultiBuffer::make_progress (int slice, bool is_blocking) {

    if (m_is_serial) {
        if (is_blocking) {
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
        if (is_blocking) {
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

    if (m_datanodes[slice].m_metadata_progress == comm_progress::sent) {
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
        if (is_blocking) {
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
        if (is_blocking) {
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
            m_datanodes[slice].m_progress = comm_progress::received;
        } else {
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

    if (m_datanodes[slice].m_progress == comm_progress::receive_started) {
        if (is_blocking) {
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

    if (is_blocking) {
        AMREX_ALWAYS_ASSERT(m_datanodes[slice].m_metadata_progress == comm_progress::received);
        AMREX_ALWAYS_ASSERT(m_datanodes[slice].m_progress == comm_progress::received);
    }

#endif
}

void MultiBuffer::get_data (int slice, MultiBeam& beams, MultiLaser& laser, int beam_slice) {
    HIPACE_PROFILE("MultiBuffer::get_data()");
    if (m_datanodes[slice].m_progress == comm_progress::ready_to_define) {
        for (int b = 0; b < m_nbeams; ++b) {
            beams.getBeam(b).intializeSlice(slice, beam_slice);
        }
        if (m_use_laser) {
            using namespace WhichLaserSlice;
            const int laser_comp = (beam_slice == WhichBeamSlice::Next) ? n00jp2_r : n00j00_r;
            laser.InitSliceEnvelope(slice, laser_comp);
        }
    } else {
        make_progress(slice, true);
        if (m_datanodes[slice].m_buffer_size != 0) {
            unpack_data(slice, beams, laser, beam_slice);
            free_buffer(slice);
        }
    }
    m_datanodes[slice].m_progress = comm_progress::in_use;
    m_datanodes[slice].m_metadata_progress = comm_progress::in_use;
}

void MultiBuffer::put_data (int slice, MultiBeam& beams, MultiLaser& laser, int beam_slice,
                            bool is_last_time_step) {
    HIPACE_PROFILE("MultiBuffer::put_data()");
    if (is_last_time_step) {
        m_datanodes[slice].m_progress = comm_progress::sim_completed;
        m_datanodes[slice].m_metadata_progress = comm_progress::sim_completed;
    } else {
        write_metadata(slice, beams, beam_slice);
        if (m_datanodes[slice].m_buffer_size != 0) {
            allocate_buffer(slice);
            pack_data(slice, beams, laser, beam_slice);
        }
        m_datanodes[slice].m_progress = comm_progress::ready_to_send;
        m_datanodes[slice].m_metadata_progress = comm_progress::ready_to_send;
    }

    make_progress(slice, false);

    for (int p=comm_progress::async_progress_end-1; p>comm_progress::async_progress_begin; --p) {
        if (p == comm_progress::async_progress_end-1) {
            if (m_async_metadata_slice[p] == slice) {
                if (slice == 0) {
                    m_async_metadata_slice[p] = m_nslices - 1;
                } else {
                    --m_async_metadata_slice[p];
                }
            }
        } else {
            if ((m_async_metadata_slice[p+1] < slice) == (m_async_metadata_slice[p] <= slice)) {
                if (m_async_metadata_slice[p+1] < m_async_metadata_slice[p]) {
                    m_async_metadata_slice[p] = m_async_metadata_slice[p+1];
                }
            } else if (m_async_metadata_slice[p+1] > slice && m_async_metadata_slice[p] <= slice) {
                m_async_metadata_slice[p] = m_async_metadata_slice[p+1];
            }
        }

        for (int i = m_async_metadata_slice[p]; i!=slice; (i==0) ? i=m_nslices-1 : --i) {
            m_async_metadata_slice[p] = i;
            if (m_datanodes[i].m_metadata_progress < p) {
                make_progress(i, false);
            }
            if (m_datanodes[i].m_metadata_progress < p) {
                break;
            }
        }
    }

    for (int p=comm_progress::async_progress_end-1; p>comm_progress::async_progress_begin; --p) {
        if (p == comm_progress::async_progress_end-1) {
            if (m_async_data_slice[p] == slice) {
                if (slice == 0) {
                    m_async_data_slice[p] = m_nslices - 1;
                } else {
                    --m_async_data_slice[p];
                }
            }
        } else {
            if ((m_async_data_slice[p+1] < slice) == (m_async_data_slice[p] <= slice)) {
                if (m_async_data_slice[p+1] < m_async_data_slice[p]) {
                    m_async_data_slice[p] = m_async_data_slice[p+1];
                }
            } else if (m_async_data_slice[p+1] > slice && m_async_data_slice[p] <= slice) {
                m_async_data_slice[p] = m_async_data_slice[p+1];
            }
        }

        for (int i = m_async_data_slice[p]; i!=slice; (i==0) ? i=m_nslices-1 : --i) {
            m_async_data_slice[p] = i;
            if (m_datanodes[i].m_progress < p) {
                make_progress(i, false);
            }
            if (m_datanodes[i].m_progress < p) {
                break;
            }
        }
    }
}

amrex::Real MultiBuffer::get_time () {
    HIPACE_PROFILE("MultiBuffer::get_time()");

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
    HIPACE_PROFILE("MultiBuffer::put_time()");

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

void MultiBuffer::write_metadata (int slice, MultiBeam& beams, int beam_slice) {
    std::size_t offset = 0;
    for (int b = 0; b < m_nbeams; ++b) {
        const int num_particles = beams.getBeam(b).getNumParticles(beam_slice);
        get_metadata_location(slice)[b + 1] = num_particles;
        offset += ((num_particles + buffer_size_roundup - 1)
                    / buffer_size_roundup * buffer_size_roundup)
                    * BeamIdx::real_nattribs_in_buffer * sizeof(amrex::Real);
        offset += ((num_particles + buffer_size_roundup - 1)
                    / buffer_size_roundup * buffer_size_roundup)
                    * BeamIdx::int_nattribs_in_buffer * sizeof(int);
    }
    if (m_use_laser) {
        offset += (m_laser_slice_box.numPts() * m_laser_ncomp) * sizeof(amrex::Real);
    }
    get_metadata_location(slice)[0] = (offset+sizeof(storage_type)-1) / sizeof(storage_type);
    m_datanodes[slice].m_buffer_size = get_metadata_location(slice)[0];
    AMREX_ALWAYS_ASSERT(get_metadata_location(slice)[0] < std::numeric_limits<int>::max());
}

std::size_t MultiBuffer::get_buffer_offset_real (int slice, int ibeam, int rcomp) {
    std::size_t offset = 0;
    for (int b = 0; b < ibeam; ++b) {
        offset += ((get_metadata_location(slice)[b + 1] + buffer_size_roundup - 1)
                    / buffer_size_roundup * buffer_size_roundup)
                    * BeamIdx::real_nattribs_in_buffer * sizeof(amrex::Real);
        offset += ((get_metadata_location(slice)[b + 1] + buffer_size_roundup - 1)
                    / buffer_size_roundup * buffer_size_roundup)
                    * BeamIdx::int_nattribs_in_buffer * sizeof(int);
    }
    offset += ((get_metadata_location(slice)[ibeam + 1] + buffer_size_roundup - 1)
                / buffer_size_roundup * buffer_size_roundup)
                * rcomp * sizeof(amrex::Real);
    return offset;
}

std::size_t MultiBuffer::get_buffer_offset_int (int slice, int ibeam, int icomp) {
    std::size_t offset = 0;
    for (int b = 0; b < ibeam; ++b) {
        offset += ((get_metadata_location(slice)[b + 1] + buffer_size_roundup - 1)
                    / buffer_size_roundup * buffer_size_roundup)
                    * BeamIdx::real_nattribs_in_buffer * sizeof(amrex::Real);
        offset += ((get_metadata_location(slice)[b + 1] + buffer_size_roundup - 1)
                    / buffer_size_roundup * buffer_size_roundup)
                    * BeamIdx::int_nattribs_in_buffer * sizeof(int);
    }
    offset += ((get_metadata_location(slice)[ibeam + 1] + buffer_size_roundup - 1)
                / buffer_size_roundup * buffer_size_roundup)
                * BeamIdx::real_nattribs_in_buffer * sizeof(amrex::Real);
    offset += ((get_metadata_location(slice)[ibeam + 1] + buffer_size_roundup - 1)
                / buffer_size_roundup * buffer_size_roundup)
                * icomp * sizeof(int);
    return offset;
}

std::size_t MultiBuffer::get_buffer_offset_laser (int slice, int icomp) {
    AMREX_ALWAYS_ASSERT(m_use_laser);
    std::size_t offset = 0;
    for (int b = 0; b < m_nbeams; ++b) {
        offset += ((get_metadata_location(slice)[b + 1] + buffer_size_roundup - 1)
                    / buffer_size_roundup * buffer_size_roundup)
                    * BeamIdx::real_nattribs_in_buffer * sizeof(amrex::Real);
        offset += ((get_metadata_location(slice)[b + 1] + buffer_size_roundup - 1)
                    / buffer_size_roundup * buffer_size_roundup)
                    * BeamIdx::int_nattribs_in_buffer * sizeof(int);
    }
    offset += (m_laser_slice_box.numPts() * icomp) * sizeof(amrex::Real);
    return offset;
}

void MultiBuffer::memcpy_to_buffer (int slice, std::size_t buffer_offset,
                                    const void* src_ptr, std::size_t num_bytes) {
#ifdef AMREX_USE_GPU
    if (m_datanodes[slice].m_location == memory_location::pinned) {
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
    if (m_datanodes[slice].m_location == memory_location::pinned) {
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

void MultiBuffer::pack_data (int slice, MultiBeam& beams, MultiLaser& laser, int beam_slice) {
    for (int b = 0; b < m_nbeams; ++b) {
        const int num_particles = beams.getBeam(b).getNumParticles(beam_slice);
        auto& soa = beams.getBeam(b).getBeamSlice(beam_slice).GetStructOfArrays();
        for (int rcomp = 0; rcomp < BeamIdx::real_nattribs_in_buffer; ++rcomp) {
            memcpy_to_buffer(slice, get_buffer_offset_real(slice, b, rcomp),
                             soa.GetRealData(rcomp).dataPtr(),
                             num_particles * sizeof(amrex::Real));
        }
        for (int icomp = 0; icomp < BeamIdx::int_nattribs_in_buffer; ++icomp) {
            memcpy_to_buffer(slice, get_buffer_offset_int(slice, b, icomp),
                             soa.GetIntData(icomp).dataPtr(),
                             num_particles * sizeof(int));
        }
    }
    if (m_use_laser) {
        using namespace WhichLaserSlice;
        const int laser_comp_0_1 = (beam_slice == WhichBeamSlice::Next) ? np1jp2_r : np1j00_r;
        const int laser_comp_2_3 = (beam_slice == WhichBeamSlice::Next) ? n00jp2_r : n00j00_r;
        memcpy_to_buffer(slice, get_buffer_offset_laser(slice, 0),
                         laser.getSlices()[0].dataPtr(laser_comp_0_1),
                         2 * m_laser_slice_box.numPts() * sizeof(amrex::Real));
        memcpy_to_buffer(slice, get_buffer_offset_laser(slice, 2),
                         laser.getSlices()[0].dataPtr(laser_comp_2_3),
                         2 * m_laser_slice_box.numPts() * sizeof(amrex::Real));
    }
    amrex::Gpu::streamSynchronize();
    for (int b = 0; b < m_nbeams; ++b) {
        beams.getBeam(b).resize(beam_slice, 0, 0);
    }
}

void MultiBuffer::unpack_data (int slice, MultiBeam& beams, MultiLaser& laser, int beam_slice) {
    for (int b = 0; b < m_nbeams; ++b) {
        const int num_particles = get_metadata_location(slice)[b + 1];
        beams.getBeam(b).resize(beam_slice, num_particles, 0);
        auto& soa = beams.getBeam(b).getBeamSlice(beam_slice).GetStructOfArrays();
        for (int rcomp = 0; rcomp < BeamIdx::real_nattribs_in_buffer; ++rcomp) {
            memcpy_from_buffer(slice, get_buffer_offset_real(slice, b, rcomp),
                               soa.GetRealData(rcomp).dataPtr(),
                               num_particles * sizeof(amrex::Real));
        }
        for (int rcomp = BeamIdx::real_nattribs_in_buffer; rcomp<BeamIdx::real_nattribs; ++rcomp) {
            amrex::Real* data_ptr = soa.GetRealData(rcomp).dataPtr();
            amrex::ParallelFor(num_particles, [=] AMREX_GPU_DEVICE (int i) noexcept {
                data_ptr[i] = amrex::Real(0.);
            });
        }
        for (int icomp = 0; icomp < BeamIdx::int_nattribs_in_buffer; ++icomp) {
            memcpy_from_buffer(slice, get_buffer_offset_int(slice, b, icomp),
                               soa.GetIntData(icomp).dataPtr(),
                               num_particles * sizeof(int));
        }
        for (int icomp = BeamIdx::int_nattribs_in_buffer; icomp < BeamIdx::int_nattribs; ++icomp) {
            int* data_ptr = soa.GetIntData(icomp).dataPtr();
            amrex::ParallelFor(num_particles, [=] AMREX_GPU_DEVICE (int i) noexcept {
                data_ptr[i] = 0;
            });
        }
    }
    if (m_use_laser) {
        using namespace WhichLaserSlice;
        const int laser_comp_0_1 = (beam_slice == WhichBeamSlice::Next) ? n00jp2_r : n00j00_r;
        const int laser_comp_2_3 = (beam_slice == WhichBeamSlice::Next) ? nm1jp2_r : nm1j00_r;
        memcpy_from_buffer(slice, get_buffer_offset_laser(slice, 0),
                           laser.getSlices()[0].dataPtr(laser_comp_0_1),
                           2 * m_laser_slice_box.numPts() * sizeof(amrex::Real));
        memcpy_from_buffer(slice, get_buffer_offset_laser(slice, 2),
                           laser.getSlices()[0].dataPtr(laser_comp_2_3),
                           2 * m_laser_slice_box.numPts() * sizeof(amrex::Real));
    }
    amrex::Gpu::streamSynchronize();
}
