/* Copyright 2026
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn
 * License: BSD-3-Clause-LBNL
 */
#ifndef ZAXIS_H_
#define ZAXIS_H_

#include <AMReX_AmrCore.H>

#include <map>

struct ZAxis
{
    amrex::Real m_zlo;
    amrex::Real m_zhi;
    int m_total_num_slices;
    int m_lev0_num_slices;
    int m_helm_num_slices;
    int m_helm_lo;
    int m_helm_hi;
    int m_helm_ref_ratio;

    ZAxis (const amrex::Geometry& lev0_geom, const amrex::Geometry& helm_geom, bool use_helm) {
        m_zlo = lev0_geom.ProbLo(2);
        m_zhi = lev0_geom.ProbHi(2);
        m_lev0_num_slices = lev0_geom.Domain().length(2);
        if (use_helm) {
            m_helm_num_slices = helm_geom.Domain().length(2);
            m_helm_lo = helm_geom.Domain().smallEnd(2);
            m_helm_hi = helm_geom.Domain().bigEnd(2);
            m_helm_ref_ratio = static_cast<int>(std::round(
                helm_geom.InvCellSize(2) * lev0_geom.CellSize(2)));

            m_total_num_slices = m_lev0_num_slices + m_helm_num_slices -
                m_helm_num_slices / m_helm_ref_ratio;
        } else {
            m_total_num_slices = m_lev0_num_slices;
            m_helm_num_slices = 0;
            m_helm_lo = 0;
            m_helm_hi = -1;
            m_helm_ref_ratio = 1;
        }
    }

    ZAxis ()=default;

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real Lev0CellSize () const {
        return (m_zhi - m_zlo) / m_lev0_num_slices;
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real HelmCellSize () const {
        return (m_zhi - m_zlo) / (amrex::Real(m_lev0_num_slices) * m_helm_ref_ratio);
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real Lev0InvCellSize () const {
        return amrex::Real(m_lev0_num_slices) / (m_zhi - m_zlo);
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real HelmInvCellSize () const {
        return (amrex::Real(m_lev0_num_slices) * m_helm_ref_ratio) / (m_zhi - m_zlo);
    }

    std::pair<int, int> SplitSlice (int islice_global) const {
        if (m_helm_num_slices > 0) {
            if (islice_global < (m_helm_lo / m_helm_ref_ratio)) {
                return {islice_global, -1};
            } else if (islice_global < (m_helm_lo / m_helm_ref_ratio) + m_helm_num_slices) {
                const int islice_helm = islice_global - (m_helm_lo / m_helm_ref_ratio)
                    + m_helm_lo;
                const int islice_lev0 = islice_helm % m_helm_ref_ratio == m_helm_ref_ratio / 2 ?
                    islice_helm / m_helm_ref_ratio : -1;
                return {islice_lev0, islice_helm};
            } else {
                return {
                    islice_global - m_helm_num_slices + m_helm_num_slices / m_helm_ref_ratio,
                    -1
                };
            }
        } else {
            return {islice_global, -1};
        }
    }

    amrex::Real TotalMinZ (int islice_global) const {
        if (m_helm_num_slices > 0) {
            if (islice_global < (m_helm_lo / m_helm_ref_ratio)) {
                return m_zlo + islice_global * Lev0CellSize();
            } else if (islice_global < (m_helm_lo / m_helm_ref_ratio) + m_helm_num_slices) {
                const int islice_helm = islice_global - (m_helm_lo / m_helm_ref_ratio) + m_helm_lo;
                return m_zlo + islice_helm  * HelmCellSize();
            } else {
                return m_zlo + (islice_global - m_helm_num_slices
                    + m_helm_num_slices / m_helm_ref_ratio) * Lev0CellSize();
            }
        } else {
            return m_zlo + islice_global * Lev0CellSize();
        }
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    int ZToGlobalSlice (amrex::Real z) const {
        if (m_helm_num_slices > 0) {
            int idx_helm = static_cast<int>(std::floor((z - m_zlo) * HelmInvCellSize()));
            if (idx_helm < m_helm_lo) {
                return static_cast<int>(std::floor((z - m_zlo) * Lev0InvCellSize()));
            } else if (idx_helm <= m_helm_hi) {
                return idx_helm - m_helm_lo + (m_helm_lo / m_helm_ref_ratio);
            } else {
                return static_cast<int>(std::floor((z - m_zlo) * Lev0InvCellSize()))
                    + m_helm_num_slices - m_helm_num_slices / m_helm_ref_ratio;
            }
        } else {
            return static_cast<int>(std::floor((z - m_zlo) * Lev0InvCellSize()));
        }
    }
};

#endif // ZAXIS_H_
