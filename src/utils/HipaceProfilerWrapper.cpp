/* Copyright 2022
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn, MaxThevenet
 * License: BSD-3-Clause-LBNL
 */

#include "HipaceProfilerWrapper.H"
#include <AMReX.H>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <algorithm>



StreamProfilerNode* StreamProfilerNode::m_root = nullptr;
StreamProfilerNode* StreamProfilerNode::m_stack_location = nullptr;

void StreamProfilerNode::Initialize () {
    StreamProfilerNode::m_root = new StreamProfilerNode();
    StreamProfilerNode::m_stack_location = StreamProfilerNode::m_root;
    m_root->start();
}

void StreamProfilerNode::Finalize () {
    m_root->stop();
    StreamProfilerNode::m_root->FinishEvents();
    StreamProfilerNode::m_root->GetExcData();

    std::vector<std::array<std::string,6>> str_arr{};
    m_root->PrintTime(str_arr, "", "", m_root->m_time_inc);

    str_arr[0] = {"Name", "NCalls", "Inc. Time", "Inc. std %", "Excl. Time", "Excl. %"};

    std::array<int, 6> num_chars{0,0,0,0,0,0};
    for (int i=0; i<str_arr.size(); ++i) {
        for (int j=0; j<num_chars.size(); ++j) {
            num_chars[j] = std::max<int>(num_chars[j], str_arr[i][j].size());
        }
    }

    for (int j=1; j<num_chars.size(); ++j) {
        num_chars[j] += 2;
    }

    amrex::OutStream() << "\n\n";

    for (int i=0; i<str_arr.size(); ++i) {
        if (i==0 ||i==1) {
            for (int j=0; j<num_chars.size(); ++j) {
                for (int c=0; c<num_chars[j]; ++c) {
                    amrex::OutStream() << '-';
                }
            }
            amrex::OutStream() << '\n';
        }

        for (int j=0; j<num_chars.size(); ++j) {
            if (j==0) {
                amrex::OutStream() << std::setw(num_chars[j]) << std::left << str_arr[i][j];
            } else {
                amrex::OutStream() << std::setw(num_chars[j]) << std::right << str_arr[i][j];
            }
        }
        amrex::OutStream() << '\n';
    }


    delete StreamProfilerNode::m_root;
}

void StreamProfilerNode::GetExcData () {
    m_time_exc = m_time_inc;
    for (auto& [fname, node] : m_children) {
        m_time_exc -= node.m_time_inc;
        node.GetExcData();
    }
}

void StreamProfilerNode::FinishEvents () {
    for (auto ncalls = m_ncalls; ncalls<(m_ncalls+m_nevents); ++ncalls) {
        const int ievent = ncalls % m_nevents;
        if (ncalls >= m_nevents) {
            float time = 0.;
            AMREX_CUDA_SAFE_CALL(cudaEventSynchronize(m_stop[ievent]));
            AMREX_CUDA_SAFE_CALL(cudaEventElapsedTime(&time, m_start[ievent], m_stop[ievent]));
            m_time_inc += time/1000.;
            m_time_inc_sq += time/1000.*time/1000.;
        }
    }

    for (auto& [fname, node] : m_children) {
        node.FinishEvents();
    }
}

void StreamProfilerNode::PrintTime (std::vector<std::array<std::string,6>>& str_arr,
                                    const std::string& prefix, const std::string& name,
                                    double time_total) {

    std::stringstream ss1{};
    ss1 << std::setprecision(4) << m_time_inc;
    std::stringstream ss2{};
    ss2 << "+-" << std::fixed << std::setprecision(2) << (100.*std::sqrt(
        std::max(m_ncalls*m_time_inc_sq - m_time_inc*m_time_inc, 0.)/m_time_inc)) << "%";
    std::stringstream ss3{};
    ss3 << std::setprecision(4) << m_time_exc;
    std::stringstream ss4{};
    ss4 << std::fixed << std::setprecision(2) << (100.*m_time_exc/time_total) << "%";

    str_arr.push_back({
        prefix + name,
        std::to_string(m_ncalls),
        ss1.str(),
        ss2.str(),
        ss3.str(),
        ss4.str()
    });

    std::vector<decltype(m_children)::value_type*> node_vect{m_children.size()};

    unsigned long long i = 0;
    for (auto& val : m_children) {
        node_vect[i++] = &val;
    }

    std::sort(node_vect.begin(), node_vect.end(), [](auto* a, auto* b){
        return a->second.m_time_inc > b->second.m_time_inc;
    });

    for (i=0; i<node_vect.size(); ++i) {
        std::string new_prefix = prefix + "|-";
        if (prefix.size() >= 2) {
            if (prefix[prefix.size()-1] == '-' && prefix[prefix.size()-2] == '|') {
                new_prefix[prefix.size()-2] = '|';
                new_prefix[prefix.size()-1] = ' ';
            } else if (prefix[prefix.size()-2] == '\\') {
                new_prefix[prefix.size()-2] = ' ';
                new_prefix[prefix.size()-1] = ' ';
            }
        }
        if (i+1==node_vect.size()) {
            new_prefix[new_prefix.size()-2] = '\\';
            new_prefix[new_prefix.size()-1] = '-';
        }

        node_vect[i]->second.PrintTime(str_arr, new_prefix, node_vect[i]->first, time_total);
    }
}