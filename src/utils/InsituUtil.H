/* Copyright 2022
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn
 * License: BSD-3-Clause-LBNL
 */
#ifndef HIPACE_IOUTIL_H_
#define HIPACE_IOUTIL_H_

#include <AMReX_AmrCore.H>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace insitu_utils
{

struct DataNode {
    std::string m_name;
    std::string m_format;
    const char * m_data_loacation;
    std::size_t m_data_size;

    template<class T>
    DataNode (const std::string& name, const T* data, std::size_t size = 1) : m_name{name} {
        std::string type_letter = "";
        if (std::is_integral_v<T> && std::is_signed_v<T>) type_letter = "i";
        else if (std::is_integral_v<T> && std::is_unsigned_v<T>) type_letter = "u";
        else if (std::is_floating_point_v<T>) type_letter = "f";
        else amrex::Abort("unknown type");

        m_format = "\"" + (size > 1 ? "(" + std::to_string(size) + ",)" : std::string(""))
                 + "<" + type_letter + std::to_string(sizeof(T)) + "\"";
        m_data_loacation = reinterpret_cast<const char *>(data);
        m_data_size = sizeof(T) * size;
    }

    DataNode (const std::string& name, const std::string& format)
        : m_name{name}, m_format{format}, m_data_loacation{nullptr}, m_data_size{0} {}
};

std::string get_header (const amrex::Vector<DataNode>& nodes, const std::string& indent = "") {
    std::stringstream ss;
    ss << "{\n" << indent << "  \"names\":[\n";
    for (unsigned int i=0; i<nodes.size(); ++i) {
        ss << indent << "    \"" << nodes[i].m_name << "\"" << (i+1==nodes.size() ? "\n" : ",\n");
    }
    ss << indent << "  ],\n" << indent << "  \"formats\":[\n";
    for (unsigned int i=0; i<nodes.size(); ++i) {
        ss << indent << "    " << nodes[i].m_format << (i+1==nodes.size() ? "\n" : ",\n");
    }
    ss << indent << "  ]\n" << indent << "}";
    return ss.str();
}

void write_data (const amrex::Vector<DataNode>& nodes, std::ofstream& ofs) {
    for (auto dn : nodes) {
        if (dn.m_data_loacation != nullptr) {
            ofs.write(dn.m_data_loacation, dn.m_data_size);
        }
    }
}

}

#endif
