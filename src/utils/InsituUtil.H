/* Copyright 2022
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn
 * License: BSD-3-Clause-LBNL
 */
#ifndef HIPACE_INSITUUTIL_H_
#define HIPACE_INSITUUTIL_H_

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
    const char * m_data_location;
    std::size_t m_data_size;
    amrex::Vector<DataNode> m_structured_format;

    template<class T>
    DataNode (const std::string& name, const T* data, std::size_t size = 1) : m_name{name} {
        std::string type_letter = "";
        if (std::is_integral_v<T> && std::is_signed_v<T>) type_letter = "i";
        else if (std::is_integral_v<T> && std::is_unsigned_v<T>) type_letter = "u";
        else if (std::is_floating_point_v<T>) type_letter = "f";
        else amrex::Abort("DataNode (): unknown type T");

        m_format = (size > 1 ? "(" + std::to_string(size) + ",)" : "")
                   + "<" + type_letter + std::to_string(sizeof(T));
        m_data_location = reinterpret_cast<const char *>(data);
        m_data_size = sizeof(T) * size;
    }

    DataNode (const std::string& name, const amrex::Vector<DataNode>& structured_format)
        : m_name{name}, m_data_location{nullptr}, m_structured_format{structured_format} {}
};

inline void write_header (const amrex::Vector<DataNode>& nodes, std::ofstream& ofs,
                          const std::string& indent = "") {
    ofs << indent << "{\n" << indent << "    \"names\": [\n";
    for (int i=0; i<nodes.size(); ++i) {
        ofs << indent << "        \"" << nodes[i].m_name << "\"" << (i+1==nodes.size()?"\n":",\n");
    }
    ofs << indent << "    ],\n" << indent << "    \"formats\": [\n";
    for (int i=0; i<nodes.size(); ++i) {
        if (nodes[i].m_data_location) {
            ofs << indent << "        \"" << nodes[i].m_format << "\"";
        } else {
            write_header(nodes[i].m_structured_format, ofs, "        ");
        }
        ofs << (i+1==nodes.size()?"\n":",\n");
    }
    ofs << indent << "    ]\n" << indent << "}";
}

inline void write_data (const amrex::Vector<DataNode>& nodes, std::ofstream& ofs) {
    for (auto& dn : nodes) {
        if (dn.m_data_location) {
            ofs.write(dn.m_data_location, dn.m_data_size);
        } else {
            write_data(dn.m_structured_format, ofs);
        }
    }
}

}

#endif
