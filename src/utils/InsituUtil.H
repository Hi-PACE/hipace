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

// generate NumPy structured datatypes, both JSON describing the type and binary data of the type
struct DataNode {
    std::string m_name; // name of a field
    std::string m_format; // NumPy simple datatype
    const char * m_data_location; // pointer to write binary data
    std::size_t m_data_size; // size in bytes to write binary data
    amrex::Vector<DataNode> m_structured_format; // used if this node contains a structured datatype

    // constructor for nodes containing a simple datatype
    template<class T>
    DataNode (const std::string& name, const T* data, std::size_t size = 1) : m_name{name} {
        std::string type_letter = "";
        if (std::is_integral_v<T> && std::is_signed_v<T>) type_letter = "i"; // int
        else if (std::is_integral_v<T> && std::is_unsigned_v<T>) type_letter = "u"; // unsiged int
        else if (std::is_same_v<T, amrex::GpuComplex<amrex::Real>>) type_letter = "c"; // complex
        else if (std::is_floating_point_v<T>) type_letter = "f"; // float, double
        else amrex::Abort("DataNode (): unknown type T");

        // "(size,)" for sub-arrays, "<" for little-endian
        m_format = (size > 1 ? "(" + std::to_string(size) + ",)" : "")
                   + "<" + type_letter + std::to_string(sizeof(T));
        m_data_location = reinterpret_cast<const char *>(data);
        m_data_size = sizeof(T) * size;
    }

    // constructor for nodes containing a structured datatype
    DataNode (const std::string& name, const amrex::Vector<DataNode>& structured_format)
        : m_name{name}, m_data_location{nullptr}, m_structured_format{structured_format} {}
};

// write JSON header describing the datatype
inline void write_header (const amrex::Vector<DataNode>& nodes, std::ofstream& ofs,
                          const std::string& indent = "") {
    //{
    //    "names": [
    //        ... list of names (strings)
    //    ],
    //    "formats": [
    //        ... list of datatypes (strings or another structured datatype, same as this)
    //    ]
    //}
    ofs << indent << "{\n" << indent << "    \"names\": [\n";
    for (int i=0; i<nodes.size(); ++i) {
        ofs << indent << "        \"" << nodes[i].m_name << "\"" << (i+1==nodes.size()?"\n":",\n");
    }
    ofs << indent << "    ],\n" << indent << "    \"formats\": [\n";
    for (int i=0; i<nodes.size(); ++i) {
        if (nodes[i].m_data_location) {
            ofs << indent << "        \"" << nodes[i].m_format << "\"";
        } else {
            // recursive call for nested structured datatypes
            write_header(nodes[i].m_structured_format, ofs, "        ");
        }
        ofs << (i+1==nodes.size()?"\n":",\n");
    }
    ofs << indent << "    ]\n" << indent << "}";
}

// write binary data in the order of the structured datatype
inline void write_data (const amrex::Vector<DataNode>& nodes, std::ofstream& ofs) {
    for (auto& dn : nodes) {
        if (dn.m_data_location) {
            ofs.write(dn.m_data_location, dn.m_data_size);
        } else {
            // recursive call for nested structured datatypes
            write_data(dn.m_structured_format, ofs);
        }
    }
}

// merge two structured datatypes into onodes
inline void merge_data (amrex::Vector<DataNode>& onodes, const amrex::Vector<DataNode>& inodes) {
    for (auto& idn : inodes) {
        if (idn.m_data_location) {
            onodes.push_back(idn);
        } else {
            bool merged = false;
            for (auto& odn : onodes) {
                if (!odn.m_data_location && odn.m_name == idn.m_name) {
                    // recursive call for nested structured datatypes
                    merge_data(odn.m_structured_format, idn.m_structured_format);
                    merged = true;
                    break;
                }
            }
            if (!merged) {
                onodes.push_back(idn);
            }
        }
    }
}

}

#endif
