/* Copyright 2021
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn
 * License: BSD-3-Clause-LBNL
 */
#ifndef HIPACE_Parser_H_
#define HIPACE_Parser_H_

#include "Constants.H"

#include <AMReX_ParmParse.H>
#include <AMReX_Parser.H>
#include <AMReX_RealVect.H>
#include <AMReX_IntVect.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>
#include <set>
#include <string>
#include <sstream>


template<class T>
inline bool
queryWithParser (const amrex::ParmParse& pp, char const * const str, T& val);

namespace Parser
{
    /** \brief add Physical constants to Parser constants */
    inline void
    addConstantsToParser () {
        amrex::ParmParse::SetParserPrefix("my_constants");
        amrex::ParmParse pp{"my_constants"};
        pp.add("pi",        MathConst::pi);
        pp.add("true",      1);
        pp.add("false",     0);
        pp.add("clight",    PhysConstSI::c);
        pp.add("epsilon0",  PhysConstSI::ep0);
        pp.add("mu0",       PhysConstSI::mu0);
        pp.add("q_e",       PhysConstSI::q_e);
        pp.add("m_e",       PhysConstSI::m_e);
        pp.add("m_p",       PhysConstSI::m_p);
        pp.add("hbar",      PhysConstSI::hbar);
        pp.add("r_e",       PhysConstSI::r_e);
    }

    /** \brief replace AMReX input parameters with Parsed version
     *
     * AMReX uses a slightly different way to get user constants when reading these parameters
     */
    inline void
    replaceAmrexParamsWithParser () {
        amrex::ParmParse pp_geom("geometry");
        for (auto str : std::array{"prob_lo", "prob_hi"}) {
            amrex::RealVect val{};
            if(queryWithParser(pp_geom, str, val)) {
                pp_geom.remove(str);
                std::vector<double> vect{val[0], val[1], val[2]};
                pp_geom.addarr(str, vect);
            }
        }
    }

    /** \brief set default ParmParse parameters before AMReX is initialized
     */
    inline void
    setDefaultParams () {
        amrex::ParmParse pp_amrex("amrex");

        // https://amrex-codes.github.io/amrex/docs_html/GPU.html#inputs-parameters
        bool the_arena_is_managed = false;  // AMReX' default: true
        pp_amrex.queryAdd("the_arena_is_managed", the_arena_is_managed);

        // https://amrex-codes.github.io/amrex/docs_html/InputsComputeBackends.html
        std::string omp_threads = "nosmt";  // AMReX' default: system
        pp_amrex.queryAdd("omp_threads", omp_threads);
    }

    /** \brief fill second argument val with a value obtained through Parsing str
     * for std::string: val is same as str
     *
     * \param[in] pp ParmParse object
     * \param[in] str expression in string form
     * \param[out] val value parsed, can be double, float, int, bool or std::string
     */
    inline void
    fillWithParser (const amrex::ParmParse& pp, std::string const& str, double& val) {
        amrex::Parser parser = pp.makeParser(str, {});
        val = parser.compileHost<0>()();
    }

    inline void
    fillWithParser (const amrex::ParmParse& pp, std::string const& str, float& val) {
        amrex::Parser parser = pp.makeParser(str, {});
        double v = parser.compileHost<0>()();
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            v <= std::numeric_limits<float>::max() &&
            v >= std::numeric_limits<float>::lowest(),
            "Error: Overflow detected when casting to float"
        );
        val = static_cast<float>(v);
    }

    inline void
    fillWithParser (const amrex::ParmParse& pp, std::string const& str, int& val) {
        amrex::IParser parser = pp.makeIParser(str, {});
        amrex::Long v = parser.compileHost<0>()();
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            v <= std::numeric_limits<int>::max() &&
            v >= std::numeric_limits<int>::lowest(),
            "Error: Overflow detected when casting to int"
        );
        val = static_cast<int>(v);
    }

    inline void
    fillWithParser (const amrex::ParmParse& pp, std::string const& str, amrex::Long& val) {
        amrex::IParser parser = pp.makeIParser(str, {});
        val = parser.compileHost<0>()();
    }

    inline void
    fillWithParser (const amrex::ParmParse& pp, std::string const& str, bool& val) {
        amrex::IParser parser = pp.makeIParser(str, {});
        val = static_cast<bool>(parser.compileHost<0>()());
    }

    inline void
    fillWithParser (const amrex::ParmParse& pp, std::string const& str, std::string& val,
                    bool do_escape_backtransform=true) {
        std::string loc_str = str;

        // replace {{ and }} with 1 and 2
        std::string::size_type pos = 0;
        while ( (pos=loc_str.find("{{")) != std::string::npos) {
            loc_str.replace(pos, 2, std::string{char(1)});
        }
        while ( (pos=loc_str.rfind("}}")) != std::string::npos) {
            loc_str.replace(pos, 2, std::string{char(2)});
        }

        // replace {expression} with to_string(parser(expression))
        static std::set<std::string> recursive_symbols{};
        while ( (pos=loc_str.rfind("{")) != std::string::npos) {
            const auto pos_end = loc_str.find("}", pos);
            if (pos_end == std::string::npos) {
                amrex::ErrorStream() << "Bad format for input '"
                                     << str
                                     << "', unclosed brace!\n";
                AMREX_ALWAYS_ASSERT(pos_end != std::string::npos);
            }
            auto pos_start_trim = pos + 1;
            auto pos_count_trim = pos_end - pos - 1;
            // remove leading and trailing spaces
            while (loc_str.at(pos_start_trim) == ' ' && pos_count_trim > 0) {
                ++pos_start_trim;
                --pos_count_trim;
            }
            while (loc_str.at(pos_start_trim + pos_count_trim - 1) == ' ' && pos_count_trim > 0) {
                --pos_count_trim;
            }
            const std::string parse_string = loc_str.substr(pos_start_trim, pos_count_trim);

            amrex::ParmParse pp_my_constants("my_constants");
            if (pos_count_trim == 0) {
                loc_str.replace(pos, pos_end-pos+1, "");
            } else if (pp_my_constants.contains(parse_string.c_str())) {
                // use my_constants directly (with recursive string parsing) if available
                if (recursive_symbols.count(parse_string) != 0) {
                    amrex::ErrorStream() << "Expression '"
                                         << str
                                         << "' contains recursive symbol '"
                                         << parse_string
                                         << "'!\n";
                    AMREX_ALWAYS_ASSERT(recursive_symbols.count(parse_string) == 0);
                }
                std::string replacer;
                pp_my_constants.get(parse_string.c_str(), replacer);
                std::string parse_val;
                recursive_symbols.insert(parse_string);
                // for proper escape handling
                fillWithParser(pp, replacer, parse_val, false);
                recursive_symbols.erase(parse_string);
                loc_str.replace(pos, pos_end-pos+1, parse_val);
            } else {
                // else parse as number
                double parse_val = 0.;
                fillWithParser(pp, parse_string, parse_val);
                std::stringstream ss{};
                ss << parse_val;
                loc_str.replace(pos, pos_end-pos+1, ss.str());
            }
        }

        if (do_escape_backtransform) {
            // replace 1 and 2 with { and }
            while ( (pos=loc_str.find(char(1))) != std::string::npos) {
                loc_str.replace(pos, 1, "{");
            }
            while ( (pos=loc_str.rfind(char(2))) != std::string::npos) {
                loc_str.replace(pos, 1, "}");
            }
        }

        val = loc_str;
    }

    /** \brief fill second argument val of array type with a value obtained through Parsing str_arr
     * if val is just a single value, str_arr is reduced to a single string with spaces as
     * separators
     *
     * \param[in] pp ParmParse object
     * \param[in] str_arr vector of expressions to be parsed
     * \param[out] val value parsed, can be scalar, std::vector, amrex::Vector or std::array
     */
    template<class T>
    inline void
    fillWithParserArr (const amrex::ParmParse& pp, std::vector<std::string> const& str_arr,
                       T& val) {
        std::string str{};
        if (!str_arr.empty()) {
            str = str_arr[0];
            for (auto i=1ul ; i != str_arr.size() ; ++i) {
                str += ' ' + str_arr[i];
            }
        }
        fillWithParser(pp, str, val);
    }

    template<class T>
    inline void
    fillWithParserArr (const amrex::ParmParse& pp, std::vector<std::string> const& str_arr,
                       std::vector<T>& val_arr) {
        auto const n = str_arr.size();
        val_arr.resize(n);
        for (auto i=0ul ; i != n ; ++i) {
            fillWithParser(pp, str_arr[i], val_arr[i]);
        }
    }

    template<class T>
    inline void
    fillWithParserArr (const amrex::ParmParse& pp, std::vector<std::string> const& str_arr,
                       amrex::Vector<T>& val_arr) {
        auto const n = str_arr.size();
        val_arr.resize(n);
        for (auto i=0ul ; i != n ; ++i) {
            fillWithParser(pp, str_arr[i], val_arr[i]);
        }
    }

    template<class T, std::size_t size>
    inline void
    fillWithParserArr (const amrex::ParmParse& pp, std::vector<std::string> const& str_arr,
                       std::array<T,size>& val_arr) {
        const auto n = str_arr.size();
        if (n != size) {
            for( auto const& str : str_arr) {
                amrex::ErrorStream() << str << ' ';
            }
            amrex::ErrorStream() << "has wrong length " << n << " should be " << size << '\n';
        }
        AMREX_ALWAYS_ASSERT( n == size );
        for (auto i=0ul ; i != n ; ++i) {
            fillWithParser(pp, str_arr[i], val_arr[i]);
        }
    }

    inline void
    fillWithParserArr (const amrex::ParmParse& pp, std::vector<std::string> const& str_arr,
                       amrex::RealVect& val_arr) {
        const auto n = str_arr.size();
        if (n != AMREX_SPACEDIM) {
            for( auto const& str : str_arr) {
                amrex::ErrorStream() << str << ' ';
            }
            amrex::ErrorStream() << "has wrong length " << n
                                 << " should be " << AMREX_SPACEDIM << '\n';
        }
        AMREX_ALWAYS_ASSERT( n == AMREX_SPACEDIM );
        for (auto i=0ul ; i != n ; ++i) {
            fillWithParser(pp, str_arr[i], val_arr[i]);
        }
    }

    inline void
    fillWithParserArr (const amrex::ParmParse& pp, std::vector<std::string> const& str_arr,
                       amrex::IntVect& val_arr) {
        const auto n = str_arr.size();
        if (n != AMREX_SPACEDIM) {
            for( auto const& str : str_arr) {
                amrex::ErrorStream() << str << ' ';
            }
            amrex::ErrorStream() << "has wrong length " << n
                                 << " should be " << AMREX_SPACEDIM << '\n';
        }
        AMREX_ALWAYS_ASSERT( n == AMREX_SPACEDIM );
        for (auto i=0ul ; i != n ; ++i) {
            fillWithParser(pp, str_arr[i], val_arr[i]);
        }
    }
}

/** \brief fill val with the evaluated expression from the input file
 *
 * \param[in] pp ParmParse that is searched for the expression
 * \param[in] str name of expression
 * \param[out] val value to be filled, see fillWithParserArr and fillWithParser for supported types
 */
template<class T>
inline void
getWithParser (const amrex::ParmParse& pp, char const * const str, T& val) {
    std::vector<std::string> f;
    pp.getarr(str, f);
    Parser::fillWithParserArr(pp, f, val);
}

/** \brief return if input file contains the expression, if so it is parsed into val
 *
 * \param[in] pp ParmParse that is searched for the expression
 * \param[in] str name of expression
 * \param[out] val value to be filled, see fillWithParserArr and fillWithParser for supported types
 */
template<class T>
inline bool
queryWithParser (const amrex::ParmParse& pp, char const * const str, T& val) {
    std::vector<std::string> f;
    const int is_specified = pp.queryarr(str, f);
    if (is_specified) {
        Parser::fillWithParserArr(pp, f, val);
    }
    return is_specified;
}

/** \brief return if input file contains the expression, if so it is parsed into val.
 * The input is searched in pp first, if it is not found pp_alt is searched
 *
 * \param[in] pp ParmParse that is first searched for the expression
 * \param[in] str name of expression
 * \param[out] val value to be filled, see fillWithParserArr and fillWithParser for supported types
 * \param[in] pp_alt Alternative ParmParse that is searched for the expression with lower priority
 */
template<class T>
inline bool
queryWithParserAlt (const amrex::ParmParse& pp, char const * const str, T& val,
                    const amrex::ParmParse& pp_alt) {
    // uses boolean short circuiting
    return queryWithParser(pp, str, val) || queryWithParser(pp_alt, str, val);
}

/** \brief fill val with the evaluated expression from the input file
 * The input is searched in pp first, if it is not found pp_alt is searched
 *
 * \param[in] pp ParmParse that is first searched for the expression
 * \param[in] str name of expression
 * \param[out] val value to be filled, see fillWithParserArr and fillWithParser for supported types
 * \param[in] pp_alt Alternative ParmParse that is searched for the expression with lower priority
 */
template<class T>
inline void
getWithParserAlt (const amrex::ParmParse& pp, char const * const str, T& val,
                  const amrex::ParmParse& pp_alt) {
    if (!queryWithParserAlt(pp, str, val, pp_alt)) {
        // Use error massage of the get function with the original pp
        getWithParser(pp, str, val);
    }
}

/** \brief return function object for Host and Device from the input file
 *
 * \param[in] func_str string that gets Parsed to a function
 * \param[out] parser Parser which owns the data of the returned function
 * \param[in] varnames names of the N arguments used in the parsed function
 */
template<int N>
inline auto
makeFunctionWithParser (std::string const& func_str,
                        amrex::Parser& parser,
                        amrex::Vector<std::string> const& varnames)
{
    std::string clean_str = func_str;
    for (char& c : clean_str) {
        if (c=='\n' || c=='\r' || c=='\t') c = ' ';
    }
    parser = amrex::ParmParse{}.makeParser(clean_str, varnames);
    return parser.compile<N>();
}

#endif
