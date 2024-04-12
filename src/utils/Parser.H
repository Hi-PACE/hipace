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
    // Cache for evaluated constants
    extern std::map<std::string, double> my_constants_cache;

    // Physical / Numerical Constants available to parsed expressions
    extern std::map<std::string, double> hipace_constants;
    // {"pi", MathConst::pi},
    // {"true", 1},
    // {"false", 0}

    /** \brief add Physical constants to Parser constants */
    inline void
    addConstantsToParser () {
        hipace_constants.insert({"clight",   PhysConstSI::c   });
        hipace_constants.insert({"epsilon0", PhysConstSI::ep0 });
        hipace_constants.insert({"mu0",      PhysConstSI::mu0 });
        hipace_constants.insert({"q_e",      PhysConstSI::q_e });
        hipace_constants.insert({"m_e",      PhysConstSI::m_e });
        hipace_constants.insert({"m_p",      PhysConstSI::m_p });
        hipace_constants.insert({"hbar",     PhysConstSI::hbar});
        hipace_constants.insert({"r_e",      PhysConstSI::r_e});
    }

    /** \brief replace ParmParse input with a Parsed version
     *
     * \param[in] pp ParmParse object
     * \param[in] str name of input value to replace
     */
    template<class T>
    inline void
    replaceWithParser (amrex::ParmParse& pp, char const * const str) {
        T val{};
        if(queryWithParser(pp, str, val)) {
            pp.remove(str);
            pp.add(str, val);
        }
    }

    /** \brief array version of replaceWithParser
     *
     * \param[in] pp ParmParse object
     * \param[in] str name of input value to replace
     */
    template<class T>
    inline void
    replaceArrWithParser (amrex::ParmParse& pp, char const * const str) {
        std::vector<T> val{};
        if(queryWithParser(pp, str, val)) {
            pp.remove(str);
            pp.addarr(str, val);
        }
    }

    /** \brief replace AMReX input parameters with Parsed version
     *
     * AMReX only uses ParmParse when reading in these parameter, not a Parser
     */
    inline void
    replaceAmrexParamsWithParser () {
        amrex::ParmParse pp_geom("geometry");
        replaceArrWithParser<int>(pp_geom, "is_periodic");
        replaceArrWithParser<double>(pp_geom, "prob_lo");
        replaceArrWithParser<double>(pp_geom, "prob_hi");
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

    /** \brief return valid int, asserts if inf or NaN
     *
     * \param[in] x value to cast
     * \param[in] real_name name of value for error message
     */
    inline int
    safeCastToInt (const double x, const std::string& real_name) {
        int result = 0;
        bool error_detected = false;
        std::string assert_msg;
        // (2.0*(numeric_limits<int>::max()/2+1)) converts numeric_limits<int>::max()+1 to a real
        // ensuring accuracy to all digits. This accepts x = 2**31-1 but rejects 2**31.
        if (x < (2.0*(std::numeric_limits<int>::max()/2+1))) {
            if (std::ceil(x) >= std::numeric_limits<int>::min()) {
                result = static_cast<int>(x);
            } else {
                error_detected = true;
                assert_msg = "Error: Negative overflow detected when casting " +
                             real_name + " = " + std::to_string(x) + " to int";
            }
        } else if (x > 0) {
            error_detected = true;
            assert_msg = "Error: Overflow detected when casting " +
                         real_name + " = " + std::to_string(x) + " to int";
        } else {
            error_detected = true;
            assert_msg =  "Error: NaN detected when casting " + real_name + " to int";
        }
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!error_detected, assert_msg);
        return result;
    }

    /** \brief return valid Long, asserts if inf or NaN
     *
     * \param[in] x value to cast
     * \param[in] real_name name of value for error message
     */
    inline amrex::Long
    safeCastToLong (const double x, const std::string& real_name) {
        amrex::Long result = 0;
        bool error_detected = false;
        std::string assert_msg;
        // (2.0*(numeric_limits<amrex::Long>::max()/2+1)) converts
        // numeric_limits<amrex::Long>::max()+1 to a real
        // ensuring accuracy to all digits. This accepts x = 2**31-1 but rejects 2**31.
        if (x < (2.0*(std::numeric_limits<amrex::Long>::max()/2+1))) {
            if (std::ceil(x) >= std::numeric_limits<amrex::Long>::min()) {
                result = static_cast<amrex::Long>(x);
            } else {
                error_detected = true;
                assert_msg = "Error: Negative overflow detected when casting " +
                             real_name + " = " + std::to_string(x) + " to amrex::Long";
            }
        } else if (x > 0) {
            error_detected = true;
            assert_msg = "Error: Overflow detected when casting " +
                         real_name + " = " + std::to_string(x) + " to amrex::Long";
        } else {
            error_detected = true;
            assert_msg =  "Error: NaN detected when casting " + real_name + " to amrex::Long";
        }
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!error_detected, assert_msg);
        return result;
    }

    /** \brief init Parser ready to compile
     *
     * \param[in,out] parser Parser that has been defined
     * \param[in] varnames names of input variables if a function is Parsed
     */
    inline void
    initParser (amrex::Parser& parser, amrex::Vector<std::string> const& varnames) {
        // Since queryWithParser recursively calls this routine, keep track of symbols
        // in case an infinite recursion is found (a symbol's value depending on itself).
        static std::set<std::string> recursive_symbols{};

        parser.registerVariables(varnames);

        std::set<std::string> symbols = parser.symbols();
        for (auto const& v : varnames) symbols.erase(v);

        // User can provide inputs under this name, through which expressions
        // can be provided for arbitrary variables. This potentially
        // includes variable names that match physical or mathematical
        // constants, in case the user wishes to enforce a different
        // system of units or some form of quasi-physical behavior in the
        // simulation. Thus, this needs to override any built-in
        // constants.
        amrex::ParmParse pp_my_constants("my_constants");

        for (auto const& s : symbols) {

            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(recursive_symbols.count(s)==0,
                                             "Expressions contains recursive symbol " + s);

            auto usr_constant = my_constants_cache.find(s);
            if (usr_constant != my_constants_cache.end()) {
                parser.setConstant(s, usr_constant->second);
                continue;
            }

            double v = 0.;
            recursive_symbols.insert(s);
            const bool is_input = queryWithParser(pp_my_constants, s.c_str(), v);
            recursive_symbols.erase(s);

            if (is_input) {
                my_constants_cache.insert({s, v});
                parser.setConstant(s, v);
                continue;
            }

            auto phy_constant = hipace_constants.find(s);
            if (phy_constant != hipace_constants.end()) {
                parser.setConstant(s, phy_constant->second);
                continue;
            }

            amrex::Abort("makeParser::Unknown symbol " + s);
        }
    }

    /** \brief fill second argument val with a value obtained through Parsing str
     * for std::string: val is same as str
     *
     * \param[in] str expression in string form
     * \param[out] val value parsed, can be double, float, int, bool or std::string
     */
    inline void
    fillWithParser (std::string const& str, double& val) {
        amrex::Parser parser(str);
        initParser(parser, {});
        val = parser.compileHost<0>()();
    }

    inline void
    fillWithParser (std::string const& str, float& val) {
        amrex::Parser parser(str);
        initParser(parser, {});
        val = static_cast<float>(parser.compileHost<0>()());
    }

    inline void
    fillWithParser (std::string const& str, int& val) {
        amrex::Parser parser(str);
        initParser(parser, {});
        val = safeCastToInt(std::round(parser.compileHost<0>()()),str);
    }

    inline void
    fillWithParser (std::string const& str, amrex::Long& val) {
        amrex::Parser parser(str);
        initParser(parser, {});
        val = safeCastToLong(std::round(parser.compileHost<0>()()),str);
    }

    inline void
    fillWithParser (std::string const& str, bool& val) {
        amrex::Parser parser(str);
        initParser(parser, {});
        val = parser.compileHost<0>()();
    }

    inline void
    fillWithParser (std::string const& str, std::string& val, bool do_escape_backtransform=true) {
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
                fillWithParser(replacer, parse_val, false);
                recursive_symbols.erase(parse_string);
                loc_str.replace(pos, pos_end-pos+1, parse_val);
            } else {
                // else parse as number
                double parse_val = 0.;
                fillWithParser(parse_string, parse_val);
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
     * \param[in] str_arr vector of expressions to be parsed
     * \param[out] val value parsed, can be scalar, std::vector, amrex::Vector or std::array
     */
    template<class T>
    inline void
    fillWithParserArr (std::vector<std::string> const& str_arr, T& val) {
        std::string str{};
        if (!str_arr.empty()) {
            str = str_arr[0];
            for (auto i=1ul ; i != str_arr.size() ; ++i) {
                str += ' ' + str_arr[i];
            }
        }
        fillWithParser(str, val);
    }

    template<class T>
    inline void
    fillWithParserArr (std::vector<std::string> const& str_arr, std::vector<T>& val_arr) {
        auto const n = str_arr.size();
        val_arr.resize(n);
        for (auto i=0ul ; i != n ; ++i) {
            fillWithParser(str_arr[i], val_arr[i]);
        }
    }

    template<class T>
    inline void
    fillWithParserArr (std::vector<std::string> const& str_arr, amrex::Vector<T>& val_arr) {
        auto const n = str_arr.size();
        val_arr.resize(n);
        for (auto i=0ul ; i != n ; ++i) {
            fillWithParser(str_arr[i], val_arr[i]);
        }
    }

    template<class T, std::size_t size>
    inline void
    fillWithParserArr (std::vector<std::string> const& str_arr, std::array<T,size>& val_arr) {
        const auto n = str_arr.size();
        if (n != size) {
            for( auto const& str : str_arr) {
                amrex::ErrorStream() << str << ' ';
            }
            amrex::ErrorStream() << "has wrong length " << n << " should be " << size << '\n';
        }
        AMREX_ALWAYS_ASSERT( n == size );
        for (auto i=0ul ; i != n ; ++i) {
            fillWithParser(str_arr[i], val_arr[i]);
        }
    }

    inline void
    fillWithParserArr (std::vector<std::string> const& str_arr, amrex::RealVect& val_arr) {
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
            fillWithParser(str_arr[i], val_arr[i]);
        }
    }

    inline void
    fillWithParserArr (std::vector<std::string> const& str_arr, amrex::IntVect& val_arr) {
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
            fillWithParser(str_arr[i], val_arr[i]);
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
    Parser::fillWithParserArr(f, val);
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
        Parser::fillWithParserArr(f, val);
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
    parser.define(clean_str);
    Parser::initParser(parser, varnames);
    return parser.compile<N>();
}

#endif
