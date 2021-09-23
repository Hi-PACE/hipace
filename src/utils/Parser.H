#ifndef HIPACE_Parser_H_
#define HIPACE_Parser_H_

#include "Constants.H"

#include <AMReX_ParmParse.H>
#include <AMReX_Parser.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>
#include <set>
#include <string>


template<class T>
inline int
queryWithParser (const amrex::ParmParse& pp, char const * const str, T& val);

namespace Parser
{
    // Cache for evaluated constants
    static std::map<std::string, double> my_constants_cache{};

    // Physical / Numerical Constants available to parsed expressions
    static std::map<std::string, double> hipace_constants
        {
            {"pi", MathConst::pi},
            {"true", 1},
            {"false", 0}
        };

    /** \brief add Physical constants to Parser constants
     *
     * \param[in] phys_const the constants to add
     */
    inline void
    addConstantsToParser (const PhysConst& phys_const) {
        hipace_constants.insert({"clight", phys_const.c});
        hipace_constants.insert({"epsilon0", phys_const.ep0});
        hipace_constants.insert({"mu0", phys_const.mu0});
        hipace_constants.insert({"q_e", phys_const.q_e});
        hipace_constants.insert({"m_e", phys_const.m_e});
        hipace_constants.insert({"m_p", phys_const.m_p});
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
        amrex::ParmParse pp_amr("amr");
        replaceArrWithParser<int>(pp_amr, "n_cell");
        replaceWithParser<int>(pp_amr, "blocking_factor");
        replaceWithParser<int>(pp_amr, "max_level");
        amrex::ParmParse pp_geom("geometry");
        replaceWithParser<int>(pp_geom, "coord_sys");
        replaceArrWithParser<int>(pp_geom, "is_periodic");
        replaceArrWithParser<double>(pp_geom, "prob_lo");
        replaceArrWithParser<double>(pp_geom, "prob_hi");
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

    /** \brief init Parser ready to compile
     *
     * \param[in,out] parser Parser that has been defined
     * \param[in] varnames names of input variables if a function is Parsed
     */
    static void
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

            double v;
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
     * for std::string: val is same as str without spaces
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
    fillWithParser (std::string const& str, bool& val) {
        amrex::Parser parser(str);
        initParser(parser, {});
        val = parser.compileHost<0>()();
    }

    inline void
    fillWithParser (std::string const& str, std::string& val) {
        val = str;
        auto end_pos = std::remove(val.begin(), val.end(), ' ');
        val.erase(end_pos, val.end());
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
        for (auto const& s : str_arr) {
            str += s + ' ';
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

    template<class T, long unsigned int size>
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
inline int
queryWithParser (const amrex::ParmParse& pp, char const * const str, T& val) {
    std::vector<std::string> f;
    const int is_specified = pp.queryarr(str, f);
    if (is_specified) {
        Parser::fillWithParserArr(f, val);
    }
    return is_specified;
}

/** \brief return function object for Host and Device from the input file
 *
 * \param[in] pp ParmParse that is searched for the function
 * \param[in] str name of the function in pp
 * \param[out] parser Parser which owns the data of the returned function
 * \param[in] varnames names of the N arguments used in the parsed function
 */
template<int N>
inline auto
getFunctionWithParser (const amrex::ParmParse& pp,
                       char const * const str,
                       amrex::Parser& parser,
                       amrex::Vector<std::string> const& varnames)
{
    std::string func_str{};
    std::vector<std::string> f;
    pp.getarr(str, f);
    for (auto const& s : f) {
        func_str += s + ' ';
    }
    parser.define(func_str);
    Parser::initParser(parser, varnames);
    return parser.compile<N>();
}

#endif
