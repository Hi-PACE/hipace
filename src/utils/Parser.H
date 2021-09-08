#ifndef HIPACE_Parser_H_
#define HIPACE_Parser_H_

#include "Constants.H"

#include <AMReX.H>
#include <AMReX_REAL.H>
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
inline void
getWithParser (const amrex::ParmParse& pp, char const * const str, T& val);

template<class T>
inline int
queryWithParser (const amrex::ParmParse& pp, char const * const str, T& val);

inline int
safeCastToInt(const double x, const std::string& real_name) {
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

static amrex::Parser&
initParser (amrex::Parser& parser, amrex::Vector<std::string> const& varnames)
{
    // Since queryWithParser recursively calls this routine, keep track of symbols
    // in case an infinite recursion is found (a symbol's value depending on itself).
    static std::set<std::string> recursive_symbols{};

    //amrex::Parser parser(parse_function);
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

    // Cache for evalutated constants
    static std::map<std::string, double> my_constants_cache{};

    // Physical / Numerical Constants available to parsed expressions
    const auto phys_const = get_phys_const();

    static std::map<std::string, double> hipace_constants
        {
            {"clight", phys_const.c},
            {"epsilon0", phys_const.ep0},
            {"mu0", phys_const.mu0},
            {"q_e", phys_const.q_e},
            {"m_e", phys_const.m_e},
            {"m_p", phys_const.m_p},
            {"pi", MathConst::pi}
        };

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
    return parser;
}

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
    initParser(parser, varnames);
    parser.print();
    return parser.compile<N>();
}

inline void
fillWithParser (std::string const& str, double& val) {
    amrex::Parser parser(str);
    val = initParser(parser, {}).compileHost<0>()();
}

inline void
fillWithParser (std::string const& str, float& val) {
    amrex::Parser parser(str);
    val = static_cast<float>(initParser(parser, {}).compileHost<0>()());
}

inline void
fillWithParser (std::string const& str, int& val) {
    amrex::Parser parser(str);
    val = safeCastToInt(std::round(initParser(parser, {}).compileHost<0>()()),str);
}

inline void
fillWithParser (std::string const& str, bool& val) {
    amrex::Parser parser(str);
    val = initParser(parser, {}).compileHost<0>()();
}

inline void
fillWithParser (std::string const& str, std::string& val) {
    val = str;
    auto end_pos = std::remove(val.begin(), val.end(), ' ');
    val.erase(end_pos, val.end());
}

template<class T>
inline void
fillWithParserArr (std::vector<std::string> const& str_arr, T& val) {
    std::string str{};
    for (auto const& s : str_arr) {
        str += s + ' ';
    }
    fillWithParser(str, val);
    amrex::Print() << " " << val;
}

template<class T>
inline void
fillWithParserArr (std::vector<std::string> const& str_arr, std::vector<T>& val_arr) {
    auto const n = str_arr.size();
    val_arr.resize(n);
    for (auto i=0ul ; i != n ; ++i) {
        fillWithParser(str_arr[i], val_arr[i]);
        amrex::Print() << " " << val_arr[i];
    }
}

template<class T>
inline void
fillWithParserArr (std::vector<std::string> const& str_arr, amrex::Vector<T>& val_arr) {
    auto const n = str_arr.size();
    val_arr.resize(n);
    for (auto i=0ul ; i != n ; ++i) {
        fillWithParser(str_arr[i], val_arr[i]);
        amrex::Print() << " " << val_arr[i];
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
        amrex::Print() << " " << val_arr[i];
    }
}

template<class T>
inline void
getWithParser (const amrex::ParmParse& pp, char const * const str, T& val)
{
    std::vector<std::string> f;
    pp.getarr(str, f);
    amrex::Print() << str << " :";
    fillWithParserArr(f, val);
    amrex::Print() << '\n';
}

template<class T>
inline int
queryWithParser (const amrex::ParmParse& pp, char const * const str, T& val)
{
    std::vector<std::string> f;
    const int is_specified = pp.queryarr(str, f);
    amrex::Print() << str << " :";
    if (is_specified)
    {
        fillWithParserArr(f, val);
    }
    amrex::Print() << '\n';
    return is_specified;
}

#endif
