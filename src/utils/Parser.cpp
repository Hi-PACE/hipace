#include "Parser.H"
#include "Constants.H"

#include <map>
#include <string>

namespace Parser
{
    // Cache for evaluated constants
    std::map<std::string, double> my_constants_cache{};

    // Physical / Numerical Constants available to parsed expressions
    std::map<std::string, double> hipace_constants
        {
            {"pi", MathConst::pi},
            {"true", 1},
            {"false", 0}
        };
}
