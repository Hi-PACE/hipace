#include "Constants.H"
#include "Hipace.H"

PhysConst get_phys_const ()
{
    Hipace& hipace = Hipace::GetInstance();
    return hipace.get_phys_const ();
}


namespace Parser {
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
