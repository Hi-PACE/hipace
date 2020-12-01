#include "Constants.H"
#include "Hipace.H"

PhysConst get_phys_const ()
{
    Hipace& hipace = Hipace::GetInstance();
    return hipace.get_phys_const ();
}
