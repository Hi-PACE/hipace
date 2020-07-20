
#include "Hipace.H"

#include <AMReX.H>
#include <AMReX_BLProfiler.H>

using namespace amrex;

int main (int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    BL_PROFILE_VAR("main()", pmain);
    {
        Hipace hipace;
        hipace.InitData();
        hipace.Evolve();
    }
    BL_PROFILE_VAR_STOP(pmain);
    amrex::Finalize();
}
