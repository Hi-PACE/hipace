
#include "Hipace.H"
#include "HipaceProfilerWrapper.H"

#include <AMReX.H>

using namespace amrex;

int main (int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    HIPACE_PROFILE_VAR("main()", pmain);
    {
        Hipace hipace;
        hipace.InitData();
        hipace.Evolve();
    }
    HIPACE_PROFILE_VAR_STOP(pmain);
    amrex::Finalize();
}
