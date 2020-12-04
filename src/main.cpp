
#include "Hipace.H"
#include "utils/HipaceProfilerWrapper.H"

#include <AMReX.H>

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
