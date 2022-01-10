
#include "Hipace.H"
#include "utils/HipaceProfilerWrapper.H"

#include <AMReX.H>

int main (int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    {
        HIPACE_PROFILE("main()");
        Hipace hipace;
        hipace.InitData();
        hipace.Evolve();
    }
    amrex::Finalize();
}
