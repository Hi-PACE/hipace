
#include "Hipace.H"

#include <AMReX.H>

using namespace amrex;

int main (int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    {
        Hipace hipace;
        hipace.InitData();
    }
    amrex::Finalize();
}
