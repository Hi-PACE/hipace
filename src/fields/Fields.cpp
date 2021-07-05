#include "Fields.H"
#include "fft_poisson_solver/FFTPoissonSolverPeriodic.H"
#include "fft_poisson_solver/FFTPoissonSolverDirichlet.H"
#include "Hipace.H"
#include "utils/HipaceProfilerWrapper.H"
#include "utils/Constants.H"
#include <cstdlib>
#include <math.h>

Fields::Fields (Hipace const* a_hipace)
    : m_slices(a_hipace->maxLevel()+1)
{
    amrex::ParmParse ppf("fields");
    ppf.query("do_dirichlet_poisson", m_do_dirichlet_poisson);
}

void
Fields::AllocData (
    int lev, amrex::Vector<amrex::Geometry> const& geom, const amrex::BoxArray& slice_ba,
    const amrex::DistributionMapping& slice_dm)
{
    HIPACE_PROFILE("Fields::AllocData()");
    // Need at least 1 guard cell transversally for transverse derivative
    int nguards_xy = std::max(1, Hipace::m_depos_order_xy);
    m_slices_nguards = {nguards_xy, nguards_xy, 0};

    for (int islice=0; islice<WhichSlice::N; islice++) {
        m_slices[lev][islice].define(
            slice_ba, slice_dm, Comps[islice]["N"], m_slices_nguards,
            amrex::MFInfo().SetArena(amrex::The_Arena()));
        m_slices[lev][islice].setVal(0.0);
    }

    // The Poisson solver operates on transverse slices only.
    // The constructor takes the BoxArray and the DistributionMap of a slice,
    // so the FFTPlans are built on a slice.
    if (m_do_dirichlet_poisson){
        m_poisson_solver.push_back(std::unique_ptr<FFTPoissonSolverDirichlet>(
            new FFTPoissonSolverDirichlet(getSlices(lev, WhichSlice::This).boxArray(),
                                          getSlices(lev, WhichSlice::This).DistributionMap(),
                                          geom[lev])) );
    } else {
        m_poisson_solver.push_back(std::unique_ptr<FFTPoissonSolverPeriodic>(
            new FFTPoissonSolverPeriodic(getSlices(lev, WhichSlice::This).boxArray(),
                                         getSlices(lev, WhichSlice::This).DistributionMap(),
                                         geom[lev]))  );
    }
}

void
Fields::TransverseDerivative (const amrex::MultiFab& src, amrex::MultiFab& dst, const int direction,
                              const amrex::Real dx, const amrex::Real mult_coeff,
                              const SliceOperatorType slice_operator,
                              const int scomp, const int dcomp)
{
    HIPACE_PROFILE("Fields::TransverseDerivative()");
    using namespace amrex::literals;

    AMREX_ALWAYS_ASSERT((direction == Direction::x) || (direction == Direction::y));
    for ( amrex::MFIter mfi(dst, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi ){
        const amrex::Box& bx = mfi.tilebox();
        amrex::Array4<amrex::Real const> const & src_array = src.array(mfi);
        amrex::Array4<amrex::Real> const & dst_array = dst.array(mfi);
        amrex::ParallelFor(
            bx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                if (direction == Direction::x){
                    /* finite difference along x */
                    if (slice_operator==SliceOperatorType::Assign)
                    {
                        dst_array(i,j,k,dcomp) = mult_coeff / (2.0_rt*dx) *
                          (src_array(i+1, j, k, scomp) - src_array(i-1, j, k, scomp));
                    }
                    else /* SliceOperatorType::Add */
                    {
                        dst_array(i,j,k,dcomp) += mult_coeff / (2.0_rt*dx) *
                          (src_array(i+1, j, k, scomp) - src_array(i-1, j, k, scomp));
                    }
                } else /* Direction::y */ {
                    /* finite difference along y */
                    if (slice_operator==SliceOperatorType::Assign)
                    {
                        dst_array(i,j,k,dcomp) = mult_coeff / (2.0_rt*dx) *
                          (src_array(i, j+1, k, scomp) - src_array(i, j-1, k, scomp));
                    }
                    else /* SliceOperatorType::Add */
                    {
                        dst_array(i,j,k,dcomp) += mult_coeff / (2.0_rt*dx) *
                          (src_array(i, j+1, k, scomp) - src_array(i, j-1, k, scomp));
                    }
                }
            }
            );
    }
}

void
Fields::LongitudinalDerivative (const amrex::MultiFab& src1, const amrex::MultiFab& src2,
                                amrex::MultiFab& dst, const amrex::Real dz,
                                const amrex::Real mult_coeff,
                                const SliceOperatorType slice_operator,
                                const int s1comp, const int s2comp, const int dcomp)
{
    HIPACE_PROFILE("Fields::LongitudinalDerivative()");
    using namespace amrex::literals;
    for ( amrex::MFIter mfi(dst, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi ){
        const amrex::Box& bx = mfi.tilebox();
        amrex::Array4<amrex::Real const> const & src1_array = src1.array(mfi);
        amrex::Array4<amrex::Real const> const & src2_array = src2.array(mfi);
        amrex::Array4<amrex::Real> const & dst_array = dst.array(mfi);
        amrex::ParallelFor(
            bx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                if (slice_operator==SliceOperatorType::Assign)
                {
                    dst_array(i,j,k,dcomp) = mult_coeff / (2.0_rt*dz) *
                      (src1_array(i, j, k, s1comp) - src2_array(i, j, k, s2comp));
                }
                else /* SliceOperatorType::Add */
                {
                    dst_array(i,j,k,dcomp) += mult_coeff / (2.0_rt*dz) *
                      (src1_array(i, j, k, s1comp) - src2_array(i, j, k, s2comp));
                }

            }
            );
    }
}


void
Fields::Copy (int lev, int i_slice, FieldCopyType copy_type, int slice_comp, int full_comp,
              int ncomp, amrex::FArrayBox& fab, int slice_dir, amrex::Geometry geom)
{
    using namespace amrex::literals;
    HIPACE_PROFILE("Fields::Copy()");
    auto& slice_mf = m_slices[lev][WhichSlice::This]; // copy from/to the current slice
    amrex::Array4<amrex::Real> slice_array; // There is only one Box.
    for (amrex::MFIter mfi(slice_mf); mfi.isValid(); ++mfi) {
        auto& slice_fab = slice_mf[mfi];
        amrex::Box slice_box = slice_fab.box();
        slice_box.setSmall(Direction::z, i_slice);
        slice_box.setBig  (Direction::z, i_slice);
        slice_array = amrex::makeArray4(slice_fab.dataPtr(), slice_box, slice_fab.nComp());
        // slice_array's longitude index is i_slice.
    }

    amrex::Box const& vbx = fab.box();
    if (vbx.smallEnd(Direction::z) <= i_slice and
        vbx.bigEnd  (Direction::z) >= i_slice)
    {
        amrex::Box copy_box = vbx;
        copy_box.setSmall(Direction::z, i_slice);
        copy_box.setBig  (Direction::z, i_slice);

        amrex::Array4<amrex::Real> const& full_array = fab.array();

        const amrex::IntVect ncells_global = geom.Domain().length();
        const bool nx_even = ncells_global[0] % 2 == 0;
        const bool ny_even = ncells_global[1] % 2 == 0;

        if (copy_type == FieldCopyType::FtoS) {
            amrex::ParallelFor(copy_box, ncomp,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
            {
                slice_array(i,j,k,n+slice_comp) = full_array(i,j,k,n+full_comp);
            });
        } else {
            amrex::ParallelFor(copy_box, ncomp,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
            {
                if        (slice_dir ==-1 /* 3D data */){
                    full_array(i,j,k,n+full_comp) = slice_array(i,j,k,n+slice_comp);
                } else if (slice_dir == 0 /* yz slice */){
                    full_array(i,j,k,n+full_comp) =
                        nx_even ? 0.5_rt * (slice_array(i-1,j,k,n+slice_comp) +
                                            slice_array(i,j,k,n+slice_comp))
                        : slice_array(i,j,k,n+slice_comp);
                } else /* slice_dir == 1, xz slice */{
                    full_array(i,j,k,n+full_comp) =
                        ny_even ? 0.5_rt * ( slice_array(i,j-1,k,n+slice_comp) +
                                             slice_array(i,j,k,n+slice_comp))
                        : slice_array(i,j,k,n+slice_comp);
                }
            });
        }
    }
}

void
Fields::ShiftSlices (int lev)
{
    HIPACE_PROFILE("Fields::ShiftSlices()");
    amrex::MultiFab::Copy(
        getSlices(lev, WhichSlice::Previous2), getSlices(lev, WhichSlice::Previous1),
        Comps[WhichSlice::Previous1]["Bx"], Comps[WhichSlice::Previous2]["Bx"],
        2, m_slices_nguards);
    amrex::MultiFab::Copy(
        getSlices(lev, WhichSlice::Previous1), getSlices(lev, WhichSlice::This),
        Comps[WhichSlice::This]["Bx"], Comps[WhichSlice::Previous1]["Bx"],
        2, m_slices_nguards);
    amrex::MultiFab::Copy(
        getSlices(lev, WhichSlice::Previous1), getSlices(lev, WhichSlice::This),
        Comps[WhichSlice::This]["jx"], Comps[WhichSlice::Previous1]["jx"],
        4, m_slices_nguards);
}

void
Fields::AddRhoIons (const int lev, bool inverse)
{
    HIPACE_PROFILE("Fields::AddRhoIons()");
    if (!inverse){
        amrex::MultiFab::Add(getSlices(lev, WhichSlice::This), getSlices(lev, WhichSlice::RhoIons),
                             Comps[WhichSlice::RhoIons]["rho"], Comps[WhichSlice::This]["rho"], 1, 0);
    } else {
        amrex::MultiFab::Subtract(getSlices(lev, WhichSlice::This), getSlices(lev, WhichSlice::RhoIons),
                                  Comps[WhichSlice::RhoIons]["rho"], Comps[WhichSlice::This]["rho"], 1, 0);
    }
}

void
Fields::AddBeamCurrents (const int lev, const int which_slice)
{
    HIPACE_PROFILE("Fields::AddBeamCurrents()");
    amrex::MultiFab& S = getSlices(lev, which_slice);
    // we add the beam currents to the full currents, as mostly the full currents are needed
    amrex::MultiFab::Add(S, S, Comps[which_slice]["jx_beam"], Comps[which_slice]["jx"], 1,
                         {Hipace::m_depos_order_xy, Hipace::m_depos_order_xy, 0});
    amrex::MultiFab::Add(S, S, Comps[which_slice]["jy_beam"], Comps[which_slice]["jy"], 1,
                         {Hipace::m_depos_order_xy, Hipace::m_depos_order_xy, 0});
    if (which_slice == WhichSlice::This) {
        amrex::MultiFab::Add(S, S, Comps[which_slice]["jz_beam"], Comps[which_slice]["jz"], 1,
                             {Hipace::m_depos_order_xy, Hipace::m_depos_order_xy, 0});
    }
}

void
Fields::SolvePoissonExmByAndEypBx (amrex::Vector<amrex::Geometry> const& geom,
                                   const MPI_Comm& m_comm_xy, const int lev)
{
    /* Solves Laplacian(Psi) =  1/episilon0 * -(rho-Jz/c) and
     * calculates Ex-c By, Ey + c Bx from  grad(-Psi)
     */
    HIPACE_PROFILE("Fields::SolveExmByAndEypBx()");

    PhysConst phys_const = get_phys_const();

    // Left-Hand Side for Poisson equation is Psi in the slice MF
    amrex::MultiFab lhs(getSlices(lev, WhichSlice::This), amrex::make_alias,
                        Comps[WhichSlice::This]["Psi"], 1);

    // calculating the right-hand side 1/episilon0 * -(rho-Jz/c)
    amrex::MultiFab::Copy(m_poisson_solver[lev]->StagingArea(), getSlices(lev, WhichSlice::This),
                              Comps[WhichSlice::This]["jz"], 0, 1, 0);
    m_poisson_solver[lev]->StagingArea().mult(-1./phys_const.c);
    amrex::MultiFab::Add(m_poisson_solver[lev]->StagingArea(), getSlices(lev, WhichSlice::This),
                          Comps[WhichSlice::This]["rho"], 0, 1, 0);
    m_poisson_solver[lev]->StagingArea().mult(-1./phys_const.ep0);
     
    /*Setting non-zero boundary conditions*/
    const auto plo = geom[lev].ProbLoArray();
    //amrex::Print()<<plo[0]<<"\n"; 
    const auto xmin =geom[lev].ProbLo(0);
    const auto xmax =geom[lev].ProbHi(0);
    const auto ymin =geom[lev].ProbLo(1);
    const auto ymax=geom[lev].ProbHi(1);
    const auto dx = geom[lev].CellSizeArray();
    
    if (lev!=0)
    {   
        //amrex::Print()<<xmin<<"\n";
        //Interpolation 
         amrex::MultiFab lhs_coarse(getSlices(lev-1, WhichSlice::This), amrex::make_alias,
                        Comps[WhichSlice::This]["Psi"], 1);
        //amrex::Print()<<"Refined Mesh \n";
        for (amrex::MFIter mfi( m_poisson_solver[lev]->StagingArea(),false); mfi.isValid(); ++mfi)
        {
            const amrex::Box & bx = mfi.tilebox();

            /*
                Get information of the mesh 
            */
           
            const amrex::Box & bx_fine = mfi.tilebox();
            const amrex::IntVect& small = bx_fine.smallEnd(); //Get the small end of the Box
            const auto nx_fine_low  =  small[0];
            const auto ny_fine_low = small[1]; 
           
            const amrex::IntVect& high = bx.bigEnd();  // Get the big end of the Box 
            const auto nx_fine_high = high[0];
            const auto ny_fine_high = high[1];
            

            //

            
            const auto x_min_fine = plo[0]+ (nx_fine_low+0.5)*dx[0];
            const auto x_max_fine = plo[0]+ (nx_fine_high+0.5)*dx[0];
            //amrex::Print()<<"x_max: "<<nx_fine_low<<"x_min: "<<nx_fine_high<<"\n";
            const auto y_min_fine = plo[1]+ (ny_fine_low+0.5)*dx[1];
            const auto y_max_fine = plo[1]+ (ny_fine_high+0.5)*dx[1];

            
            //
            //const auto x_min_fine =
            amrex::Array4<amrex::Real >  data_array = m_poisson_solver[lev]->StagingArea().array(mfi);
            amrex::Array4<amrex::Real >  data_array_coarse = m_poisson_solver[lev-1]->StagingArea().array(mfi);
            amrex::ParallelFor(
                bx,
                [=] AMREX_GPU_DEVICE(int i, int j , int k) noexcept
                {
    
                    //Compute coordinate
                    const auto plo_coarse = geom[lev-1].ProbLoArray(); 
                    const auto dx_coarse = geom[lev-1].CellSizeArray(); //
                    const auto refinement_ratio = dx_coarse[0]/dx[0];
                    
                    if (j==ny_fine_low|| j == ny_fine_high)
                    {
                        //amrex::Print()<<"dx coarse "<< dx_coarse[0]<< " dx fine "<< dx[0]<<"\n";
                        amrex::Real x = plo[0] + (i+0.5) *dx[0];
                        amrex::Real y = plo[1] + (j+0.5) *dx[1];
                        amrex::Real x_neighbor_left = x - dx[0];
                        amrex::Real x_neighbor_right = x + dx[0];
                        if(x_neighbor_left <= x_min_fine) {x_neighbor_left = x_min_fine;}
                        if(x_neighbor_right >= x_max_fine) {x_neighbor_right = x_max_fine;}

                        double ind_left = (x_neighbor_left-plo[0]-0.5*dx_coarse[0])/dx_coarse[0]; //-0.5*dx_coarse[0]
                        double ind_right = (x_neighbor_right-plo[0]-0.5*dx_coarse[0])/dx_coarse[0];
                        int count_2=0; 
                        double epsilon = 1E-5;
                        amrex::Print()<<"Start Loop initial position: "<< x_neighbor_left<<"\n";
                        while ( ind_left-floor(ind_left) > epsilon && count_2 <refinement_ratio)
                        {
                            amrex::Print()<< "Floor: "<<floor(ind_left)<<" Real value "<<ind_left<<" "<<x_neighbor_left<<"\n";
                            x_neighbor_left = x_neighbor_left - dx[0];
                            ind_left = (x_neighbor_left-plo[0]-0.5*dx_coarse[0])/(dx_coarse[0]);
                            count_2 +=1;
                            //amrex::Print()<<"Indices in loop left "<<ind_left<<"\n";
                            //amrex::Print()<<"Position in Loop left "<<x_neighbor_left<<"\n";
                            
                        }
                        amrex::Print()<<"End loop indice: "<<ind_left<<"\n";
                        int count_1=0;
                        while (  ind_right-floor(ind_right) > epsilon && count_1 <refinement_ratio){
                            //amrex::Print()<<std::floor(ind_right)!= ind_right<<"\n";
                            x_neighbor_right = x_neighbor_right + dx[0];
                            ind_right = (x_neighbor_right-plo[0]-0.5*dx_coarse[0])/(dx_coarse[0]);
                            count_1 +=1;
                            /*
                            amrex::Print()<<"Position "<<x_neighbor_right<<"\n";
                            amrex::Print()<<"Iterator "<<count_1<<"\n";
                            amrex::Print()<<"Indices "<<ind_right<<"\n";
                            */

                            /*
                            amrex::Print()<<"Indices in loop right  "<<ind_right<<"\n";
                            amrex::Print()<<"Position in Loop right "<<x_neighbor_right<<"\n";
                            */
                        }
                        
                        if(count_2>=refinement_ratio){
                            amrex::Print()<<" Problem !! \n";
                            //amrex::Print()<<"Position Left "<<x_neighbor_left<<"\n";
                            //amrex::Print()<<"Indice "<<ind_left<<"\n";
                        }
                        /*
                        if(count_1>=4){
                            amrex::Print()<<"Position right "<<x_neighbor_right<<"\n";
                            amrex::Print()<<"Indice "<<ind_right<<"\n";
                        }
                        */
                        if(x_neighbor_left <= x_min_fine){ x_neighbor_left = x_min_fine; ind_left = ny_fine_low;}
                        if(x_neighbor_right >= x_max_fine){ x_neighbor_right = x_max_fine; ind_right = ny_fine_high;}
                        //amrex::Print()<<x_max_fine<<"\n";
                        //amrex::Print()<<"Left neighbor in coarse grid: "<<x_neighbor_left<<"\n Point: "<<x<<" Right neighbor in coarse grid: "<<x_neighbor_right<<"\n";
                        //amrex::Print()<<"Left neighbor "<<x_neighbor_left<<"\n";
                        //amrex::Print()<<"Index left "<<ind_left<<" Index right "<<ind_right<<"\n";
                        /*
                            Interpolation 
                        */
                        const auto val_left = data_array_coarse(std::floor(ind_left),j,k);
                        const auto val_right = data_array_coarse(std::floor(ind_right),j,k);
                        const auto left_term = val_left*(x-x_neighbor_right)/(x_neighbor_left-x_neighbor_right) ;
                        const auto right_term = val_right*(x-x_neighbor_left)/(x_neighbor_right-x_neighbor_left) ;
                        data_array(i,j,k) += left_term+right_term/(pow(dx[0],2));

                    }
                   
                    if(i==nx_fine_low || i== nx_fine_high)
                    {
                        //amrex::Print()<<"dx coarse "<< dx_coarse[0]<< " dx fine "<< dx[0]<<"\n";
                        amrex::Real x = plo[0] + (i) *dx[0];
                        amrex::Real y = plo[1] + (j) *dx[1];
                        amrex::Real y_neighbor_left = y - dx[1];
                        amrex::Real y_neighbor_right = y + dx[1];
                        double ind_left = (y_neighbor_left-plo[1])/dx_coarse[1]; //-0.5*dx_coarse[1]
                        double ind_right = (y_neighbor_right-plo[1])/dx_coarse[1];
                        double epsilon = 1E-5;

                        while (ind_left-floor(ind_left) > epsilon)
                        {
                            y_neighbor_left -= dx[1];
                            ind_left = (y_neighbor_left-plo[1])/dx_coarse[1];
                        }

                        if(y_neighbor_left <= y_min_fine){ y_neighbor_left = y_min_fine; ind_left = nx_fine_low;}

                        while (ind_right-floor(ind_right) > epsilon)
                        {
                            y_neighbor_right += dx[1];
                            ind_right = (y_neighbor_right-plo[1])/dx_coarse[1];
                        }
                        if(y_neighbor_right >= y_max_fine){ y_neighbor_right = y_max_fine; ind_right = nx_fine_high;}
                        //amrex::Print()<<y_max_fine<<"\n";
                        //amrex::Print()<<"Left neighbor in coarse grid "<<y_neighbor_left<<"\n Point: "<<y<<" Right neighbor in coarse grid: "<<y_neighbor_right<<"\n";
                        //amrex::Print()<<"Index left "<<ind_left<<" Index right "<<ind_right<<"\n";
                        /*
                            Interpolation 
                        */
                        const auto val_left = data_array_coarse(i,std::floor(ind_left),k);
                        const auto val_right = data_array_coarse(i,std::floor(ind_right),k);
                        const auto left_term = val_left*(y-y_neighbor_right)/(y_neighbor_left-y_neighbor_right) ;
                        const auto right_term = val_right*(y-y_neighbor_left)/(y_neighbor_right-y_neighbor_left) ;
                        data_array(i,j,k) += left_term+right_term/(pow(dx[0],2)); 
                    }

                    /*
                    if (y-dx_coarse[0] <= y_max_fine && x >= x_min_fine && x<= x_max_fine)
                    {   
                        amrex::Real x_neighbor_left = x - dx_coarse[0];
                        amrex::Real x_neighbor_right = x + dx_coarse[0];
                        //if (x_neighbor_left < x_min_fine) {x_neighbor_left = x_min_fine;}
                        //if (x_neighbor_right > x_max_fine ) {x_neighbor_right = x_max_fine;}
                        //Or high i have to check 
                        const auto index_row = 4;
                        //amrex::Print()<<x_neighbor_left<<"Begin\n";
                        for(int m = 0; m < 2*(refinement_ratio+1); m++)
                        {   
                            double x_point = x_neighbor_left + m*dx[0];
                            double col_ind =  (x_point-x_min_fine)/dx[0];
                            //amrex::Print()<<col_ind<<" "<<x_point<<"\n";
                            //amrex::Print()<<col_ind<<"\n";
                            //Interpolation we use polynomial interpolation 
                            const auto val_mid = data_array_coarse(i,j,k);
                            const auto val_left = data_array_coarse(i-1,j,k);
                            const auto val_right = data_array_coarse(i+1,j,k);
                            const auto first_term = val_mid*(x_point-x_neighbor_right)*(x_point-x_neighbor_left)/((x-x_neighbor_left)*(x-x_neighbor_right));
                            const auto second_term = val_left*(x_point-x)*(x_point-x_neighbor_right)/((x_neighbor_left-x)*(x_neighbor_left-x_neighbor_right));
                            const auto third_term = val_right*(x_point-x)*(x_point-x_neighbor_left)/((x_neighbor_right-x)*(x_neighbor_right-x_neighbor_left));
                            data_array(col_ind,index_row,k,0) += first_term+second_term+third_term;
                        }
                        //amrex::Print()<<"End\n";
                    }
                    if (y+dx_coarse[1] >= y_min_fine && x >= x_min_fine && x<= x_max_fine )
                    {   
                        amrex::Real x_neighbor_left = x - dx_coarse[0];
                        amrex::Real x_neighbor_right = x + dx_coarse[0];
                        //if (x_neighbor_left < x_min_fine) {x_neighbor_left = x_min_fine;}
                        //if (x_neighbor_right > x_max_fine ) {x_neighbor_right = x_max_fine;}
                        //Or high i have to check 
                        const auto index_row = 15;
                        //amrex::Print()<<x_neighbor_left<<"Begin\n";
                        for(int m = 0; m < 2*(refinement_ratio+1); m++)
                        {   
                            double x_point = x_neighbor_left + m*dx[0];
                            double col_ind =  (x_point-x_min_fine)/dx[0];
                            //amrex::Print()<<col_ind<<" "<<x_point<<"\n";
                            //amrex::Print()<<col_ind<<"\n";
                            //Interpolation we use polynomial interpolation 
                            const auto val_mid = data_array_coarse(i,j,k);
                            const auto val_left = data_array_coarse(i-1,j,k);
                            const auto val_right = data_array_coarse(i+1,j,k);
                            const auto first_term = val_mid*(x_point-x_neighbor_right)*(x_point-x_neighbor_left)/((x-x_neighbor_left)*(x-x_neighbor_right));
                            const auto second_term = val_left*(x_point-x)*(x_point-x_neighbor_right)/((x_neighbor_left-x)*(x_neighbor_left-x_neighbor_right));
                            const auto third_term = val_right*(x_point-x)*(x_point-x_neighbor_left)/((x_neighbor_right-x)*(x_neighbor_right-x_neighbor_left));
                            data_array(col_ind,index_row,k,0) += first_term+second_term+third_term;
                        }
                        //amrex::Print()<<"End\n";
                        
                    }

                    if (x+dx_coarse[0]>= x_min_fine && y >= y_min_fine && y<= y_max_fine)
                    {   
                        amrex::Real y_neighbor_left = y - dx_coarse[1];
                        amrex::Real y_neighbor_right = y + dx_coarse[1];
                        //if (x_neighbor_left < x_min_fine) {x_neighbor_left = x_min_fine;}
                        //if (x_neighbor_right > x_max_fine ) {x_neighbor_right = x_max_fine;}
                        //Or high i have to check 
                        const auto index_line = 15;
                        //amrex::Print()<<x_neighbor_left<<"Begin\n";
                        for(int m = 0; m < 2*(refinement_ratio+1); m++)
                        {   
                            double y_point = y_neighbor_left + m*dx[1];
                            double line_ind =  (y_point-y_min_fine)/dx[1];
                            //amrex::Print()<<col_ind<<" "<<x_point<<"\n";
                            //amrex::Print()<<col_ind<<"\n";
                            //Interpolation we use polynomial interpolation 
                            const auto val_mid = data_array_coarse(i,j,k);
                            const auto val_left = data_array_coarse(i-1,j,k);
                            const auto val_right = data_array_coarse(i+1,j,k);
                            const auto first_term = val_mid*(y_point-y_neighbor_right)*(y_point-y_neighbor_left)/((y-y_neighbor_left)*(y-y_neighbor_right));
                            const auto second_term = val_left*(y_point-y)*(y_point-y_neighbor_right)/((y_neighbor_left-y)*(y_neighbor_left-y_neighbor_right));
                            const auto third_term = val_right*(y_point-y)*(y_point-y_neighbor_left)/((y_neighbor_right-y)*(y_neighbor_right-y_neighbor_left));
                            data_array(index_line,line_ind,k,0) += first_term+second_term+third_term;
                        }
                        //amrex::Print()<<"End\n";
                    }
                    if (x-dx_coarse[0]<= x_max_fine && y >= y_min_fine && y<= y_max_fine)
                    {   
                        amrex::Real y_neighbor_left = y - dx_coarse[1];
                        amrex::Real y_neighbor_right = y + dx_coarse[1];
                        //if (x_neighbor_left < x_min_fine) {x_neighbor_left = x_min_fine;}
                        //if (x_neighbor_right > x_max_fine ) {x_neighbor_right = x_max_fine;}
                        //Or high i have to check 
                        const auto index_line = 4;
                        //amrex::Print()<<x_neighbor_left<<"Begin\n";
                        for(int m = 0; m < 2*(refinement_ratio+1); m++)
                        {   
                            double y_point = y_neighbor_left + m*dx[1];
                            double line_ind =  (y_point-y_min_fine)/dx[1];
                            //amrex::Print()<<col_ind<<" "<<x_point<<"\n";
                            //amrex::Print()<<col_ind<<"\n";
                            //Interpolation we use polynomial interpolation 
                            const auto val_mid = data_array_coarse(i,j,k);
                            const auto val_left = data_array_coarse(i-1,j,k);
                            const auto val_right = data_array_coarse(i+1,j,k);
                            const auto first_term = val_mid*(y_point-y_neighbor_right)*(y_point-y_neighbor_left)/((y-y_neighbor_left)*(y-y_neighbor_right));
                            const auto second_term = val_left*(y_point-y)*(y_point-y_neighbor_right)/((y_neighbor_left-y)*(y_neighbor_left-y_neighbor_right));
                            const auto third_term = val_right*(y_point-y)*(y_point-y_neighbor_left)/((y_neighbor_right-y)*(y_neighbor_right-y_neighbor_left));
                            data_array(index_line,line_ind,k,0) += first_term+second_term+third_term;
                        }
                        //amrex::Print()<<"End\n";
                    }
                    */
                
                }

        );
        }
    }

    m_poisson_solver[lev]->SolvePoissonEquation(lhs);

    /* ---------- Transverse FillBoundary Psi ---------- */
    amrex::ParallelContext::push(m_comm_xy);
    lhs.FillBoundary(geom[lev].periodicity());
    amrex::ParallelContext::pop();

    /* Compute ExmBy and Eypbx from grad(-psi) */
    TransverseDerivative(
        getSlices(lev, WhichSlice::This),
        getSlices(lev, WhichSlice::This),
        Direction::x,
        geom[lev].CellSize(Direction::x),
        -1.,
        SliceOperatorType::Assign,
        Comps[WhichSlice::This]["Psi"],
        Comps[WhichSlice::This]["ExmBy"]);

    TransverseDerivative(
        getSlices(lev, WhichSlice::This),
        getSlices(lev, WhichSlice::This),
        Direction::y,
        geom[lev].CellSize(Direction::y),
        -1.,
        SliceOperatorType::Assign,
        Comps[WhichSlice::This]["Psi"],
        Comps[WhichSlice::This]["EypBx"]);
}


void
Fields::SolvePoissonEz (amrex::Vector<amrex::Geometry> const& geom, const int lev)
{
    /* Solves Laplacian(Ez) =  1/(episilon0 *c0 )*(d_x(jx) + d_y(jy)) */
    HIPACE_PROFILE("Fields::SolvePoissonEz()");

    PhysConst phys_const = get_phys_const();
    // Left-Hand Side for Poisson equation is Bz in the slice MF
    amrex::MultiFab lhs(getSlices(lev, WhichSlice::This), amrex::make_alias,
                        Comps[WhichSlice::This]["Ez"], 1);
    // Right-Hand Side for Poisson equation: compute 1/(episilon0 *c0 )*(d_x(jx) + d_y(jy))
    // from the slice MF, and store in the staging area of poisson_solver
    TransverseDerivative(
        getSlices(lev, WhichSlice::This),
        m_poisson_solver[lev]->StagingArea(),
        Direction::x,
        geom[lev].CellSize(Direction::x),
        1./(phys_const.ep0*phys_const.c),
        SliceOperatorType::Assign,
        Comps[WhichSlice::This]["jx"]);

    TransverseDerivative(
        getSlices(lev, WhichSlice::This),
        m_poisson_solver[lev]->StagingArea(),
        Direction::y,
        geom[lev].CellSize(Direction::y),
        1./(phys_const.ep0*phys_const.c),
        SliceOperatorType::Add,
        Comps[WhichSlice::This]["jy"]);
    // Solve Poisson equation.
    // The RHS is in the staging area of poisson_solver.
    // The LHS will be returned as lhs.
    m_poisson_solver[lev]->SolvePoissonEquation(lhs);
}

void
Fields::SolvePoissonBx (amrex::MultiFab& Bx_iter, amrex::Vector<amrex::Geometry> const& geom,
                        const int lev)
{
    /* Solves Laplacian(Bx) = mu_0*(- d_y(jz) + d_z(jy) ) */
    HIPACE_PROFILE("Fields::SolvePoissonBx()");

    PhysConst phys_const = get_phys_const();
    // Right-Hand Side for Poisson equation: compute -mu_0*d_y(jz) from the slice MF,
    // and store in the staging area of poisson_solver
    TransverseDerivative(
        getSlices(lev, WhichSlice::This),
        m_poisson_solver[lev]->StagingArea(),
        Direction::y,
        geom[lev].CellSize(Direction::y),
        -phys_const.mu0,
        SliceOperatorType::Assign,
        Comps[WhichSlice::This]["jz"]);

    LongitudinalDerivative(
        getSlices(lev, WhichSlice::Previous1),
        getSlices(lev, WhichSlice::Next),
        m_poisson_solver[lev]->StagingArea(),
        geom[lev].CellSize(Direction::z),
        phys_const.mu0,
        SliceOperatorType::Add,
        Comps[WhichSlice::Previous1]["jy"],
        Comps[WhichSlice::Next]["jy"]);
    // Solve Poisson equation.
    // The RHS is in the staging area of poisson_solver.
    // The LHS will be returned as lhs.
    m_poisson_solver[lev]->SolvePoissonEquation(Bx_iter);
}

void
Fields::SolvePoissonBy (amrex::MultiFab& By_iter, amrex::Vector<amrex::Geometry> const& geom,
                        const int lev)
{
    /* Solves Laplacian(By) = mu_0*(d_x(jz) - d_z(jx) ) */
    HIPACE_PROFILE("Fields::SolvePoissonBy()");

    PhysConst phys_const = get_phys_const();
    // Right-Hand Side for Poisson equation: compute mu_0*d_x(jz) from the slice MF,
    // and store in the staging area of poisson_solver
    TransverseDerivative(
        getSlices(lev, WhichSlice::This),
        m_poisson_solver[lev]->StagingArea(),
        Direction::x,
        geom[lev].CellSize(Direction::x),
        phys_const.mu0,
        SliceOperatorType::Assign,
        Comps[WhichSlice::This]["jz"]);

    LongitudinalDerivative(
        getSlices(lev, WhichSlice::Previous1),
        getSlices(lev, WhichSlice::Next),
        m_poisson_solver[lev]->StagingArea(),
        geom[lev].CellSize(Direction::z),
        -phys_const.mu0,
        SliceOperatorType::Add,
        Comps[WhichSlice::Previous1]["jx"],
        Comps[WhichSlice::Next]["jx"]);
    // Solve Poisson equation.
    // The RHS is in the staging area of poisson_solver.
    // The LHS will be returned as lhs.
    m_poisson_solver[lev]->SolvePoissonEquation(By_iter);
}

void
Fields::SolvePoissonBz (amrex::Vector<amrex::Geometry> const& geom, const int lev)
{
    /* Solves Laplacian(Bz) = mu_0*(d_y(jx) - d_x(jy)) */
    HIPACE_PROFILE("Fields::SolvePoissonBz()");

    PhysConst phys_const = get_phys_const();
    // Left-Hand Side for Poisson equation is Bz in the slice MF
    amrex::MultiFab lhs(getSlices(lev, WhichSlice::This), amrex::make_alias,
                        Comps[WhichSlice::This]["Bz"], 1);
    // Right-Hand Side for Poisson equation: compute mu_0*(d_y(jx) - d_x(jy))
    // from the slice MF, and store in the staging area of m_poisson_solver
    TransverseDerivative(
        getSlices(lev, WhichSlice::This),
        m_poisson_solver[lev]->StagingArea(),
        Direction::y,
        geom[lev].CellSize(Direction::y),
        phys_const.mu0,
        SliceOperatorType::Assign,
        Comps[WhichSlice::This]["jx"]);

    TransverseDerivative(
        getSlices(lev, WhichSlice::This),
        m_poisson_solver[lev]->StagingArea(),
        Direction::x,
        geom[lev].CellSize(Direction::x),
        -phys_const.mu0,
        SliceOperatorType::Add,
        Comps[WhichSlice::This]["jy"]);
    // Solve Poisson equation.
    // The RHS is in the staging area of m_poisson_solver.
    // The LHS will be returned as lhs.
    m_poisson_solver[lev]->SolvePoissonEquation(lhs);
}

void
Fields::InitialBfieldGuess (const amrex::Real relative_Bfield_error,
                            const amrex::Real predcorr_B_error_tolerance, const int lev)
{
    /* Sets the initial guess of the B field from the two previous slices
     */
    HIPACE_PROFILE("Fields::InitialBfieldGuess()");

    const amrex::Real mix_factor_init_guess = exp(-0.5 * pow(relative_Bfield_error /
                                              ( 2.5 * predcorr_B_error_tolerance ), 2));

    amrex::MultiFab::LinComb(
        getSlices(lev, WhichSlice::This),
        1+mix_factor_init_guess, getSlices(lev, WhichSlice::Previous1), Comps[WhichSlice::Previous1]["Bx"],
        -mix_factor_init_guess, getSlices(lev, WhichSlice::Previous2), Comps[WhichSlice::Previous2]["Bx"],
        Comps[WhichSlice::This]["Bx"], 1, 0);

    amrex::MultiFab::LinComb(
        getSlices(lev, WhichSlice::This),
        1+mix_factor_init_guess, getSlices(lev, WhichSlice::Previous1), Comps[WhichSlice::Previous1]["By"],
        -mix_factor_init_guess, getSlices(lev, WhichSlice::Previous2), Comps[WhichSlice::Previous2]["By"],
        Comps[WhichSlice::This]["By"], 1, 0);
}

void
Fields::MixAndShiftBfields (const amrex::MultiFab& B_iter, amrex::MultiFab& B_prev_iter,
                            const int field_comp, const amrex::Real relative_Bfield_error,
                            const amrex::Real relative_Bfield_error_prev_iter,
                            const amrex::Real predcorr_B_mixing_factor, const int lev)
{
    /* Mixes the B field according to B = a*B + (1-a)*( c*B_iter + d*B_prev_iter),
     * with a,c,d mixing coefficients.
     */
    HIPACE_PROFILE("Fields::MixAndShiftBfields()");

    /* Mixing factors to mix the current and previous iteration of the B field */
    amrex::Real weight_B_iter;
    amrex::Real weight_B_prev_iter;
    /* calculating the weight for mixing the current and previous iteration based
     * on their respective errors. Large errors will induce a small weight of and vice-versa  */
    if (relative_Bfield_error != 0.0 || relative_Bfield_error_prev_iter != 0.0)
    {
        weight_B_iter = relative_Bfield_error_prev_iter /
                        ( relative_Bfield_error + relative_Bfield_error_prev_iter );
        weight_B_prev_iter = relative_Bfield_error /
                             ( relative_Bfield_error + relative_Bfield_error_prev_iter );
    }
    else
    {
        weight_B_iter = 0.5;
        weight_B_prev_iter = 0.5;
    }

    /* calculating the mixed temporary B field  B_prev_iter = c*B_iter + d*B_prev_iter.
     * This is temporarily stored in B_prev_iter just to avoid additional memory allocation.
     * B_prev_iter is overwritten at the end of this function */
    amrex::MultiFab::LinComb(
        B_prev_iter,
        weight_B_iter, B_iter, 0,
        weight_B_prev_iter, B_prev_iter, 0,
        0, 1, 0);

    /* calculating the mixed B field  B = a*B + (1-a)*B_prev_iter */
    amrex::MultiFab::LinComb(
        getSlices(lev, WhichSlice::This),
        1-predcorr_B_mixing_factor, getSlices(lev, WhichSlice::This), field_comp,
        predcorr_B_mixing_factor, B_prev_iter, 0,
        field_comp, 1, 0);

    /* Shifting the B field from the current iteration to the previous iteration */
    amrex::MultiFab::Copy(B_prev_iter, B_iter, 0, 0, 1, 0);

}

amrex::Real
Fields::ComputeRelBFieldError (
    const amrex::MultiFab& Bx, const amrex::MultiFab& By, const amrex::MultiFab& Bx_iter,
    const amrex::MultiFab& By_iter, const int Bx_comp, const int By_comp, const int Bx_iter_comp,
    const int By_iter_comp, const amrex::Geometry& geom)
{
    // calculates the relative B field error between two B fields
    // for both Bx and By simultaneously
    HIPACE_PROFILE("Fields::ComputeRelBFieldError()");

    amrex::Real norm_Bdiff = 0;
    amrex::Gpu::DeviceScalar<amrex::Real> gpu_norm_Bdiff(norm_Bdiff);
    amrex::Real* p_norm_Bdiff = gpu_norm_Bdiff.dataPtr();

    amrex::Real norm_B = 0;
    amrex::Gpu::DeviceScalar<amrex::Real> gpu_norm_B(norm_B);
    amrex::Real* p_norm_B = gpu_norm_B.dataPtr();

    for ( amrex::MFIter mfi(Bx, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi ){
        const amrex::Box& bx = mfi.tilebox();
        amrex::Array4<amrex::Real const> const & Bx_array = Bx.array(mfi);
        amrex::Array4<amrex::Real const> const & Bx_iter_array = Bx_iter.array(mfi);
        amrex::Array4<amrex::Real const> const & By_array = By.array(mfi);
        amrex::Array4<amrex::Real const> const & By_iter_array = By_iter.array(mfi);

        amrex::ParallelFor(amrex::Gpu::KernelInfo().setReduction(true), bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, amrex::Gpu::Handler const& handler) noexcept
        {
            amrex::Gpu::deviceReduceSum(p_norm_B, std::sqrt(
                                        Bx_array(i, j, k, Bx_comp) * Bx_array(i, j, k, Bx_comp) +
                                        By_array(i, j, k, By_comp) * By_array(i, j, k, By_comp)),
                                        handler);
            amrex::Gpu::deviceReduceSum(p_norm_Bdiff, std::sqrt(
                            ( Bx_array(i, j, k, Bx_comp) - Bx_iter_array(i, j, k, Bx_iter_comp) ) *
                            ( Bx_array(i, j, k, Bx_comp) - Bx_iter_array(i, j, k, Bx_iter_comp) ) +
                            ( By_array(i, j, k, By_comp) - By_iter_array(i, j, k, By_iter_comp) ) *
                            ( By_array(i, j, k, By_comp) - By_iter_array(i, j, k, By_iter_comp) )),
                            handler);
        }
        );
    }
    // no cudaDeviceSynchronize required here, as there is one in the MFIter destructor called above.
    norm_Bdiff = gpu_norm_Bdiff.dataValue();
    norm_B = gpu_norm_B.dataValue();

    const int numPts_transverse = geom.Domain().length(0) * geom.Domain().length(1);

    // calculating the relative error
    // Warning: this test might be not working in SI units!
    const amrex::Real relative_Bfield_error = (norm_B/numPts_transverse > 1e-10)
                                               ? norm_Bdiff/norm_B : 0.;

    return relative_Bfield_error;
}
