
#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_MultiFab.H>

#include <Prob.H>

// Non-uniform MultiFab setup for FillBoundary testing.
void setup(amrex::MultiFab &mfab, const amrex::Geometry &geom)
{
    // Setup the data outside the FillBoundary timers.
    // Ensures data is moved to GPU.
    for (MFIter mfi(mfab); mfi.isValid(); ++mfi)
    {
        const Box bx = mfi.validbox();
        Array4<Real> phi = mfab[mfi].array(); 
        GeometryData geomData = geom.data();

        amrex::launch(bx,
        [=] AMREX_GPU_DEVICE (Box const& tbx)
        {
            initdata(tbx, phi, geomData);
        });
    }    
}


// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&


using namespace amrex;

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
    
        // AMREX_SPACEDIM: number of dimensions
        int nsteps, Ncomp;
        Vector<int> n_cell(AMREX_SPACEDIM), max_grid_size(AMREX_SPACEDIM), Nghost_arr(AMREX_SPACEDIM, 1);
        Vector<int> is_periodic(AMREX_SPACEDIM,1);  // periodic in all direction by default
    
        // inputs parameters
        {
            // ParmParse is way of reading inputs from the inputs file
            ParmParse pp;
    
            // We need to get n_cell from the inputs file - this is the number of cells on each side of 
            //   a square (or cubic) domain.
            pp.getarr("n_cell",n_cell);
    
            // The domain is broken into boxes of size max_grid_size
            pp.getarr("max_grid_size",max_grid_size);
    
            // The number of ghost cells and components of the MultiFab
            pp.queryarr("nghost", Nghost_arr); 
    
            pp.query("ncomp", Ncomp); 
    
            // Default nsteps to 0, allow us to set it to something else in the inputs file
            nsteps = 10;
            pp.query("nsteps",nsteps);
    
            // Periodic in all directions by default
            pp.queryarr("is_periodic", is_periodic);

            ParmParse pp_a("amrex");
            int cag = 0;
            pp.query("use_gpu_aware_mpi",cag);
            amrex::Print() << "CUDA-aware MPI = " << cag << std::endl;
        }
    
        // make BoxArray and Geometry
        BoxArray ba;
        Geometry geom;
        {
            IntVect dom_lo(AMREX_D_DECL(          0,           0,           0));
            IntVect dom_hi(AMREX_D_DECL(n_cell[0]-1, n_cell[1]-1, n_cell[2]-1));
            Box domain(dom_lo, dom_hi);
    
            // Initialize the boxarray "ba" from the single box "bx"
            ba.define(domain);
            // Break up boxarray "ba" into chunks no larger than "max_grid_size" along a direction
            ba.maxSize(IntVect{max_grid_size});
    
           // This defines the physical box, [-1,1] in each direction.
            RealBox real_box({AMREX_D_DECL(-1.0,-1.0,-1.0)},
                             {AMREX_D_DECL( 1.0, 1.0, 1.0)});
    
            // This defines a Geometry object
            geom.define(domain,&real_box,CoordSys::cartesian,is_periodic.data());
        }
    
        // How Boxes are distrubuted among MPI processes
        DistributionMapping dm(ba);
     
        amrex::Print() << "========================================================" << std::endl; 
        amrex::Print() << " Fill Boundary CUDA Graph Test " << std::endl;
        amrex::Print() << " Domain size: " << n_cell[0] << " "
                                           << n_cell[1] << " " 
                                           << n_cell[2] << std::endl;
        amrex::Print() << " Max grid size: " << max_grid_size[0] << " " 
                                             << max_grid_size[1] << " "
                                             << max_grid_size[2] << std::endl;
        amrex::Print() << " Nghost: " << Nghost_arr[0] << " "
                                      << Nghost_arr[1] << " "
                                      << Nghost_arr[2] << std::endl;
        amrex::Print() << " Periodicity: " << is_periodic[0] << " "
                                           << is_periodic[1] << " "
                                           << is_periodic[2] << std::endl;
        amrex::Print() << " Ncomp: " << Ncomp << std::endl;
        amrex::Print() << " Boxes: " << ba.size() << std::endl << std::endl;
        amrex::Print() << "========================================================" << std::endl; 
    
        IntVect Nghost{Nghost_arr};
        MultiFab mf_graph(ba, dm, Ncomp, Nghost);
        MultiFab mf_gpu  (ba, dm, Ncomp, Nghost);
        MultiFab mf_cpu  (ba, dm, Ncomp, Nghost);
    
        Real start_time, end_time;
        Real cpu_avg, gpu_avg, graph_avg, graph_init, gpu_even;
    
        // With GPUs and Graphs
        Gpu::setLaunchRegion(true); 
        Gpu::setGraphRegion(true); 
        {
            // Setup the data outside the FillBoundary timers.
            // Ensures data is moved to GPU.
            setup(mf_graph, geom);
    
            // First FillBoundary will create graph and run.
            // Timed separately.
            // -------------------------------------
            ParallelDescriptor::Barrier();
            BL_PROFILE_VAR("GRAPH: Create Graph and Run", makeandrungraph);
            start_time = amrex::second();
            amrex::Print() << "** GRAPH ***************************************************" << std::endl; 
   
            mf_graph.FillBoundary(geom.periodicity());
    
            ParallelDescriptor::Barrier();
            end_time = amrex::second();
            BL_PROFILE_VAR_STOP(makeandrungraph);
    
            graph_init = end_time - start_time;
            amrex::Print() << "Time for 1st graphed FillBoundary (Recorded, Instantiated Ran) = " << graph_init << std::endl;
            // -------------------------------------
   
            // Run the remainder of the FillBoundarys (nsteps-1)
            ParallelDescriptor::Barrier();
            BL_PROFILE_VAR("GRAPH: Run Graph", rungraph);
            start_time = amrex::second();
    
            for (int i=1; i<nsteps; ++i)
            {
                mf_graph.FillBoundary(geom.periodicity());
            }
    
            ParallelDescriptor::Barrier();
            end_time = amrex::second();
            BL_PROFILE_VAR_STOP(rungraph);
    
            graph_avg = (end_time - start_time)/nsteps;
            amrex::Print() << "Average time per graph-only FillBoundary = " << graph_avg << std::endl;
        }
        amrex::Print() << "*** CPU *****************************************************" << std::endl; 
   
        // With CPU
        Gpu::setLaunchRegion(false); 
        {
            // Setup the data outside the FillBoundary timers.
            setup(mf_cpu, geom);
    
            // Run the remainder of the FillBoundary's (nsteps-1)
            // -------------------------------------
            ParallelDescriptor::Barrier();
            BL_PROFILE_VAR("CPU: FillBoundary", CPUFB);
            start_time = amrex::second();
    
            for (int i=0; i<nsteps; ++i)
            {
                mf_cpu.FillBoundary(geom.periodicity());
            }
    
            ParallelDescriptor::Barrier();
            end_time = amrex::second();
            BL_PROFILE_VAR_STOP(CPUFB);
    
            cpu_avg = (end_time - start_time)/nsteps;
            amrex::Print() << "Average time per CPU FillBoundary = " << cpu_avg << std::endl;
        }
        amrex::Print() << "** GPU *******************************************************" << std::endl; 
   
        // With GPU and no graphs
        Gpu::setLaunchRegion(true); 
        Gpu::setGraphRegion(false); 
        {
            // Setup the data outside the FillBoundary timers.
            // Ensures data is moved to GPU.
            setup(mf_gpu, geom);
    
            // Run the nstep FillBoundaries
            // -------------------------------------
    
            ParallelDescriptor::Barrier();
            BL_PROFILE_VAR("GPU: FillBoundary", GPUFB);
            start_time = amrex::second();
    
            for (int i=0; i<nsteps; ++i)
            {
                mf_gpu.FillBoundary(geom.periodicity());
            }
    
            ParallelDescriptor::Barrier();
            end_time = amrex::second();
            BL_PROFILE_VAR_STOP(GPUFB);
    
            gpu_avg = (end_time - start_time)/nsteps;
            gpu_even = (graph_init)/(gpu_avg - graph_avg);
            amrex::Print() << "Average time per GPU FillBoundary = " << gpu_avg << std::endl;
            amrex::Print() << "   Graphed FillBoundary(s) needed to break even = " << gpu_even << std::endl;
        }
    
        // Check the results of Graph vs. CPU and GPU.
        // Maximum of difference of all cells.
        {
            amrex::Real max_error = 0;
            MultiFab mf_error (ba, dm, Ncomp, Nghost);
    
            MultiFab::Copy(mf_error, mf_cpu, 0, 0, Ncomp, Nghost);
            MultiFab::Subtract(mf_error, mf_gpu, 0, 0, Ncomp, Nghost);
            int min_ghost = Nghost.min();

            for (int i = 0; i<Ncomp; ++i)
            {
                max_error = std::max(max_error, mf_error.norm0(0, min_ghost));
            }
            amrex::Print() << std::endl;
            amrex::Print() << "Max difference between CPU and GPU: " << max_error << std::endl; 
    
            MultiFab::Copy(mf_error, mf_graph, 0, 0, Ncomp, Nghost);
            MultiFab::Subtract(mf_error, mf_cpu, 0, 0, Ncomp, Nghost);
            for (int i = 0; i<Ncomp; ++i)
            {
                max_error = std::max(max_error, mf_error.norm0(0, min_ghost));
            }
            amrex::Print() << "Max difference between CPU and Graph: " << max_error << std::endl; 
    
            amrex::Print() << "========================================================" << std::endl << std::endl;
        }

        // With GPU and no graphs
        Gpu::setLaunchRegion(true); 
        Gpu::setGraphRegion(false); 
        {
            // Setup the data outside the FillBoundary timers.
            // Ensures data is moved to GPU.
            setup(mf_gpu, geom);
    
            // Run the nstep FillBoundaries
            // -------------------------------------
    
            ParallelDescriptor::Barrier();
            BL_PROFILE_VAR("GPU: FillBoundary", GPUFB);
            start_time = amrex::second();
    
            for (int i=0; i<nsteps; ++i)
            {
                mf_gpu.FillBoundary(geom.periodicity());
            }
    
            ParallelDescriptor::Barrier();
            end_time = amrex::second();
            BL_PROFILE_VAR_STOP(GPUFB);
    
            gpu_avg = (end_time - start_time)/nsteps;
            gpu_even = (graph_init)/(gpu_avg - graph_avg);
            amrex::Print() << "Average time per GPU FillBoundary = " << gpu_avg << std::endl;
            amrex::Print() << "   Graphed FillBoundary(s) needed to break even = " << gpu_even << std::endl;
        }
    
        // Check the results of Graph vs. CPU and GPU.
        // Maximum of difference of all cells.
        {
            amrex::Real max_error = 0;
            MultiFab mf_error (ba, dm, Ncomp, Nghost);
    
            MultiFab::Copy(mf_error, mf_cpu, 0, 0, Ncomp, Nghost);
            MultiFab::Subtract(mf_error, mf_gpu, 0, 0, Ncomp, Nghost);
            int min_ghost = Nghost.min();

            for (int i = 0; i<Ncomp; ++i)
            {
                max_error = std::max(max_error, mf_error.norm0(0, min_ghost));
            }
            amrex::Print() << std::endl;
            amrex::Print() << "Max difference between CPU and GPU: " << max_error << std::endl; 
    
            MultiFab::Copy(mf_error, mf_graph, 0, 0, Ncomp, Nghost);
            MultiFab::Subtract(mf_error, mf_cpu, 0, 0, Ncomp, Nghost);
            for (int i = 0; i<Ncomp; ++i)
            {
                max_error = std::max(max_error, mf_error.norm0(0, min_ghost));
            }
            amrex::Print() << "Max difference between CPU and Graph: " << max_error << std::endl; 
    
            amrex::Print() << "========================================================" << std::endl << std::endl;
        }





    }
    amrex::Finalize();
}
