
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
        FabArrayBase::CpOp op = FabArrayBase::COPY;
        bool op_add = false;
        int n_cell, max_grid_size_src, max_grid_size_dst, nsteps, Nghost, Nghost_src, Nghost_dst, Ncomp;
        Vector<int> is_periodic(AMREX_SPACEDIM,1);  // periodic in all direction by default
    
        // inputs parameters
        {
            // ParmParse is way of reading inputs from the inputs file
            ParmParse pp;
    
            // We need to get n_cell from the inputs file - this is the number of cells on each side of 
            //   a square (or cubic) domain.
            pp.get("n_cell",n_cell);
    
            // The src fab is broken into boxes of size max_grid_size_src
            pp.get("max_grid_size_src", max_grid_size_src);

            // The dst fab is broken into boxes of size max_grid_size_dst
            pp.get("max_grid_size_dst", max_grid_size_dst);
 
            // The number of ghost cells and components of the MultiFab
            // For ParallelCopy, ghost cells may vary between  
            // "nghost" is used for src and dst, unless superceded by specific settings.
            Nghost = 0;
            pp.query("nghost", Nghost); 
   
            Nghost_src = Nghost;
            pp.query("nghost_src", Nghost_src);

            Nghost_dst = Nghost;
            pp.query("nghost_dst", Nghost_dst);
 
            Ncomp = 1;
            pp.query("ncomp", Ncomp); 
    
            // Default nsteps to 0, allow us to set it to something else in the inputs file
            nsteps = 10;
            pp.query("nsteps",nsteps);
    
            // Periodic in all directions by default
            pp.queryarr("is_periodic", is_periodic);

            // Toggle between ParallelCopy and ParallelAdd
            pp.query("add", op_add);
            if (op_add) { op = FabArrayBase::ADD; }
        }
    
        // make BoxArray and Geometry
        BoxArray ba_src, ba_dst;
        Geometry geom;
        {
            IntVect dom_lo(AMREX_D_DECL(       0,        0,        0));
            IntVect dom_hi(AMREX_D_DECL(n_cell-1, n_cell-1, n_cell-1));
            Box domain(dom_lo, dom_hi);
    
            // Initialize the boxarray "ba" from the single box "bx"
            ba_src.define(domain);
            ba_dst.define(domain);

            // Break up boxarray "ba" into chunks no larger than "max_grid_size" along a direction
            ba_src.maxSize(max_grid_size_src);
            ba_dst.maxSize(max_grid_size_dst);
   
           // This defines the physical box, [-1,1] in each direction.
            RealBox real_box({AMREX_D_DECL(-1.0,-1.0,-1.0)},
                             {AMREX_D_DECL( 1.0, 1.0, 1.0)});
    
            // This defines a Geometry object
            geom.define(domain,&real_box,CoordSys::cartesian,is_periodic.data());
        }
    
        // How Boxes are distrubuted among MPI processes
        DistributionMapping dm_src(ba_src);
        DistributionMapping dm_dst(ba_dst);
    
        amrex::Print() << "========================================================" << std::endl; 
        if (op == FabArrayBase::COPY)
          { amrex::Print() << " Parallel Copy CUDA Graph Test " << std::endl; }
        else
          { amrex::Print() << " Parallel Add CUDA Graph Test " << std::endl; }
        amrex::Print() << " Domain size: " << n_cell << "^3." << std::endl;
        amrex::Print() << std::endl; 
        amrex::Print() << " Max grid size for source: " << max_grid_size_src << " cells." << std::endl;
        amrex::Print() << " Boxes in source: " << ba_src.size() << std::endl; 
        amrex::Print() << " Number of ghost cells in source: " << Nghost_src << std::endl;
        amrex::Print() << std::endl; 
        amrex::Print() << " Max grid size for destination: " << max_grid_size_dst << " cells." << std::endl;
        amrex::Print() << " Boxes in destination: " << ba_dst.size() << std::endl;
        amrex::Print() << " Number of ghost cells in destination: " << Nghost_dst << std::endl;
        amrex::Print() << "========================================================" << std::endl; 

        MultiFab mf_init (ba_src, dm_src, Ncomp, Nghost_src);
        MultiFab mf_graph(ba_dst, dm_dst, Ncomp, Nghost_dst);
        MultiFab mf_gpu  (ba_dst, dm_dst, Ncomp, Nghost_dst);
        MultiFab mf_cpu  (ba_dst, dm_dst, Ncomp, Nghost_dst);
    
        Real start_time, end_time;
        Real cpu_avg, gpu_avg, graph_avg, graph_init, gpu_even;
  
        mf_graph.setVal(0.0);
        mf_gpu.setVal(0.0);
        mf_cpu.setVal(0.0); 

        // With GPUs and Graphs
        Gpu::setLaunchRegion(true); 
        Gpu::setGraphRegion(true); 
        {
            setup(mf_init, geom);

            // First FillBoundary will create graph and run.
            // Timed separately.
            // -------------------------------------
            ParallelDescriptor::Barrier();
            BL_PROFILE_VAR("GRAPH: Create Graph and Run", makeandrungraph);
            start_time = amrex::second();
    
            mf_graph.ParallelCopy(mf_init,0,0,Ncomp,Nghost_src,Nghost_dst,geom.periodicity(),op);
    
            ParallelDescriptor::Barrier();
            end_time = amrex::second();
            BL_PROFILE_VAR_STOP(makeandrungraph);
    
            graph_init = end_time - start_time;
            amrex::Print() << "Time for 1st graphed ParallelCopy/Add (Recorded, Instantiated Ran) = " << graph_init << std::endl;
            // -------------------------------------
    
            // Run the remainder of the FillBoundarys (nsteps-1)
            ParallelDescriptor::Barrier();
            BL_PROFILE_VAR("GRAPH: Run Graph", rungraph);
            start_time = amrex::second();
    
            for (int i=1; i<nsteps; ++i)
            {
                mf_graph.ParallelCopy(mf_init,0,0,Ncomp,Nghost_src,Nghost_dst,geom.periodicity(),op);
            }
    
            ParallelDescriptor::Barrier();
            end_time = amrex::second();
            BL_PROFILE_VAR_STOP(rungraph);
    
            graph_avg = (end_time - start_time)/nsteps;
            amrex::Print() << "Average time per graph-only ParallelCopy/Add = " << graph_avg << std::endl;
        }
    
        // With CPU
        Gpu::setLaunchRegion(false); 
        {
            setup(mf_init, geom);

            // Run the remainder of the FillBoundary's (nsteps-1)
            // -------------------------------------
            ParallelDescriptor::Barrier();
            BL_PROFILE_VAR("CPU: ParallelCopy", CPUFB);
            start_time = amrex::second();
    
            for (int i=0; i<nsteps; ++i)
            {
                mf_cpu.ParallelCopy(mf_init,0,0,Ncomp,Nghost_src,Nghost_dst,geom.periodicity(),op);
            }
    
            ParallelDescriptor::Barrier();
            end_time = amrex::second();
            BL_PROFILE_VAR_STOP(CPUFB);
    
            cpu_avg = (end_time - start_time)/nsteps;
            amrex::Print() << "Average time per CPU ParallelCopy/Add = " << cpu_avg << std::endl;
        }
    
        // With GPU and no graphs
        Gpu::setLaunchRegion(true); 
        Gpu::setGraphRegion(false); 
        {
            setup(mf_init, geom);

            // Run the nstep FillBoundaries
            // -------------------------------------
    
            ParallelDescriptor::Barrier();
            BL_PROFILE_VAR("GPU: ParallelCopy", GPUFB);
            start_time = amrex::second();
    
            for (int i=0; i<nsteps; ++i)
            {
                mf_gpu.ParallelCopy(mf_init,0,0,Ncomp,Nghost_src,Nghost_dst,geom.periodicity(),op);
            }
    
            ParallelDescriptor::Barrier();
            end_time = amrex::second();
            BL_PROFILE_VAR_STOP(GPUFB);
    
            gpu_avg = (end_time - start_time)/nsteps;
            gpu_even = (graph_init)/(gpu_avg - graph_avg);
            amrex::Print() << "Average time per GPU ParallelCopy/Add = " << gpu_avg << std::endl;
            amrex::Print() << "   Graphed ParallelCopy/Add(s) needed to break even = " << gpu_even << std::endl;
        }
    
        // Check the results of Graph vs. CPU and GPU.
        // Maximum of difference of all cells.
        {
            amrex::Real max_error = 0;
            MultiFab mf_error (ba_dst, dm_dst, Ncomp, Nghost_dst);
    
            MultiFab::Copy(mf_error, mf_cpu, 0, 0, Ncomp, Nghost_dst);
            MultiFab::Subtract(mf_error, mf_gpu, 0, 0, Ncomp, Nghost_dst);
            for (int i = 0; i<Ncomp; ++i)
            {
                max_error = std::max(max_error, mf_error.norm0(0, Nghost_dst));
            }
            amrex::Print() << std::endl;
            amrex::Print() << "Max difference between CPU and GPU: " << max_error << std::endl; 
    
            MultiFab::Copy(mf_error, mf_graph, 0, 0, Ncomp, Nghost_dst);
            MultiFab::Subtract(mf_error, mf_cpu, 0, 0, Ncomp, Nghost_dst);
            for (int i = 0; i<Ncomp; ++i)
            {
                max_error = std::max(max_error, mf_error.norm0(0, Nghost_dst));
            }
            amrex::Print() << "Max difference between CPU and Graph: " << max_error << std::endl; 
    
            amrex::Print() << "========================================================" << std::endl << std::endl;
        }

    }

    amrex::Finalize();
}
