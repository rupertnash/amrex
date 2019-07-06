#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_MultiFab.H>

using namespace amrex;

constexpr double epsilon = DBL_EPSILON * 4;

// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

// Single launch version, no simultaneous boxes.
// Kept to possibly test overhead in this version compared to simul version.
AMREX_GPU_GLOBAL
void copy (int size, amrex::Box* bx, amrex::Dim3* offset,
           amrex::Array4<Real>* src, amrex::Array4<Real>* dst, 
           int scomp, int dcomp, int ncomp)
{
    // Should add these where used instead of here
    //    to reduce register count?
    const int tid = blockDim.x*blockIdx.x+threadIdx.x;
    const int stride = blockDim.x*gridDim.x;

    for (int l=0; l<size; ++l)
    {
        int ncells = bx[l].numPts();
        const auto lo = amrex::lbound(bx[l]);
        const auto len = amrex::length(bx[l]);
 
        for (int icell = tid; icell < ncells; icell += stride) {
            int k =  icell /   (len.x*len.y);
            int j = (icell - k*(len.x*len.y)) /   len.x;
            int i = (icell - k*(len.x*len.y)) - j*len.x;
            i += lo.x;
            j += lo.y;
            k += lo.z;
            for (int n = 0; n < ncomp; ++n) {
                (dst[l])(i,j,k,dcomp+n) = (src[l])(i+offset[l].x,j+offset[l].y,k+offset[l].z,scomp+n); 
            }
        }
    }
}

// Single launch, simultaneous box calc version
AMREX_GPU_GLOBAL
void copy (int size, amrex::Box* bx, amrex::Dim3* offset,
           amrex::Array4<Real>* src, amrex::Array4<Real>* dst, 
           int scomp, int dcomp, int ncomp, int simul)
{
    // Break up the threads evenly across the boxes.

    // Should add these where used instead of here
    //    to reduce registery count?
    int tid = blockDim.x*blockIdx.x+threadIdx.x;
    int stride = (blockDim.x*gridDim.x)/simul;
    int bidx = tid/stride;

    for (int bid = bidx; bid < size; bid+=simul)
    {
        int ncells = bx[bid].numPts();
        const auto lo  = amrex::lbound(bx[bid]);
        const auto len = amrex::length(bx[bid]);
 
        for (int icell = (tid%ncells); icell < ncells; icell += stride) {
            int k =  icell /   (len.x*len.y);
            int j = (icell - k*(len.x*len.y)) /   len.x;
            int i = (icell - k*(len.x*len.y)) - j*len.x;
            i += lo.x;
            j += lo.y;
            k += lo.z;
            for (int n = 0; n < ncomp; ++n) {
                (dst[bid])(i,j,k,dcomp+n) = (src[bid])(i+offset[bid].x,j+offset[bid].y,k+offset[bid].z,scomp+n); 
            }
        }
    }
}

// Single box at a time version
//    Launch within MFIter loop 
AMREX_GPU_GLOBAL
void copy (amrex::Dim3 lo, amrex::Dim3 len, int ncells,
           amrex::Dim3 offset, amrex::Array4<Real> src, amrex::Array4<Real> dst, 
           int scomp, int dcomp, int ncomp)
{
    for (int icell = blockDim.x*blockIdx.x+threadIdx.x, stride = blockDim.x*gridDim.x;
             icell < ncells; icell += stride) {
        int k =  icell /   (len.x*len.y);
        int j = (icell - k*(len.x*len.y)) /   len.x;
        int i = (icell - k*(len.x*len.y)) - j*len.x;
        i += lo.x;
        j += lo.y;
        k += lo.z;
        for (int n = 0; n < ncomp; ++n) {
            dst(i,j,k,dcomp+n) = src(i+offset.x,j+offset.y,k+offset.z,scomp+n); 
        }
    }
}

// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

std::string buildMFs(MultiFab& src_fab, MultiFab& dst_fab, 
                     IntVect n_cell, IntVect max_grid_size, int Ncomp, int Nghost) 
{
    BoxArray ba;
    {
        IntVect dom_lo(AMREX_D_DECL(          0,           0,           0));
        IntVect dom_hi(AMREX_D_DECL(n_cell[0]-1, n_cell[1]-1, n_cell[2]-1));
        Box domain(dom_lo, dom_hi);

        // Initialize the boxarray "ba" from the single box "bx"
        ba.define(domain);
        // Break up boxarray "ba" into chunks no larger than "max_grid_size" along a direction
        ba.maxSize(max_grid_size);
    }

    DistributionMapping dm(ba);

    src_fab.define(ba, dm, Ncomp, Nghost);
    dst_fab.define(ba, dm, Ncomp, Nghost);

    // Extra first touch to make sure the MultiFab data is moved to the device.
    src_fab.setVal(0.0);
    dst_fab.setVal(0.0);

    std::string mf_label = AMREX_D_TERM( '(' + std::to_string(n_cell[0]) ,
                                       + ',' + std::to_string(n_cell[1]) ,
                                       + ',' + std::to_string(n_cell[2]) + ')') + 'x' +
                           AMREX_D_TERM( '(' + std::to_string(max_grid_size[0]) ,
                                       + ',' + std::to_string(max_grid_size[1]) ,
                                       + ',' + std::to_string(max_grid_size[2]) + ')') + '-'
                                       + std::to_string(Ncomp) + "C/" 
                                       + std::to_string(Nghost) + "G";

    return mf_label;

}

// -------------------------------------------------------------

void errorCheck(std::string label, MultiFab& src_fab, MultiFab& dst_fab)
{
    Real src_sum = src_fab.sum();
    Real dst_sum = dst_fab.sum();

    if ( (std::abs(src_sum - dst_sum)/src_sum) > epsilon )
    {
        amrex::Print() << "*********** " << label << " error found." << std::endl;
        amrex::Print() << " ---- dst = " << dst_sum 
                       << "; src = "     << src_sum  
                       << "; delta = "   << std::abs(dst_sum - src_sum)/src_sum << std::endl;

        // Full error check. Search for and print all cells with errors. Uncomment if needed.
/*
        for (MFIter mfi(src_fab); mfi.isValid(); ++mfi)
        {
            const Box bx = mfi.validbox();
            Array4<Real> const& src = src_fab.array(mfi);
            Array4<Real> const& dst = dst_fab.array(mfi);

            amrex::ParallelFor(bx, src_fab.nComp(),
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
            {
                if (src(i,j,k,n) != dst(i,j,k,n))
                {
                    printf("Error on: %i,%i,%i,%i: %f != %f\n",i,j,k,n,src(i,j,k,n),dst(i,j,k,n));
                }
            });
        }
*/
    }
}

// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

void loopLaunch(IntVect n_cells, IntVect max_grid_size, int Ncomp, int Nghost, 
                int cpt = 1, int synchs = 0, int num_streams = Gpu::Device::numGpuStreams())
{
    MultiFab src_fab, dst_fab;
    std::string mf_label = buildMFs(src_fab, dst_fab, n_cells, max_grid_size, Ncomp, Nghost);

    src_fab.setVal(amrex::Random());
    dst_fab.setVal(amrex::Random());

    if (num_streams > Gpu::Device::numGpuStreams())
    {
        num_streams = Gpu::Device::numGpuStreams();
        std::cout << "Too many streams requested. Using maximum value of " << num_streams << std::endl;
    }

    int ips = 0;    // (iterations per synch)
    if ( (synchs <= 0) || (synchs > src_fab.local_size()) )
    {
        synchs = 0;
        ips = src_fab.local_size();         // No synchs.
    }
    else
    {
        ips = src_fab.local_size() / synchs;
    }

    std::string timer_name = "LOOP:" + mf_label + ": " +
                               std::to_string(num_streams) + " streams, " + 
                               std::to_string(cpt) + " CPT, " +
                               std::to_string(synchs) + " synchs";

    double timer_start = amrex::second();
    BL_PROFILE_VAR(timer_name, loop);
    for (MFIter mfi(src_fab); mfi.isValid(); ++mfi)
    {
        const int idx = mfi.LocalIndex();
        Gpu::Device::setStreamIndex(idx % num_streams);
        const Box bx = mfi.validbox();

        const auto src = src_fab.array(mfi);
        const auto dst = dst_fab.array(mfi);

        const int ncells = bx.numPts();
        const auto lo  = amrex::lbound(bx);
        const auto len = amrex::length(bx);
        const Dim3 offset = {0,0,0};

        const auto ec = Gpu::ExecutionConfig((bx.numPts()+cpt-1)/cpt);
        AMREX_GPU_LAUNCH_GLOBAL(ec, copy,
                                lo, len, ncells,
                                offset, src, dst,
                                0, 0, Ncomp);

        if ((idx % ips) == 0)
        {
            Gpu::Device::synchronize();
        }
    }
    BL_PROFILE_VAR_STOP(loop);
    double timer_end = amrex::second();

    amrex::Print() << timer_name << " = " << timer_end-timer_start << " seconds." << std::endl;
    errorCheck(timer_name, src_fab, dst_fab);
}

// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

void fusedLaunch(IntVect n_cells, IntVect max_grid_size, int Ncomp, int Nghost, 
                      int cpt = 1, int simul = 1, int num_launches = 1)
{
    MultiFab src_fab, dst_fab;
    std::string mf_label = buildMFs(src_fab, dst_fab, n_cells, max_grid_size, Ncomp, Nghost);

    src_fab.setVal(amrex::Random());
    dst_fab.setVal(amrex::Random());

    int lsize = src_fab.local_size();

    if (num_launches > lsize)
    {
        num_launches = lsize; 
        std::cout << "Too many launches requested. Using one box per launch: " << dst_fab.local_size() << std::endl;
    }
    if (simul > lsize)
    {
        num_launches = lsize; 
        std::cout << "Too many simultaneous boxes requested. Using total number of boxes: " << dst_fab.local_size() << std::endl;
    }

    std::string timer_name = "FUSED:" + mf_label + ": " +
                               std::to_string(num_launches) + " launches, " + 
                               std::to_string(cpt) + " CPT, " +
                               std::to_string(simul) + " simul";

    double timer_start = amrex::second();
//    BL_PROFILE_REGION_START(std::string(timer_name + " region"));
    BL_PROFILE_VAR(timer_name, fused);
    BL_PROFILE_VAR("FUSED: ALLOC", fusedalloc);

    Box* bx_h =           static_cast<Box*>         (The_Pinned_Arena()->alloc(lsize*sizeof(Box)));
    Dim3* offset_h =      static_cast<Dim3*>        (The_Pinned_Arena()->alloc(lsize*sizeof(Dim3)));
    Array4<Real>* src_h = static_cast<Array4<Real>*>(The_Pinned_Arena()->alloc(lsize*sizeof(Array4<Real>)));
    Array4<Real>* dst_h = static_cast<Array4<Real>*>(The_Pinned_Arena()->alloc(lsize*sizeof(Array4<Real>)));

    Box* bx_d =           static_cast<Box*>         (The_Device_Arena()->alloc(lsize*sizeof(Box)));
    Dim3* offset_d =      static_cast<Dim3*>        (The_Device_Arena()->alloc(lsize*sizeof(Dim3)));
    Array4<Real>* src_d = static_cast<Array4<Real>*>(The_Device_Arena()->alloc(lsize*sizeof(Array4<Real>)));
    Array4<Real>* dst_d = static_cast<Array4<Real>*>(The_Device_Arena()->alloc(lsize*sizeof(Array4<Real>))); 

    BL_PROFILE_VAR_STOP(fusedalloc);
    BL_PROFILE_VAR("FUSED: SETUP", fusedsetup);

    int max_box_size = 0;

    for (MFIter mfi(src_fab); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.validbox();
	int idx = mfi.LocalIndex(); 

        bx_h[idx] = bx;
        offset_h[idx] = {0,0,0};
        src_h[idx] = src_fab.array(mfi);
        dst_h[idx] = dst_fab.array(mfi);

        if (bx.numPts() > max_box_size)
        { 
            max_box_size = bx.numPts();
        }
    }
 
    BL_PROFILE_VAR_STOP(fusedsetup);
    BL_PROFILE_VAR("FUSED: COPY+LAUNCH", fusedl);
                     
    AMREX_GPU_SAFE_CALL(cudaMemcpyAsync(bx_d,     bx_h,     lsize*sizeof(Box),          cudaMemcpyHostToDevice));
    AMREX_GPU_SAFE_CALL(cudaMemcpyAsync(offset_d, offset_h, lsize*sizeof(Dim3),         cudaMemcpyHostToDevice));
    AMREX_GPU_SAFE_CALL(cudaMemcpyAsync(src_d,    src_h,    lsize*sizeof(Array4<Real>), cudaMemcpyHostToDevice));
    AMREX_GPU_SAFE_CALL(cudaMemcpyAsync(dst_d,    dst_h,    lsize*sizeof(Array4<Real>), cudaMemcpyHostToDevice));

    // For this simple test, assume all boxes have the same size. Otherwise, launch on the biggest box.
    const auto ec = Gpu::ExecutionConfig(((max_box_size*simul)+cpt-1)/cpt);
    int l_start = 0;

    for(int lid = 0; lid < num_launches; ++lid) 
    {
        amrex::Gpu::Device::setStreamIndex(lid);
        int bx_num = (lsize/num_launches) + (lid < (lsize%num_launches)); 

        if (bx_num > 0)
        {
            AMREX_GPU_LAUNCH_GLOBAL(ec, copy, bx_num, 
                                    bx_d+l_start,  offset_d+l_start,
                                    src_d+l_start, dst_d+l_start,
                                    0, 0, Ncomp);
        }
        l_start += bx_num;
    }
    amrex::Gpu::Device::resetStreamIndex();
    amrex::Gpu::Device::synchronize();

    BL_PROFILE_VAR_STOP(fusedl);
    BL_PROFILE_VAR_STOP(fused);
//    BL_PROFILE_REGION_STOP(std::string(timer_name + " region"));
    double timer_end = amrex::second();

    amrex::Print() << timer_name << " = " << timer_end-timer_start << " seconds." << std::endl;
    errorCheck(timer_name, src_fab, dst_fab);

    The_Pinned_Arena()->free(bx_h);
    The_Pinned_Arena()->free(offset_h);
    The_Pinned_Arena()->free(src_h);
    The_Pinned_Arena()->free(dst_h);

    The_Device_Arena()->free(bx_d);
    The_Device_Arena()->free(offset_d);
    The_Device_Arena()->free(src_d);
    The_Device_Arena()->free(dst_d);
}

// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

void loopSuite(IntVect n_cells, IntVect max_grid_size, int Ncomp, int Nghost,
                   int cpt_start, int cpt_end, int cpt_stride,
                   int synchs_start, int synchs_end, int synchs_stride,
                   int nstreams_start, int nstreams_end, int nstreams_stride)
{
    for (int cpt = cpt_start; cpt <= cpt_end; cpt *= cpt_stride)
    {
        for (int synchs = synchs_start; synchs <= synchs_end; synchs += synchs_stride)
        {
            for(int nstreams = nstreams_start; nstreams <= nstreams_end; nstreams *= nstreams_stride)
            {
                loopLaunch(n_cells, max_grid_size, Ncomp, Nghost, cpt, synchs, nstreams);
            }
        }
    }
}

// -------------------------------------------------------------

void fusedSuite(IntVect n_cells, IntVect max_grid_size, int Ncomp, int Nghost,
                int cpt_start, int cpt_end, int cpt_stride,
                int simul_start, int simul_end, int simul_stride,
                int nlaunches_start, int nlaunches_end, int nlaunches_stride)
{
    for (int cpt = cpt_start; cpt <= cpt_end; cpt *= cpt_stride)
    {
        for (int simul = simul_start; simul <= simul_end; simul *= simul_stride)
        {
            for(int nlaunches = nlaunches_start; nlaunches <= nlaunches_end; nlaunches *= nlaunches_stride)
            {
                fusedLaunch(n_cells, max_grid_size, Ncomp, Nghost, cpt, simul, nlaunches);
            }
        }
    }
}


// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {

        // Test entire launch range for a given problem size.
        IntVect n_cells(256);
        IntVect max_grid_size(16);
        int Ncomp = 1;
        int Nghost = 0;

        amrex::Print() << std::endl << "Loop Tests" << std::endl;

        loopSuite(n_cells, max_grid_size, Ncomp, Nghost,
                      1,  32,  2,    // cells per thread
                      0,  10,  1,    // evenly-distributed synchs ( setup for none )
                      1,  16,  2);   // number of threads

        amrex::Print() << std::endl << "Fused Tests" << std::endl;

        fusedSuite(n_cells, max_grid_size, Ncomp, Nghost,
                   1, 32, 2,    // cells per thread
                   1, 32, 2,    // simultaneous boxes 
                   1, 32, 2);   // number of launches

/*
        // Call Syntax: Defaults in { }
        // loopLaunch (n_cells, max_grid_size, Ncomp, Nghost, cpt{1}, synchs{0}, num_streams{Gpu::Device::numGpuStreams})
        // fusedLaunch (n_cells, max_grid_size, Ncomp, Nghost, cpt{1}, simul{1}, num_launches{1})


        // loopLaunch (n_cells, max_grid_size, Ncomp, Nghost, cpt{1}, synchs{0}, num_streams{Gpu::Device::numGpuStreams})
        loopLaunch(        256,            64,     1,      0);
        loopLaunch(        256,            61,     1,      0);
        loopLaunch(        256,            16,     1,      0);
        loopLaunch(        256,            14,     1,      0);
        loopLaunch(         64,            16,     1,      0);
        loopLaunch(         64,            14,     1,      0);

        amrex::Print() << std::endl << std::endl;

        // fusedLaunch (n_cells, max_grid_size, Ncomp, Nghost, cpt{1}, simul{1}, num_launches{1})
        fusedLaunch(        256,            64,     1,      0);
        fusedLaunch(        256,            61,     1,      0);
        fusedLaunch(        256,            16,     1,      0);
        fusedLaunch(        256,            14,     1,      0);
        fusedLaunch(         64,            16,     1,      0);
        fusedLaunch(         64,            14,     1,      0);

        fusedLaunch(        256,            14,     1,      0,      1,        2);
        fusedLaunch(        256,            14,     1,      0,      1,        3);
        fusedLaunch(        256,            14,     1,      0,      1,        4);
        fusedLaunch(        256,            14,     1,      0,      1,        8);
*/
    }
    amrex::Finalize();
}
