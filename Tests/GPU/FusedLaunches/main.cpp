#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_MultiFab.H>

using namespace amrex;

// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

// Single launch version
AMREX_GPU_GLOBAL
void copy (int size, amrex::Box* bx, amrex::Dim3* offset,
           amrex::Array4<Real>* src, amrex::Array4<Real>* dst, 
           int scomp, int dcomp, int ncomp)
{
    for (int l=0; l<size; ++l)
    {
        int ncells = bx[l].numPts();
        const auto lo = amrex::lbound(bx[l]);
        const auto len = amrex::length(bx[l]);
 
        for (int icell = blockDim.x*blockIdx.x+threadIdx.x, stride = blockDim.x*gridDim.x;
                 icell < ncells; icell += stride) {
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

// Single launch, simutaneous box calc version
AMREX_GPU_GLOBAL
void copy (int size, amrex::Box* bx, amrex::Dim3* offset,
           amrex::Array4<Real>* src, amrex::Array4<Real>* dst, 
           int scomp, int dcomp, int ncomp, int simul)
{
    // Works because boxes are currently of equal size.
    //    When box size varies: calc or pass in maximum box size.
    //    Will need to adjust threads.
    int ncells = bx[0].numPts();
    int tid = blockDim.x*blockIdx.x+threadIdx.x;
    int bidx = tid/ncells;

    if (bidx < simul)
    {
        for (int l = 0; l < size; l+=simul)
        {
            int bid = l+bidx;
            const auto lo  = amrex::lbound(bx[bid]);
            const auto len = amrex::length(bx[bid]);
 
            for (int icell = (tid%ncells), stride = (blockDim.x*gridDim.x)/simul; icell < ncells; icell += stride) {
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

void buildMFs(MultiFab& src_fab, MultiFab& dst_fab, 
              int n_cell, int max_grid_size, int Ncomp, int Nghost) 
{
    BoxArray ba;
    {
        IntVect dom_lo(AMREX_D_DECL(       0,        0,        0));
        IntVect dom_hi(AMREX_D_DECL(n_cell-1, n_cell-1, n_cell-1));
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
}

// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

void standardLaunch(int n_cells, int max_grid_size, int Ncomp, int Nghost, 
                                      Real src_val, Real dst_val,
                                      int cpt = 1, int ips = 0,
                                      int num_streams = Gpu::Device::numGpuStreams())
{
    MultiFab src_fab, dst_fab;
    buildMFs(src_fab, dst_fab, n_cells, max_grid_size, Ncomp, Nghost);

    std::string mf_label = "(" + std::to_string(n_cells) + "x" + std::to_string(max_grid_size) + ","
                               + std::to_string(Ncomp) + "C/" + std::to_string(Nghost) + "G)";

    src_fab.setVal(src_val);
    dst_fab.setVal(dst_val);

    // If CPT > CPB test/adjust?

    if (num_streams > Gpu::Device::numGpuStreams())
    {
        num_streams = Gpu::Device::numGpuStreams();
        std::cout << "Too many streams requested. Using maximum value of " << num_streams << std::endl;
    }

    if ( (ips <= 0) || (ips > src_fab.local_size()) )
    {
        ips = src_fab.local_size();
    }

    std::string timer_name = "STANDARD" + mf_label + ": " +
                               std::to_string(num_streams) + " streams, " + 
                               std::to_string(cpt) + " CPT, " +
                               std::to_string(ips) + " IPS";

    double timer_start = amrex::second();
    BL_PROFILE_VAR(timer_name, standard);
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
    BL_PROFILE_VAR_STOP(standard);
    double timer_end = amrex::second();

    amrex::Print() << timer_name << " = " << timer_end-timer_start << " seconds." << std::endl;

    // Error check
    Real src_sum = src_fab.sum();
    Real dst_sum = dst_fab.sum();
    if (src_sum != dst_sum)
    {
        amrex::Print() << "*********** " << timer_name << " error found." << std::endl;
        amrex::Print() << " ---- dst = " << dst_sum 
                       << "; src = " << src_sum << std::endl;
    }
}

// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&


void fusedLaunch(int n_cells, int max_grid_size, int Ncomp, int Nghost, 
                                   Real src_val, Real dst_val,
                                   int cpt = 1, int simul = 1, int num_launches = 1)
{
    MultiFab src_fab, dst_fab;
    buildMFs(src_fab, dst_fab, n_cells, max_grid_size, Ncomp, Nghost);

    std::string mf_label = "(" + std::to_string(n_cells) + "x" + std::to_string(max_grid_size) + ","
                               + std::to_string(Ncomp) + "C/" + std::to_string(Nghost) + "G)";

    src_fab.setVal(src_val);
    dst_fab.setVal(dst_val);

    int lsize = src_fab.local_size();

    // Note: Not a parallel copy, so src_ba = dst_ba.
    if (num_launches > lsize)
    {
        num_launches = lsize; 
        std::cout << "Too many launches requested. Using one box per launch: " << dst_fab.local_size() << std::endl;
    }

    std::string timer_name = "FUSED" + mf_label + ": " +
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
                                    0, 0, Ncomp, simul);
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

    Real src_sum = src_fab.sum();
    Real dst_sum = dst_fab.sum();
    if (src_sum != dst_sum)
    {
        amrex::Print() << "*********** " << timer_name << " error found." << std::endl;
        amrex::Print() << " ---- dst = " << dst_sum 
                       << "; src = " << src_sum << std::endl;
    }

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

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        amrex::Print() << " Testing with uneven boxes " << std::endl << std::endl;

        // Call Syntax: Defualts in { }
        // standardLaunch (domain_size, max_grid_size, Ncomp, Nghost, src_val, dst_val, cpt{1}, ips{0}, num_streams{Gpu::Device::numGpuStreams})
        standardLaunch(            256,            64,     1,      1,     4.5,     2.1);
        standardLaunch(            256,            61,     1,      1,     3.5,    3.14);
        standardLaunch(            256,            16,     1,      1,     4.1,     1.2);
        standardLaunch(            256,            14,     1,      1,    0.27,   1.768);
        standardLaunch(             64,            16,     1,      1,     3.1,     9.2);
        standardLaunch(             64,            14,     1,      1,    0.17,    0.68);

        amrex::Print() << std::endl;

        // fusedLaunch (domain_size, max_grid_size, Ncomp, Nghost, src_val, dst_val, cpt{1}, simul{1}, num_launches{1})
        fusedLaunch(            256,            64,     1,      1,    0.33,    1.22);
        fusedLaunch(            256,            61,     1,      1,    0.63,    1.26);
        fusedLaunch(            256,            16,     1,      1,    0.37,    0.45);
        fusedLaunch(            256,            14,     1,      1,    0.93,    0.55);
        fusedLaunch(             64,            16,     1,      1,    0.31,    0.15);
        fusedLaunch(             64,            14,     1,      1,    0.91,    0.51);
   }

    amrex::Finalize();
}
