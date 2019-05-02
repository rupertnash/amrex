#include <AMReX_Utility.H>
#include <AMReX_GpuTimer.H>

namespace amrex {

GpuTimer::GpuTimer () noexcept
{
#ifdef AMREX_USE_CUDA
    cudaEventCreate(&m_start);
    cudaEventCreate(&m_stop);
#endif

    start();
}

GpuTimer::~GpuTimer () noexcept
{
    stop();
#ifdef AMREX_USE_CUDA
    cudaEventDestroy(m_start);
    cudaEventDestroy(m_stop);
#endif
}
    
void
GpuTimer::start () noexcept
{
#ifdef AMREX_USE_CUDA
    cudaEventRecord(m_start);
#else
    m_start = amrex::second();
#endif
}

Real
GpuTimer::stop () noexcept
{
#ifdef AMREX_USE_CUDA
    cudaEventRecord(m_stop);
    
    cudaEventSynchronize(m_stop);
    
    float time = 0.0;
    cudaEventElapsedTime(&time, m_start, m_stop);
    return time;
#else
    return amrex::second() - m_start;
#endif    
}

}
