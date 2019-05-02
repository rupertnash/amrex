#ifndef AMREX_GPU_TIMER_H_
#define AMREX_GPU_TIMER_H_

#include <AMReX_REAL.H>

#ifdef AMREX_USE_CUDA
#include "nvToolsExt.h"
#endif

namespace amrex {

    /** \brief A timer that uses Cuda events to measure GPU execution time.
     *  When AMREX_USE_CUDA=FALSE, reverts to measuring CPU time.
     */
class GpuTimer
{
public:
    GpuTimer () noexcept;
    ~GpuTimer ();

    void start () noexcept;
    Real stop () noexcept;

private:

#ifdef AMREX_USE_CUDA
    cudaEvent_t m_start, m_stop;
#else
    Real m_start;
#endif
};

}

#endif
