
#include <iostream>
#ifdef NVML
#include <nvml.h>
#include <map>
#endif
#include <AMReX_Device.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>

bool amrex::Device::in_device_launch_region = false;
int amrex::Device::device_id = 0;

int amrex::Device::verbose = 0;

#if defined(AMREX_USE_CUDA) && defined(__CUDACC__)
cudaStream_t amrex::Device::cuda_streams[max_cuda_streams];
cudaStream_t amrex::Device::cuda_stream;

dim3 amrex::Device::numThreadsMin = dim3(1, 1, 1);
#endif

void
amrex::Device::initialize_cuda_c () {

    for (int i = 0; i < max_cuda_streams; ++i)
        CudaAPICheck(cudaStreamCreate(&cuda_streams[i]));

    cuda_stream = cuda_streams[0];
}

cudaStream_t
amrex::Device::stream_from_index(int idx) {

    if (idx < 0)
        return 0;
    else
        return cuda_streams[idx % max_cuda_streams];

}


void
amrex::Device::initialize_device() {

    ParmParse pp("device");

    pp.query("v", verbose);
    pp.query("verbose", verbose);

#ifdef AMREX_USE_CUDA

    const int n_procs = ParallelDescriptor::NProcs();
    const int my_rank = ParallelDescriptor::MyProc();
    const int ioproc  = ParallelDescriptor::IOProcessorNumber();

    int total_count = 1;

#if (defined(NVML) && defined(BL_USE_MPI))

    nvmlReturn_t nvml_err;

    nvml_err = nvmlInit();

    unsigned int nvml_device_count;
    nvml_err = nvmlDeviceGetCount(&nvml_device_count);

    int cuda_device_count;
    cudaGetDeviceCount(&cuda_device_count);

    // Get total number of GPUs seen by all ranks.

    total_count = (int) nvml_device_count;

    amrex::ParallelDescriptor::ReduceIntSum(total_count);

    total_count = total_count / n_procs;

    std::map<std::string, int> device_uuid;

    unsigned int char_length = 256;
    char uuid[char_length];

    std::string s;

    for (unsigned int i = 0; i < nvml_device_count; ++i) {

	nvmlDevice_t handle;
	nvml_err = nvmlDeviceGetHandleByIndex(i, &handle);

	nvml_err = nvmlDeviceGetUUID(handle, uuid, char_length);
	s = uuid;

        bool present = false;

        if (device_uuid.find(s) != device_uuid.end())
            present = true;

	if (present) {

	    device_uuid[s] += 1;

	} else {

            // Add this device to our list, but only if it's
            // visible to CUDA.

            nvmlPciInfo_t pci_info;
            nvml_err = nvmlDeviceGetPciInfo(handle, &pci_info);

            char bus_id[char_length];

            for (int cuda_device = 0; cuda_device < cuda_device_count; ++cuda_device) {

                CudaAPICheck(cudaDeviceGetPCIBusId(bus_id, char_length, cuda_device));
                CudaAPICheck(cudaDeviceReset());

                std::string s_n(pci_info.busId);
                std::string s_c(bus_id);

                if (s_c == s_n) {
                    device_uuid[s] = 1;
                    break;
                }

            }

        }

    }

    // Gather the list of device UUID's from all ranks. For simplicitly we'll
    // assume that every rank sees the same number of devices and every device
    // UUID has the same number of characters; this assumption could be lifted
    // later in situations with unequal distributions of devices per node.

    int strlen = s.size();
    int len = cuda_device_count * strlen;

    char sendbuf[len];
    char recvbuf[n_procs][len];

    int i = 0;
    for (auto it = device_uuid.begin(); it != device_uuid.end(); ++it) {
        for (int j = 0; j < strlen; ++j) {
            sendbuf[i * strlen + j] = it->first[j];
        }
        ++i;
    }

    MPI_Allgather(sendbuf, len, MPI_CHAR, recvbuf, len, MPI_CHAR, MPI_COMM_WORLD);

    // Count up the number of ranks that share each GPU, and record their rank number.

    std::map<std::string, std::vector<int>> uuid_to_rank;

    for (auto it = device_uuid.begin(); it != device_uuid.end(); ++it) {
        it->second = 0;
        for (int i = 0; i < n_procs; ++i) {
            for (int j = 0; j < cuda_device_count; ++j) {
                std::string temp_s(&recvbuf[i][j * strlen], strlen);
                if (it->first == temp_s) {
                    it->second += 1;
                    uuid_to_rank[it->first].push_back(i);
                }
            }
        }
    }

    // For each rank that shares a GPU, use round-robin assignment
    // to assign MPI ranks to GPUs. We will arbitrarily assign
    // ranks to GPUs. It would be nice to do better here and be
    // socket-aware, but this is complicated to get right. It is
    // better for this to be offloaded to the job launcher.

    device_id = -1;
    int j = 0;
    for (auto it = uuid_to_rank.begin(); it != uuid_to_rank.end(); ++it) {
        for (int i = 0; i < it->second.size(); ++i) {
            if (it->second[i] == my_rank && i % cuda_device_count == j) {

                device_id = j;

                break;
            }
        }
        j += 1;
        if (device_id != -1) break;
    }
    if (device_id == -1 || device_id >= cuda_device_count)
        amrex::Abort("Could not associate MPI rank with GPU");

#endif

    ParallelDescriptor::Barrier();

    initialize_cuda(&device_id, &my_rank, &total_count, &n_procs, &ioproc, &verbose);

    initialize_cuda_c();

#endif

}

void
amrex::Device::finalize_device() {

#ifdef AMREX_USE_CUDA
    finalize_cuda();

    set_is_program_running(0);
#endif

}

int
amrex::Device::deviceId() {

    return device_id;

}

void
amrex::Device::set_stream_index(const int idx) {

#ifdef AMREX_USE_CUDA
    set_stream_idx(idx);
#endif

    cuda_stream = cuda_streams[idx % max_cuda_streams + 1];

}

int
amrex::Device::get_stream_index() {

    int index = -1;

#ifdef AMREX_USE_CUDA
    get_stream_idx(&index);
#endif

    return index;

}

void
amrex::Device::prepare_for_launch(const int* lo, const int* hi) {

    // Sets the number of threads and blocks in Fortran.

#if defined(AMREX_USE_CUDA) && defined(__CUDACC__)
    int txmin = numThreadsMin.x;
    int tymin = numThreadsMin.y;
    int tzmin = numThreadsMin.z;

    set_threads_and_blocks(lo, hi, &txmin, &tymin, &tzmin);
#endif

}

void*
amrex::Device::get_host_pointer(const void* ptr) {

    void* r = const_cast<void*>(ptr);
#ifdef AMREX_USE_CUDA
    gpu_host_device_ptr(&r, ptr);
#endif
    return r;

}

void
amrex::Device::check_for_errors() {

#if defined(AMREX_USE_CUDA) && defined(__CUDACC__)
    CudaErrorCheck();
#endif

}

void
amrex::Device::synchronize() {

#ifdef AMREX_USE_CUDA
    gpu_synchronize();
#endif

}

void
amrex::Device::stream_synchronize(const int idx) {

#ifdef AMREX_USE_CUDA
    gpu_stream_synchronize(idx);
#endif

}

void*
amrex::Device::device_malloc(const std::size_t sz) {

    void* ptr = NULL;

#ifdef AMREX_USE_CUDA
    gpu_malloc(&ptr, &sz);
#endif

    return ptr;

}

void
amrex::Device::device_free(void* ptr) {

#ifdef AMREX_USE_CUDA
    gpu_free(ptr);
#endif

}

void
amrex::Device::device_htod_memcpy_async(void* p_d, const void* p_h, const std::size_t sz, const int idx) {

#ifdef AMREX_USE_CUDA
    gpu_htod_memcpy_async(p_d, p_h, &sz, &idx);
#endif

}

void
amrex::Device::device_dtoh_memcpy_async(void* p_h, const void* p_d, const std::size_t sz, const int idx) {

#ifdef AMREX_USE_CUDA
    gpu_dtoh_memcpy_async(p_h, p_d, &sz, &idx);
#endif

}

void
amrex::Device::start_profiler() {

#ifdef AMREX_USE_CUDA
    gpu_start_profiler();
#endif

}

void
amrex::Device::stop_profiler() {

#ifdef AMREX_USE_CUDA
    gpu_stop_profiler();
#endif

}

#if (defined(AMREX_USE_CUDA) && defined(__CUDACC__))
void
amrex::Device::c_threads_and_blocks(const int* lo, const int* hi, dim3& numBlocks, dim3& numThreads) {

    int bx, by, bz, tx, ty, tz;

    int txmin = numThreadsMin.x;
    int tymin = numThreadsMin.y;
    int tzmin = numThreadsMin.z;

    get_threads_and_blocks(lo, hi, &bx, &by, &bz, &tx, &ty, &tz, &txmin, &tymin, &tzmin);

    numBlocks.x = bx;
    numBlocks.y = by;
    numBlocks.z = bz;

    numThreads.x = tx;
    numThreads.y = ty;
    numThreads.z = tz;

}
#endif
