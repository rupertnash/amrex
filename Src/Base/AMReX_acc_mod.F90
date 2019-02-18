
module amrex_acc_module

  implicit none

  integer, save :: acc_async_queue

contains

  subroutine amrex_initialize_acc (id) bind(c,name='amrex_initialize_acc')
#ifdef AMREX_USE_ACC
    use openacc, only : acc_init, acc_set_device_num, acc_device_nvidia
#endif
    integer, intent(in), value :: id
#ifdef AMREX_USE_ACC
    call acc_init(acc_device_nvidia)
    call acc_set_device_num(id, acc_device_nvidia)
#endif
  end subroutine amrex_initialize_acc

  subroutine amrex_finalize_acc () bind(c,name='amrex_finalize_acc')
#ifdef AMREX_USE_ACC
    use openacc, only: acc_shutdown, acc_device_nvidia
    call acc_shutdown(acc_device_nvidia)
#endif
  end subroutine amrex_finalize_acc

  subroutine amrex_set_acc_queue (queue, stream) bind(c,name='amrex_set_acc_queue')

    use iso_c_binding, only: c_ptr
#ifdef AMREX_USE_ACC
    use openacc, only: acc_set_cuda_stream, acc_async_sync
#endif

    implicit none

    integer, intent(in), value :: queue
    type(c_ptr), intent(in), value :: stream

    acc_async_queue = queue
#ifdef AMREX_USE_ACC
    if (queue == -1) then
       acc_async_queue = acc_async_sync
    end if
#endif
    call acc_set_cuda_stream(acc_async_queue, stream)

  end subroutine amrex_set_acc_queue

end module amrex_acc_module
