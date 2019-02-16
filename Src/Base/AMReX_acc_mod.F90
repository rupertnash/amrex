
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

  subroutine amrex_set_acc_queue (queue) bind(c,name='amrex_set_acc_queue')

    implicit none

    integer, intent(in), value :: queue

    acc_async_queue = queue

  end subroutine amrex_set_acc_queue

end module amrex_acc_module
