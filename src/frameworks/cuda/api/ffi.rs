//! Describes the foreign function interface of the CUDA DRIVER API
//!
#![allow(non_camel_case_types, non_snake_case)]
// Created by bindgen

pub type size_t = ::libc::c_ulong;
pub type wchar_t = ::libc::c_int;
pub type __u_char = ::libc::c_uchar;
pub type __u_short = ::libc::c_ushort;
pub type __u_int = ::libc::c_uint;
pub type __u_long = ::libc::c_ulong;
pub type __int8_t = ::libc::c_char;
pub type __uint8_t = ::libc::c_uchar;
pub type __int16_t = ::libc::c_short;
pub type __uint16_t = ::libc::c_ushort;
pub type __int32_t = ::libc::c_int;
pub type __uint32_t = ::libc::c_uint;
pub type __int64_t = ::libc::c_long;
pub type __uint64_t = ::libc::c_ulong;
pub type __quad_t = ::libc::c_long;
pub type __u_quad_t = ::libc::c_ulong;
pub type __dev_t = ::libc::c_ulong;
pub type __uid_t = ::libc::c_uint;
pub type __gid_t = ::libc::c_uint;
pub type __ino_t = ::libc::c_ulong;
pub type __ino64_t = ::libc::c_ulong;
pub type __mode_t = ::libc::c_uint;
pub type __nlink_t = ::libc::c_ulong;
pub type __off_t = ::libc::c_long;
pub type __off64_t = ::libc::c_long;
pub type __pid_t = ::libc::c_int;
#[repr(C)]
#[derive(Copy)]
pub struct Struct_Unnamed1 {
    pub __val: [::libc::c_int; 2usize],
}
impl ::std::clone::Clone for Struct_Unnamed1 {
    fn clone(&self) -> Self { *self }
}
impl ::std::default::Default for Struct_Unnamed1 {
    fn default() -> Self { unsafe { ::std::mem::zeroed() } }
}
pub type __fsid_t = Struct_Unnamed1;
pub type __clock_t = ::libc::c_long;
pub type __rlim_t = ::libc::c_ulong;
pub type __rlim64_t = ::libc::c_ulong;
pub type __id_t = ::libc::c_uint;
pub type __time_t = ::libc::c_long;
pub type __useconds_t = ::libc::c_uint;
pub type __suseconds_t = ::libc::c_long;
pub type __daddr_t = ::libc::c_int;
pub type __key_t = ::libc::c_int;
pub type __clockid_t = ::libc::c_int;
pub type __timer_t = *mut ::libc::c_void;
pub type __blksize_t = ::libc::c_long;
pub type __blkcnt_t = ::libc::c_long;
pub type __blkcnt64_t = ::libc::c_long;
pub type __fsblkcnt_t = ::libc::c_ulong;
pub type __fsblkcnt64_t = ::libc::c_ulong;
pub type __fsfilcnt_t = ::libc::c_ulong;
pub type __fsfilcnt64_t = ::libc::c_ulong;
pub type __fsword_t = ::libc::c_long;
pub type __ssize_t = ::libc::c_long;
pub type __syscall_slong_t = ::libc::c_long;
pub type __syscall_ulong_t = ::libc::c_ulong;
pub type __loff_t = __off64_t;
pub type __qaddr_t = *mut __quad_t;
pub type __caddr_t = *mut ::libc::c_char;
pub type __intptr_t = ::libc::c_long;
pub type __socklen_t = ::libc::c_uint;
#[repr(C)]
#[derive(Copy)]
pub struct Union_wait {
    pub _bindgen_data_: [u32; 1usize],
}
impl Union_wait {
    pub unsafe fn w_status(&mut self) -> *mut ::libc::c_int {
        let raw: *mut u8 = ::std::mem::transmute(&self._bindgen_data_);
        ::std::mem::transmute(raw.offset(0))
    }
    pub unsafe fn __wait_terminated(&mut self) -> *mut Struct_Unnamed2 {
        let raw: *mut u8 = ::std::mem::transmute(&self._bindgen_data_);
        ::std::mem::transmute(raw.offset(0))
    }
    pub unsafe fn __wait_stopped(&mut self) -> *mut Struct_Unnamed3 {
        let raw: *mut u8 = ::std::mem::transmute(&self._bindgen_data_);
        ::std::mem::transmute(raw.offset(0))
    }
}
impl ::std::clone::Clone for Union_wait {
    fn clone(&self) -> Self { *self }
}
impl ::std::default::Default for Union_wait {
    fn default() -> Self { unsafe { ::std::mem::zeroed() } }
}
#[repr(C)]
#[derive(Copy)]
pub struct Struct_Unnamed2 {
    pub _bindgen_bitfield_1_: ::libc::c_uint,
}
impl ::std::clone::Clone for Struct_Unnamed2 {
    fn clone(&self) -> Self { *self }
}
impl ::std::default::Default for Struct_Unnamed2 {
    fn default() -> Self { unsafe { ::std::mem::zeroed() } }
}
#[repr(C)]
#[derive(Copy)]
pub struct Struct_Unnamed3 {
    pub _bindgen_bitfield_1_: ::libc::c_uint,
}
impl ::std::clone::Clone for Struct_Unnamed3 {
    fn clone(&self) -> Self { *self }
}
impl ::std::default::Default for Struct_Unnamed3 {
    fn default() -> Self { unsafe { ::std::mem::zeroed() } }
}
#[repr(C)]
#[derive(Copy)]
pub struct Union_Unnamed4 {
    pub _bindgen_data_: [u64; 1usize],
}
impl Union_Unnamed4 {
    pub unsafe fn __uptr(&mut self) -> *mut *mut Union_wait {
        let raw: *mut u8 = ::std::mem::transmute(&self._bindgen_data_);
        ::std::mem::transmute(raw.offset(0))
    }
    pub unsafe fn __iptr(&mut self) -> *mut *mut ::libc::c_int {
        let raw: *mut u8 = ::std::mem::transmute(&self._bindgen_data_);
        ::std::mem::transmute(raw.offset(0))
    }
}
impl ::std::clone::Clone for Union_Unnamed4 {
    fn clone(&self) -> Self { *self }
}
impl ::std::default::Default for Union_Unnamed4 {
    fn default() -> Self { unsafe { ::std::mem::zeroed() } }
}
pub type __WAIT_STATUS = Union_Unnamed4;
#[repr(C)]
#[derive(Copy)]
pub struct Struct_Unnamed5 {
    pub quot: ::libc::c_int,
    pub rem: ::libc::c_int,
}
impl ::std::clone::Clone for Struct_Unnamed5 {
    fn clone(&self) -> Self { *self }
}
impl ::std::default::Default for Struct_Unnamed5 {
    fn default() -> Self { unsafe { ::std::mem::zeroed() } }
}
pub type div_t = Struct_Unnamed5;
#[repr(C)]
#[derive(Copy)]
pub struct Struct_Unnamed6 {
    pub quot: ::libc::c_long,
    pub rem: ::libc::c_long,
}
impl ::std::clone::Clone for Struct_Unnamed6 {
    fn clone(&self) -> Self { *self }
}
impl ::std::default::Default for Struct_Unnamed6 {
    fn default() -> Self { unsafe { ::std::mem::zeroed() } }
}
pub type ldiv_t = Struct_Unnamed6;
#[repr(C)]
#[derive(Copy)]
pub struct Struct_Unnamed7 {
    pub quot: ::libc::c_longlong,
    pub rem: ::libc::c_longlong,
}
impl ::std::clone::Clone for Struct_Unnamed7 {
    fn clone(&self) -> Self { *self }
}
impl ::std::default::Default for Struct_Unnamed7 {
    fn default() -> Self { unsafe { ::std::mem::zeroed() } }
}
pub type lldiv_t = Struct_Unnamed7;
pub type u_char = __u_char;
pub type u_short = __u_short;
pub type u_int = __u_int;
pub type u_long = __u_long;
pub type quad_t = __quad_t;
pub type u_quad_t = __u_quad_t;
pub type fsid_t = __fsid_t;
pub type loff_t = __loff_t;
pub type ino_t = __ino_t;
pub type dev_t = __dev_t;
pub type gid_t = __gid_t;
pub type mode_t = __mode_t;
pub type nlink_t = __nlink_t;
pub type uid_t = __uid_t;
pub type off_t = __off_t;
pub type pid_t = __pid_t;
pub type id_t = __id_t;
pub type ssize_t = __ssize_t;
pub type daddr_t = __daddr_t;
pub type caddr_t = __caddr_t;
pub type key_t = __key_t;
pub type clock_t = __clock_t;
pub type time_t = __time_t;
pub type clockid_t = __clockid_t;
pub type timer_t = __timer_t;
pub type ulong = ::libc::c_ulong;
pub type ushort = ::libc::c_ushort;
pub type _uint = ::libc::c_uint;
pub type int8_t = ::libc::c_char;
pub type int16_t = ::libc::c_short;
pub type int32_t = ::libc::c_int;
pub type int64_t = ::libc::c_long;
pub type u_int8_t = ::libc::c_uchar;
pub type u_int16_t = ::libc::c_ushort;
pub type u_int32_t = ::libc::c_uint;
pub type u_int64_t = ::libc::c_ulong;
pub type register_t = ::libc::c_long;
pub type __sig_atomic_t = ::libc::c_int;
#[repr(C)]
#[derive(Copy)]
pub struct Struct_Unnamed8 {
    pub __val: [::libc::c_ulong; 16usize],
}
impl ::std::clone::Clone for Struct_Unnamed8 {
    fn clone(&self) -> Self { *self }
}
impl ::std::default::Default for Struct_Unnamed8 {
    fn default() -> Self { unsafe { ::std::mem::zeroed() } }
}
pub type __sigset_t = Struct_Unnamed8;
pub type sigset_t = __sigset_t;
#[repr(C)]
#[derive(Copy)]
pub struct Struct_timespec {
    pub tv_sec: __time_t,
    pub tv_nsec: __syscall_slong_t,
}
impl ::std::clone::Clone for Struct_timespec {
    fn clone(&self) -> Self { *self }
}
impl ::std::default::Default for Struct_timespec {
    fn default() -> Self { unsafe { ::std::mem::zeroed() } }
}
#[repr(C)]
#[derive(Copy)]
pub struct Struct_timeval {
    pub tv_sec: __time_t,
    pub tv_usec: __suseconds_t,
}
impl ::std::clone::Clone for Struct_timeval {
    fn clone(&self) -> Self { *self }
}
impl ::std::default::Default for Struct_timeval {
    fn default() -> Self { unsafe { ::std::mem::zeroed() } }
}
pub type suseconds_t = __suseconds_t;
pub type __fd_mask = ::libc::c_long;
#[repr(C)]
#[derive(Copy)]
pub struct Struct_Unnamed9 {
    pub __fds_bits: [__fd_mask; 16usize],
}
impl ::std::clone::Clone for Struct_Unnamed9 {
    fn clone(&self) -> Self { *self }
}
impl ::std::default::Default for Struct_Unnamed9 {
    fn default() -> Self { unsafe { ::std::mem::zeroed() } }
}
pub type fd_set = Struct_Unnamed9;
pub type fd_mask = __fd_mask;
pub type blksize_t = __blksize_t;
pub type blkcnt_t = __blkcnt_t;
pub type fsblkcnt_t = __fsblkcnt_t;
pub type fsfilcnt_t = __fsfilcnt_t;
pub type pthread_t = ::libc::c_ulong;
#[repr(C)]
#[derive(Copy)]
pub struct Union_pthread_attr_t {
    pub _bindgen_data_: [u64; 7usize],
}
impl Union_pthread_attr_t {
    pub unsafe fn __size(&mut self) -> *mut [::libc::c_char; 56usize] {
        let raw: *mut u8 = ::std::mem::transmute(&self._bindgen_data_);
        ::std::mem::transmute(raw.offset(0))
    }
    pub unsafe fn __align(&mut self) -> *mut ::libc::c_long {
        let raw: *mut u8 = ::std::mem::transmute(&self._bindgen_data_);
        ::std::mem::transmute(raw.offset(0))
    }
}
impl ::std::clone::Clone for Union_pthread_attr_t {
    fn clone(&self) -> Self { *self }
}
impl ::std::default::Default for Union_pthread_attr_t {
    fn default() -> Self { unsafe { ::std::mem::zeroed() } }
}
pub type pthread_attr_t = Union_pthread_attr_t;
#[repr(C)]
#[derive(Copy)]
pub struct Struct___pthread_internal_list {
    pub __prev: *mut Struct___pthread_internal_list,
    pub __next: *mut Struct___pthread_internal_list,
}
impl ::std::clone::Clone for Struct___pthread_internal_list {
    fn clone(&self) -> Self { *self }
}
impl ::std::default::Default for Struct___pthread_internal_list {
    fn default() -> Self { unsafe { ::std::mem::zeroed() } }
}
pub type __pthread_list_t = Struct___pthread_internal_list;
#[repr(C)]
#[derive(Copy)]
pub struct Union_Unnamed10 {
    pub _bindgen_data_: [u64; 5usize],
}
impl Union_Unnamed10 {
    pub unsafe fn __data(&mut self) -> *mut Struct___pthread_mutex_s {
        let raw: *mut u8 = ::std::mem::transmute(&self._bindgen_data_);
        ::std::mem::transmute(raw.offset(0))
    }
    pub unsafe fn __size(&mut self) -> *mut [::libc::c_char; 40usize] {
        let raw: *mut u8 = ::std::mem::transmute(&self._bindgen_data_);
        ::std::mem::transmute(raw.offset(0))
    }
    pub unsafe fn __align(&mut self) -> *mut ::libc::c_long {
        let raw: *mut u8 = ::std::mem::transmute(&self._bindgen_data_);
        ::std::mem::transmute(raw.offset(0))
    }
}
impl ::std::clone::Clone for Union_Unnamed10 {
    fn clone(&self) -> Self { *self }
}
impl ::std::default::Default for Union_Unnamed10 {
    fn default() -> Self { unsafe { ::std::mem::zeroed() } }
}
#[repr(C)]
#[derive(Copy)]
pub struct Struct___pthread_mutex_s {
    pub __lock: ::libc::c_int,
    pub __count: ::libc::c_uint,
    pub __owner: ::libc::c_int,
    pub __nusers: ::libc::c_uint,
    pub __kind: ::libc::c_int,
    pub __spins: ::libc::c_short,
    pub __elision: ::libc::c_short,
    pub __list: __pthread_list_t,
}
impl ::std::clone::Clone for Struct___pthread_mutex_s {
    fn clone(&self) -> Self { *self }
}
impl ::std::default::Default for Struct___pthread_mutex_s {
    fn default() -> Self { unsafe { ::std::mem::zeroed() } }
}
pub type pthread_mutex_t = Union_Unnamed10;
#[repr(C)]
#[derive(Copy)]
pub struct Union_Unnamed11 {
    pub _bindgen_data_: [u32; 1usize],
}
impl Union_Unnamed11 {
    pub unsafe fn __size(&mut self) -> *mut [::libc::c_char; 4usize] {
        let raw: *mut u8 = ::std::mem::transmute(&self._bindgen_data_);
        ::std::mem::transmute(raw.offset(0))
    }
    pub unsafe fn __align(&mut self) -> *mut ::libc::c_int {
        let raw: *mut u8 = ::std::mem::transmute(&self._bindgen_data_);
        ::std::mem::transmute(raw.offset(0))
    }
}
impl ::std::clone::Clone for Union_Unnamed11 {
    fn clone(&self) -> Self { *self }
}
impl ::std::default::Default for Union_Unnamed11 {
    fn default() -> Self { unsafe { ::std::mem::zeroed() } }
}
pub type pthread_mutexattr_t = Union_Unnamed11;
#[repr(C)]
#[derive(Copy)]
pub struct Union_Unnamed12 {
    pub _bindgen_data_: [u64; 6usize],
}
impl Union_Unnamed12 {
    pub unsafe fn __data(&mut self) -> *mut Struct_Unnamed13 {
        let raw: *mut u8 = ::std::mem::transmute(&self._bindgen_data_);
        ::std::mem::transmute(raw.offset(0))
    }
    pub unsafe fn __size(&mut self) -> *mut [::libc::c_char; 48usize] {
        let raw: *mut u8 = ::std::mem::transmute(&self._bindgen_data_);
        ::std::mem::transmute(raw.offset(0))
    }
    pub unsafe fn __align(&mut self) -> *mut ::libc::c_longlong {
        let raw: *mut u8 = ::std::mem::transmute(&self._bindgen_data_);
        ::std::mem::transmute(raw.offset(0))
    }
}
impl ::std::clone::Clone for Union_Unnamed12 {
    fn clone(&self) -> Self { *self }
}
impl ::std::default::Default for Union_Unnamed12 {
    fn default() -> Self { unsafe { ::std::mem::zeroed() } }
}
#[repr(C)]
#[derive(Copy)]
pub struct Struct_Unnamed13 {
    pub __lock: ::libc::c_int,
    pub __futex: ::libc::c_uint,
    pub __total_seq: ::libc::c_ulonglong,
    pub __wakeup_seq: ::libc::c_ulonglong,
    pub __woken_seq: ::libc::c_ulonglong,
    pub __mutex: *mut ::libc::c_void,
    pub __nwaiters: ::libc::c_uint,
    pub __broadcast_seq: ::libc::c_uint,
}
impl ::std::clone::Clone for Struct_Unnamed13 {
    fn clone(&self) -> Self { *self }
}
impl ::std::default::Default for Struct_Unnamed13 {
    fn default() -> Self { unsafe { ::std::mem::zeroed() } }
}
pub type pthread_cond_t = Union_Unnamed12;
#[repr(C)]
#[derive(Copy)]
pub struct Union_Unnamed14 {
    pub _bindgen_data_: [u32; 1usize],
}
impl Union_Unnamed14 {
    pub unsafe fn __size(&mut self) -> *mut [::libc::c_char; 4usize] {
        let raw: *mut u8 = ::std::mem::transmute(&self._bindgen_data_);
        ::std::mem::transmute(raw.offset(0))
    }
    pub unsafe fn __align(&mut self) -> *mut ::libc::c_int {
        let raw: *mut u8 = ::std::mem::transmute(&self._bindgen_data_);
        ::std::mem::transmute(raw.offset(0))
    }
}
impl ::std::clone::Clone for Union_Unnamed14 {
    fn clone(&self) -> Self { *self }
}
impl ::std::default::Default for Union_Unnamed14 {
    fn default() -> Self { unsafe { ::std::mem::zeroed() } }
}
pub type pthread_condattr_t = Union_Unnamed14;
pub type pthread_key_t = ::libc::c_uint;
pub type pthread_once_t = ::libc::c_int;
#[repr(C)]
#[derive(Copy)]
pub struct Union_Unnamed15 {
    pub _bindgen_data_: [u64; 7usize],
}
impl Union_Unnamed15 {
    pub unsafe fn __data(&mut self) -> *mut Struct_Unnamed16 {
        let raw: *mut u8 = ::std::mem::transmute(&self._bindgen_data_);
        ::std::mem::transmute(raw.offset(0))
    }
    pub unsafe fn __size(&mut self) -> *mut [::libc::c_char; 56usize] {
        let raw: *mut u8 = ::std::mem::transmute(&self._bindgen_data_);
        ::std::mem::transmute(raw.offset(0))
    }
    pub unsafe fn __align(&mut self) -> *mut ::libc::c_long {
        let raw: *mut u8 = ::std::mem::transmute(&self._bindgen_data_);
        ::std::mem::transmute(raw.offset(0))
    }
}
impl ::std::clone::Clone for Union_Unnamed15 {
    fn clone(&self) -> Self { *self }
}
impl ::std::default::Default for Union_Unnamed15 {
    fn default() -> Self { unsafe { ::std::mem::zeroed() } }
}
#[repr(C)]
#[derive(Copy)]
pub struct Struct_Unnamed16 {
    pub __lock: ::libc::c_int,
    pub __nr_readers: ::libc::c_uint,
    pub __readers_wakeup: ::libc::c_uint,
    pub __writer_wakeup: ::libc::c_uint,
    pub __nr_readers_queued: ::libc::c_uint,
    pub __nr_writers_queued: ::libc::c_uint,
    pub __writer: ::libc::c_int,
    pub __shared: ::libc::c_int,
    pub __rwelision: ::libc::c_char,
    pub __pad1: [::libc::c_uchar; 7usize],
    pub __pad2: ::libc::c_ulong,
    pub __flags: ::libc::c_uint,
}
impl ::std::clone::Clone for Struct_Unnamed16 {
    fn clone(&self) -> Self { *self }
}
impl ::std::default::Default for Struct_Unnamed16 {
    fn default() -> Self { unsafe { ::std::mem::zeroed() } }
}
pub type pthread_rwlock_t = Union_Unnamed15;
#[repr(C)]
#[derive(Copy)]
pub struct Union_Unnamed17 {
    pub _bindgen_data_: [u64; 1usize],
}
impl Union_Unnamed17 {
    pub unsafe fn __size(&mut self) -> *mut [::libc::c_char; 8usize] {
        let raw: *mut u8 = ::std::mem::transmute(&self._bindgen_data_);
        ::std::mem::transmute(raw.offset(0))
    }
    pub unsafe fn __align(&mut self) -> *mut ::libc::c_long {
        let raw: *mut u8 = ::std::mem::transmute(&self._bindgen_data_);
        ::std::mem::transmute(raw.offset(0))
    }
}
impl ::std::clone::Clone for Union_Unnamed17 {
    fn clone(&self) -> Self { *self }
}
impl ::std::default::Default for Union_Unnamed17 {
    fn default() -> Self { unsafe { ::std::mem::zeroed() } }
}
pub type pthread_rwlockattr_t = Union_Unnamed17;
pub type pthread_spinlock_t = ::libc::c_int;
#[repr(C)]
#[derive(Copy)]
pub struct Union_Unnamed18 {
    pub _bindgen_data_: [u64; 4usize],
}
impl Union_Unnamed18 {
    pub unsafe fn __size(&mut self) -> *mut [::libc::c_char; 32usize] {
        let raw: *mut u8 = ::std::mem::transmute(&self._bindgen_data_);
        ::std::mem::transmute(raw.offset(0))
    }
    pub unsafe fn __align(&mut self) -> *mut ::libc::c_long {
        let raw: *mut u8 = ::std::mem::transmute(&self._bindgen_data_);
        ::std::mem::transmute(raw.offset(0))
    }
}
impl ::std::clone::Clone for Union_Unnamed18 {
    fn clone(&self) -> Self { *self }
}
impl ::std::default::Default for Union_Unnamed18 {
    fn default() -> Self { unsafe { ::std::mem::zeroed() } }
}
pub type pthread_barrier_t = Union_Unnamed18;
#[repr(C)]
#[derive(Copy)]
pub struct Union_Unnamed19 {
    pub _bindgen_data_: [u32; 1usize],
}
impl Union_Unnamed19 {
    pub unsafe fn __size(&mut self) -> *mut [::libc::c_char; 4usize] {
        let raw: *mut u8 = ::std::mem::transmute(&self._bindgen_data_);
        ::std::mem::transmute(raw.offset(0))
    }
    pub unsafe fn __align(&mut self) -> *mut ::libc::c_int {
        let raw: *mut u8 = ::std::mem::transmute(&self._bindgen_data_);
        ::std::mem::transmute(raw.offset(0))
    }
}
impl ::std::clone::Clone for Union_Unnamed19 {
    fn clone(&self) -> Self { *self }
}
impl ::std::default::Default for Union_Unnamed19 {
    fn default() -> Self { unsafe { ::std::mem::zeroed() } }
}
pub type pthread_barrierattr_t = Union_Unnamed19;
#[repr(C)]
#[derive(Copy)]
pub struct Struct_random_data {
    pub fptr: *mut int32_t,
    pub rptr: *mut int32_t,
    pub state: *mut int32_t,
    pub rand_type: ::libc::c_int,
    pub rand_deg: ::libc::c_int,
    pub rand_sep: ::libc::c_int,
    pub end_ptr: *mut int32_t,
}
impl ::std::clone::Clone for Struct_random_data {
    fn clone(&self) -> Self { *self }
}
impl ::std::default::Default for Struct_random_data {
    fn default() -> Self { unsafe { ::std::mem::zeroed() } }
}
#[repr(C)]
#[derive(Copy)]
pub struct Struct_drand48_data {
    pub __x: [::libc::c_ushort; 3usize],
    pub __old_x: [::libc::c_ushort; 3usize],
    pub __c: ::libc::c_ushort,
    pub __init: ::libc::c_ushort,
    pub __a: ::libc::c_ulonglong,
}
impl ::std::clone::Clone for Struct_drand48_data {
    fn clone(&self) -> Self { *self }
}
impl ::std::default::Default for Struct_drand48_data {
    fn default() -> Self { unsafe { ::std::mem::zeroed() } }
}
pub type __compar_fn_t =
    ::std::option::Option<unsafe extern "C" fn(arg1: *const ::libc::c_void,
                                               arg2: *const ::libc::c_void)
                              -> ::libc::c_int>;
pub type CUdeviceptr = ::libc::c_ulonglong;
pub type CUdevice = ::libc::c_int;
pub enum Struct_CUctx_st { }
pub type CUcontext = *mut Struct_CUctx_st;
pub enum Struct_CUmod_st { }
pub type CUmodule = *mut Struct_CUmod_st;
pub enum Struct_CUfunc_st { }
pub type CUfunction = *mut Struct_CUfunc_st;
pub enum Struct_CUarray_st { }
pub type CUarray = *mut Struct_CUarray_st;
pub enum Struct_CUmipmappedArray_st { }
pub type CUmipmappedArray = *mut Struct_CUmipmappedArray_st;
pub enum Struct_CUtexref_st { }
pub type CUtexref = *mut Struct_CUtexref_st;
pub enum Struct_CUsurfref_st { }
pub type CUsurfref = *mut Struct_CUsurfref_st;
pub enum Struct_CUevent_st { }
pub type CUevent = *mut Struct_CUevent_st;
pub enum Struct_CUstream_st { }
pub type CUstream = *mut Struct_CUstream_st;
pub enum Struct_CUgraphicsResource_st { }
pub type CUgraphicsResource = *mut Struct_CUgraphicsResource_st;
pub type CUtexObject = ::libc::c_ulonglong;
pub type CUsurfObject = ::libc::c_ulonglong;
#[repr(C)]
#[derive(Copy)]
pub struct Struct_CUuuid_st {
    pub bytes: [::libc::c_char; 16usize],
}
impl ::std::clone::Clone for Struct_CUuuid_st {
    fn clone(&self) -> Self { *self }
}
impl ::std::default::Default for Struct_CUuuid_st {
    fn default() -> Self { unsafe { ::std::mem::zeroed() } }
}
pub type CUuuid = Struct_CUuuid_st;
#[repr(C)]
#[derive(Copy)]
pub struct Struct_CUipcEventHandle_st {
    pub reserved: [::libc::c_char; 64usize],
}
impl ::std::clone::Clone for Struct_CUipcEventHandle_st {
    fn clone(&self) -> Self { *self }
}
impl ::std::default::Default for Struct_CUipcEventHandle_st {
    fn default() -> Self { unsafe { ::std::mem::zeroed() } }
}
pub type CUipcEventHandle = Struct_CUipcEventHandle_st;
#[repr(C)]
#[derive(Copy)]
pub struct Struct_CUipcMemHandle_st {
    pub reserved: [::libc::c_char; 64usize],
}
impl ::std::clone::Clone for Struct_CUipcMemHandle_st {
    fn clone(&self) -> Self { *self }
}
impl ::std::default::Default for Struct_CUipcMemHandle_st {
    fn default() -> Self { unsafe { ::std::mem::zeroed() } }
}
pub type CUipcMemHandle = Struct_CUipcMemHandle_st;
pub type Enum_CUipcMem_flags_enum = ::libc::c_uint;
pub const CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS: ::libc::c_uint = 1;
pub type CUipcMem_flags = Enum_CUipcMem_flags_enum;
pub type Enum_CUmemAttach_flags_enum = ::libc::c_uint;
pub const CU_MEM_ATTACH_GLOBAL: ::libc::c_uint = 1;
pub const CU_MEM_ATTACH_HOST: ::libc::c_uint = 2;
pub const CU_MEM_ATTACH_SINGLE: ::libc::c_uint = 4;
pub type CUmemAttach_flags = Enum_CUmemAttach_flags_enum;
pub type Enum_CUctx_flags_enum = ::libc::c_uint;
pub const CU_CTX_SCHED_AUTO: ::libc::c_uint = 0;
pub const CU_CTX_SCHED_SPIN: ::libc::c_uint = 1;
pub const CU_CTX_SCHED_YIELD: ::libc::c_uint = 2;
pub const CU_CTX_SCHED_BLOCKING_SYNC: ::libc::c_uint = 4;
pub const CU_CTX_BLOCKING_SYNC: ::libc::c_uint = 4;
pub const CU_CTX_SCHED_MASK: ::libc::c_uint = 7;
pub const CU_CTX_MAP_HOST: ::libc::c_uint = 8;
pub const CU_CTX_LMEM_RESIZE_TO_MAX: ::libc::c_uint = 16;
pub const CU_CTX_FLAGS_MASK: ::libc::c_uint = 31;
pub type CUctx_flags = Enum_CUctx_flags_enum;
pub type Enum_CUstream_flags_enum = ::libc::c_uint;
pub const CU_STREAM_DEFAULT: ::libc::c_uint = 0;
pub const CU_STREAM_NON_BLOCKING: ::libc::c_uint = 1;
pub type CUstream_flags = Enum_CUstream_flags_enum;
pub type Enum_CUevent_flags_enum = ::libc::c_uint;
pub const CU_EVENT_DEFAULT: ::libc::c_uint = 0;
pub const CU_EVENT_BLOCKING_SYNC: ::libc::c_uint = 1;
pub const CU_EVENT_DISABLE_TIMING: ::libc::c_uint = 2;
pub const CU_EVENT_INTERPROCESS: ::libc::c_uint = 4;
pub type CUevent_flags = Enum_CUevent_flags_enum;
pub type Enum_CUoccupancy_flags_enum = ::libc::c_uint;
pub const CU_OCCUPANCY_DEFAULT: ::libc::c_uint = 0;
pub const CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE: ::libc::c_uint = 1;
pub type CUoccupancy_flags = Enum_CUoccupancy_flags_enum;
pub type Enum_CUarray_format_enum = ::libc::c_uint;
pub const CU_AD_FORMAT_UNSIGNED_INT8: ::libc::c_uint = 1;
pub const CU_AD_FORMAT_UNSIGNED_INT16: ::libc::c_uint = 2;
pub const CU_AD_FORMAT_UNSIGNED_INT32: ::libc::c_uint = 3;
pub const CU_AD_FORMAT_SIGNED_INT8: ::libc::c_uint = 8;
pub const CU_AD_FORMAT_SIGNED_INT16: ::libc::c_uint = 9;
pub const CU_AD_FORMAT_SIGNED_INT32: ::libc::c_uint = 10;
pub const CU_AD_FORMAT_HALF: ::libc::c_uint = 16;
pub const CU_AD_FORMAT_FLOAT: ::libc::c_uint = 32;
pub type CUarray_format = Enum_CUarray_format_enum;
pub type Enum_CUaddress_mode_enum = ::libc::c_uint;
pub const CU_TR_ADDRESS_MODE_WRAP: ::libc::c_uint = 0;
pub const CU_TR_ADDRESS_MODE_CLAMP: ::libc::c_uint = 1;
pub const CU_TR_ADDRESS_MODE_MIRROR: ::libc::c_uint = 2;
pub const CU_TR_ADDRESS_MODE_BORDER: ::libc::c_uint = 3;
pub type CUaddress_mode = Enum_CUaddress_mode_enum;
pub type Enum_CUfilter_mode_enum = ::libc::c_uint;
pub const CU_TR_FILTER_MODE_POINT: ::libc::c_uint = 0;
pub const CU_TR_FILTER_MODE_LINEAR: ::libc::c_uint = 1;
pub type CUfilter_mode = Enum_CUfilter_mode_enum;
#[derive(PartialEq, Debug)]
#[repr(C)]
pub enum CUdevice_attribute {
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 3,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 4,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = 6,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = 7,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8,
    CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY = 9,
    CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10,
    CU_DEVICE_ATTRIBUTE_MAX_PITCH = 11,
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 12,
    CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13,
    CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT = 14,
    CU_DEVICE_ATTRIBUTE_GPU_OVERLAP = 15,
    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16,
    CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT = 17,
    CU_DEVICE_ATTRIBUTE_INTEGRATED = 18,
    CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY = 19,
    CU_DEVICE_ATTRIBUTE_COMPUTE_MODE = 20,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH = 21,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH = 22,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT = 23,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH = 24,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT = 25,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH = 26,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH = 27,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT = 28,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS = 29,
    CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT = 30,
    CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS = 31,
    CU_DEVICE_ATTRIBUTE_ECC_ENABLED = 32,
    CU_DEVICE_ATTRIBUTE_PCI_BUS_ID = 33,
    CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = 34,
    CU_DEVICE_ATTRIBUTE_TCC_DRIVER = 35,
    CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36,
    CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = 37,
    CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 38,
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39,
    CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT = 40,
    CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING = 41,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH = 42,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS = 43,
    CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER = 44,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH = 45,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT = 46,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE = 47,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE = 48,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE = 49,
    CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID = 50,
    CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT = 51,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH = 52,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH = 53,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS = 54,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH = 55,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH = 56,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT = 57,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH = 58,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT = 59,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH = 60,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH = 61,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS = 62,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH = 63,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT = 64,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS = 65,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH = 66,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH = 67,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS = 68,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH = 69,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH = 70,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT = 71,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH = 72,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH = 73,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT = 74,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH = 77,
    CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED = 78,
    CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED = 79,
    CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED = 80,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 81,
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR = 82,
    CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY = 83,
    CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD = 84,
    CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID = 85,
    CU_DEVICE_ATTRIBUTE_MAX = 86,
    // Not really in the Cuda specifaction. Added for load_device_info convenience.
    CU_DEVICE_NAME = 999,
    CU_DEVICE_MEMORY_TOTAL = 1000,
}

#[repr(C)]
#[derive(Copy)]
pub struct Struct_CUdevprop_st {
    pub maxThreadsPerBlock: ::libc::c_int,
    pub maxThreadsDim: [::libc::c_int; 3usize],
    pub maxGridSize: [::libc::c_int; 3usize],
    pub sharedMemPerBlock: ::libc::c_int,
    pub totalConstantMemory: ::libc::c_int,
    pub SIMDWidth: ::libc::c_int,
    pub memPitch: ::libc::c_int,
    pub regsPerBlock: ::libc::c_int,
    pub clockRate: ::libc::c_int,
    pub textureAlign: ::libc::c_int,
}
impl ::std::clone::Clone for Struct_CUdevprop_st {
    fn clone(&self) -> Self { *self }
}
impl ::std::default::Default for Struct_CUdevprop_st {
    fn default() -> Self { unsafe { ::std::mem::zeroed() } }
}
pub type CUdevprop = Struct_CUdevprop_st;
pub type Enum_CUpointer_attribute_enum = ::libc::c_uint;
pub const CU_POINTER_ATTRIBUTE_CONTEXT: ::libc::c_uint = 1;
pub const CU_POINTER_ATTRIBUTE_MEMORY_TYPE: ::libc::c_uint = 2;
pub const CU_POINTER_ATTRIBUTE_DEVICE_POINTER: ::libc::c_uint = 3;
pub const CU_POINTER_ATTRIBUTE_HOST_POINTER: ::libc::c_uint = 4;
pub const CU_POINTER_ATTRIBUTE_P2P_TOKENS: ::libc::c_uint = 5;
pub const CU_POINTER_ATTRIBUTE_SYNC_MEMOPS: ::libc::c_uint = 6;
pub const CU_POINTER_ATTRIBUTE_BUFFER_ID: ::libc::c_uint = 7;
pub const CU_POINTER_ATTRIBUTE_IS_MANAGED: ::libc::c_uint = 8;
pub type CUpointer_attribute = Enum_CUpointer_attribute_enum;
pub type Enum_CUfunction_attribute_enum = ::libc::c_uint;
pub const CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK: ::libc::c_uint = 0;
pub const CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES: ::libc::c_uint = 1;
pub const CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES: ::libc::c_uint = 2;
pub const CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES: ::libc::c_uint = 3;
pub const CU_FUNC_ATTRIBUTE_NUM_REGS: ::libc::c_uint = 4;
pub const CU_FUNC_ATTRIBUTE_PTX_VERSION: ::libc::c_uint = 5;
pub const CU_FUNC_ATTRIBUTE_BINARY_VERSION: ::libc::c_uint = 6;
pub const CU_FUNC_ATTRIBUTE_CACHE_MODE_CA: ::libc::c_uint = 7;
pub const CU_FUNC_ATTRIBUTE_MAX: ::libc::c_uint = 8;
pub type CUfunction_attribute = Enum_CUfunction_attribute_enum;
pub type Enum_CUfunc_cache_enum = ::libc::c_uint;
pub const CU_FUNC_CACHE_PREFER_NONE: ::libc::c_uint = 0;
pub const CU_FUNC_CACHE_PREFER_SHARED: ::libc::c_uint = 1;
pub const CU_FUNC_CACHE_PREFER_L1: ::libc::c_uint = 2;
pub const CU_FUNC_CACHE_PREFER_EQUAL: ::libc::c_uint = 3;
pub type CUfunc_cache = Enum_CUfunc_cache_enum;
pub type Enum_CUsharedconfig_enum = ::libc::c_uint;
pub const CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE: ::libc::c_uint = 0;
pub const CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE: ::libc::c_uint = 1;
pub const CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE: ::libc::c_uint = 2;
pub type CUsharedconfig = Enum_CUsharedconfig_enum;
pub type Enum_CUmemorytype_enum = ::libc::c_uint;
pub const CU_MEMORYTYPE_HOST: ::libc::c_uint = 1;
pub const CU_MEMORYTYPE_DEVICE: ::libc::c_uint = 2;
pub const CU_MEMORYTYPE_ARRAY: ::libc::c_uint = 3;
pub const CU_MEMORYTYPE_UNIFIED: ::libc::c_uint = 4;
pub type CUmemorytype = Enum_CUmemorytype_enum;
pub type Enum_CUcomputemode_enum = ::libc::c_uint;
pub const CU_COMPUTEMODE_DEFAULT: ::libc::c_uint = 0;
pub const CU_COMPUTEMODE_EXCLUSIVE: ::libc::c_uint = 1;
pub const CU_COMPUTEMODE_PROHIBITED: ::libc::c_uint = 2;
pub const CU_COMPUTEMODE_EXCLUSIVE_PROCESS: ::libc::c_uint = 3;
pub type CUcomputemode = Enum_CUcomputemode_enum;
pub type Enum_CUjit_option_enum = ::libc::c_uint;
pub const CU_JIT_MAX_REGISTERS: ::libc::c_uint = 0;
pub const CU_JIT_THREADS_PER_BLOCK: ::libc::c_uint = 1;
pub const CU_JIT_WALL_TIME: ::libc::c_uint = 2;
pub const CU_JIT_INFO_LOG_BUFFER: ::libc::c_uint = 3;
pub const CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES: ::libc::c_uint = 4;
pub const CU_JIT_ERROR_LOG_BUFFER: ::libc::c_uint = 5;
pub const CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES: ::libc::c_uint = 6;
pub const CU_JIT_OPTIMIZATION_LEVEL: ::libc::c_uint = 7;
pub const CU_JIT_TARGET_FROM_CUCONTEXT: ::libc::c_uint = 8;
pub const CU_JIT_TARGET: ::libc::c_uint = 9;
pub const CU_JIT_FALLBACK_STRATEGY: ::libc::c_uint = 10;
pub const CU_JIT_GENERATE_DEBUG_INFO: ::libc::c_uint = 11;
pub const CU_JIT_LOG_VERBOSE: ::libc::c_uint = 12;
pub const CU_JIT_GENERATE_LINE_INFO: ::libc::c_uint = 13;
pub const CU_JIT_CACHE_MODE: ::libc::c_uint = 14;
pub const CU_JIT_NUM_OPTIONS: ::libc::c_uint = 15;
pub type CUjit_option = Enum_CUjit_option_enum;
pub type Enum_CUjit_target_enum = ::libc::c_uint;
pub const CU_TARGET_COMPUTE_10: ::libc::c_uint = 10;
pub const CU_TARGET_COMPUTE_11: ::libc::c_uint = 11;
pub const CU_TARGET_COMPUTE_12: ::libc::c_uint = 12;
pub const CU_TARGET_COMPUTE_13: ::libc::c_uint = 13;
pub const CU_TARGET_COMPUTE_20: ::libc::c_uint = 20;
pub const CU_TARGET_COMPUTE_21: ::libc::c_uint = 21;
pub const CU_TARGET_COMPUTE_30: ::libc::c_uint = 30;
pub const CU_TARGET_COMPUTE_32: ::libc::c_uint = 32;
pub const CU_TARGET_COMPUTE_35: ::libc::c_uint = 35;
pub const CU_TARGET_COMPUTE_37: ::libc::c_uint = 37;
pub const CU_TARGET_COMPUTE_50: ::libc::c_uint = 50;
pub const CU_TARGET_COMPUTE_52: ::libc::c_uint = 52;
pub type CUjit_target = Enum_CUjit_target_enum;
pub type Enum_CUjit_fallback_enum = ::libc::c_uint;
pub const CU_PREFER_PTX: ::libc::c_uint = 0;
pub const CU_PREFER_BINARY: ::libc::c_uint = 1;
pub type CUjit_fallback = Enum_CUjit_fallback_enum;
pub type Enum_CUjit_cacheMode_enum = ::libc::c_uint;
pub const CU_JIT_CACHE_OPTION_NONE: ::libc::c_uint = 0;
pub const CU_JIT_CACHE_OPTION_CG: ::libc::c_uint = 1;
pub const CU_JIT_CACHE_OPTION_CA: ::libc::c_uint = 2;
pub type CUjit_cacheMode = Enum_CUjit_cacheMode_enum;
pub type Enum_CUjitInputType_enum = ::libc::c_uint;
pub const CU_JIT_INPUT_CUBIN: ::libc::c_uint = 0;
pub const CU_JIT_INPUT_PTX: ::libc::c_uint = 1;
pub const CU_JIT_INPUT_FATBINARY: ::libc::c_uint = 2;
pub const CU_JIT_INPUT_OBJECT: ::libc::c_uint = 3;
pub const CU_JIT_INPUT_LIBRARY: ::libc::c_uint = 4;
pub const CU_JIT_NUM_INPUT_TYPES: ::libc::c_uint = 5;
pub type CUjitInputType = Enum_CUjitInputType_enum;
pub enum Struct_CUlinkState_st { }
pub type CUlinkState = *mut Struct_CUlinkState_st;
pub type Enum_CUgraphicsRegisterFlags_enum = ::libc::c_uint;
pub const CU_GRAPHICS_REGISTER_FLAGS_NONE: ::libc::c_uint = 0;
pub const CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY: ::libc::c_uint = 1;
pub const CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD: ::libc::c_uint = 2;
pub const CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST: ::libc::c_uint = 4;
pub const CU_GRAPHICS_REGISTER_FLAGS_TEXTURE_GATHER: ::libc::c_uint = 8;
pub type CUgraphicsRegisterFlags = Enum_CUgraphicsRegisterFlags_enum;
pub type Enum_CUgraphicsMapResourceFlags_enum = ::libc::c_uint;
pub const CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE: ::libc::c_uint = 0;
pub const CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY: ::libc::c_uint = 1;
pub const CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD: ::libc::c_uint = 2;
pub type CUgraphicsMapResourceFlags = Enum_CUgraphicsMapResourceFlags_enum;
pub type Enum_CUarray_cubemap_face_enum = ::libc::c_uint;
pub const CU_CUBEMAP_FACE_POSITIVE_X: ::libc::c_uint = 0;
pub const CU_CUBEMAP_FACE_NEGATIVE_X: ::libc::c_uint = 1;
pub const CU_CUBEMAP_FACE_POSITIVE_Y: ::libc::c_uint = 2;
pub const CU_CUBEMAP_FACE_NEGATIVE_Y: ::libc::c_uint = 3;
pub const CU_CUBEMAP_FACE_POSITIVE_Z: ::libc::c_uint = 4;
pub const CU_CUBEMAP_FACE_NEGATIVE_Z: ::libc::c_uint = 5;
pub type CUarray_cubemap_face = Enum_CUarray_cubemap_face_enum;
pub type Enum_CUlimit_enum = ::libc::c_uint;
pub const CU_LIMIT_STACK_SIZE: ::libc::c_uint = 0;
pub const CU_LIMIT_PRINTF_FIFO_SIZE: ::libc::c_uint = 1;
pub const CU_LIMIT_MALLOC_HEAP_SIZE: ::libc::c_uint = 2;
pub const CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH: ::libc::c_uint = 3;
pub const CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT: ::libc::c_uint = 4;
pub const CU_LIMIT_MAX: ::libc::c_uint = 5;
pub type CUlimit = Enum_CUlimit_enum;
pub type Enum_CUresourcetype_enum = ::libc::c_uint;
pub const CU_RESOURCE_TYPE_ARRAY: ::libc::c_uint = 0;
pub const CU_RESOURCE_TYPE_MIPMAPPED_ARRAY: ::libc::c_uint = 1;
pub const CU_RESOURCE_TYPE_LINEAR: ::libc::c_uint = 2;
pub const CU_RESOURCE_TYPE_PITCH2D: ::libc::c_uint = 3;
pub type CUresourcetype = Enum_CUresourcetype_enum;
#[derive(PartialEq, Debug)]
#[repr(C)]
pub enum CUresult {
    CUDA_SUCCESS = 0,
    CUDA_ERROR_INVALID_VALUE = 1,
    CUDA_ERROR_OUT_OF_MEMORY = 2,
    CUDA_ERROR_NOT_INITIALIZED = 3,
    CUDA_ERROR_DEINITIALIZED = 4,
    CUDA_ERROR_PROFILER_DISABLED = 5,
    CUDA_ERROR_PROFILER_NOT_INITIALIZED = 6,
    CUDA_ERROR_PROFILER_ALREADY_STARTED = 7,
    CUDA_ERROR_PROFILER_ALREADY_STOPPED = 8,
    CUDA_ERROR_NO_DEVICE = 100,
    CUDA_ERROR_INVALID_DEVICE = 101,
    CUDA_ERROR_INVALID_IMAGE = 200,
    CUDA_ERROR_INVALID_CONTEXT = 201,
    CUDA_ERROR_CONTEXT_ALREADY_CURRENT = 202,
    CUDA_ERROR_MAP_FAILED = 205,
    CUDA_ERROR_UNMAP_FAILED = 206,
    CUDA_ERROR_ARRAY_IS_MAPPED = 207,
    CUDA_ERROR_ALREADY_MAPPED = 208,
    CUDA_ERROR_NO_BINARY_FOR_GPU = 209,
    CUDA_ERROR_ALREADY_ACQUIRED = 210,
    CUDA_ERROR_NOT_MAPPED = 211,
    CUDA_ERROR_NOT_MAPPED_AS_ARRAY = 212,
    CUDA_ERROR_NOT_MAPPED_AS_POINTER = 213,
    CUDA_ERROR_ECC_UNCORRECTABLE = 214,
    CUDA_ERROR_UNSUPPORTED_LIMIT = 215,
    CUDA_ERROR_CONTEXT_ALREADY_IN_USE = 216,
    CUDA_ERROR_PEER_ACCESS_UNSUPPORTED = 217,
    CUDA_ERROR_INVALID_PTX = 218,
    CUDA_ERROR_INVALID_GRAPHICS_CONTEXT = 219,
    CUDA_ERROR_INVALID_SOURCE = 300,
    CUDA_ERROR_FILE_NOT_FOUND = 301,
    CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = 302,
    CUDA_ERROR_SHARED_OBJECT_INIT_FAILED = 303,
    CUDA_ERROR_OPERATING_SYSTEM = 304,
    CUDA_ERROR_INVALID_HANDLE = 400,
    CUDA_ERROR_NOT_FOUND = 500,
    CUDA_ERROR_NOT_READY = 600,
    CUDA_ERROR_ILLEGAL_ADDRESS = 700,
    CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES = 701,
    CUDA_ERROR_LAUNCH_TIMEOUT = 702,
    CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING = 703,
    CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED = 704,
    CUDA_ERROR_PEER_ACCESS_NOT_ENABLED = 705,
    CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE = 708,
    CUDA_ERROR_CONTEXT_IS_DESTROYED = 709,
    CUDA_ERROR_ASSERT = 710,
    CUDA_ERROR_TOO_MANY_PEERS = 711,
    CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = 712,
    CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED = 713,
    CUDA_ERROR_HARDWARE_STACK_ERROR = 714,
    CUDA_ERROR_ILLEGAL_INSTRUCTION = 715,
    CUDA_ERROR_MISALIGNED_ADDRESS = 716,
    CUDA_ERROR_INVALID_ADDRESS_SPACE = 717,
    CUDA_ERROR_INVALID_PC = 718,
    CUDA_ERROR_LAUNCH_FAILED = 719,
    CUDA_ERROR_NOT_PERMITTED = 800,
    CUDA_ERROR_NOT_SUPPORTED = 801,
    CUDA_ERROR_UNKNOWN = 999,
}
pub type CUstreamCallback =
    ::std::option::Option<unsafe extern "C" fn(hStream: CUstream,
                                               status: CUresult,
                                               userData: *mut ::libc::c_void)
                              -> ()>;
pub type CUoccupancyB2DSize =
    ::std::option::Option<extern "C" fn(blockSize: ::libc::c_int) -> size_t>;
#[repr(C)]
#[derive(Copy)]
pub struct Struct_CUDA_MEMCPY2D_st {
    pub srcXInBytes: size_t,
    pub srcY: size_t,
    pub srcMemoryType: CUmemorytype,
    pub srcHost: *const ::libc::c_void,
    pub srcDevice: CUdeviceptr,
    pub srcArray: CUarray,
    pub srcPitch: size_t,
    pub dstXInBytes: size_t,
    pub dstY: size_t,
    pub dstMemoryType: CUmemorytype,
    pub dstHost: *mut ::libc::c_void,
    pub dstDevice: CUdeviceptr,
    pub dstArray: CUarray,
    pub dstPitch: size_t,
    pub WidthInBytes: size_t,
    pub Height: size_t,
}
impl ::std::clone::Clone for Struct_CUDA_MEMCPY2D_st {
    fn clone(&self) -> Self { *self }
}
impl ::std::default::Default for Struct_CUDA_MEMCPY2D_st {
    fn default() -> Self { unsafe { ::std::mem::zeroed() } }
}
pub type CUDA_MEMCPY2D = Struct_CUDA_MEMCPY2D_st;
#[repr(C)]
#[derive(Copy)]
pub struct Struct_CUDA_MEMCPY3D_st {
    pub srcXInBytes: size_t,
    pub srcY: size_t,
    pub srcZ: size_t,
    pub srcLOD: size_t,
    pub srcMemoryType: CUmemorytype,
    pub srcHost: *const ::libc::c_void,
    pub srcDevice: CUdeviceptr,
    pub srcArray: CUarray,
    pub reserved0: *mut ::libc::c_void,
    pub srcPitch: size_t,
    pub srcHeight: size_t,
    pub dstXInBytes: size_t,
    pub dstY: size_t,
    pub dstZ: size_t,
    pub dstLOD: size_t,
    pub dstMemoryType: CUmemorytype,
    pub dstHost: *mut ::libc::c_void,
    pub dstDevice: CUdeviceptr,
    pub dstArray: CUarray,
    pub reserved1: *mut ::libc::c_void,
    pub dstPitch: size_t,
    pub dstHeight: size_t,
    pub WidthInBytes: size_t,
    pub Height: size_t,
    pub Depth: size_t,
}
impl ::std::clone::Clone for Struct_CUDA_MEMCPY3D_st {
    fn clone(&self) -> Self { *self }
}
impl ::std::default::Default for Struct_CUDA_MEMCPY3D_st {
    fn default() -> Self { unsafe { ::std::mem::zeroed() } }
}
pub type CUDA_MEMCPY3D = Struct_CUDA_MEMCPY3D_st;
#[repr(C)]
#[derive(Copy)]
pub struct Struct_CUDA_MEMCPY3D_PEER_st {
    pub srcXInBytes: size_t,
    pub srcY: size_t,
    pub srcZ: size_t,
    pub srcLOD: size_t,
    pub srcMemoryType: CUmemorytype,
    pub srcHost: *const ::libc::c_void,
    pub srcDevice: CUdeviceptr,
    pub srcArray: CUarray,
    pub srcContext: CUcontext,
    pub srcPitch: size_t,
    pub srcHeight: size_t,
    pub dstXInBytes: size_t,
    pub dstY: size_t,
    pub dstZ: size_t,
    pub dstLOD: size_t,
    pub dstMemoryType: CUmemorytype,
    pub dstHost: *mut ::libc::c_void,
    pub dstDevice: CUdeviceptr,
    pub dstArray: CUarray,
    pub dstContext: CUcontext,
    pub dstPitch: size_t,
    pub dstHeight: size_t,
    pub WidthInBytes: size_t,
    pub Height: size_t,
    pub Depth: size_t,
}
impl ::std::clone::Clone for Struct_CUDA_MEMCPY3D_PEER_st {
    fn clone(&self) -> Self { *self }
}
impl ::std::default::Default for Struct_CUDA_MEMCPY3D_PEER_st {
    fn default() -> Self { unsafe { ::std::mem::zeroed() } }
}
pub type CUDA_MEMCPY3D_PEER = Struct_CUDA_MEMCPY3D_PEER_st;
#[repr(C)]
#[derive(Copy)]
pub struct Struct_CUDA_ARRAY_DESCRIPTOR_st {
    pub Width: size_t,
    pub Height: size_t,
    pub Format: CUarray_format,
    pub NumChannels: ::libc::c_uint,
}
impl ::std::clone::Clone for Struct_CUDA_ARRAY_DESCRIPTOR_st {
    fn clone(&self) -> Self { *self }
}
impl ::std::default::Default for Struct_CUDA_ARRAY_DESCRIPTOR_st {
    fn default() -> Self { unsafe { ::std::mem::zeroed() } }
}
pub type CUDA_ARRAY_DESCRIPTOR = Struct_CUDA_ARRAY_DESCRIPTOR_st;
#[repr(C)]
#[derive(Copy)]
pub struct Struct_CUDA_ARRAY3D_DESCRIPTOR_st {
    pub Width: size_t,
    pub Height: size_t,
    pub Depth: size_t,
    pub Format: CUarray_format,
    pub NumChannels: ::libc::c_uint,
    pub Flags: ::libc::c_uint,
}
impl ::std::clone::Clone for Struct_CUDA_ARRAY3D_DESCRIPTOR_st {
    fn clone(&self) -> Self { *self }
}
impl ::std::default::Default for Struct_CUDA_ARRAY3D_DESCRIPTOR_st {
    fn default() -> Self { unsafe { ::std::mem::zeroed() } }
}
pub type CUDA_ARRAY3D_DESCRIPTOR = Struct_CUDA_ARRAY3D_DESCRIPTOR_st;
#[repr(C)]
#[derive(Copy)]
pub struct Struct_CUDA_RESOURCE_DESC_st {
    pub resType: CUresourcetype,
    pub res: Union_Unnamed20,
    pub flags: ::libc::c_uint,
}
impl ::std::clone::Clone for Struct_CUDA_RESOURCE_DESC_st {
    fn clone(&self) -> Self { *self }
}
impl ::std::default::Default for Struct_CUDA_RESOURCE_DESC_st {
    fn default() -> Self { unsafe { ::std::mem::zeroed() } }
}
#[repr(C)]
#[derive(Copy)]
pub struct Union_Unnamed20 {
    pub _bindgen_data_: [u64; 16usize],
}
impl Union_Unnamed20 {
    pub unsafe fn array(&mut self) -> *mut Struct_Unnamed21 {
        let raw: *mut u8 = ::std::mem::transmute(&self._bindgen_data_);
        ::std::mem::transmute(raw.offset(0))
    }
    pub unsafe fn mipmap(&mut self) -> *mut Struct_Unnamed22 {
        let raw: *mut u8 = ::std::mem::transmute(&self._bindgen_data_);
        ::std::mem::transmute(raw.offset(0))
    }
    pub unsafe fn linear(&mut self) -> *mut Struct_Unnamed23 {
        let raw: *mut u8 = ::std::mem::transmute(&self._bindgen_data_);
        ::std::mem::transmute(raw.offset(0))
    }
    pub unsafe fn pitch2D(&mut self) -> *mut Struct_Unnamed24 {
        let raw: *mut u8 = ::std::mem::transmute(&self._bindgen_data_);
        ::std::mem::transmute(raw.offset(0))
    }
    pub unsafe fn reserved(&mut self) -> *mut Struct_Unnamed25 {
        let raw: *mut u8 = ::std::mem::transmute(&self._bindgen_data_);
        ::std::mem::transmute(raw.offset(0))
    }
}
impl ::std::clone::Clone for Union_Unnamed20 {
    fn clone(&self) -> Self { *self }
}
impl ::std::default::Default for Union_Unnamed20 {
    fn default() -> Self { unsafe { ::std::mem::zeroed() } }
}
#[repr(C)]
#[derive(Copy)]
pub struct Struct_Unnamed21 {
    pub hArray: CUarray,
}
impl ::std::clone::Clone for Struct_Unnamed21 {
    fn clone(&self) -> Self { *self }
}
impl ::std::default::Default for Struct_Unnamed21 {
    fn default() -> Self { unsafe { ::std::mem::zeroed() } }
}
#[repr(C)]
#[derive(Copy)]
pub struct Struct_Unnamed22 {
    pub hMipmappedArray: CUmipmappedArray,
}
impl ::std::clone::Clone for Struct_Unnamed22 {
    fn clone(&self) -> Self { *self }
}
impl ::std::default::Default for Struct_Unnamed22 {
    fn default() -> Self { unsafe { ::std::mem::zeroed() } }
}
#[repr(C)]
#[derive(Copy)]
pub struct Struct_Unnamed23 {
    pub devPtr: CUdeviceptr,
    pub format: CUarray_format,
    pub numChannels: ::libc::c_uint,
    pub sizeInBytes: size_t,
}
impl ::std::clone::Clone for Struct_Unnamed23 {
    fn clone(&self) -> Self { *self }
}
impl ::std::default::Default for Struct_Unnamed23 {
    fn default() -> Self { unsafe { ::std::mem::zeroed() } }
}
#[repr(C)]
#[derive(Copy)]
pub struct Struct_Unnamed24 {
    pub devPtr: CUdeviceptr,
    pub format: CUarray_format,
    pub numChannels: ::libc::c_uint,
    pub width: size_t,
    pub height: size_t,
    pub pitchInBytes: size_t,
}
impl ::std::clone::Clone for Struct_Unnamed24 {
    fn clone(&self) -> Self { *self }
}
impl ::std::default::Default for Struct_Unnamed24 {
    fn default() -> Self { unsafe { ::std::mem::zeroed() } }
}
#[repr(C)]
#[derive(Copy)]
pub struct Struct_Unnamed25 {
    pub reserved: [::libc::c_int; 32usize],
}
impl ::std::clone::Clone for Struct_Unnamed25 {
    fn clone(&self) -> Self { *self }
}
impl ::std::default::Default for Struct_Unnamed25 {
    fn default() -> Self { unsafe { ::std::mem::zeroed() } }
}
pub type CUDA_RESOURCE_DESC = Struct_CUDA_RESOURCE_DESC_st;
#[repr(C)]
#[derive(Copy)]
pub struct Struct_CUDA_TEXTURE_DESC_st {
    pub addressMode: [CUaddress_mode; 3usize],
    pub filterMode: CUfilter_mode,
    pub flags: ::libc::c_uint,
    pub maxAnisotropy: ::libc::c_uint,
    pub mipmapFilterMode: CUfilter_mode,
    pub mipmapLevelBias: ::libc::c_float,
    pub minMipmapLevelClamp: ::libc::c_float,
    pub maxMipmapLevelClamp: ::libc::c_float,
    pub reserved: [::libc::c_int; 16usize],
}
impl ::std::clone::Clone for Struct_CUDA_TEXTURE_DESC_st {
    fn clone(&self) -> Self { *self }
}
impl ::std::default::Default for Struct_CUDA_TEXTURE_DESC_st {
    fn default() -> Self { unsafe { ::std::mem::zeroed() } }
}
pub type CUDA_TEXTURE_DESC = Struct_CUDA_TEXTURE_DESC_st;
pub type Enum_CUresourceViewFormat_enum = ::libc::c_uint;
pub const CU_RES_VIEW_FORMAT_NONE: ::libc::c_uint = 0;
pub const CU_RES_VIEW_FORMAT_UINT_1X8: ::libc::c_uint = 1;
pub const CU_RES_VIEW_FORMAT_UINT_2X8: ::libc::c_uint = 2;
pub const CU_RES_VIEW_FORMAT_UINT_4X8: ::libc::c_uint = 3;
pub const CU_RES_VIEW_FORMAT_SINT_1X8: ::libc::c_uint = 4;
pub const CU_RES_VIEW_FORMAT_SINT_2X8: ::libc::c_uint = 5;
pub const CU_RES_VIEW_FORMAT_SINT_4X8: ::libc::c_uint = 6;
pub const CU_RES_VIEW_FORMAT_UINT_1X16: ::libc::c_uint = 7;
pub const CU_RES_VIEW_FORMAT_UINT_2X16: ::libc::c_uint = 8;
pub const CU_RES_VIEW_FORMAT_UINT_4X16: ::libc::c_uint = 9;
pub const CU_RES_VIEW_FORMAT_SINT_1X16: ::libc::c_uint = 10;
pub const CU_RES_VIEW_FORMAT_SINT_2X16: ::libc::c_uint = 11;
pub const CU_RES_VIEW_FORMAT_SINT_4X16: ::libc::c_uint = 12;
pub const CU_RES_VIEW_FORMAT_UINT_1X32: ::libc::c_uint = 13;
pub const CU_RES_VIEW_FORMAT_UINT_2X32: ::libc::c_uint = 14;
pub const CU_RES_VIEW_FORMAT_UINT_4X32: ::libc::c_uint = 15;
pub const CU_RES_VIEW_FORMAT_SINT_1X32: ::libc::c_uint = 16;
pub const CU_RES_VIEW_FORMAT_SINT_2X32: ::libc::c_uint = 17;
pub const CU_RES_VIEW_FORMAT_SINT_4X32: ::libc::c_uint = 18;
pub const CU_RES_VIEW_FORMAT_FLOAT_1X16: ::libc::c_uint = 19;
pub const CU_RES_VIEW_FORMAT_FLOAT_2X16: ::libc::c_uint = 20;
pub const CU_RES_VIEW_FORMAT_FLOAT_4X16: ::libc::c_uint = 21;
pub const CU_RES_VIEW_FORMAT_FLOAT_1X32: ::libc::c_uint = 22;
pub const CU_RES_VIEW_FORMAT_FLOAT_2X32: ::libc::c_uint = 23;
pub const CU_RES_VIEW_FORMAT_FLOAT_4X32: ::libc::c_uint = 24;
pub const CU_RES_VIEW_FORMAT_UNSIGNED_BC1: ::libc::c_uint = 25;
pub const CU_RES_VIEW_FORMAT_UNSIGNED_BC2: ::libc::c_uint = 26;
pub const CU_RES_VIEW_FORMAT_UNSIGNED_BC3: ::libc::c_uint = 27;
pub const CU_RES_VIEW_FORMAT_UNSIGNED_BC4: ::libc::c_uint = 28;
pub const CU_RES_VIEW_FORMAT_SIGNED_BC4: ::libc::c_uint = 29;
pub const CU_RES_VIEW_FORMAT_UNSIGNED_BC5: ::libc::c_uint = 30;
pub const CU_RES_VIEW_FORMAT_SIGNED_BC5: ::libc::c_uint = 31;
pub const CU_RES_VIEW_FORMAT_UNSIGNED_BC6H: ::libc::c_uint = 32;
pub const CU_RES_VIEW_FORMAT_SIGNED_BC6H: ::libc::c_uint = 33;
pub const CU_RES_VIEW_FORMAT_UNSIGNED_BC7: ::libc::c_uint = 34;
pub type CUresourceViewFormat = Enum_CUresourceViewFormat_enum;
#[repr(C)]
#[derive(Copy)]
pub struct Struct_CUDA_RESOURCE_VIEW_DESC_st {
    pub format: CUresourceViewFormat,
    pub width: size_t,
    pub height: size_t,
    pub depth: size_t,
    pub firstMipmapLevel: ::libc::c_uint,
    pub lastMipmapLevel: ::libc::c_uint,
    pub firstLayer: ::libc::c_uint,
    pub lastLayer: ::libc::c_uint,
    pub reserved: [::libc::c_uint; 16usize],
}
impl ::std::clone::Clone for Struct_CUDA_RESOURCE_VIEW_DESC_st {
    fn clone(&self) -> Self { *self }
}
impl ::std::default::Default for Struct_CUDA_RESOURCE_VIEW_DESC_st {
    fn default() -> Self { unsafe { ::std::mem::zeroed() } }
}
pub type CUDA_RESOURCE_VIEW_DESC = Struct_CUDA_RESOURCE_VIEW_DESC_st;
#[repr(C)]
#[derive(Copy)]
pub struct Struct_CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_st {
    pub p2pToken: ::libc::c_ulonglong,
    pub vaSpaceToken: ::libc::c_uint,
}
impl ::std::clone::Clone for Struct_CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_st {
    fn clone(&self) -> Self { *self }
}
impl ::std::default::Default for Struct_CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_st {
    fn default() -> Self { unsafe { ::std::mem::zeroed() } }
}
pub type CUDA_POINTER_ATTRIBUTE_P2P_TOKENS =
    Struct_CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_st;
#[link(name = "cuda")]
extern "C" {
    // CUDA ERROR HANDLING
    pub fn cuGetErrorString(error: CUresult, pStr: *mut *const ::libc::c_char) -> CUresult;

    pub fn cuGetErrorName(error: CUresult, pStr: *mut *const ::libc::c_char)-> CUresult;

    // CUDA INITIALIZATION
    pub fn cuInit(Flags: ::libc::c_uint) -> CUresult;

    // CUDA VERSION MANAGEMENT
    pub fn cuDriverGetVersion(driverVersion: *mut ::libc::c_int) -> CUresult;

    // CUDA DEVICE MANAGEMENT
    pub fn cuDeviceGet(device: *mut CUdevice, ordinal: ::libc::c_int) -> CUresult;

    pub fn cuDeviceGetCount(count: *mut ::libc::c_int) -> CUresult;

    pub fn cuDeviceGetName(
        name: *mut ::libc::c_char,
        len: ::libc::c_int,
        dev: CUdevice
    ) -> CUresult;

    pub fn cuDeviceTotalMem_v2(bytes: *mut size_t, dev: CUdevice) -> CUresult;

    pub fn cuDeviceGetAttribute(
        pi: *mut ::libc::c_int,
        attrib: CUdevice_attribute,
        dev: CUdevice
    ) -> CUresult;

    // CUDA PRIMARY CONTEXT MANAGEMENT
    pub fn cuDevicePrimaryCtxRetain(pctx: *mut CUcontext, dev: CUdevice) -> CUresult;
    pub fn cuDevicePrimaryCtxRelease(dev: CUdevice) -> CUresult;
    pub fn cuDevicePrimaryCtxSetFlags(dev: CUdevice, flags: ::libc::c_uint) -> CUresult;

    pub fn cuDevicePrimaryCtxGetState(
        dev: CUdevice,
        flags: *mut ::libc::c_uint,
        active: *mut ::libc::c_int
    ) -> CUresult;

    pub fn cuDevicePrimaryCtxReset(dev: CUdevice) -> CUresult;

    // CUDA CONTEXT MANAGEMENT
    pub fn cuCtxCreate_v2(
        pctx: *mut CUcontext,
        flags: ::libc::c_uint,
        dev: CUdevice
    ) -> CUresult;

    pub fn cuCtxDestroy_v2(ctx: CUcontext) -> CUresult;

    pub fn cuCtxPushCurrent_v2(ctx: CUcontext) -> CUresult;

    pub fn cuCtxPopCurrent_v2(pctx: *mut CUcontext) -> CUresult;

    pub fn cuCtxSetCurrent(ctx: CUcontext) -> CUresult;

    pub fn cuCtxGetCurrent(pctx: *mut CUcontext) -> CUresult;

    pub fn cuCtxGetDevice(device: *mut CUdevice) -> CUresult;

    pub fn cuCtxGetFlags(flags: *mut ::libc::c_uint) -> CUresult;

    pub fn cuCtxSynchronize() -> CUresult;

    pub fn cuCtxSetLimit(limit: CUlimit, value: size_t) -> CUresult;

    pub fn cuCtxGetLimit(pvalue: *mut size_t, limit: CUlimit) -> CUresult;

    pub fn cuCtxGetCacheConfig(pconfig: *mut CUfunc_cache) -> CUresult;

    pub fn cuCtxSetCacheConfig(config: CUfunc_cache) -> CUresult;

    pub fn cuCtxGetSharedMemConfig(pConfig: *mut CUsharedconfig) -> CUresult;

    pub fn cuCtxSetSharedMemConfig(config: CUsharedconfig) -> CUresult;

    pub fn cuCtxGetApiVersion(ctx: CUcontext, version: *mut ::libc::c_uint) -> CUresult;

    pub fn cuCtxGetStreamPriorityRange(
        leastPriority: *mut ::libc::c_int,
        greatestPriority: *mut ::libc::c_int
    ) -> CUresult;


    // CUDA MODULE MANAGEMENT
    pub fn cuModuleLoad(
        module: *mut CUmodule,
        fname: *const ::libc::c_char
    ) -> CUresult;

    pub fn cuModuleLoadData(
        module: *mut CUmodule,
        image: *const ::libc::c_void
    ) -> CUresult;

    pub fn cuModuleLoadDataEx(
        module: *mut CUmodule,
        image: *const ::libc::c_void,
        numOptions: ::libc::c_uint,
        options: *mut CUjit_option,
        optionValues: *mut *mut ::libc::c_void
    ) -> CUresult;

    pub fn cuModuleLoadFatBinary(
        module: *mut CUmodule,
        fatCubin: *const ::libc::c_void
    ) -> CUresult;

    pub fn cuModuleUnload(hmod: CUmodule) -> CUresult;

    pub fn cuModuleGetFunction(
        hfunc: *mut CUfunction,
        hmod: CUmodule,
        name: *const ::libc::c_char
    ) -> CUresult;

    pub fn cuModuleGetGlobal_v2(
        dptr: *mut CUdeviceptr,
        bytes: *mut size_t,
        hmod: CUmodule,
        name: *const ::libc::c_char
    ) -> CUresult;

    pub fn cuModuleGetTexRef(
        pTexRef: *mut CUtexref,
        hmod: CUmodule,
        name: *const ::libc::c_char
    ) -> CUresult;

    pub fn cuModuleGetSurfRef(
        pSurfRef: *mut CUsurfref,
        hmod: CUmodule,
        name: *const ::libc::c_char
    ) -> CUresult;

    pub fn cuLinkCreate_v2(
        numOptions: ::libc::c_uint,
        options: *mut CUjit_option,
        optionValues: *mut *mut ::libc::c_void,
        stateOut: *mut CUlinkState
    ) -> CUresult;

    pub fn cuLinkAddData_v2(
        state: CUlinkState,
        _type: CUjitInputType,
        data: *mut ::libc::c_void,
        size: size_t,
        name: *const ::libc::c_char,
        numOptions: ::libc::c_uint,
        options: *mut CUjit_option,
        optionValues: *mut *mut ::libc::c_void
    ) -> CUresult;

    pub fn cuLinkAddFile_v2(
        state: CUlinkState,
        _type: CUjitInputType,
        path: *const ::libc::c_char,
        numOptions: ::libc::c_uint,
        options: *mut CUjit_option,
        optionValues: *mut *mut ::libc::c_void
    ) -> CUresult;

    pub fn cuLinkComplete(
        state: CUlinkState,
        cubinOut: *mut *mut ::libc::c_void,
        sizeOut: *mut size_t
    ) -> CUresult;

    pub fn cuLinkDestroy(state: CUlinkState) -> CUresult;

    // CUDA MEMORY MANAGEMENT
    pub fn cuMemGetInfo_v2(free: *mut size_t, total: *mut size_t) -> CUresult;

    pub fn cuMemAlloc_v2(dptr: *mut CUdeviceptr, bytesize: size_t) -> CUresult;

    pub fn cuMemAllocPitch_v2(
        dptr: *mut CUdeviceptr,
        pPitch: *mut size_t,
        WidthInBytes: size_t,
        Height: size_t,
        ElementSizeBytes: ::libc::c_uint
    ) -> CUresult;

    pub fn cuMemFree_v2(dptr: CUdeviceptr) -> CUresult;

    pub fn cuMemGetAddressRange_v2(
        pbase: *mut CUdeviceptr,
        psize: *mut size_t,
        dptr: CUdeviceptr
    ) -> CUresult;

    pub fn cuMemAllocHost_v2(pp: *mut *mut ::libc::c_void, bytesize: size_t) -> CUresult;

    pub fn cuMemFreeHost(p: *mut ::libc::c_void) -> CUresult;

    pub fn cuMemHostAlloc(
        pp: *mut *mut ::libc::c_void,
        bytesize: size_t,
        Flags: ::libc::c_uint
    ) -> CUresult;

    pub fn cuMemHostGetDevicePointer_v2(
        pdptr: *mut CUdeviceptr,
        p: *mut ::libc::c_void,
        Flags: ::libc::c_uint
    ) -> CUresult;

    pub fn cuMemHostGetFlags(
        pFlags: *mut ::libc::c_uint,
        p: *mut ::libc::c_void
    ) -> CUresult;

    pub fn cuMemAllocManaged(
        dptr: *mut CUdeviceptr,
        bytesize: size_t,
        flags: ::libc::c_uint
    ) -> CUresult;

    pub fn cuDeviceGetByPCIBusId(
        dev: *mut CUdevice,
        pciBusId: *const ::libc::c_char
    ) -> CUresult;

    pub fn cuDeviceGetPCIBusId(
        pciBusId: *mut ::libc::c_char,
        len: ::libc::c_int,
        dev: CUdevice
    ) -> CUresult;

    pub fn cuIpcGetEventHandle(
        pHandle: *mut CUipcEventHandle,
        event: CUevent
    ) -> CUresult;

    pub fn cuIpcOpenEventHandle(
        phEvent: *mut CUevent,
        handle: CUipcEventHandle
    ) -> CUresult;

    pub fn cuIpcGetMemHandle(pHandle: *mut CUipcMemHandle, dptr: CUdeviceptr) -> CUresult;

    pub fn cuIpcOpenMemHandle(
        pdptr: *mut CUdeviceptr,
        handle: CUipcMemHandle,
        Flags: ::libc::c_uint
    ) -> CUresult;

    pub fn cuIpcCloseMemHandle(dptr: CUdeviceptr) -> CUresult;

    pub fn cuMemHostRegister_v2(
        p: *mut ::libc::c_void,
        bytesize: size_t,
        Flags: ::libc::c_uint
    ) -> CUresult;

    pub fn cuMemHostUnregister(p: *mut ::libc::c_void) -> CUresult;

    pub fn cuMemcpy(dst: CUdeviceptr, src: CUdeviceptr, ByteCount: size_t) -> CUresult;

    pub fn cuMemcpyPeer(
        dstDevice: CUdeviceptr,
        dstContext: CUcontext,
        srcDevice: CUdeviceptr,
        srcContext: CUcontext,
        ByteCount: size_t
    ) -> CUresult;

    pub fn cuMemcpyHtoD_v2(
        dstDevice: CUdeviceptr,
        srcHost: *const ::libc::c_void,
        ByteCount: size_t
    ) -> CUresult;

    pub fn cuMemcpyDtoH_v2(
        dstHost: *mut ::libc::c_void,
        srcDevice: CUdeviceptr,
        ByteCount: size_t
    ) -> CUresult;

    pub fn cuMemcpyDtoD_v2(
        dstDevice: CUdeviceptr,
        srcDevice: CUdeviceptr,
        ByteCount: size_t
    ) -> CUresult;

    pub fn cuMemcpyDtoA_v2(
        dstArray: CUarray,
        dstOffset: size_t,
        srcDevice: CUdeviceptr,
        ByteCount: size_t
    ) -> CUresult;

    pub fn cuMemcpyAtoD_v2(
        dstDevice: CUdeviceptr,
        srcArray: CUarray,
        srcOffset: size_t, ByteCount: size_t
    ) -> CUresult;

    pub fn cuMemcpyHtoA_v2(
        dstArray: CUarray,
        dstOffset: size_t,
        srcHost: *const ::libc::c_void,
        ByteCount: size_t
    ) -> CUresult;

    pub fn cuMemcpyAtoH_v2(
        dstHost: *mut ::libc::c_void,
        srcArray: CUarray,
        srcOffset: size_t,
        ByteCount: size_t
    ) -> CUresult;

    pub fn cuMemcpyAtoA_v2(
        dstArray: CUarray,
        dstOffset: size_t,
        srcArray: CUarray,
        srcOffset: size_t,
        ByteCount: size_t
    ) -> CUresult;

    pub fn cuMemcpy2D_v2(pCopy: *const CUDA_MEMCPY2D) -> CUresult;

    pub fn cuMemcpy2DUnaligned_v2(pCopy: *const CUDA_MEMCPY2D) -> CUresult;

    pub fn cuMemcpy3D_v2(pCopy: *const CUDA_MEMCPY3D) -> CUresult;

    pub fn cuMemcpy3DPeer(pCopy: *const CUDA_MEMCPY3D_PEER) -> CUresult;

    pub fn cuMemcpyAsync(
        dst: CUdeviceptr,
        src: CUdeviceptr,
        ByteCount: size_t,
        hStream: CUstream
    ) -> CUresult;

    pub fn cuMemcpyPeerAsync(
        dstDevice: CUdeviceptr,
        dstContext: CUcontext,
        srcDevice: CUdeviceptr,
        srcContext: CUcontext,
        ByteCount: size_t,
        hStream: CUstream
    ) -> CUresult;

    pub fn cuMemcpyHtoDAsync_v2(
        dstDevice: CUdeviceptr,
        srcHost: *const ::libc::c_void,
        ByteCount: size_t,
        hStream: CUstream
    ) -> CUresult;

    pub fn cuMemcpyDtoHAsync_v2(
        dstHost: *mut ::libc::c_void,
        srcDevice: CUdeviceptr,
        ByteCount: size_t,
        hStream: CUstream
    ) -> CUresult;

    pub fn cuMemcpyDtoDAsync_v2(
        dstDevice: CUdeviceptr,
        srcDevice: CUdeviceptr,
        ByteCount: size_t,
        hStream: CUstream
    ) -> CUresult;

    pub fn cuMemcpyHtoAAsync_v2(
        dstArray: CUarray,
        dstOffset: size_t,
        srcHost: *const ::libc::c_void,
        ByteCount: size_t,
        hStream: CUstream
    ) -> CUresult;

    pub fn cuMemcpyAtoHAsync_v2(
        dstHost: *mut ::libc::c_void,
        srcArray: CUarray,
        srcOffset: size_t,
        ByteCount: size_t,
        hStream: CUstream
    ) -> CUresult;
    pub fn cuMemcpy2DAsync_v2(pCopy: *const CUDA_MEMCPY2D, hStream: CUstream) -> CUresult;

    pub fn cuMemcpy3DAsync_v2(
        pCopy: *const CUDA_MEMCPY3D,
        hStream: CUstream
    ) -> CUresult;

    pub fn cuMemcpy3DPeerAsync(
        pCopy: *const CUDA_MEMCPY3D_PEER,
        hStream: CUstream
    ) -> CUresult;

    pub fn cuMemsetD8_v2(
        dstDevice: CUdeviceptr,
        uc: ::libc::c_uchar,
        N: size_t
    ) -> CUresult;

    pub fn cuMemsetD16_v2(
        dstDevice: CUdeviceptr,
        us: ::libc::c_ushort,
        N: size_t
    ) -> CUresult;

    pub fn cuMemsetD32_v2(
        dstDevice: CUdeviceptr,
        ui: ::libc::c_uint,
        N: size_t
    ) -> CUresult;

    pub fn cuMemsetD2D8_v2(
        dstDevice: CUdeviceptr,
        dstPitch: size_t,
        uc: ::libc::c_uchar,
        Width: size_t,
        Height: size_t
    ) -> CUresult;

    pub fn cuMemsetD2D16_v2(
        dstDevice: CUdeviceptr,
        dstPitch: size_t,
        us: ::libc::c_ushort,
        Width: size_t,
        Height: size_t
    ) -> CUresult;

    pub fn cuMemsetD2D32_v2(
        dstDevice: CUdeviceptr,
        dstPitch: size_t,
        ui: ::libc::c_uint,
        Width: size_t,
        Height: size_t
    ) -> CUresult;

    pub fn cuMemsetD8Async(
        dstDevice: CUdeviceptr,
        uc: ::libc::c_uchar,
        N: size_t,
        hStream: CUstream
    ) -> CUresult;

    pub fn cuMemsetD16Async(
        dstDevice: CUdeviceptr,
        us: ::libc::c_ushort,
        N: size_t,
        hStream: CUstream
    ) -> CUresult;

    pub fn cuMemsetD32Async(
        dstDevice: CUdeviceptr,
        ui: ::libc::c_uint,
        N: size_t,
        hStream: CUstream
    ) -> CUresult;

    pub fn cuMemsetD2D8Async(
        dstDevice: CUdeviceptr,
        dstPitch: size_t,
        uc: ::libc::c_uchar,
        Width: size_t,
        Height: size_t,
        hStream: CUstream
    ) -> CUresult;

    pub fn cuMemsetD2D16Async(
        dstDevice: CUdeviceptr,
        dstPitch: size_t,
        us: ::libc::c_ushort,
        Width: size_t,
        Height: size_t,
        hStream: CUstream
    ) -> CUresult;

    pub fn cuMemsetD2D32Async(
        dstDevice: CUdeviceptr,
        dstPitch: size_t,
        ui: ::libc::c_uint,
        Width: size_t,
        Height: size_t,
        hStream: CUstream
    ) -> CUresult;

    pub fn cuArrayCreate_v2(
        pHandle: *mut CUarray,
        pAllocateArray: *const CUDA_ARRAY_DESCRIPTOR
    ) -> CUresult;

    pub fn cuArrayGetDescriptor_v2(
        pArrayDescriptor: *mut CUDA_ARRAY_DESCRIPTOR,
        hArray: CUarray
    ) -> CUresult;

    pub fn cuArrayDestroy(hArray: CUarray) -> CUresult;

    pub fn cuArray3DCreate_v2(
        pHandle: *mut CUarray,
        pAllocateArray: *const CUDA_ARRAY3D_DESCRIPTOR
    ) -> CUresult;

    pub fn cuArray3DGetDescriptor_v2(
        pArrayDescriptor: *mut CUDA_ARRAY3D_DESCRIPTOR,
        hArray: CUarray
    ) -> CUresult;

    pub fn cuMipmappedArrayCreate(
        pHandle: *mut CUmipmappedArray,
        pMipmappedArrayDesc: *const CUDA_ARRAY3D_DESCRIPTOR,
        numMipmapLevels: ::libc::c_uint
    ) -> CUresult;

    pub fn cuMipmappedArrayGetLevel(
        pLevelArray: *mut CUarray,
        hMipmappedArray: CUmipmappedArray,
        level: ::libc::c_uint
    ) -> CUresult;

    pub fn cuMipmappedArrayDestroy(hMipmappedArray: CUmipmappedArray) -> CUresult;

    pub fn cuPointerGetAttribute(
        data: *mut ::libc::c_void,
        attribute: CUpointer_attribute,
        ptr: CUdeviceptr
    ) -> CUresult;

    pub fn cuPointerSetAttribute(
        value: *const ::libc::c_void,
        attribute: CUpointer_attribute,
        ptr: CUdeviceptr
    ) -> CUresult;

    pub fn cuPointerGetAttributes(
        numAttributes: ::libc::c_uint,
        attributes: *mut CUpointer_attribute,
        data: *mut *mut ::libc::c_void,
        ptr: CUdeviceptr
    ) -> CUresult;

    pub fn cuStreamCreate(
        phStream: *mut CUstream,
        Flags: ::libc::c_uint
    ) -> CUresult;

    pub fn cuStreamCreateWithPriority(
        phStream: *mut CUstream,
        flags: ::libc::c_uint,
        priority: ::libc::c_int
    ) -> CUresult;

    pub fn cuStreamGetPriority(
        hStream: CUstream,
        priority: *mut ::libc::c_int
    ) -> CUresult;

    pub fn cuStreamGetFlags(
        hStream: CUstream,
        flags: *mut ::libc::c_uint
    ) -> CUresult;

    pub fn cuStreamWaitEvent(
        hStream: CUstream,
        hEvent: CUevent,
        Flags: ::libc::c_uint
    ) -> CUresult;

    pub fn cuStreamAddCallback(
        hStream: CUstream,
        callback: CUstreamCallback,
        userData: *mut ::libc::c_void,
        flags: ::libc::c_uint
    ) -> CUresult;

    pub fn cuStreamAttachMemAsync(
        hStream: CUstream,
        dptr: CUdeviceptr,
        length: size_t,
        flags: ::libc::c_uint
    ) -> CUresult;

    pub fn cuStreamQuery(hStream: CUstream) -> CUresult;

    pub fn cuStreamSynchronize(hStream: CUstream) -> CUresult;

    pub fn cuStreamDestroy_v2(hStream: CUstream) -> CUresult;

    pub fn cuEventCreate(
        phEvent: *mut CUevent,
        Flags: ::libc::c_uint
    ) -> CUresult;

    pub fn cuEventRecord(hEvent: CUevent, hStream: CUstream) -> CUresult;

    pub fn cuEventQuery(hEvent: CUevent) -> CUresult;

    pub fn cuEventSynchronize(hEvent: CUevent) -> CUresult;

    pub fn cuEventDestroy_v2(hEvent: CUevent) -> CUresult;

    pub fn cuEventElapsedTime(
        pMilliseconds: *mut ::libc::c_float,
        hStart: CUevent,
        hEnd: CUevent
    ) -> CUresult;

    pub fn cuFuncGetAttribute(
        pi: *mut ::libc::c_int,
        attrib: CUfunction_attribute,
        hfunc: CUfunction
    ) -> CUresult;

    pub fn cuFuncSetCacheConfig(hfunc: CUfunction, config: CUfunc_cache) -> CUresult;
    pub fn cuFuncSetSharedMemConfig(hfunc: CUfunction, config: CUsharedconfig) -> CUresult;

    pub fn cuLaunchKernel(
        f: CUfunction,
        gridDimX: ::libc::c_uint,
        gridDimY: ::libc::c_uint,
        gridDimZ: ::libc::c_uint,
        blockDimX: ::libc::c_uint,
        blockDimY: ::libc::c_uint,
        blockDimZ: ::libc::c_uint,
        sharedMemBytes: ::libc::c_uint, hStream: CUstream,
        kernelParams: *mut *mut ::libc::c_void,
        extra: *mut *mut ::libc::c_void
    ) -> CUresult;

    pub fn cuOccupancyMaxActiveBlocksPerMultiprocessor(
        numBlocks: *mut ::libc::c_int,
        func: CUfunction,
        blockSize: ::libc::c_int,
        dynamicSMemSize: size_t
    ) -> CUresult;

    pub fn cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
        numBlocks: *mut ::libc::c_int,
        func: CUfunction,
        blockSize: ::libc::c_int,
        dynamicSMemSize: size_t,
        flags: ::libc::c_uint
    ) -> CUresult;

    pub fn cuOccupancyMaxPotentialBlockSize(
        minGridSize: *mut ::libc::c_int,
        blockSize: *mut ::libc::c_int,
        func: CUfunction,
        blockSizeToDynamicSMemSize: CUoccupancyB2DSize,
        dynamicSMemSize: size_t,
        blockSizeLimit: ::libc::c_int
    ) -> CUresult;

    pub fn cuOccupancyMaxPotentialBlockSizeWithFlags(
        minGridSize: *mut ::libc::c_int,
        blockSize: *mut ::libc::c_int,
        func: CUfunction,
        blockSizeToDynamicSMemSize: CUoccupancyB2DSize,
        dynamicSMemSize: size_t,
        blockSizeLimit: ::libc::c_int,
        flags: ::libc::c_uint
    ) -> CUresult;

    pub fn cuTexRefSetArray(
        hTexRef: CUtexref,
        hArray: CUarray,
        Flags: ::libc::c_uint
    ) -> CUresult;

    pub fn cuTexRefSetMipmappedArray(
        hTexRef: CUtexref,
        hMipmappedArray: CUmipmappedArray,
        Flags: ::libc::c_uint
    ) -> CUresult;

    pub fn cuTexRefSetAddress_v2(
        ByteOffset: *mut size_t,
        hTexRef: CUtexref,
        dptr: CUdeviceptr, bytes: size_t
    ) -> CUresult;

    pub fn cuTexRefSetAddress2D_v3(
        hTexRef: CUtexref,
        desc: *const CUDA_ARRAY_DESCRIPTOR,
        dptr: CUdeviceptr, Pitch: size_t
    ) -> CUresult;

    pub fn cuTexRefSetFormat(
        hTexRef: CUtexref,
        fmt: CUarray_format,
        NumPackedComponents: ::libc::c_int
    ) -> CUresult;

    pub fn cuTexRefSetAddressMode(
        hTexRef: CUtexref,
        dim: ::libc::c_int,
        am: CUaddress_mode
    ) -> CUresult;

    pub fn cuTexRefSetFilterMode(hTexRef: CUtexref, fm: CUfilter_mode) -> CUresult;

    pub fn cuTexRefSetMipmapFilterMode(hTexRef: CUtexref, fm: CUfilter_mode) -> CUresult;

    pub fn cuTexRefSetMipmapLevelBias(
        hTexRef: CUtexref,
        bias: ::libc::c_float
    ) -> CUresult;

    pub fn cuTexRefSetMipmapLevelClamp(
        hTexRef: CUtexref,
        minMipmapLevelClamp: ::libc::c_float,
        maxMipmapLevelClamp: ::libc::c_float
    ) -> CUresult;

    pub fn cuTexRefSetMaxAnisotropy(hTexRef: CUtexref, maxAniso: ::libc::c_uint) -> CUresult;

    pub fn cuTexRefSetFlags(hTexRef: CUtexref, Flags: ::libc::c_uint) -> CUresult;

    pub fn cuTexRefGetAddress_v2(pdptr: *mut CUdeviceptr, hTexRef: CUtexref) -> CUresult;

    pub fn cuTexRefGetArray(phArray: *mut CUarray, hTexRef: CUtexref) -> CUresult;

    pub fn cuTexRefGetMipmappedArray(
        phMipmappedArray: *mut CUmipmappedArray,
        hTexRef: CUtexref
    ) -> CUresult;

    pub fn cuTexRefGetAddressMode(
        pam: *mut CUaddress_mode,
        hTexRef: CUtexref,
        dim: ::libc::c_int
    ) -> CUresult;

    pub fn cuTexRefGetFilterMode(pfm: *mut CUfilter_mode, hTexRef: CUtexref) -> CUresult;

    pub fn cuTexRefGetFormat(
        pFormat: *mut CUarray_format,
        pNumChannels: *mut ::libc::c_int,
        hTexRef: CUtexref
    ) -> CUresult;

    pub fn cuTexRefGetMipmapFilterMode(
        pfm: *mut CUfilter_mode,
        hTexRef: CUtexref
    ) -> CUresult;

    pub fn cuTexRefGetMipmapLevelBias(
        pbias: *mut ::libc::c_float,
        hTexRef: CUtexref
    ) -> CUresult;

    pub fn cuTexRefGetMipmapLevelClamp(
        pminMipmapLevelClamp: *mut ::libc::c_float,
        pmaxMipmapLevelClamp: *mut ::libc::c_float,
        hTexRef: CUtexref
    ) -> CUresult;

    pub fn cuTexRefGetMaxAnisotropy(
        pmaxAniso: *mut ::libc::c_int,
        hTexRef: CUtexref
    ) -> CUresult;

    pub fn cuTexRefGetFlags(pFlags: *mut ::libc::c_uint, hTexRef: CUtexref) -> CUresult;

    pub fn cuSurfRefSetArray(
        hSurfRef: CUsurfref, hArray: CUarray,
        Flags: ::libc::c_uint
    ) -> CUresult;

    pub fn cuSurfRefGetArray(
        phArray: *mut CUarray,
        hSurfRef: CUsurfref
    ) -> CUresult;

    pub fn cuTexObjectCreate(
        pTexObject: *mut CUtexObject,
        pResDesc: *const CUDA_RESOURCE_DESC,
        pTexDesc: *const CUDA_TEXTURE_DESC,
        pResViewDesc: *const CUDA_RESOURCE_VIEW_DESC
    ) -> CUresult;

    pub fn cuTexObjectDestroy(texObject: CUtexObject) -> CUresult;

    pub fn cuTexObjectGetResourceDesc(
        pResDesc: *mut CUDA_RESOURCE_DESC,
        texObject: CUtexObject
    ) -> CUresult;

    pub fn cuTexObjectGetTextureDesc(
        pTexDesc: *mut CUDA_TEXTURE_DESC,
        texObject: CUtexObject
    ) -> CUresult;

    pub fn cuTexObjectGetResourceViewDesc(
        pResViewDesc: *mut CUDA_RESOURCE_VIEW_DESC,
        texObject: CUtexObject
    ) -> CUresult;

    pub fn cuSurfObjectCreate(
        pSurfObject: *mut CUsurfObject,
        pResDesc: *const CUDA_RESOURCE_DESC
    ) -> CUresult;

    pub fn cuSurfObjectDestroy(surfObject: CUsurfObject) -> CUresult;

    pub fn cuSurfObjectGetResourceDesc(
        pResDesc: *mut CUDA_RESOURCE_DESC,
        surfObject: CUsurfObject
    ) -> CUresult;

    pub fn cuDeviceCanAccessPeer(
        canAccessPeer: *mut ::libc::c_int,
        dev: CUdevice, peerDev: CUdevice
    ) -> CUresult;

    pub fn cuCtxEnablePeerAccess(
        peerContext: CUcontext,
        Flags: ::libc::c_uint
    ) -> CUresult;

    pub fn cuCtxDisablePeerAccess(peerContext: CUcontext) -> CUresult;

    pub fn cuGraphicsUnregisterResource(resource: CUgraphicsResource) -> CUresult;

    pub fn cuGraphicsSubResourceGetMappedArray(
        pArray: *mut CUarray,
        resource: CUgraphicsResource,
        arrayIndex: ::libc::c_uint,
        mipLevel: ::libc::c_uint
    ) -> CUresult;

    pub fn cuGraphicsResourceGetMappedMipmappedArray(
        pMipmappedArray: *mut CUmipmappedArray,
        resource: CUgraphicsResource
    ) -> CUresult;

    pub fn cuGraphicsResourceGetMappedPointer_v2(
        pDevPtr: *mut CUdeviceptr,
        pSize: *mut size_t,
        resource: CUgraphicsResource
    ) -> CUresult;

    pub fn cuGraphicsResourceSetMapFlags_v2(
        resource: CUgraphicsResource,
        flags: ::libc::c_uint
    ) -> CUresult;

    pub fn cuGraphicsMapResources(
        count: ::libc::c_uint,
        resources: *mut CUgraphicsResource,
        hStream: CUstream
    ) -> CUresult;

    pub fn cuGraphicsUnmapResources(
        count: ::libc::c_uint,
        resources: *mut CUgraphicsResource,
        hStream: CUstream
    ) -> CUresult;

    pub fn cuGetExportTable(
        ppExportTable: *mut *const ::libc::c_void,
        pExportTableId: *const CUuuid
    ) -> CUresult;
}
