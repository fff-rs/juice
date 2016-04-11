//! Provides useful macros for easier NN implementation for CUDA/cuDNN.

macro_rules! read {
    ($x:ident, $slf:ident) => (
        try!($x.read($slf.device())).as_cuda()
            .expect("Broken invariant: not a CUDA memory")
    )
}

macro_rules! read_write {
    ($x:ident, $slf:ident) => (
        try!($x.read_write($slf.device())).as_cuda()
            .expect("Broken invariant: not a CUDA memory")
    )
}

macro_rules! write_only {
    ($x:ident, $slf:ident) => (
        try!($x.write_only($slf.device())).as_cuda()
            .expect("Broken invariant: not a CUDA memory")
    )
}

// trans! cannot be inlined into macros above, because `$mem` would become
// intermidiate variable and `*mut $t` will outlive it.
macro_rules! trans {
    ($mem:ident) => (
        unsafe { ::std::mem::transmute::<u64, *const ::libc::c_void>(*$mem.id_c()) }
    )
}

macro_rules! trans_mut {
    ($mem:ident) => (
        unsafe { ::std::mem::transmute::<u64, *mut ::libc::c_void>(*$mem.id_c()) }
    )
}
