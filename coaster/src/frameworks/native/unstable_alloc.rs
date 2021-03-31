use alloc::raw_vec::RawVec;

/// Alternative way to allocate memory, requiring unstable RawVec.
pub fn allocate_boxed_slice(cap: usize) -> Box<[u8]> {
    let raw = RawVec::with_capacity(cap);
    unsafe { raw.into_box() }
}
