/// Traditional allocation via creation of a filled Vec.
pub fn allocate_boxed_slice(cap: usize) -> Box<[u8]> {
    let vec: Vec<u8> = vec![0; cap];
    let bx: Box<[u8]> = vec.into_boxed_slice();
    bx
}
