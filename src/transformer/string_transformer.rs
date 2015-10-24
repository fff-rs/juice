use super::Transformer;
use murmurhash3::murmurhash3_x86_32 as murmur3;

/// Transformer for strings
///
/// # Example
///
/// ```
/// # extern crate cuticula;
/// # fn main() {
/// use cuticula::{ Transformer, StringTransformer };
/// let some_string = " test ".to_string();
/// let str_t = StringTransformer { string: some_string.trim().to_string() };
/// println!("{:?}", str_t.transform(1));
/// # }
/// ```
///
pub struct StringTransformer {
    /// The string
    pub string: String,
}

impl Transformer for StringTransformer {
    fn transform(&self, dimensions: u32) -> Option<Vec<u32>> {
        match dimensions {
            0 => None,
            1 => Some(vec![murmur3(self.string.as_bytes(), 0)]),
            _ => None,
        }
    }
}
