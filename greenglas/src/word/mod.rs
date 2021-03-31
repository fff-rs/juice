use crate::murmur3::murmurhash3_x86_32 as murmur3;
use crate::{Set, Transformer};

/// The Modifiers for `Word`
pub mod modifiers;

#[derive(Debug)]
/// The Transformable Data Type `Word`
pub struct Word {
    value: String,
}

impl Set for Word {}

impl Transformer for Word {
    fn transform_to_vec(&self) -> Vec<f32> {
        vec![murmur3(self.value.as_bytes(), 0) as f32]
    }
}

impl Word {
    /// Creates a new `Word`
    pub fn new(word: String) -> Word {
        Word { value: word }
    }
}
