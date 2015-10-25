use murmur3::murmurhash3_x86_32 as murmur3;
use {Set, Transformer};
pub use self::modifiers::*;

pub mod modifiers;

pub struct Word {
    value: String,
}

impl Set for Word {}

impl Transformer for Word {
    fn transform(&self, dimensions: u32) -> Option<Vec<u32>> {
        match dimensions {
            0 => None,
            1 => Some(vec![murmur3(self.value.as_bytes(), 0)]),
            _ => None,
        }
    }
}

impl Word {
    pub fn new(word: String) -> Word {
        Word { value: word }
    }
}
