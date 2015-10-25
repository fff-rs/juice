use std::path::Path;
use image_lib::{DynamicImage, open};
use {Set, Transformer};
pub use self::modifiers::*;

pub mod modifiers;

pub struct Image {
    value: DynamicImage,
}

impl Set for Image {}

impl Transformer for Image {
    fn transform(&self, dimensions: u32) -> Option<Vec<u32>> {
        match dimensions {
            0 => None,
            1 => Some(self.value.raw_pixels().iter().map(|&e| e as u32).collect()),
            _ => None,
        }
    }
}

impl Image {

    /// Create a new Image from a Path
    pub fn from_path<P>(path: P) -> Image
        where P: AsRef<Path>
    {
        Image { value: open(path).unwrap() }
    }
}
