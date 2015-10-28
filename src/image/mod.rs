use std::path::Path;
use image_lib::{DynamicImage, open};
use {Set, Transformer};
pub use self::modifiers::*;

/// The Modifiers form `Image`
pub mod modifiers;

#[allow(missing_debug_implementations)]
/// The Transformable Data Type `Image`
pub struct Image {
    value: DynamicImage,
}

impl Set for Image {}

impl Transformer for Image {
    fn transform_to_vec(&self) -> Vec<f32> {
        self.value.raw_pixels().iter().map(|&e| e as f32).collect()
    }

    fn write_into_blob_data(&self, blob_data: &mut Vec<f32>) {
        for (i, &e) in self.value.raw_pixels().iter().enumerate() {
            blob_data[i] = e as f32;
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
