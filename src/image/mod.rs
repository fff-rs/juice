use std::path::Path;
use image_lib::{DynamicImage, ImageBuffer, open, load_from_memory};
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
}

impl Image {

    /// Create a new Image from a DynamicImage
    pub fn new(image: DynamicImage) -> Image {
        Image {
            value: image
        }
    }

    /// Create a new Image from a Path
    pub fn from_path<P>(path: P) -> Image
        where P: AsRef<Path>
    {
        Image { value: open(path).unwrap() }
    }

    /// Create a new Image from Buffer
    pub fn from_buffer(buf: &[u8]) -> Image
    {
        Image { value: load_from_memory(buf).unwrap() }
    }

    /// Create a new Image from RGB style pixel container such as `Vec`
    pub fn from_rgb_pixels(w: u32, h: u32, buf: Vec<u8>) -> Image {
        Image {
            value: ImageBuffer::from_raw(w, h, buf).map(DynamicImage::ImageRgb8).unwrap()
        }
    }

    /// Create a new Image from RGBa style pixel container such as `Vec`
    pub fn from_rgba_pixels(w: u32, h: u32, buf: Vec<u8>) -> Image {
        Image {
            value: ImageBuffer::from_raw(w, h, buf).map(DynamicImage::ImageRgba8).unwrap()
        }
    }
}
