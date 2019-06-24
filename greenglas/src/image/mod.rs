use std::path::Path;
use crate::image_crate::{DynamicImage, ImageBuffer, open, load_from_memory};

use crate::{Set, Transformer};
use crate::transformer::TransformerError;
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
    pub fn from_rgb_pixels(w: u32, h: u32, buf: Vec<u8>) -> Result<Image, TransformerError> {
        let dynamic_image = ImageBuffer::from_raw(w, h, buf).map(DynamicImage::ImageRgb8);
        match dynamic_image {
            Some(image) => Ok(Image { value: image }),
            None => Err(TransformerError::InvalidRgbPixels)
        }
    }

    /// Create a new Image from RGBa style pixel container such as `Vec`
    pub fn from_rgba_pixels(w: u32, h: u32, buf: Vec<u8>) -> Result<Image, TransformerError> {
        let dynamic_image = ImageBuffer::from_raw(w, h, buf).map(DynamicImage::ImageRgba8);
        match dynamic_image {
            Some(image) => Ok(Image { value: image }),
            None => Err(TransformerError::InvalidRgbaPixels)
        }
    }

    /// Create a new Image from Luma (greyscale) style pixel container such as `Vec`
    pub fn from_luma_pixels(w: u32, h: u32, buf: Vec<u8>) -> Result<Image, TransformerError> {
        let dynamic_image = ImageBuffer::from_raw(w, h, buf).map(DynamicImage::ImageLuma8);
        match dynamic_image {
            Some(image) => Ok(Image { value: image }),
            None => Err(TransformerError::InvalidLumaPixels)
        }
    }

    /// Create a new Image from LumaA style pixel container such as `Vec`
    pub fn from_lumaa_pixels(w: u32, h: u32, buf: Vec<u8>) -> Result<Image, TransformerError> {
        let dynamic_image = ImageBuffer::from_raw(w, h, buf).map(DynamicImage::ImageLumaA8);
        match dynamic_image {
            Some(image) => Ok(Image { value: image }),
            None => Err(TransformerError::InvalidLumaAlphaPixels)
        }
    }
}
