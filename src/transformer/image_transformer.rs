use std::path::Path;
use image::{DynamicImage, FilterType, open};
use super::Transformer;

/// Transformer for images
///
/// Supports transforming images for most of the popular image formats including `PNG`,
/// `JPEG (non-progressive)`, `GIF`, and `BMP`. For all formats and more details take a look at
/// the [`image` crate](https://github.com/PistonDevelopers/image).
pub struct ImageTransformer {
    /// The image
    image: DynamicImage,
}

impl Transformer for ImageTransformer {
    fn transform(&self, dimensions: u32) -> Option<Vec<u32>> {
        match dimensions {
            0 => None,
            1 => Some(self.image.raw_pixels().iter().map(|&e| e as u32).collect()),
            _ => None,
        }
    }
}

impl ImageTransformer {

    /// Create a new ImageTransformer
    ///
    /// For convenience you can specify a Path to the image
    pub fn new<P>(path: P) -> ImageTransformer
        where P: AsRef<Path>
    {
        let img = open(path).unwrap();
        ImageTransformer { image: img }
    }

    /// Resizes to the new specfied dimensions
    ///
    /// It will try to preserve the aspect ratio. It uses the `Nearest` Filter.
    pub fn resize(&self, width: u32, height: u32) -> ImageTransformer {
        ImageTransformer { image: self.image.resize(width, height, FilterType::Nearest) }
    }

    /// Crops the image
    ///
    /// Use (x, y, width, height) to specify the rectangle for the new image
    pub fn crop(&mut self, x: u32, y: u32, width: u32, height: u32) -> ImageTransformer {
        ImageTransformer { image: self.image.crop(x, y, width, height) }
    }
}
