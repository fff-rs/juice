use modifier::Modifier;
use image_crate::FilterType;
use super::Image;

#[derive(Debug, Clone, Copy)]
/// Resize Modifier for `Image`
pub struct Resize {
    /// The resized width of the new Image
    pub width: u32,
    /// The resized heigt of the new Image
    pub height: u32,
}

impl Modifier<Image> for Resize {
    fn modify(self, image: &mut Image) {
        image.value = image.value.resize(self.width, self.height, FilterType::Nearest)
    }
}

#[derive(Debug, Clone, Copy)]
/// Crop Modifier for `Image`
pub struct Crop {
    /// The x value from where the new Image should start
    pub x: u32,
    /// The y value from where the new Image should start
    pub y: u32,
    /// The width for the new Image
    pub width: u32,
    /// The height for the new Image
    pub height: u32,
}

impl Modifier<Image> for Crop {
    fn modify(self, image: &mut Image) {
        image.value = image.value.crop(self.x, self.y, self.width, self.height)
    }
}

#[derive(Debug, Clone, Copy)]
/// Grayscale Modifier for `Image`
pub struct Grayscale;

impl Modifier<Image> for Grayscale {
    fn modify(self, image: &mut Image) {
        image.value = image.value.grayscale();
    }
}
