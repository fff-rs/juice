use modifier::Modifier;
use image_lib::FilterType;
use super::Image;

pub struct Resize {
    pub width: u32,
    pub height: u32,
}

pub struct Crop {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl Modifier<Image> for Resize {
    fn modify(self, image: &mut Image) {
        image.value = image.value.resize(self.width, self.height, FilterType::Nearest)
    }
}

impl Modifier<Image> for Crop {
    fn modify(self, image: &mut Image) {
        image.value = image.value.crop(self.x, self.y, self.width, self.height)
    }
}
