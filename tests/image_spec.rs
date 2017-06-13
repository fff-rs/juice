extern crate cuticula;

#[cfg(test)]
mod image_spec {

    use cuticula::{Set, Transformer, Image};
    use cuticula::image::{Resize, Crop};
    use std::path::Path;

    fn expected_result() -> Vec<f32> {
        vec![255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 0.0, 0.0, 0.0]
    }

    fn expected_result_with_alpha() -> Vec<f32> {
        vec![255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 0.0, 0.0, 0.0, 255.0]
    }

    fn expected_result_resize() -> Vec<f32> { vec![191.0, 191.0, 191.0] }
    fn expected_result_crop() -> Vec<f32> { vec![255.0, 255.0, 255.0] }

    #[test]
    fn it_works_for_pixels_rgb() {
        let buffer: Vec<u8> = vec![255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0];
        let img = Image::from_rgb_pixels(2, 2, buffer);
        match img {
            Ok(i) => assert_eq!(expected_result(), i.transform_to_vec()),
            Err(_) => assert!(false)
        }
    }

    #[test]
    fn it_works_for_pixels_rgba() {
        let buffer: Vec<u8> = vec![255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 255];
        let img = Image::from_rgba_pixels(2, 2, buffer);
        match img {
            Ok(i) => assert_eq!(expected_result_with_alpha(), i.transform_to_vec()),
            Err(_) => assert!(false)
        }
    }

    #[test]
    fn it_works_for_pixels_greyscale() {
        let buffer: Vec<u8> = vec![255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0];
        let img = Image::from_luma_pixels(3, 4, buffer);
        match img {
            Ok(i) => assert_eq!(expected_result(), i.transform_to_vec()),
            Err(_) => assert!(false)
        }
    }

    #[test]
    fn it_works_for_pixels_greyscale_alpha() {
        let buffer: Vec<u8> = vec![255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0];
        let img = Image::from_lumaa_pixels(3, 2, buffer);
        match img {
            Ok(i) => assert_eq!(expected_result(), i.transform_to_vec()),
            Err(_) => assert!(false)
        }
    }

    #[test]
    fn it_works_for_png() {
        let path = Path::new("tests/assets/test_image.png");
        let img = Image::from_path(&path);
        assert_eq!(expected_result(), img.transform_to_vec());
    }

    #[test]
    #[should_panic]
    fn it_works_not_for_progressive_jpeg() {
        let path = Path::new("tests/assets/test_image.jpeg");
        let img = Image::from_path(&path);
        assert_eq!(expected_result(), img.transform_to_vec());
    }

    #[test]
    fn it_works_for_baseline_jpeg() {
        let path = Path::new("tests/assets/test_image.baseline.jpeg");
        let img = Image::from_path(&path);
        assert_eq!(expected_result(), img.transform_to_vec());
    }

    #[test]
    fn it_works_for_gif() {
        let path = Path::new("tests/assets/test_image.gif");
        let img = Image::from_path(&path);
        assert_eq!(expected_result_with_alpha(), img.transform_to_vec());
    }

    #[test]
    fn it_works_for_bmp() {
        let path = Path::new("tests/assets/test_image.bmp");
        let img = Image::from_path(&path);
        assert_eq!(expected_result(), img.transform_to_vec());
    }

    #[test]
    fn it_works_to_resize() {
        let path = Path::new("tests/assets/test_image.png");
        let mut img = Image::from_path(&path);
        let resize = Resize { width: 1, height: 1 };
        img = img.set(resize);
        assert_eq!(expected_result_resize(), img.transform_to_vec());
    }

    #[test]
    fn it_works_to_crop() {
        let path = Path::new("tests/assets/test_image.png");
        let mut img = Image::from_path(&path);
        let crop = Crop { x: 0, y: 0, width: 1, height: 1 };
        img = img.set(crop);
        assert_eq!(expected_result_crop(), img.transform_to_vec());
    }
}
