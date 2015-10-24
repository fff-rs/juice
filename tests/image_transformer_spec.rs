extern crate cuticula;

#[cfg(test)]
mod image_transformer_spec {

    use cuticula::{ Transformer, ImageTransformer };
    use std::path::Path;

    fn expected_result() -> Vec<u32> {
        vec![255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0]
    }

    // Additional Alpha Channel for GIF
    fn expected_result_gif() -> Vec<u32> {
        vec![255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0]
    }

    fn expected_result_resize() -> Vec<u32> { vec![191, 191, 191] }
    fn expected_result_crop() -> Vec<u32> { vec![255, 255, 255] }

    #[test]
    fn it_works_for_png() {
        let path = Path::new("tests/assets/test_image.png");
        let img = ImageTransformer::new(&path);
        assert_eq!(expected_result(), img.transform(1).unwrap());
    }

    #[test]
    #[should_panic]
    fn it_works_not_for_progressive_jpeg() {
        let path = Path::new("tests/assets/test_image.jpeg");
        let img = ImageTransformer::new(&path);
        assert_eq!(expected_result(), img.transform(1).unwrap());
    }

    #[test]
    fn it_works_for_baseline_jpeg() {
        let path = Path::new("tests/assets/test_image.baseline.jpeg");
        let img = ImageTransformer::new(&path);
        assert_eq!(expected_result(), img.transform(1).unwrap());
    }

    #[test]
    fn it_works_for_gif() {
        let path = Path::new("tests/assets/test_image.gif");
        let img = ImageTransformer::new(&path);
        assert_eq!(expected_result_gif(), img.transform(1).unwrap());
    }

    #[test]
    fn it_works_for_bmp() {
        let path = Path::new("tests/assets/test_image.bmp");
        let img = ImageTransformer::new(&path);
        assert_eq!(expected_result(), img.transform(1).unwrap());
    }

    #[test]
    fn it_works_to_resize() {
        let path = Path::new("tests/assets/test_image.png");
        let img = ImageTransformer::new(&path).resize(1, 1);
        assert_eq!(expected_result_resize(), img.transform(1).unwrap());
    }

    #[test]
    fn it_works_to_crop() {
        let path = Path::new("tests/assets/test_image.png");
        let img = ImageTransformer::new(&path).crop(0, 0, 1, 1);
        assert_eq!(expected_result_crop(), img.transform(1).unwrap());
    }
}
