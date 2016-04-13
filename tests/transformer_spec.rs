extern crate cuticula;
extern crate collenchyma;

#[cfg(test)]
mod transformer_spec {

    use cuticula::{Set, Transformer, Image};
    use cuticula::image::{Crop};
    use cuticula::transformer::TransformerError;
    use collenchyma::prelude::*;
    use std::path::Path;

    fn expected_result() -> Vec<f32> {
        vec![255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 0.0, 0.0, 0.0]
    }

    #[test]
    fn transform_returns_a_valid_result() {
        let path = Path::new("tests/assets/test_image.png");
        let img = Image::from_path(&path);
        match img.transform(&vec![2, 2, 3]) {
            Ok(_) => assert!(true),
            _ => assert!(false)
        }
    }

    #[test]
    fn transform_returns_a_tensor() {
        let path = Path::new("tests/assets/test_image.png");
        let img = Image::from_path(&path);
        match img.transform(&vec![2, 2, 3]) {
            Ok(tensor) => {
                let native_backend = Backend::<Native>::default().unwrap();
                let data = tensor.get(native_backend.device()).unwrap().as_native().unwrap().as_slice();
                assert_eq!(expected_result(), data);
            },
            _ => assert!(false)
        }
    }

    #[test]
    fn transform_returns_an_error_when_different_shape() {
        let path = Path::new("tests/assets/test_image.png");
        let img = Image::from_path(&path);
        match img.transform(&vec![3, 3, 3]) {
            Err(TransformerError::InvalidShape) => assert!(true),
            _ => assert!(false)
        }
    }

    #[test]
    fn transform_returns_a_valid_result_with_modifiers() {
        let path = Path::new("tests/assets/test_image.png");
        let img = Image::from_path(&path);
        let crop = Crop { x: 0, y: 0, width: 1, height: 1 };
        match img.set(crop).transform(&vec![1, 1, 3]) {
            Ok(_) => assert!(true),
            _ => assert!(false)
        }
    }
}
