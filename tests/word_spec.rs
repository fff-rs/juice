extern crate cuticula;

#[cfg(test)]
mod word_spec {

    use cuticula::{ Transformer, Word };

    fn expected_result() -> Vec<f32> {
        vec![3127628307.0]
    }

    #[test]
    fn it_works() {
        assert_eq!(expected_result(), Word::new("test".to_string()).transform_to_vec());
    }
}
