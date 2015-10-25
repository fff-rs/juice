extern crate cuticula;

#[cfg(test)]
mod word_spec {

    use cuticula::{ Transformer, Word };

    fn expected_result() -> Vec<u32> {
        vec![3127628307]
    }

    #[test]
    fn it_works() {
        assert_eq!(expected_result(), Word::new("test".to_string()).transform(1).unwrap());
    }
}
