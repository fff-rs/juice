extern crate cuticula;

#[cfg(test)]
mod string_transformer_spec {

    use cuticula::{ Transformer, StringTransformer };

    fn expected_result() -> Vec<u32> {
        vec![3127628307]
    }

    #[test]
    fn it_works() {
        let st = StringTransformer { string: "test".to_string() };
        assert_eq!(expected_result(), st.transform(1).unwrap());
    }
}
