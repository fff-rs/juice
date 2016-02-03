extern crate collenchyma as co;

#[cfg(test)]
mod tensor_spec {
    use co::*;

    #[test]
    fn it_returns_correct_tensor_desc_stride() {
        let tensor_desc_r0: TensorDesc = vec!();
        let tensor_desc_r1: TensorDesc = vec!(5);
        let tensor_desc_r2: TensorDesc = vec!(2, 4);
        let tensor_desc_r3: TensorDesc = vec!(2, 2, 4);
        let tensor_desc_r4: TensorDesc = vec!(2, 2, 4, 4);
        let r0: Vec<usize> = vec!();
        assert_eq!(r0, tensor_desc_r0.default_stride());
        assert_eq!(vec![1], tensor_desc_r1.default_stride());
        assert_eq!(vec![4, 1], tensor_desc_r2.default_stride());
        assert_eq!(vec![8, 4, 1], tensor_desc_r3.default_stride());
        assert_eq!(vec![32, 16, 4, 1], tensor_desc_r4.default_stride());
    }

    #[test]
    fn it_returns_correct_size_for_rank_0() {
        // In order for correct memory allocation of scala Tensor, the size should never return less than 1.
        let tensor_desc_r0: TensorDesc = vec!();
        assert_eq!(1, tensor_desc_r0.size());

        let tensor_desc_r0_into = <() as IntoTensorDesc>::into(&());
        assert_eq!(1, tensor_desc_r0_into.size());
    }
}
