use std::fmt;

use co::prelude::*;
use co::plugin::numeric_helpers::Float;

use plugin::{Convolution, LRN, Pooling,
             ConvForwardAlgo, ConvBackwardFilterAlgo, ConvBackwardDataAlgo};
use tests::{Epsilon, filled_tensor, tensor_assert_eq};

pub fn test_pooling_max<T, F: IFramework>(backend: Backend<F>)
    where T: Float + Epsilon + fmt::Debug,
          Backend<F>: Pooling<T> + IBackend {
    let mut inp = vec![1.0; 256];
    inp[0] = 2.0;

    let x  = filled_tensor(&[4, 4, 4, 4], &inp);
    let mut r = SharedTensor::<T>::new(&[4, 4, 2, 2]);
    let conf = Pooling::<T>::new_pooling_config(&backend, &[2, 2], &[0, 0], &[2, 1])
        .unwrap();

    backend.pooling_max(&x, &mut r, &conf).unwrap();

    let mut r_test = vec![1.0; 64];
    r_test[0] = 2.0;
    tensor_assert_eq(&r, &r_test, 3.0);
}

pub fn test_pooling_max_grad<T, F: IFramework>(backend: Backend<F>)
    where T: Float + Epsilon + fmt::Debug,
          Backend<F>: Pooling<T> + IBackend {
    let mut inp = vec![1.0; 256];
    inp[0] = 2.0;

    let x  = filled_tensor(&[4, 4, 4, 4], &inp);
    let dx = filled_tensor(&[4, 4, 4, 4], &inp);
    let r  = filled_tensor(&[4, 4, 2, 2], &inp[0..64]);
    let mut dr = SharedTensor::<T>::new(&[4, 4, 2, 2]);
    let conf = Pooling::<T>::new_pooling_config(&backend, &[2, 2], &[0, 0], &[2, 1])
        .unwrap();

    backend.pooling_max_grad(&x, &dx, &r, &mut dr, &conf).unwrap();

    let dr_test = [
        2.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0,
        1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0,
        1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0,
        1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0];
    tensor_assert_eq(&dr, &dr_test, 3.0);
}

pub fn test_pooling_avg<T, F: IFramework>(backend: Backend<F>)
    where T: Float + Epsilon + fmt::Debug,
          Backend<F>: Pooling<T> + IBackend {
    let mut inp = vec![1.0; 256];
    inp[0] = 5.0;

    let x  = filled_tensor(&[4, 4, 4, 4], &inp);
    let mut r = SharedTensor::<T>::new(&[4, 4, 2, 2]);
    let conf = Pooling::<T>::new_pooling_config(&backend, &[2, 2], &[0, 0], &[2, 1])
        .unwrap();

    backend.pooling_avg(&x, &mut r, &conf).unwrap();

    let mut r_test = vec![1.0; 64];
    r_test[0] = 2.0;
    tensor_assert_eq(&r, &r_test, 3.0);
}

pub fn test_pooling_avg_grad<T, F: IFramework>(backend: Backend<F>)
    where T: Float + Epsilon + fmt::Debug,
          Backend<F>: Pooling<T> + IBackend {
    let mut inp = vec![1.0; 256];
    inp[0] = 2.0;

    let x  = filled_tensor(&[4, 4, 4, 4], &inp);
    let dx = filled_tensor(&[4, 4, 4, 4], &inp);
    let r  = filled_tensor(&[4, 4, 2, 2], &inp[0..64]);
    let mut dr = SharedTensor::<T>::new(&[4, 4, 2, 2]);
    let conf = Pooling::<T>::new_pooling_config(&backend, &[2, 2], &[0, 0], &[2, 1])
        .unwrap();

    backend.pooling_avg_grad(&x, &dx, &r, &mut dr, &conf).unwrap();

    tensor_assert_eq(&dr, &inp, 1.0);
}

mod cuda {
    use super::*;
    test_cuda!(test_pooling_avg, pooling_avg_f32, pooling_avg_f64);
    test_cuda!(test_pooling_avg_grad, pooling_avg_grad_f32, pooling_avg_grad_f64);
    test_cuda!(test_pooling_max, pooling_max_f32, pooling_max_f64);
    test_cuda!(test_pooling_max_grad, pooling_max_grad_f32, pooling_max_grad_f64);
}

