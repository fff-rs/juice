use std::fmt;

use co::prelude::*;
use co::plugin::numeric_helpers::Float;

use plugin::{Convolution, LRN, Pooling,
             ConvForwardAlgo, ConvBackwardFilterAlgo, ConvBackwardDataAlgo};
use tests::{Epsilon, filled_tensor, tensor_assert_eq};


pub fn test_lrn<T, F: IFramework>(backend: Backend<F>)
    where T: Float + Epsilon + fmt::Debug,
          Backend<F>: LRN<T> + IBackend {

    let x  = filled_tensor(&[1, 1, 3], &[1.0, 1.0, 2.0]);
    let mut r = SharedTensor::<T>::new(&[1, 1, 3]);
    let conf = LRN::<T>::new_lrn_config(&backend, 1u32, 1e-4f64, 0.75f64, 2f64)
        .unwrap();

    backend.lrn(&x, &mut r, &conf).unwrap();

    let r_test = [0.594581260843431, 0.594581260843431, 1.1890287651464355];
    tensor_assert_eq(&r, &r_test, 3.0);
}

pub fn test_lrn_grad<T, F: IFramework>(backend: Backend<F>)
    where T: Float + Epsilon + fmt::Debug,
          Backend<F>: LRN<T> + IBackend {

    let x  = filled_tensor(&[1, 1, 3], &[1.0, 1.0, 2.0]);
    let dx = filled_tensor(&[1, 1, 3], &[1.0, 1.0, 2.0]);
    let r  = filled_tensor(&[1, 1, 3], &[1.0, 1.0, 2.0]);
    let mut dr = SharedTensor::<T>::new(&[1, 1, 3]);

    let conf = LRN::<T>::new_lrn_config(&backend, 1u32, 1e-4f64, 0.75f64, 2f64)
        .unwrap();

    backend.lrn_grad(&x, &dx, &r, &mut dr, &conf).unwrap();

    let dr_test = [0.594536669478436, 0.594536669478436, 1.188672127844352];
    tensor_assert_eq(&dr, &dr_test, 3.0);
}


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
        2.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0,
        1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0,
        1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0,
        1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0];
    tensor_assert_eq(&dr, &dr_test, 3.0);
}

pub fn test_convolution<T, F: IFramework>(backend: Backend<F>)
    where T: Float + Epsilon + fmt::Debug,
          Backend<F>: Convolution<T> + IBackend {
    let batch = 4;
    let w1 = 9;
    let h1 = 9;
    let d1 = 3;
    let k = 6;
    let f = 3;
    let w2 = (w1 - f + 0) / 1;
    let h2 = (h1 - f + 0) / 1;

    let x_val = vec![1.0; batch * d1 * h1 * w1];
    let f_val = vec![1.0; k * d1 * f * f];

    let x  = filled_tensor(&[batch, d1, h1, w1], &x_val);
    let f  = filled_tensor(&[k, d1, f,  f], &f_val);
    let mut r  = SharedTensor::<T>::new(&[batch, k, h2, w2]);
    let mut ws = SharedTensor::<u8>::new(&[4]);

    let conf = backend.new_convolution_config(
        &x, &r, &f,
        ConvForwardAlgo::ImplicitGEMM,
        ConvBackwardFilterAlgo::ImplicitGEMM,
        ConvBackwardDataAlgo::ImplicitGEMM,
        &[1,1], &[0,0]).unwrap();

    backend.convolution(&f, &x, &mut r, &mut ws, &conf).unwrap();
}


// TODO
// pub fn test_convolution_grad<T, F: IFramework>(backend: Backend<F>)
//     where T: Float + Epsilon + fmt::Debug,
//           Backend<F>: Convolution<T> + IBackend {
//     let batch = 4;
//     let w1 = 9;
//     let h1 = 9;
//     let d1 = 3;
//     let k = 6;
//     let f = 3;
//     let w2 = (w1 - f + 0) / 1;
//     let h2 = (h1 - f + 0) / 1;
//
//     let mut x_val  = vec![1.0; batch * d1 * h1 * w1];
//     let mut dx_val = vec![1.0; batch * d1 * h1 * w1];
//     let mut f_val  = vec![1.0; k * d1 * f * f];
//     let mut r_val  = vec![1.0; batch * k * h2 * w2];
//     x_val[0] = 2.0;
//     f_val[0] = 2.0;
//
//     let x  = filled_tensor(&[batch, d1, h1, w1], &x_val);
//     let dx = filled_tensor(&[batch,  k, h2, w2], &x_val);
//     let f  = filled_tensor(&[k, d1, f,  f], &f_val);
//     let r  = filled_tensor(&[batch,  k, h2, w2], &f_val);
//     let mut dr  = SharedTensor::<T>::new(&[batch, k, h2, w2]);
// }

mod cuda {
    use super::*;
    test_cuda!(test_lrn, lrn_f32, lrn_f64);
    test_cuda!(test_lrn_grad, lrn_grad_f32, lrn_grad_f64);
    test_cuda!(test_pooling_max, pooling_max_f32, pooling_max_f64);
    test_cuda!(test_pooling_max_grad, pooling_max_grad_f32, pooling_max_grad_f64);
    test_cuda!(test_convolution, convolution_f32, convolution_f64);
}

