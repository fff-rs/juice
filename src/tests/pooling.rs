use std::fmt;

use co::prelude::*;
use co::plugin::numeric_helpers::Float;

use plugin::{Convolution, LRN, Pooling,
             ConvForwardAlgo, ConvBackwardFilterAlgo, ConvBackwardDataAlgo};
use tests::{Epsilon, filled_tensor, tensor_assert_eq};

// TODO re-enable this over the stupid version below
// TODO cuda verification for this one necessary
// pub fn test_pooling_max<T, F: IFramework>(backend: Backend<F>)
//     where T: Float + Epsilon + fmt::Debug,
//           Backend<F>: Pooling<T> + IBackend {

//     let test = |inp_dims: &[usize], out_dims: &[usize], window: &[i32], padding: &[i32], stride: &[i32] | {
//         let inp_size = (0..inp_dims.len()).fold(1, |mpy, x| mpy * inp_dims[x]);
//         let out_size = (0..out_dims.len()).fold(1, |mpy, x| mpy * out_dims[x]);
//         let mut inp = vec![1.0; inp_size];
//         inp[0] = 2.0;

        // FIXME TODO implement sanity checks in pooling_max
//         let x  = filled_tensor(&backend, inp_dims, &inp);
//         let mut r = SharedTensor::<T>::new(&out_dims);
//         let conf = Pooling::<T>::new_pooling_config(&backend, window, padding, stride)
//             .unwrap();

//         backend.pooling_max(&x, &mut r, &conf).unwrap();

//         let mut r_test = vec![1.0; out_size];
//         r_test[0] = 2.0;
//         tensor_assert_eq(&r, &r_test, 1.0);
//     };

    //       input dims   , output dims  ,  window, padding, stride
//     test(&[1, 1, 3, 3], &[1, 1, 2, 2], &[2, 2], &[0,0], &[1,1]);
//     test(&[1, 1, 10, 10], &[1, 1, 2, 2], &[9, 9], &[0,0], &[1,1]);
//     test(&[1, 1, 49, 49], &[1, 1, 7, 7], &[7, 7], &[0,0], &[7,7]);
//     test(&[1, 1, 4, 4], &[1, 1, 2, 2], &[2, 2], &[0,0], &[2,2]);
//     test(&[4, 1, 4, 4], &[4, 1, 2, 2], &[2, 2], &[0,0], &[2,2]);
//     test(&[1, 4, 4, 4], &[1, 4, 2, 2], &[2, 2], &[0,0], &[2,2]);
//     test(&[4, 4, 4, 4], &[4, 4, 3, 3], &[2, 2], &[1,1], &[2,2]);
// }

pub fn test_pooling_max<T, F: IFramework>(backend: Backend<F>)
    where T: Float + Epsilon + fmt::Debug,
          Backend<F>: Pooling<T> + IBackend {
    let mut inp = vec![1.0; 256];
    inp[0] = 2.0;

    let x  = filled_tensor(&backend,&[4, 4, 4, 4], &inp);
    let mut r = SharedTensor::<T>::new(&[4, 4, 2, 4]);
    let conf = Pooling::<T>::new_pooling_config(&backend, &[2, 2], &[0, 0], &[2, 1])
        .unwrap();

    backend.pooling_max(&x, &mut r, &conf).unwrap();

    let mut r_test = vec![1.0; 128];
    r_test[0] = 2.0;
    tensor_assert_eq(&r, &r_test, 3.0);
}

pub fn test_pooling_max_grad<T, F: IFramework>(backend: Backend<F>)
    where T: Float + Epsilon + fmt::Debug,
          Backend<F>: Pooling<T> + IBackend {
    let mut inp = vec![1.0; 256];
    inp[0] = 2.0;

    let x  = filled_tensor(&backend,&[4, 4, 4, 4], &inp);
    let dx = filled_tensor(&backend,&[4, 4, 4, 4], &inp);
    let r  = filled_tensor(&backend,&[4, 4, 2, 2], &inp[0..64]);
    let mut dr = SharedTensor::<T>::new(&[4, 4, 2, 2]);
    let conf = Pooling::<T>::new_pooling_config(&backend, &[2, 2], &[0, 0], &[2, 2])
        .unwrap();

    backend.pooling_max_grad(&x, &dx, &r, &mut dr, &conf).unwrap();

    let dr_test = [
        2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
    tensor_assert_eq(&dr, &dr_test, 3.0);
}

pub fn test_pooling_avg<T, F: IFramework>(backend: Backend<F>)
    where T: Float + Epsilon + fmt::Debug,
          Backend<F>: Pooling<T> + IBackend {
    let mut inp = vec![1.0; 256];
    inp[0] = 5.0;

    let x  = filled_tensor(&backend, &[4, 4, 4, 4], &inp);
    let mut r = SharedTensor::<T>::new(&[4, 4, 2, 2]);
    let conf = Pooling::<T>::new_pooling_config(&backend, &[2, 2], &[0, 0], &[2, 2])
        .unwrap();

    backend.pooling_avg(&x, &mut r, &conf).unwrap();

    let mut r_test = vec![1.0; 64];
    r_test[0] = 2.0;
    tensor_assert_eq(&r, &r_test, 3.0);
}


pub fn test_pooling_avg_grad<T, F: IFramework>(backend: Backend<F>)
    where T: Float + Epsilon + fmt::Debug,
          Backend<F>: Pooling<T> + IBackend {
    let mut inp = vec![1.0; 512];
    inp[0] = 2.0;

    let x  = filled_tensor(&backend, &[8, 4, 4, 4], &inp);
    let dx = filled_tensor(&backend, &[8, 4, 4, 4], &inp);
    let r  = filled_tensor(&backend, &[8, 4, 2, 2], &inp[0..128]);
    let mut dr = SharedTensor::<T>::new(&[8, 4, 2, 2]);
    let conf = Pooling::<T>::new_pooling_config(&backend, &[2, 2], &[0, 0], &[2, 2])
        .unwrap();

    backend.pooling_avg_grad(&x, &dx, &r, &mut dr, &conf).unwrap();

    let mut dr_test = vec![0.25; 128];
    dr_test[0] = 0.5;
    dr_test[1] = 0.5;
    dr_test[2] = 0.5;
    dr_test[3] = 0.5;
    tensor_assert_eq(&dr, &dr_test, 1.0);
}

mod cuda {
    use super::*;
    test_cuda!(test_pooling_avg, pooling_avg_f32, pooling_avg_f64);
    test_cuda!(test_pooling_avg_grad, pooling_avg_grad_f32, pooling_avg_grad_f64);
    test_cuda!(test_pooling_max, pooling_max_f32, pooling_max_f64);
    test_cuda!(test_pooling_max_grad, pooling_max_grad_f32, pooling_max_grad_f64);
}

mod native {
    use super::*;
    //test_native!(test_pooling_avg, pooling_avg_f32, pooling_avg_f64);
    //test_native!(test_pooling_avg_grad, pooling_avg_grad_f32, pooling_avg_grad_f64);
    test_native!(test_pooling_max, pooling_max_f32, pooling_max_f64);
    //test_native!(test_pooling_max_grad, pooling_max_grad_f32, pooling_max_grad_f64);
}
