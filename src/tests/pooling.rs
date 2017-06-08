use std::fmt;

use co::prelude::*;
use co::plugin::numeric_helpers::Float;

use plugin::Pooling;
use tests::{Epsilon, filled_tensor, tensor_assert_eq, tensor_assert_eq_tensor, uniformly_random_tensor};
use rand::distributions::{range, IndependentSample, Range};

pub fn test_pooling_max<T, F: IFramework>(backend: Backend<F>)
    where T: Float + Epsilon + fmt::Debug,
          Backend<F>: Pooling<T> + IBackend {

    let test = |inp_dims: &[usize], out_dims: &[usize], window: &[i32], padding: &[i32], stride: &[i32] | {
        let inp_size = (0..inp_dims.len()).fold(1, |mpy, x| mpy * inp_dims[x]);
        let out_size = (0..out_dims.len()).fold(1, |mpy, x| mpy * out_dims[x]);
        let mut inp = vec![1.0; inp_size];
        inp[0] = 2.0;

        let x  = filled_tensor(&backend, inp_dims, &inp);
        let mut r = SharedTensor::<T>::new(&out_dims);
        let conf = Pooling::<T>::new_pooling_config(&backend, window, padding, stride)
            .unwrap();

        backend.pooling_max(&x, &mut r, &conf).unwrap();

        let mut r_test = vec![1.0; out_size];
        r_test[0] = 2.0;
        tensor_assert_eq(&r, &r_test, 1.0);
    };

    //   input dims   , output dims  ,  window, stride, padding
    test(&[1, 1, 3, 3], &[1, 1, 2, 2], &[2, 2], &[1,1], &[0,0]);
    test(&[1, 1, 10, 10], &[1, 1, 2, 2], &[9, 9], &[1,1], &[0,0]);
    test(&[1, 1, 49, 49], &[1, 1, 7, 7], &[7, 7], &[7,7], &[0,0]);
    test(&[1, 1, 4, 4], &[1, 1, 2, 2], &[2, 2], &[2,2], &[0,0]);
    test(&[4, 1, 4, 4], &[4, 1, 2, 2], &[2, 2], &[2,2], &[0,0]);
    test(&[1, 4, 4, 4], &[1, 4, 2, 2], &[2, 2], &[2,2], &[0,0]);
    test(&[4, 4, 4, 4], &[4, 4, 3, 3], &[2, 2], &[2,2], &[1,1]);
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
    let conf = Pooling::<T>::new_pooling_config(&backend, &[2, 2], &[2, 2], &[0, 0])
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
    let conf = Pooling::<T>::new_pooling_config(&backend, &[2, 2], &[2, 2], &[0, 0])
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
    let conf = Pooling::<T>::new_pooling_config(&backend, &[2, 2], &[2, 2], &[0, 0])
        .unwrap();

    backend.pooling_avg_grad(&x, &dx, &r, &mut dr, &conf).unwrap();

    let mut dr_test = vec![0.25; 128];
    dr_test[0] = 0.5;
    dr_test[1] = 0.5;
    dr_test[2] = 0.5;
    dr_test[3] = 0.5;
    tensor_assert_eq(&dr, &dr_test, 1.0);
}

pub fn cross_test_pooling_max<F: IFramework, G: IFramework>(backend_a: Backend<F>, backend_b: Backend<G>)
        where
          Backend<F>: Pooling<f32> + IBackend,
          Backend<G>: Pooling<f32> + IBackend {

    let mut inp = vec![1.0; 192];
    inp[0] = 2.0;

    let stride = &[2, 1];
    let padding = &[0, 0];
    let window = &[2, 2];

    let dim_in = &[4, 3, 4, 4];
    let dim_out = &[4, 3, 2, 4];

    let lower : f32 = -128.;
    let upper : f32 = 127.;
    let x = uniformly_random_tensor(&backend_a, dim_in, lower, upper);

    let mut r_a = SharedTensor::<f32>::new(dim_out);
    let mut r_b = SharedTensor::<f32>::new(dim_out);

    let conf_a = Pooling::<f32>::new_pooling_config(&backend_a, window, stride, padding)
        .unwrap();
    let conf_b = Pooling::<f32>::new_pooling_config(&backend_b, window, stride, padding)
        .unwrap();

    backend_a.pooling_max(&x, &mut r_a, &conf_a).unwrap();
    backend_b.pooling_max(&x, &mut r_b, &conf_b).unwrap();
    tensor_assert_eq_tensor(&r_a, &r_b, 3.0);
}


pub fn cross_test_pooling_max_grad<F: IFramework, G: IFramework>(backend_a: Backend<F>, backend_b: Backend<G>)
        where
          Backend<F>: Pooling<f32> + IBackend,
          Backend<G>: Pooling<f32> + IBackend {

    let mut inp = vec![1.0; 256];
    inp[0] = 2.0;

    let batchsize = 1;
    let channels = 1;

    let input_dims = &[batchsize,channels,2,2];
    let window = &[2,2];
    let stride = &[2,2];
    let padding = &[0,0];
    // FIXME calc dynamically
    let output_dims = &[batchsize,channels,1,1];

    let N_in = input_dims.iter().fold(1,|a, &b| a * b);
    let N_out = output_dims.iter().fold(1,|a, &b| a * b);

    let x  = filled_tensor(&backend_a, input_dims, &inp[0..N_in]);
    let dx = filled_tensor(&backend_a, input_dims, &inp[0..N_in]);
    let r  = filled_tensor(&backend_a, output_dims, &inp[0..N_out]);
    let mut dr_a = SharedTensor::<f32>::new(output_dims);
    let mut dr_b = SharedTensor::<f32>::new(output_dims);
    let conf_a = Pooling::<f32>::new_pooling_config(&backend_a, window, stride, padding)
        .unwrap();
    let conf_b = Pooling::<f32>::new_pooling_config(&backend_b, window, stride, padding)
        .unwrap();

//    backend_a.pooling_max_grad(&x, &dx, &r, &mut dr_a, &conf_a).unwrap();
    backend_b.pooling_max_grad(&x, &dx, &r, &mut dr_b, &conf_b).unwrap();

//    tensor_assert_eq_tensor(&dr_a, &dr_b, 3.0);
}

mod cross {
    use super::*;
    test_cross!(cross_test_pooling_max, cross_test_pooling_max_f32);
    test_cross!(cross_test_pooling_max_grad, cross_test_pooling_max_grad_f32);
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
