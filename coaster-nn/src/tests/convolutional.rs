use std::fmt;

use crate::co::plugin::numeric_helpers::Float;
use crate::co::prelude::*;

use crate::plugin::{
    ConvBackwardDataAlgo, ConvBackwardFilterAlgo, ConvForwardAlgo, Convolution, ConvolutionConfig,
    LRN,
};
use crate::tests::{filled_tensor, tensor_assert_eq, tensor_assert_eq_tensor, Epsilon};

pub fn test_lrn<T, F: IFramework>(backend: Backend<F>)
where
    T: Float + Epsilon + fmt::Debug,
    Backend<F>: LRN<T> + IBackend,
{
    let x = filled_tensor(&backend, &[1, 1, 3], &[1.0, 1.0, 2.0]);
    let mut r = SharedTensor::<T>::new(&[1, 1, 3]);
    let conf = LRN::<T>::new_lrn_config(&backend, 1u32, 1e-4f64, 0.75f64, 2f64).unwrap();

    backend.lrn(&x, &mut r, &conf).unwrap();

    let r_test = [0.594581260843431, 0.594581260843431, 1.1890287651464355];
    tensor_assert_eq(&r, &r_test, 3.0);
}

pub fn test_lrn_grad<T, F: IFramework>(backend: Backend<F>)
where
    T: Float + Epsilon + fmt::Debug,
    Backend<F>: LRN<T> + IBackend,
{
    let x = filled_tensor(&backend, &[1, 1, 3], &[1.0, 1.0, 2.0]);
    let dx = filled_tensor(&backend, &[1, 1, 3], &[1.0, 1.0, 2.0]);
    let r = filled_tensor(&backend, &[1, 1, 3], &[1.0, 1.0, 2.0]);
    let mut dr = SharedTensor::<T>::new(&[1, 1, 3]);

    let conf = LRN::<T>::new_lrn_config(&backend, 1u32, 1e-4f64, 0.75f64, 2f64).unwrap();

    backend.lrn_grad(&x, &dx, &r, &mut dr, &conf).unwrap();

    let dr_test = [0.594536669478436, 0.594536669478436, 1.188672127844352];
    tensor_assert_eq(&dr, &dr_test, 3.0);
}

pub fn test_convolution<T, F: IFramework>(backend: Backend<F>)
where
    T: Float + Epsilon + fmt::Debug,
    Backend<F>: Convolution<T> + IBackend,
{
    let test = |input_dim: &[usize; 4],
                filter_count: usize,
                filter_size: &[usize; 2],
                stride: &[usize; 2],
                padding: &[usize; 2]| {
        let batch = input_dim[0];
        let width = input_dim[1];
        let height = input_dim[2];
        let depth = input_dim[3];

        let result_width = (width + 2 * padding[0] - filter_size[0]) / stride[0] + 1;
        let result_height = (height + 2 * padding[1] - filter_size[1]) / stride[1] + 1;

        let f_element_count = filter_count * depth * filter_size[0] * filter_size[1];

        let x_val = vec![1.0; batch * depth * height * width];
        let f_val = vec![1.0; f_element_count];

        let x = filled_tensor(&backend, &[batch, depth, height, width], &x_val);
        let f = filled_tensor(
            &backend,
            &[filter_count, depth, filter_size[0], filter_size[1]],
            &f_val,
        );
        let mut r = SharedTensor::<T>::new(&[batch, filter_count, result_height, result_width]);

        let conf = backend
            .new_convolution_config(
                &x,
                &r,
                &f,
                ConvForwardAlgo::Auto,
                ConvBackwardFilterAlgo::Auto,
                ConvBackwardDataAlgo::Auto,
                &[stride[0] as i32, stride[1] as i32],
                &[padding[0] as i32, padding[1] as i32],
            )
            .unwrap();

        let mut ws = SharedTensor::<u8>::new(&[conf.workspace_size()]);

        assert!(backend.convolution(&f, &x, &mut r, &mut ws, &conf).is_ok());
        assert!(r.read(backend.device()).is_ok());

        // this only works because our data is all ones, if padding is non zero, this can not be applied
        let expected_val_count = batch * filter_count * result_height * result_width;
        let expected_val = depth * filter_size[0] * filter_size[1];
        let expected_val = expected_val as f64;
        let expected_vals: Vec<f64> = vec![expected_val; expected_val_count];
        let expected: SharedTensor<T> = filled_tensor(
            &backend,
            &[batch, filter_count, result_height, result_width],
            expected_vals.as_slice(),
        );

        tensor_assert_eq_tensor(&r, &expected, 3.0);
    };
    // [batchsize, width, height, depth], k_filters, [filter_size_x, filter_size_y], stride, padding
    test(&[4, 9, 9, 3], 6, &[3, 3], &[1, 1], &[0, 0]);
    test(&[2, 16, 16, 1], 1, &[4, 4], &[1, 1], &[0, 0]);
    test(&[2, 16, 16, 1], 1, &[2, 2], &[1, 1], &[0, 0]);
    test(&[2, 16, 16, 10], 10, &[2, 2], &[1, 1], &[0, 0]);
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
//     let x  = filled_tensor(&backend,&[batch, d1, h1, w1], &x_val);
//     let dx = filled_tensor(&backend,&[batch,  k, h2, w2], &x_val);
//     let f  = filled_tensor(&backend,&[k, d1, f,  f], &f_val);
//     let r  = filled_tensor(&backend,&[batch,  k, h2, w2], &f_val);
//     let mut dr  = SharedTensor::<T>::new(&[batch, k, h2, w2]);
// }

fn cross_test_convolution<F: IFramework, G: IFramework>(
    backend_a: Backend<F>,
    backend_b: Backend<G>,
) where
    Backend<F>: Convolution<f32> + IBackend,
    Backend<G>: Convolution<f32> + IBackend,
{
    // TODO add stride and padding
    // TODO use a slice for filtersize and k_filters
    let batch = 4;
    let width1 = 9;
    let height1 = 9;
    let depth1 = 3;
    let padding = &[0i32, 0i32];
    let stride = &[1i32, 1i32];
    let filter_size = 6;
    let filter_count = 3;

    let result_width =
        (width1 - filter_size + 2 * (padding[0]) as usize) / (stride[0] as usize) + 1;
    let result_height =
        (height1 - filter_size + 2 * (padding[1]) as usize) / (stride[1] as usize) + 1;

    let x_val = vec![1.0; batch * depth1 * height1 * width1];
    let f_val = vec![1.0; filter_count * depth1 * filter_size * filter_size];

    let x = filled_tensor(&backend_a, &[batch, depth1, height1, width1], &x_val);
    let f = filled_tensor(
        &backend_a,
        &[filter_count, depth1, filter_size, filter_size],
        &f_val,
    );
    let mut result_a =
        SharedTensor::<f32>::new(&[batch, filter_count, result_height, result_width]);
    let mut result_b =
        SharedTensor::<f32>::new(&[batch, filter_count, result_height, result_width]);

    let conf_a = backend_a
        .new_convolution_config(
            &x,
            &result_a,
            &f,
            ConvForwardAlgo::Auto,
            ConvBackwardFilterAlgo::Auto,
            ConvBackwardDataAlgo::Auto,
            stride,
            padding,
        )
        .unwrap();

    let mut ws = SharedTensor::<u8>::new(&[conf_a.workspace_size()]);

    backend_a
        .convolution(&f, &x, &mut result_a, &mut ws, &conf_a)
        .unwrap();

    let conf_b = backend_b
        .new_convolution_config(
            &x,
            &result_b,
            &f,
            ConvForwardAlgo::Auto,
            ConvBackwardFilterAlgo::Auto,
            ConvBackwardDataAlgo::Auto,
            stride,
            padding,
        )
        .unwrap();

    let mut ws = SharedTensor::<u8>::new(&[conf_b.workspace_size()]);

    backend_b
        .convolution(&f, &x, &mut result_b, &mut ws, &conf_b)
        .unwrap();

    tensor_assert_eq_tensor(&result_a, &result_b, 3.0);
}

mod cuda {
    use super::*;
    test_cuda!(test_lrn, lrn_f32, lrn_f64);
    test_cuda!(test_lrn_grad, lrn_grad_f32, lrn_grad_f64);
    test_cuda!(test_convolution, convolution_f32, convolution_f64);
}

mod native {
    use super::*;
    //test_native!(test_lrn, lrn_f32, lrn_f64);
    //test_native!(test_lrn_grad, lrn_grad_f32, lrn_grad_f64);
    test_native!(test_convolution, convolution_f32, convolution_f64);
}

mod cross {
    use super::*;
    test_cross!(cross_test_convolution, cross_test_convolution_f32);
}
