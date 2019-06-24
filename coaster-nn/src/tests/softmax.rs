use std::fmt;

use crate::co::prelude::*;
use crate::co::plugin::numeric_helpers::Float;

use crate::plugin::{Softmax, LogSoftmax};
use crate::tests::{Epsilon, filled_tensor, tensor_assert_eq, tensor_assert_eq_tensor};

const DIMS: [usize; 3] = [4, 1, 3];

const IN: [f64; 12] =
    [-0.3768541784373798341, -0.4190485384650235847, 0.5958971899345203651,
     1.201292917640018342, -0.2406155214817796814, -0.1324849200097359183,
     0.01328099434291760409, -0.581962897607930672, -0.5905963672681562759,
     -0.9211015102408774548, -1.368822998145939182, 0.8509696368242991619];

const OUT_GRAD: [f64; 12] =
    [-2.403764079434107295, 3.555336738840519548, -2.288944264898976203,
     1.969619340429111837, 2.804058445190456017, 1.407220298754862102,
     -3.347891193465470093, 2.189872108671865896, 1.427670874053681487,
     0.2996809826406714856, -0.937226079977424, 0.5226855345859900333];

const SOFTMAX_OUT: [f64; 12] =
    [0.05171473742304219928, 0.04957806199520956396, 0.1367964652178650133,
     0.2506079819996082626, 0.05926278008468241267, 0.06603018961553208981,
     0.07639199935558898925, 0.042124692050059925, 0.04176257521092115797,
     0.0300089438519342605, 0.01917819540548354381, 0.1765433777900725818];

const SOFTMAX_IN_GRAD: [f64; 12] =
    [-0.1486650380754509047, 0.1529179607757129824, -0.3775436589037359082,
     0.3755787205930539972, 0.1382665450812886289, 0.06182216371989694205,
     -0.2917288464701554001, 0.07240910171835037837, 0.03995516437097136814,
     -0.005139575801680075536, -0.02700625907621399834, 0.009133722067961989762];

const LOG_SOFTMAX_OUT: [f64; 12] =
    [-2.962012481550115532, -3.004206841577759283, -1.989261113178215333,
     -1.383865385472717357, -2.82577382459451538, -2.717643223122471617,
     -2.571877308769818094, -3.16712120072066637, -3.175754670380891974,
     -3.506259813353613153, -3.953981301258674881, -1.734188666288436536];

const LOG_SOFTMAX_IN_GRAD: [f64; 12] =
    [-2.672593766324700871, 3.297614171793778065, -3.000055888872397954,
     0.6668791798802118677, 2.495991626942767485, 1.063974328955915138,
     -3.745001152657859367, 1.97089453402919622, 1.210575698149443993,
     0.1436849284861279457, -1.036920451889396308, -0.3950432084930862146];


pub fn test_softmax<T, F: IFramework>(backend: Backend<F>)
    where T: Float + Epsilon + fmt::Debug,
          Backend<F>: Softmax<T> + IBackend {

    let x  = filled_tensor(&backend, &DIMS, &IN);
    let mut r = SharedTensor::<T>::new(&DIMS);

    backend.softmax(&x, &mut r).unwrap();
    tensor_assert_eq(&r, &SOFTMAX_OUT, 3.0);
}

pub fn test_softmax_grad<T, F: IFramework>(backend: Backend<F>)
    where T: Float + Epsilon + fmt::Debug,
          Backend<F>: Softmax<T> + IBackend {

    let x  = filled_tensor(&backend, &DIMS, &SOFTMAX_OUT);
    let dx = filled_tensor(&backend, &DIMS, &OUT_GRAD);
    let mut dr = SharedTensor::new(&DIMS);

    backend.softmax_grad(&x, &dx, &mut dr).unwrap();
    tensor_assert_eq(&dr, &SOFTMAX_IN_GRAD, 10.0);
}

pub fn test_log_softmax<T, F: IFramework>(backend: Backend<F>)
    where T: Float + Epsilon + fmt::Debug,
          Backend<F>: LogSoftmax<T> + IBackend {

    let x  = filled_tensor(&backend, &DIMS, &IN);
    let mut r = SharedTensor::<T>::new(&DIMS);

    backend.log_softmax(&x, &mut r).unwrap();
    tensor_assert_eq(&r, &LOG_SOFTMAX_OUT, 3.0);
}

pub fn test_log_softmax_grad<T, F: IFramework>(backend: Backend<F>)
    where T: Float + Epsilon + fmt::Debug,
          Backend<F>: LogSoftmax<T> + IBackend {

    let x  = filled_tensor(&backend, &DIMS, &LOG_SOFTMAX_OUT);
    let dx = filled_tensor(&backend, &DIMS, &OUT_GRAD);
    let mut dr = SharedTensor::new(&DIMS);

    backend.log_softmax_grad(&x, &dx, &mut dr).unwrap();
    tensor_assert_eq(&dr, &LOG_SOFTMAX_IN_GRAD, 10.0);
}

pub fn cross_test_log_softmax_grad<F: IFramework, G: IFramework>(backend_a: Backend<F>, backend_b: Backend<G>)
    where Backend<F>: LogSoftmax<f32> + IBackend,
          Backend<G>: LogSoftmax<f32> + IBackend {

    let x  = filled_tensor(&backend_a, &DIMS, &LOG_SOFTMAX_OUT);
    let dx = filled_tensor(&backend_a, &DIMS, &OUT_GRAD);
    let mut dr_a = SharedTensor::new(&DIMS);
    let mut dr_b = SharedTensor::new(&DIMS);

    {
        backend_a.log_softmax_grad(&x, &dx, &mut dr_b).unwrap();
    }
    {
        backend_b.log_softmax_grad(&x, &dx, &mut dr_a).unwrap();
    }
    tensor_assert_eq_tensor(&dr_a, &dr_b, 10.0);
}

mod cross {
    use super::*;
    test_cross!(cross_test_log_softmax_grad, cross_test_log_softmax_grad_f32);
}

mod native {
    use super::*;
    test_native!(test_softmax, softmax_f32, softmax_f64);
    test_native!(test_softmax_grad, softmax_grad_f32, softmax_grad_f64);
    test_native!(test_log_softmax, log_softmax_f32, log_softmax_f64);
    test_native!(test_log_softmax_grad, log_softmax_grad_f32, log_softmax_grad_f64);
}

mod cuda {
    use super::*;
    test_cuda!(test_softmax, softmax_f32, softmax_f64);
    test_cuda!(test_softmax_grad, softmax_grad_f32, softmax_grad_f64);
    test_cuda!(test_log_softmax, log_softmax_f32, log_softmax_f64);
    test_cuda!(test_log_softmax_grad, log_softmax_grad_f32, log_softmax_grad_f64);
}
