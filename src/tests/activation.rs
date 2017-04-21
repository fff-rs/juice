// Code for relu, sigmoid and tanh is mostly the same, but ATM I can't get
// how to abstract it better. Generic function wouldn't be shorter... Macros
// would be, but they would have to accept ~15 parameters and that is qute
// evil by itself and they'd add another level of indirection. Not nice.
use std::fmt;

use co::prelude::*;
use co::plugin::numeric_helpers::Float;

use plugin::{Relu, ReluPointwise, Sigmoid, SigmoidPointwise, Tanh, TanhPointwise};
use tests::{Epsilon, filled_tensor, tensor_assert_eq};

const DIMS:   [usize; 4] = [3, 1, 2, 2];

const IN: [f64; 12] =
    [1.121623378076182407, 0.562888119501944841, -1.339156477386188037,
     0.8759488434687463583, 0.5710683496214725947, 0.1198723930562685942,
     -0.3748904319909696, 0.2090742138343960133, -0.6626539528423519309,
     -0.918982785419555966, 1.402159805804972244, -1.978255365302346012];

const OUT_GRAD: [f64; 12] =
    [-2.332776404488865508, 1.70589003183088233, -1.639385156921041195,
     0.06062355027829264628, -2.98757598356132714, 2.299513994512549636,
     1.47030623613516523, 2.225557495134344654, -0.4007462184938826337,
     2.815467050105664459, 2.709297453597423977, -2.895567849550241764];

const RELU_OUT: [f64; 12] =
    [1.121623378076182407, 0.562888119501944841, 0.0,
     0.8759488434687463583, 0.5710683496214725947, 0.1198723930562685942,
     0.0, 0.2090742138343960133, 0.0,
     0.0, 1.402159805804972244, 0.0];

const RELU_IN_GRAD: [f64; 12] =
    [-2.332776404488865508, 1.70589003183088233, 0.0,
     0.06062355027829264628, -2.98757598356132714, 2.299513994512549636,
     0.0, 2.225557495134344654, 0.0,
     0.0, 2.709297453597423977, 0.0];

const SIGMOID_OUT: [f64; 12] =
    [0.7542897122404972129, 0.6371205316919988361, 0.2076488097696117026,
     0.7059820190977259747, 0.6390096546850780026, 0.5299322644783190903,
     0.4073598514454171755, 0.5520789850230285882, 0.3401436900840779611,
     0.2851652041111022745, 0.802526393449478207, 0.1215049398609489357];

const SIGMOID_IN_GRAD: [f64; 12] =
    [-0.432349179202473982, 0.3943982949828453617, -0.2697293211639109796,
     0.0125837156776099823, -0.6891630213721981575, 0.5728182710094757442,
     0.3549581010823164556, 0.5503531707184640608, -0.08994586979838948354,
     0.5739217257889142655, 0.4293634492369666151, -0.3090772250654995713];

const TANH_OUT: [f64; 12] =
    [0.8081328403503516179, 0.5101171538136796331, -0.8714694945764568607,
     0.7043839918457185048, 0.5161434470638732663, 0.1193015097168258505,
     -0.3582618973187905141, 0.2060802001687216763, -0.5801268964170599928,
     -0.7254158427301376402, 0.8858176038189352612, -0.9624586626492133626];

const TANH_IN_GRAD: [f64; 12] =
    [-0.8092898516580304142, 1.261984162584895754, -0.3943392139172716021,
     0.0305447630844965373, -2.191673618115657509, 2.266785356248217009,
     1.28159009724762284, 2.131040185040205402, -0.2658761943586825486,
     1.333889047346882493, 0.5833853608611378429, -0.2133261045550120892];


//----------------------------------------------------------- relu

pub fn test_relu<T, F: IFramework>(backend: Backend<F>)
    where T: Float + Epsilon + fmt::Debug,
          Backend<F>: Relu<T> + IBackend {

    let x  = filled_tensor(&backend, &DIMS, &IN);
    let mut r = SharedTensor::<T>::new(&DIMS);

    backend.relu(&x, &mut r).unwrap();
    tensor_assert_eq(&r, &RELU_OUT, 3.0);
}

pub fn test_relu_grad<T, F: IFramework>(backend: Backend<F>)
    where T: Float + Epsilon + fmt::Debug,
          Backend<F>: Relu<T> + IBackend {

    let x  = filled_tensor(&backend, &DIMS, &RELU_OUT);
    let dx = filled_tensor(&backend, &DIMS, &OUT_GRAD);
    let r  = filled_tensor(&backend, &DIMS, &IN);
    let mut dr = SharedTensor::new(&DIMS);

    backend.relu_grad(&x, &dx, &r, &mut dr).unwrap();
    tensor_assert_eq(&dr, &RELU_IN_GRAD, 3.0);
}

pub fn test_relu_pointwise<T, F: IFramework>(backend: Backend<F>)
    where T: Float + fmt::Debug + Epsilon,
          Backend<F>: ReluPointwise<T> + IBackend {

    let mut x = filled_tensor(&backend, &DIMS, &IN);
    backend.relu_pointwise(&mut x).unwrap();
    tensor_assert_eq(&x, &RELU_OUT, 3.0);
}

pub fn test_relu_pointwise_grad<T, F: IFramework>(backend: Backend<F>)
    where T: Float + fmt::Debug + Epsilon,
          Backend<F>: ReluPointwise<T> + IBackend {
    let      x = filled_tensor(&backend, &DIMS, &RELU_OUT);
    let mut dx = filled_tensor(&backend, &DIMS, &OUT_GRAD);
    backend.relu_pointwise_grad(&x, &mut dx).unwrap();
    tensor_assert_eq(&dx, &RELU_IN_GRAD, 3.0);
}

//----------------------------------------------------------- sigmoid

pub fn test_sigmoid<T, F: IFramework>(backend: Backend<F>)
    where T: Float + Epsilon + fmt::Debug,
          Backend<F>: Sigmoid<T> + IBackend {

    let x  = filled_tensor(&backend, &DIMS, &IN);
    let mut r = SharedTensor::<T>::new(&DIMS);

    backend.sigmoid(&x, &mut r).unwrap();
    tensor_assert_eq(&r, &SIGMOID_OUT, 3.0);
}

pub fn test_sigmoid_grad<T, F: IFramework>(backend: Backend<F>)
    where T: Float + Epsilon + fmt::Debug,
          Backend<F>: Sigmoid<T> + IBackend {

    let x  = filled_tensor(&backend, &DIMS, &SIGMOID_OUT);
    let dx = filled_tensor(&backend, &DIMS, &OUT_GRAD);
    let r  = filled_tensor(&backend, &DIMS, &IN);
    let mut dr = SharedTensor::new(&DIMS);

    backend.sigmoid_grad(&x, &dx, &r, &mut dr).unwrap();
    tensor_assert_eq(&dr, &SIGMOID_IN_GRAD, 3.0);
}

pub fn test_sigmoid_pointwise<T, F: IFramework>(backend: Backend<F>)
    where T: Float + fmt::Debug + Epsilon,
          Backend<F>: SigmoidPointwise<T> + IBackend {

    let mut x = filled_tensor(&backend, &DIMS, &IN);
    backend.sigmoid_pointwise(&mut x).unwrap();
    tensor_assert_eq(&x, &SIGMOID_OUT, 3.0);
}

pub fn test_sigmoid_pointwise_grad<T, F: IFramework>(backend: Backend<F>)
    where T: Float + fmt::Debug + Epsilon,
          Backend<F>: SigmoidPointwise<T> + IBackend {
    let      x = filled_tensor(&backend, &DIMS, &SIGMOID_OUT);
    let mut dx = filled_tensor(&backend, &DIMS, &OUT_GRAD);
    backend.sigmoid_pointwise_grad(&x, &mut dx).unwrap();
    tensor_assert_eq(&dx, &SIGMOID_IN_GRAD, 3.0);
}

//----------------------------------------------------------- sigmoid

pub fn test_tanh<T, F: IFramework>(backend: Backend<F>)
    where T: Float + Epsilon + fmt::Debug,
          Backend<F>: Tanh<T> + IBackend {

    let x  = filled_tensor(&backend, &DIMS, &IN);
    let mut r = SharedTensor::<T>::new(&DIMS);

    backend.tanh(&x, &mut r).unwrap();
    tensor_assert_eq(&r, &TANH_OUT, 3.0);
}

pub fn test_tanh_grad<T, F: IFramework>(backend: Backend<F>)
    where T: Float + Epsilon + fmt::Debug,
          Backend<F>: Tanh<T> + IBackend {

    let x  = filled_tensor(&backend, &DIMS, &TANH_OUT);
    let dx = filled_tensor(&backend, &DIMS, &OUT_GRAD);
    let r  = filled_tensor(&backend, &DIMS, &IN);
    let mut dr = SharedTensor::new(&DIMS);

    backend.tanh_grad(&x, &dx, &r, &mut dr).unwrap();
    tensor_assert_eq(&dr, &TANH_IN_GRAD, 10.0);
}

pub fn test_tanh_pointwise<T, F: IFramework>(backend: Backend<F>)
    where T: Float + fmt::Debug + Epsilon,
          Backend<F>: TanhPointwise<T> + IBackend {

    let mut x = filled_tensor(&backend, &DIMS, &IN);
    backend.tanh_pointwise(&mut x).unwrap();
    tensor_assert_eq(&x, &TANH_OUT, 3.0);
}

pub fn test_tanh_pointwise_grad<T, F: IFramework>(backend: Backend<F>)
    where T: Float + fmt::Debug + Epsilon,
          Backend<F>: TanhPointwise<T> + IBackend {
    let      x = filled_tensor(&backend, &DIMS, &TANH_OUT);
    let mut dx = filled_tensor(&backend, &DIMS, &OUT_GRAD);
    backend.tanh_pointwise_grad(&x, &mut dx).unwrap();
    tensor_assert_eq(&dx, &TANH_IN_GRAD, 10.0);
}


mod native {
    use super::*;
    test_native!(test_relu, relu_f32, relu_f64);
    test_native!(test_relu_grad, relu_grad_f32, relu_grad_f64);
    test_native!(test_relu_pointwise, relu_pointwise_f32, relu_pointwise_f64);
    test_native!(test_relu_pointwise_grad,
               relu_pointwise_grad_f32, relu_pointwise_grad_f64);

    test_native!(test_sigmoid, sigmoid_f32, sigmoid_f64);
    test_native!(test_sigmoid_grad, sigmoid_grad_f32, sigmoid_grad_f64);
    test_native!(test_sigmoid_pointwise, sigmoid_pointwise_f32, sigmoid_pointwise_f64);
    test_native!(test_sigmoid_pointwise_grad,
               sigmoid_pointwise_grad_f32, sigmoid_pointwise_grad_f64);

    test_native!(test_tanh, tanh_f32, tanh_f64);
    test_native!(test_tanh_grad, tanh_grad_f32, tanh_grad_f64);
    test_native!(test_tanh_pointwise, tanh_pointwise_f32, tanh_pointwise_f64);
    test_native!(test_tanh_pointwise_grad,
               tanh_pointwise_grad_f32, tanh_pointwise_grad_f64);
}

mod cuda {
    use super::*;
    test_cuda!(test_relu, relu_f32, relu_f64);
    test_cuda!(test_relu_grad, relu_grad_f32, relu_grad_f64);
    test_cuda!(test_relu_pointwise, relu_pointwise_f32, relu_pointwise_f64);
    test_cuda!(test_relu_pointwise_grad,
               relu_pointwise_grad_f32, relu_pointwise_grad_f64);

    test_cuda!(test_sigmoid, sigmoid_f32, sigmoid_f64);
    test_cuda!(test_sigmoid_grad, sigmoid_grad_f32, sigmoid_grad_f64);
    test_cuda!(test_sigmoid_pointwise, sigmoid_pointwise_f32, sigmoid_pointwise_f64);
    test_cuda!(test_sigmoid_pointwise_grad,
               sigmoid_pointwise_grad_f32, sigmoid_pointwise_grad_f64);

    test_cuda!(test_tanh, tanh_f32, tanh_f64);
    test_cuda!(test_tanh_grad, tanh_grad_f32, tanh_grad_f64);
    test_cuda!(test_tanh_pointwise, tanh_pointwise_f32, tanh_pointwise_f64);
    test_cuda!(test_tanh_pointwise_grad,
               tanh_pointwise_grad_f32, tanh_pointwise_grad_f64);
}
