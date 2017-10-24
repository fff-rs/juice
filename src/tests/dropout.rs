use std::fmt;

use co::prelude::*;
use co::plugin::numeric_helpers::Float;

use plugin::Dropout;
use tests::{Epsilon, filled_tensor, tensor_assert_eq, tensor_assert_eq_tensor, tensor_assert_ne_tensor};

pub fn test_dropout<T, F: IFramework>(backend: Backend<F>)
    where T: Float + Epsilon + fmt::Debug,
          Backend<F>: Dropout<T> + IBackend {

    let test = |dims : &[usize],
                probability : f32,
                seed : u64,
                tensor_assert : &Fn(&SharedTensor<T>, &SharedTensor<T>, f64) | {

        let conf = Dropout::<T>::new_dropout_config(&backend, probability, seed)
            .unwrap();

        let inp_element_num = dims.iter().fold(1, |factorial, f| factorial * f );

        let mut inp_vals : Vec<f64> = (0..inp_element_num).map(|i| (i*i) as f64).collect();

        let x  = filled_tensor(&backend, dims, &inp_vals);
        let mut r = SharedTensor::<T>::new(&dims);

        backend.dropout(&x,
               &mut r,
               &conf).unwrap();

        tensor_assert(&x, &r, 0.0); // TODO should not fail? or should?
    };

    test(&[1, 5, 5, 2], 0.999, 77777, &tensor_assert_ne_tensor);
    test(&[1, 1, 1, 1], 0.000, 77777, &tensor_assert_eq_tensor);
    test(&[5, 200, 200, 4], 0.5, 77777, &tensor_assert_ne_tensor);
}


// TODO
// pub fn test_dropout_grad<T, F: IFramework>(backend: Backend<F>)
//     where T: Float + Epsilon + fmt::Debug,
//           Backend<F>: Dropout<T> + IBackend {
//
// }

mod cuda {
    use super::*;
    test_cuda!(test_dropout, dropout_f32, dropout_f64);
    // TODO test_cuda!(test_dropout_grad, dropout_grad_f32, dropout_grad_f64);
}

mod native {
    use super::*;
    test_native!(test_dropout, dropout_f32, dropout_f64);
    // TODO test_native!(test_dropout_grad, dropout_grad_f32, dropout_grad_f64);
}
