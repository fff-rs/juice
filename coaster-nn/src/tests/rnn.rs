use std::fmt;

use coaster as co;
use co::prelude::*;

use crate::{Rnn, co::plugin::numeric_helpers::Float};
use crate::plugin::{
    DirectionMode,
    RnnAlgorithm, RnnConfig, RnnInputMode, RnnNetworkMode, RnnPaddingMode,
};
use crate::tests::{filled_tensor, Epsilon};


pub fn test_rnn<T, F: IFramework>(backend: Backend<F>)
where
    T: Float + Epsilon + fmt::Debug,
    Backend<F>: Rnn<T> + IBackend,
{

    let _forward_mode = rcudnn::cudnnForwardMode_t::CUDNN_FWD_MODE_TRAINING;
    let direction_mode = DirectionMode::UniDirectional;
    let _bidirectional = if direction_mode == DirectionMode::UniDirectional
    {
        1
    } else {
        2 // bidirection needs twice as much memory
    };
    let network_mode = RnnNetworkMode::LSTM;
    let algorithm = RnnAlgorithm::Standard;
    let input_mode = RnnInputMode::LinearInput;
    let sequence_length = 7;
    let hidden_size = 5;
    let num_layers = 3;
    let batch_size = 2;
    let input_size = 11;

    let dropout_probability = Some(0.05);
    let dropout_seed = Some(27_u64);

    let _x = filled_tensor::<T,_>(&backend, &[1, 1, 3], &[1.0, 1.0, 2.0]);
    let src = SharedTensor::<T>::new(&[num_layers, batch_size, input_size]);
    let mut output = SharedTensor::<T>::new(&[num_layers, batch_size, hidden_size]);
    let _weight = SharedTensor::<T>::new(&[1, 1, 3]);
    let mut workspace = SharedTensor::<u8>::new(&[1, 1, 3]);

    let rnn_config = backend.new_rnn_config(
        &src,
        dropout_probability,
        dropout_seed,
        sequence_length,
        network_mode,
        input_mode,
        direction_mode,
        algorithm,
        hidden_size as i32,
        num_layers as i32,
        batch_size as i32,
    ).unwrap();

    let filter_dimensions = backend.generate_rnn_weight_description(
        &rnn_config,
        batch_size as i32,
        hidden_size as i32,
    )
    .unwrap();

    let w = SharedTensor::<T>::new(&filter_dimensions);
    let _dw = SharedTensor::<T>::new(&filter_dimensions);

    // let filler = FillerType::Constant { value: 0.02 };

    // filler.fill(&mut weights[0]);
    // filler.fill(&mut weights[1]);

    backend.rnn_forward(
        &src,
        &mut output,
        &rnn_config,
        &w,
        &mut workspace,
    ).unwrap();

    // conf.rnn_backward_weights(src, output, filter, rnn_config, workspace).unwrap();
    // conf.rnn_backward_data(src, src_gradient, output, output_gradient, rnn_config, weight, workspace).unwrap();

}

mod cuda {
    use super::*;
    test_cuda!(test_rnn, rnn_f32, rnn_f64);
}
