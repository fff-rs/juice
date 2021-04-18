use std::fmt;

use co::prelude::*;
use coaster as co;

use crate::plugin::{
    self, DirectionMode, RnnAlgorithm, RnnConfig, RnnInputMode, RnnNetworkMode, RnnPaddingMode,
};
use crate::tests::{filled_tensor, uniformly_random_tensor, Epsilon, One, Zero};
use crate::{co::plugin::numeric_helpers::Float, Rnn};

pub fn test_rnn<T, F: IFramework>(backend: Backend<F>)
where
    T: Float + Epsilon + One + Zero + rand::distributions::uniform::SampleUniform + fmt::Debug,
    Backend<F>: Rnn<T> + IBackend,
{
    let direction_mode = DirectionMode::UniDirectional;
    let network_mode = RnnNetworkMode::LSTM;
    let algorithm = RnnAlgorithm::Standard;
    let input_mode = RnnInputMode::LinearInput;
    let sequence_length = 7;
    let hidden_size = 5;
    let num_layers = 3;
    let batch_size = 2;
    let input_size = 11;

    let dropout_probability = Some(0.05_f32);
    let dropout_seed = Some(27_u64);

    let src = filled_tensor::<T, F>(
        &backend,
        &[batch_size, input_size, 1],
        &vec![1.0f64; input_size * batch_size],
    );

    let mut output = SharedTensor::<T>::new(&[batch_size, hidden_size, 1]);

    let rnn_config = backend
        .new_rnn_config(
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
        )
        .unwrap();

    let filter_dimensions = backend
        .generate_rnn_weight_description(&rnn_config, batch_size as i32, input_size as i32)
        .unwrap();

    let w = uniformly_random_tensor::<T, F>(
        &backend,
        &filter_dimensions,
        <T as Zero>::zero(),
        <T as One>::one(),
    );
    let mut dw = SharedTensor::<T>::new(&filter_dimensions);

    let workspace_size = rnn_config.workspace_size();
    assert_ne!(workspace_size, 0);
    let mut workspace = SharedTensor::<u8>::new(&[1, 1, workspace_size]);


    backend
        .rnn_forward(&src, &mut output, &rnn_config, &w, &mut workspace)
        .expect("Forward RNN works");

    backend
        .rnn_backward_weights(
            &src,
            &output,
            &mut dw,
            &rnn_config,
            &mut workspace
        )
        .expect("Backward Weights RNN works");

            // usually computated by a weight function or the following layer
    let output_gradient = uniformly_random_tensor::<T, F>(
        &backend,
        &output.desc(),
        <T as Zero>::zero(),
        <T as One>::one(),
    );

    let mut src_gradient = SharedTensor::new(src.desc());

    backend
        .rnn_backward_data(
            &src,
            &mut src_gradient,
            &output,
            &output_gradient,
            &rnn_config,
            &w,
            &mut workspace
        )
        .expect("Backward Data RNN works");
}

mod cuda {
    use super::*;
    test_cuda!(test_rnn, rnn_f32, rnn_f64);
}
