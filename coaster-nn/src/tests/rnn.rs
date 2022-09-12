use std::fmt;

use co::prelude::*;
use coaster as co;

use super::{filled_tensor, uniformly_random_tensor, Epsilon, One, Zero};
use crate::plugin::{
    self, DirectionMode, RnnAlgorithm, RnnConfig, RnnInputMode, RnnNetworkMode, RnnPaddingMode,
};
use crate::{co::plugin::numeric_helpers::Float, Rnn};

pub fn test_rnn<T, F: IFramework>(backend: Backend<F>)
where
    T: Float + Epsilon + One + Zero + rand::distributions::uniform::SampleUniform + fmt::Debug,
    Backend<F>: Rnn<T> + IBackend,
{
    let _ = env_logger::builder()
        .is_test(true)
        .filter_level(log::LevelFilter::Trace)
        .try_init();

    let direction_mode = DirectionMode::UniDirectional;
    let network_mode = RnnNetworkMode::LSTM;
    let algorithm = RnnAlgorithm::Standard;
    let input_mode = RnnInputMode::LinearInput;
    const SEQUENCE_LENGTH: usize = 7;
    const HIDDEN_SIZE: usize = 5;
    const NUM_LAYERS: usize = 3;
    const BATCH_SIZE: usize = 2;
    const INPUT_SIZE: usize = 11;

    let dropout_probability = Some(0.05_f32);
    let dropout_seed = Some(27_u64);

    let src = filled_tensor::<T, F>(
        &backend,
        &[BATCH_SIZE, INPUT_SIZE, 1],
        &vec![1.0f64; INPUT_SIZE * BATCH_SIZE],
    );

    let mut output = SharedTensor::<T>::new(&[BATCH_SIZE, HIDDEN_SIZE, 1]);

    let rnn_config = backend
        .new_rnn_config(
            &src,
            dropout_probability,
            dropout_seed,
            SEQUENCE_LENGTH as i32,
            network_mode,
            input_mode,
            direction_mode,
            algorithm,
            HIDDEN_SIZE as i32,
            NUM_LAYERS as i32,
            BATCH_SIZE as i32,
        )
        .unwrap();

    let filter_dimensions = backend
        .generate_rnn_weight_description(&rnn_config, BATCH_SIZE as i32, INPUT_SIZE as i32)
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
        .rnn_backward_weights(&src, &output, &mut dw, &rnn_config, &mut workspace)
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
            &mut workspace,
        )
        .expect("Backward Data RNN works");
}

mod cuda {
    use super::*;
    test_cuda!(test_rnn, rnn_f32, rnn_f64);
}

mod native {
    use env_logger as _;
}
