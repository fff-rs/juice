//! Create a Recursive Layer
//!
//! Recurrent Neural Network Layer
//! A type of Neural Network that can process data in sequence, with temporal understanding of
//! one element of data flowing into the next. This type of understanding is suitable for tasks such
//! as translating a sentence, mimicking the patterns in a musical piece, or time series forecasting.
//!
//! Currently this is implemented in CUDA, but not in native or opencl.
//!
//! ## CUDA Specific Notes - Using Juice
//! CUDA currently supports GRU, LSTM, ReLU, and tanh for LSTM operations.
//! All of these can be uni or bi-directional.
//!
//! All of these perform better when Tensor Core are available, this has some pretty stringent
//! requirements (https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#tensor_ops);
//!
//! For Standard Algorithm - CUDNN_RNN_ALGO_STANDARD in Cuda Docs or RnnAlgorithm::Standard in Juice,
//! * hidden size, input size, and batch size must be a multiple of 8
//! * All user-provided tensors, workspace, and reserve space are aligned to 128 bit boundaries.
//! * Math Type CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION (MathType::TensorOPMathAllowConversion) is selected.
// TODO: Ensure workspace & reserve-space are aligned to 128 bit boundaries.
//!
//! ## CUDA Specific Notes - Developing Juice
//! The following resources are your best bet for debugging an issue within Juice.
//! Generic Docs https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html
//! API Docs https://docs.nvidia.com/deeplearning/sdk/cudnn-api/index.html
//!
//! We're aiming to support the latest features in CUDNN, and promise no support for outdated
//! versions of CUDA or CUDNN. Current code has been tested with
//! | CUDA            | CUDNN              |
//! |---                   |---                |
//! | 10.2              | 7.6.5 |
//!
//! And the following graphics cards
//! | Card            |
//! |---              |
//! | NVIDIA GeForce GTX 1070 |

use std::rc::Rc;
use std::sync::{Arc, RwLock};

use conn::{DirectionMode, RnnAlgorithm, RnnInputMode, RnnNetworkMode};

use crate::capnp_util::*;
use crate::co::prelude::*;
use crate::conn;
use crate::conn::RnnConfig as connRnnConfig;
use crate::juice_capnp::rnn_config as capnp_config;
use crate::layer::*;
use crate::util::{native_backend, ArcLock};
use crate::weight::FillerType;

#[derive(Debug, Clone)]
///
pub struct Rnn<B: conn::Rnn<f32>> {
    hidden_size: usize,
    num_layers: usize,
    dropout_probability: f32,
    dropout_seed: u64,
    rnn_type: RnnNetworkMode,
    input_mode: RnnInputMode,
    direction_mode: DirectionMode,
    workspace: Option<ArcLock<SharedTensor<u8>>>,
    rnn_config: Option<Rc<B::CRNN>>,
}

impl<B: conn::Rnn<f32>> Rnn<B> {
    /// Create a RNN from a RNNConfig
    pub fn from_config(config: &RnnConfig) -> Rnn<B> {
        Rnn {
            hidden_size: config.hidden_size,
            num_layers: config.num_layers,
            dropout_probability: config.dropout_probability,
            dropout_seed: config.dropout_seed,
            rnn_type: config.rnn_type,
            input_mode: config.input_mode,
            direction_mode: config.direction_mode,
            workspace: None,
            rnn_config: None,
        }
    }
}

impl<B: IBackend + conn::Rnn<f32>> ILayer<B> for Rnn<B> {
    impl_ilayer_common!();

    fn auto_weight_blobs(&self) -> bool {
        true
    }

    fn reshape(
        &mut self,
        backend: Rc<B>,
        input_data: &mut Vec<ArcLock<SharedTensor<f32>>>,
        input_gradient: &mut Vec<ArcLock<SharedTensor<f32>>>,
        weights_data: &mut Vec<ArcLock<SharedTensor<f32>>>,
        weights_gradient: &mut Vec<ArcLock<SharedTensor<f32>>>,
        output_data: &mut Vec<ArcLock<SharedTensor<f32>>>,
        output_gradient: &mut Vec<ArcLock<SharedTensor<f32>>>,
    ) {
        let input = input_data[0].read().unwrap();
        let mut output_data = output_data[0].write().unwrap();
        let mut output_gradient = output_gradient[0].write().unwrap();

        // Input Shape is Batch, Number of Inputs, Sequence Length
        let input_shape = input.desc();
        let batch_size = input_shape[0];
        let input_size = input_shape[1];
        let sequence_length = input_shape[2];

        let hidden_size = self.hidden_size;

        let output_shape = &[batch_size, hidden_size, self.num_layers];
        input_gradient[0].write().unwrap().resize(input_shape).unwrap();
        output_data.resize(output_shape).unwrap();
        output_gradient.resize(output_shape).unwrap();

        let config = backend
            .new_rnn_config(
                &input,
                Some(self.dropout_probability),
                Some(self.dropout_seed),
                sequence_length as i32,
                self.rnn_type,
                self.input_mode,
                self.direction_mode,
                // Standard is likely to be effective across most parameters. This should be
                // calculated internal to Juice if modified, allowing user input is likely to be
                // more confusing than helpful to the end user.
                // https://docs.nvidia.com/deeplearning/sdk/cudnn-api/index.html#cudnnRNNAlgo_t
                // Lists the differences and how we can pick between Algorithms automatically
                RnnAlgorithm::Standard,
                hidden_size as i32,
                self.num_layers as i32,
                batch_size as i32,
            )
            .unwrap();

        let filter_dimensions: TensorDesc = backend
            .generate_rnn_weight_description(&config, batch_size as i32, input_size as i32)
            .unwrap();

        // weights
        weights_data[0].write().unwrap().resize(&filter_dimensions).unwrap();
        // biases
        weights_data[1].write().unwrap().resize(&(1, self.hidden_size)).unwrap();

        let filler = FillerType::Glorot {
            input_size: filter_dimensions.clone().size(),
            output_size: batch_size * self.num_layers * self.hidden_size,
        };

        let bias_filler = FillerType::Constant { value: 1.0 };

        filler.fill(&mut weights_data[0].write().unwrap());
        bias_filler.fill(&mut weights_data[1].write().unwrap());

        weights_gradient[0].write().unwrap().resize(&filter_dimensions).unwrap();
        weights_gradient[1].write().unwrap().resize(&filter_dimensions).unwrap();

        self.rnn_config = Some(Rc::new(config));
    }

    fn resize_shared_workspace(
        &mut self,
        backend: Rc<B>,
        workspace: Option<ArcLock<SharedTensor<u8>>>,
    ) -> Option<ArcLock<SharedTensor<u8>>> {
        let required_size = self.rnn_config.as_ref().unwrap().workspace_size();

        if let Some(old_workspace) = workspace.clone() {
            let old_workspace_size = old_workspace.read().unwrap().capacity();
            if old_workspace_size >= required_size {
                return Some(old_workspace)
            }
        }
        self.workspace = Some(
            Arc::new(RwLock::new(SharedTensor::<u8>::new(&[required_size])))
        );
        self.workspace.clone()
    }
}

impl<B: IBackend + conn::Rnn<f32>> ComputeOutput<f32, B> for Rnn<B> {
    fn compute_output(
        &self,
        backend: &B,
        weights: &[&SharedTensor<f32>],
        input_data: &[&SharedTensor<f32>],
        output_data: &mut [&mut SharedTensor<f32>],
    ) {
        let input_shape = input_data[0].desc();
        let batch_size = input_shape[0];
        let input_size = input_shape[1];
        let sequence_length = input_shape[2];
        let rnn_config = self.rnn_config.as_ref().unwrap();
        let mut workspace = self.workspace.as_ref().unwrap().write().unwrap();
        backend
            .rnn_forward(&input_data[0], output_data[0], rnn_config, weights[0], &mut workspace)
            .unwrap();
    }
}

impl<B: IBackend + conn::Rnn<f32>> ComputeInputGradient<f32, B> for Rnn<B> {
    fn compute_input_gradient(
        &self,
        backend: &B,
        weights_data: &[&SharedTensor<f32>],
        output_data: &[&SharedTensor<f32>],
        output_gradients: &[&SharedTensor<f32>],
        input_data: &[&SharedTensor<f32>],
        input_gradients: &mut [&mut SharedTensor<f32>],
    ) {
        let rnn_config = self.rnn_config.as_ref().unwrap();
        let mut workspace = self.workspace.as_ref().unwrap().write().unwrap();

        let src = input_data[0];
        let input_shape = src.desc();
        let batch_size = input_shape[0];
        let input_size = input_shape[1];
        let sequence_length = input_shape[2];
        let native_backend = native_backend();
        let readable_input = src.read(native_backend.device()).unwrap().as_slice::<f32>().to_vec();

        backend
            .rnn_backward_data(
                &input_data[0],
                input_gradients[0],
                &output_data[0],
                output_gradients[0],
                rnn_config,
                weights_data[0],
                &mut workspace,
            )
            .unwrap();
    }
}

impl<B: IBackend + conn::Rnn<f32>> ComputeParametersGradient<f32, B> for Rnn<B> {
    fn compute_parameters_gradient(
        &self,
        backend: &B,
        output_data: &[&SharedTensor<f32>],
        output_gradients: &[&SharedTensor<f32>],
        input_data: &[&SharedTensor<f32>],
        parameters_gradients: &mut [&mut SharedTensor<f32>],
    ) {
        let rnn_config = self.rnn_config.as_ref().unwrap();
        let mut workspace = self.workspace.as_ref().unwrap().write().unwrap();

        // let src = input_data[0];
        // let input_shape = src.desc();
        // let batch_size = input_shape[0];
        // let input_size = input_shape[1];
        // let sequence_length = input_shape[2];

        backend
            .rnn_backward_weights(
                &input_data[0],
                &output_data[0],
                &mut parameters_gradients[0],
                rnn_config,
                &mut workspace,
            )
            .unwrap();

        backend
            .rnn_backward_weights(
                &input_data[0],
                &output_data[0],
                &mut parameters_gradients[1],
                rnn_config,
                &mut workspace,
            )
            .unwrap();
    }
}

#[derive(Debug, Clone, Copy)]
/// Specifies configuration parameters for a RNN Layer.
/// TODO: Update to RnnConfig in CUDA Layer
pub struct RnnConfig {
    /// Size of the Hidden Layer
    pub hidden_size: usize,
    /// Number of Hidden Layers
    pub num_layers: usize,
    /// Type of RNN
    pub rnn_type: RnnNetworkMode,
    /// Dropout Probability
    pub dropout_probability: f32,
    /// Dropout Seed
    pub dropout_seed: u64,
    /// Input Mode
    pub input_mode: RnnInputMode,
    /// RNN Direction
    pub direction_mode: DirectionMode,
}

impl Into<LayerType> for RnnConfig {
    fn into(self) -> LayerType {
        LayerType::Rnn(self)
    }
}

impl<'a> CapnpWrite<'a> for RnnConfig {
    type Builder = capnp_config::Builder<'a>;

    /// Write the RnnConfig into a capnp message.
    fn write_capnp(&self, builder: &mut Self::Builder) {
        builder.reborrow().set_num_layers(self.num_layers as u64);
        builder.reborrow().set_hidden_size(self.hidden_size as u64);
        builder.reborrow().set_rnn_type(&self.rnn_type.to_string());
        builder.reborrow().set_dropout_probability(self.dropout_probability);
        builder.reborrow().set_dropout_seed(self.dropout_seed);
        builder.reborrow().set_input_mode(&self.input_mode.to_string());
        builder.reborrow().set_direction_mode(&self.direction_mode.to_string());
    }
}

impl<'a> CapnpRead<'a> for RnnConfig {
    type Reader = capnp_config::Reader<'a>;

    fn read_capnp(reader: Self::Reader) -> Self {
        let read_num_layers = reader.get_num_layers() as usize;
        let read_hidden_size = reader.get_hidden_size() as usize;
        let read_dropout_probability = reader.get_dropout_probability();
        let read_dropout_seed = reader.get_dropout_seed();
        let read_rnn_type = RnnNetworkMode::from_string(reader.get_rnn_type().unwrap()).unwrap();
        let read_input_mode = RnnInputMode::from_string(reader.get_input_mode().unwrap()).unwrap();
        let read_direction_mode = DirectionMode::from_string(reader.get_direction_mode().unwrap()).unwrap();

        RnnConfig {
            hidden_size: read_hidden_size,
            num_layers: read_num_layers,
            rnn_type: read_rnn_type,
            dropout_seed: read_dropout_seed,
            dropout_probability: read_dropout_probability,
            input_mode: read_input_mode,
            direction_mode: read_direction_mode,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;

    use conn::Rnn as coRnn;
    use conn::{DirectionMode, RnnAlgorithm, RnnInputMode, RnnNetworkMode};

    #[cfg(feature = "cuda")]
    use crate::co::frameworks::cuda::get_cuda_backend as cuda_backend;
    use crate::co::*;
    use crate::layer::{ILayer, ComputeInputGradient, ComputeOutput, ComputeParametersGradient};
    use crate::util::native_backend;
    use crate::weight::FillerType;

    use super::{Rnn, RnnConfig};

    fn sample_input_64() -> Vec<f32> {
        vec![
            // Default Input Type - Batch of 8 Elements, 8 Time Parts, Width 1, Height 1.
            0.5f32;64
        ]
    }

    fn sample_input_25() -> Vec<f32> {
        vec![
            // Default Input Type - Batch of 5 Elements, 5 Time Parts, Width 1, Height 1.
            0.5f32;25
        ]
    }

    fn sample_output() -> &'static [f32] {
        [0.6639924, 0.5426032, 0.7527217, 0.3648719, 0.6244233].as_ref()
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn rnn_create_layer() {
        let cfg = RnnConfig {
            hidden_size: 8,
            num_layers: 2,
            dropout_probability: 0.5,
            dropout_seed: 0,
            rnn_type: RnnNetworkMode::LSTM,
            input_mode: RnnInputMode::LinearInput,
            direction_mode: DirectionMode::UniDirectional,
        };

        let native_backend = native_backend();
        let backend = cuda_backend();

        let batch_size = 5_usize;
        let sequence_length = 5_usize;
        let height = 1_usize;
        let width = 1_usize;

        let hidden_size = cfg.hidden_size;
        let num_layers = cfg.num_layers;

        let input_shape = &(batch_size, sequence_length, height, width);
        let mut layer = Rnn::<Backend<Cuda>>::from_config(&cfg);

        let mut input_data = SharedTensor::<f32>::new(input_shape);
        input_data
            .write_only(native_backend.device())
            .unwrap()
            .as_mut_slice()
            .copy_from_slice(&sample_input_25());

        let input_shape = input_data.desc();

        let output_shape = &[input_shape[0], input_shape[1], num_layers];
        let output_data = SharedTensor::<f32>::new(output_shape);

        layer.rnn_config = Some(Rc::from(
            backend
                .new_rnn_config(
                    &input_data,
                    None,
                    None,
                    sequence_length as i32,
                    RnnNetworkMode::LSTM,
                    RnnInputMode::LinearInput,
                    DirectionMode::UniDirectional,
                    RnnAlgorithm::Standard,
                    hidden_size as i32,
                    num_layers as i32,
                    input_shape[0] as i32,
                )
                .unwrap(),
        ));
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn rnn_roundtrip_pass() {
        let _ = env_logger::builder().is_test(true).filter_level(log::LevelFilter::Trace).try_init();

        let backend: Backend<Cuda> = cuda_backend();
        const SEQUENCE_LENGTH: usize = 7;
        const HIDDEN_SIZE: usize = 5;
        const NUM_LAYERS: usize = 3;
        const BATCH_SIZE: usize = 2;
        const INPUT_SIZE: usize = 11;

        let cfg = RnnConfig {
            hidden_size: HIDDEN_SIZE,
            num_layers: NUM_LAYERS,
            dropout_probability: 0.5,
            dropout_seed: 1337,
            rnn_type: RnnNetworkMode::LSTM,
            input_mode: RnnInputMode::LinearInput,
            direction_mode: DirectionMode::UniDirectional,
        };

        let native_backend = native_backend();
        let mut layer = Rnn::<Backend<Cuda>>::from_config(&cfg);

        let input_shape = vec![
            BATCH_SIZE,
            INPUT_SIZE,
            1,
            1,
        ];

        let mut input_data = SharedTensor::<f32>::new(&input_shape);
        let mut input_gradients = SharedTensor::<f32>::new(&input_shape);

        let data = std::iter::repeat(0.5_f32)
            .take(BATCH_SIZE * INPUT_SIZE)
            .collect::<Vec<f32>>();
        input_data
            .write_only(native_backend.device())
            .unwrap()
            .as_mut_slice()
            .copy_from_slice(&data);

        let output_shape = vec![
            BATCH_SIZE,
            HIDDEN_SIZE,
            1,
        ];

        let mut output_data = SharedTensor::<f32>::new(&output_shape);

        let config = backend
            .new_rnn_config(
                &input_data,
                None,
                None,
                SEQUENCE_LENGTH as i32,
                RnnNetworkMode::LSTM,
                RnnInputMode::LinearInput,
                DirectionMode::UniDirectional,
                RnnAlgorithm::Standard,
                HIDDEN_SIZE as i32,
                NUM_LAYERS as i32,
                BATCH_SIZE as i32,
            )
            .unwrap();

        let filter_dimensions = <Backend<Cuda> as conn::Rnn<f32>>::generate_rnn_weight_description(
            &backend,
            &config,
            BATCH_SIZE as i32,
            INPUT_SIZE as i32,
        )
        .unwrap();

        layer.rnn_config = Some(Rc::from(config));

        let mut weights_data = vec![
            SharedTensor::<f32>::new(&filter_dimensions),
            SharedTensor::<f32>::new(&filter_dimensions), // bias, XXX unused
        ];

        let weights_gradient = vec![
            SharedTensor::<f32>::new(&filter_dimensions),
            SharedTensor::<f32>::new(&(1, SEQUENCE_LENGTH)), // bias, XXX unused
        ];

        let filler = FillerType::Constant { value: 0.02 };

        filler.fill(&mut weights_data[0]);
        filler.fill(&mut weights_data[1]);

        layer.resize_shared_workspace(Rc::from(cuda_backend()), None);

        layer.compute_output(
            &backend,
            &weights_data.iter().collect::<Vec<_>>(),
            &[&input_data],
            &mut [&mut output_data],
        );

        // simulate some feedback
        let mut output_gradients = SharedTensor::<f32>::new(&output_shape);
        filler.fill(&mut output_gradients);

        layer.compute_input_gradient(
            &backend,
            &weights_data.iter().collect::<Vec<_>>(),
            &[&output_data],
            &[&output_gradients],
            &[&input_data],
            &mut[&mut input_gradients],
        );

        layer.compute_parameters_gradient(
            &backend,
            &[&output_data],
            &[&output_gradients],
            &[&input_data],
            &mut weights_data.iter_mut().collect::<Vec<_>>(),
        );
    }
}
