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
use crate::util::ArcLock;
use crate::weight::FillerType;

#[derive(Clone, Copy)]
#[allow(non_camel_case_types)]
/// Types of Supported RNNs
pub enum RnnNetworkMode_UserInput {
    /// Long Short Term Memory
    LSTM,
    /// Gated Recurrent Unit
    GRU,
    /// ReLU Recursive Unit
    ReLU,
    /// Tanh Recursive Unit
    tanh,
}

impl RnnNetworkMode_UserInput {
    fn to_text(&self) -> String {
        match self {
            RnnNetworkMode_UserInput::GRU => "GRU",
            RnnNetworkMode_UserInput::LSTM => "LSTM",
            RnnNetworkMode_UserInput::ReLU => "ReLU",
            RnnNetworkMode_UserInput::tanh => "tanh",
        }
            .to_string()
    }

    fn from_text(input: &str) -> Result<Self, &str> {
        match input {
            "GRU" => Ok(RnnNetworkMode_UserInput::GRU),
            "LSTM" => Ok(RnnNetworkMode_UserInput::LSTM),
            "ReLU" => Ok(RnnNetworkMode_UserInput::ReLU),
            "tanh" => Ok(RnnNetworkMode_UserInput::tanh),
            _ => Err("Unknown RnnType used - variants are GRU, LSTM, ReLU, and tanh"),
        }
    }

    fn to_cudnn(&self) -> RnnNetworkMode {
        match self {
            RnnNetworkMode_UserInput::GRU => RnnNetworkMode::GRU,
            RnnNetworkMode_UserInput::LSTM => RnnNetworkMode::LSTM,
            RnnNetworkMode_UserInput::ReLU => RnnNetworkMode::ReLU,
            RnnNetworkMode_UserInput::tanh => RnnNetworkMode::Tanh,
        }
    }
}

#[derive(Clone, Copy)]
#[allow(non_camel_case_types)]
/// Types of Supported RNN Input Modes
pub enum RnnInputMode_UserInput {
    /// Linear Input - A biased matrix multiplication is performed at the input of the first recurrent layer.
    Linear,
    /// Skip Input - No operation is performed to the input of the first recurrent layer
    Skip,
}

impl RnnInputMode_UserInput {
    fn to_text(&self) -> String {
        match self {
            RnnInputMode_UserInput::Skip => "Skip",
            RnnInputMode_UserInput::Linear => "Linear",
        }
            .to_string()
    }

    fn from_text(input: &str) -> Result<Self, &str> {
        match input {
            "Linear" => Ok(RnnInputMode_UserInput::Linear),
            "Skip" => Ok(RnnInputMode_UserInput::Skip),
            _ => Err("Unknown RnnInputMode used - variants are Linear, Skip"),
        }
    }

    fn to_cudnn(&self) -> RnnInputMode {
        match self {
            RnnInputMode_UserInput::Linear => RnnInputMode::LinearInput,
            RnnInputMode_UserInput::Skip => RnnInputMode::SkipInput,
        }
    }
}

#[derive(Clone, Copy)]
#[allow(non_camel_case_types)]
/// Types of Supported RNN DirectionModel
pub enum DirectionMode_UserInput {
    /// The network iterates recurrently from the first input to the last.
    UniDirectional,
    /// Each layer of the network iterates recurrently from the first input to the last and separately from the last input to the first. The outputs of the two are concatenated at each iteration giving the output of the layer.
    BiDirectional,
}

impl DirectionMode_UserInput {
    fn to_text(&self) -> String {
        match self {
            DirectionMode_UserInput::UniDirectional => "UniDirectional",
            DirectionMode_UserInput::BiDirectional => "BiDirectional",
        }
            .to_string()
    }

    fn from_text(input: &str) -> Result<Self, &str> {
        match input {
            "UniDirectional" => Ok(DirectionMode_UserInput::UniDirectional),
            "BiDirectional" => Ok(DirectionMode_UserInput::BiDirectional),
            _ => Err("Unknown DirectionMode used - variants are UniDirectional, BiDirectional"),
        }
    }

    fn to_cudnn(&self) -> DirectionMode {
        match self {
            DirectionMode_UserInput::UniDirectional => DirectionMode::UniDirectional,
            DirectionMode_UserInput::BiDirectional => DirectionMode::BiDirectional,
        }
    }
}

impl std::fmt::Debug for RnnNetworkMode_UserInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_text())
    }
}

impl std::fmt::Debug for RnnInputMode_UserInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_text())
    }
}

impl std::fmt::Debug for DirectionMode_UserInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_text())
    }
}

#[derive(Debug, Clone)]
///
pub struct Rnn<B: conn::Rnn<f32>> {
    hidden_size: usize,
    num_layers: usize,
    dropout_probability: f32,
    dropout_seed: u64,
    rnn_type: RnnNetworkMode_UserInput,
    input_mode: RnnInputMode_UserInput,
    direction_mode: DirectionMode_UserInput,
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
                self.rnn_type.to_cudnn(),
                self.input_mode.to_cudnn(),
                self.direction_mode.to_cudnn(),
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

        let filter_dimensions: TensorDesc = backend.generate_rnn_weight_description(
            &config,
            sequence_length as i32,
            batch_size as i32,
            input_size as i32,
        ).unwrap();

        weights_data[0].write().unwrap().resize(&filter_dimensions).unwrap();
        weights_data[1].write().unwrap().resize(&(1, self.hidden_size)).unwrap();

        let filler = FillerType::Glorot {
            input_size,
            output_size: self.hidden_size,
        };

        filler.fill(&mut weights_data[0].write().unwrap());
        filler.fill(&mut weights_data[1].write().unwrap());

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
        let new_workspace = if workspace.is_none() {
            Arc::new(RwLock::new(SharedTensor::<u8>::new(&[required_size])))
        } else {
            let old_workspace = workspace.as_ref().unwrap().clone();
            let old_workspace_size = old_workspace.read().unwrap().capacity();
            if old_workspace_size < required_size {
                Arc::new(RwLock::new(SharedTensor::<u8>::new(&[required_size])))
            } else {
                workspace.unwrap()
            }
        };

        self.workspace = Some(new_workspace.clone());
        Some(new_workspace)
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
        let src = input_data[0];
        let rnn_config = self.rnn_config.as_ref().unwrap();
        let mut workspace = self.workspace.as_ref().unwrap().write().unwrap();
        backend
            .rnn_forward(input_data[0], output_data[0], rnn_config, weights[0], &mut workspace)
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

        backend.rnn_backward_weights(&input_data[0],
                                     &output_data[0],
                                     &mut parameters_gradients[0],
                                     rnn_config,
                                     &mut workspace)
            .unwrap();
        backend.rnn_backward_weights(&input_data[0],
                                     &output_data[0],
                                     &mut parameters_gradients[1],
                                     rnn_config,
                                     &mut workspace)
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
    /// Dropout Probability
    pub dropout_probability: f32,
    /// Dropout Seed
    pub dropout_seed: u64,
    /// Type of RNN
    pub rnn_type: RnnNetworkMode_UserInput,
    /// Input Mode
    pub input_mode: RnnInputMode_UserInput,
    /// RNN Direction
    pub direction_mode: DirectionMode_UserInput,
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
        builder.reborrow().set_hidden_size(self.hidden_size as u64);
        builder.reborrow().set_num_layers(self.num_layers as u64);
        builder.reborrow().set_rnn_type(&self.rnn_type.to_text());
        builder.reborrow().set_dropout_probability(self.dropout_probability);
        builder.reborrow().set_dropout_seed(self.dropout_seed);
    }
}

impl<'a> CapnpRead<'a> for RnnConfig {
    type Reader = capnp_config::Reader<'a>;

    fn read_capnp(reader: Self::Reader) -> Self {
        let read_num_layers = reader.get_num_layers() as usize;
        let read_hidden_size = reader.get_hidden_size() as usize;
        let read_dropout_probability = reader.get_dropout_probability();
        let read_dropout_seed = reader.get_dropout_seed();
        let read_rnn_type = RnnNetworkMode_UserInput::from_text(reader.get_rnn_type().unwrap()).unwrap();
        let read_input_mode = RnnInputMode_UserInput::from_text(reader.get_input_mode().unwrap()).unwrap();
        let read_direction_mode = DirectionMode_UserInput::from_text(reader.get_direction_mode().unwrap()).unwrap();

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
    use conn::Rnn as coRnn;
    use conn::{DirectionMode, RnnAlgorithm, RnnInputMode, RnnNetworkMode};
    use util::{cast_vec_usize_to_i32, native_backend, native_scalar, write_batch_sample};

    use crate::co::*;

    use super::{Rnn, RnnConfig, RnnNetworkMode_UserInput};
    use layer::ILayer;
    use std::rc::Rc;
    use weight::FillerType;
    use layers::common::rnn::{RnnInputMode_UserInput, DirectionMode_UserInput};

    fn sample_input() -> &'static [f32] { [1.0_f32; 512].as_ref() }

    #[cfg(feature = "cuda")]
    fn cuda_backend() -> Backend<Cuda> {
        let framework = Cuda::new();
        let hardwares = framework.hardwares()[0..1].to_vec();
        let backend_config = BackendConfig::new(framework, &hardwares);
        Backend::new(backend_config).unwrap()
    }

    fn sample_output() -> &'static [f32] {
        [0.99, 0.99, 0.99, 0.99,
            0.99, 0.99, 0.99, 0.99].as_ref()
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn rnn_create_layer() {
        let cfg = RnnConfig {
            hidden_size: 8,
            num_layers: 8,
            dropout_probability: 0.5,
            dropout_seed: 0,
            rnn_type: RnnNetworkMode_UserInput::LSTM,
            input_mode: RnnInputMode_UserInput::Linear,
            direction_mode: DirectionMode_UserInput::UniDirectional,
        };

        let native_backend = native_backend();
        let backend = cuda_backend();
        let sequence_length: i32 = 8;
        let hidden_size = cfg.hidden_size;
        let num_layers = cfg.num_layers;
        let input_shape = &(8, 8, 8);
        let mut layer = Rnn::<Backend<Cuda>>::from_config(&cfg);

        let mut input_data = SharedTensor::<f32>::new(input_shape);
        input_data
            .write_only(native_backend.device())
            .unwrap()
            .as_mut_slice()
            .copy_from_slice(sample_input());
        let input_shape = input_data.desc();

        let output_shape = &[input_shape[0], input_shape[1], num_layers];
        let output_data = SharedTensor::<f32>::new(output_shape);

        layer.rnn_config = Some(Rc::from(
            backend
                .new_rnn_config(
                    &input_data,
                    None,
                    None,
                    sequence_length,
                    RnnNetworkMode::LSTM,
                    RnnInputMode::LinearInput,
                    DirectionMode::UniDirectional,
                    RnnAlgorithm::PersistStatic,
                    hidden_size as i32,
                    num_layers as i32,
                    input_shape[0] as i32,
                )
                .unwrap(),
        ));
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn rnn_forward_pass() {
        let backend: Backend<Cuda> = cuda_backend();
        let cfg = RnnConfig {
            hidden_size: 8,
            num_layers: 8,
            dropout_probability: 0.0,
            dropout_seed: 0,
            rnn_type: RnnNetworkMode_UserInput::LSTM,
            input_mode: RnnInputMode_UserInput::Linear,
            direction_mode: DirectionMode_UserInput::UniDirectional,
        };

        let batch_size = 8;
        let sequence_length = 8;
        let native_backend = native_backend();
        let mut layer = Rnn::<Backend<Cuda>>::from_config(&cfg);

        let input_shape = vec![batch_size, cfg.hidden_size, cfg.num_layers];

        let mut input_data = SharedTensor::<f32>::new(&input_shape);

        input_data
            .write_only(native_backend.device())
            .unwrap()
            .as_mut_slice()
            .copy_from_slice(sample_input());

        let output_shape = vec![batch_size, cfg.hidden_size, cfg.num_layers];
        let mut output_data = SharedTensor::<f32>::new(&output_shape);


        let config =
            backend
                .new_rnn_config(
                    &input_data,
                    None,
                    None,
                    sequence_length,
                    RnnNetworkMode::LSTM,
                    RnnInputMode::LinearInput,
                    DirectionMode::UniDirectional,
                    RnnAlgorithm::Standard,
                    cfg.hidden_size as i32,
                    cfg.num_layers as i32,
                    batch_size as i32,
                )
                .unwrap();

        let filter_dimensions = <Backend<Cuda> as conn::Rnn<f32>>::generate_rnn_weight_description(
            &backend,
            &layer.rnn_config.as_ref().unwrap(),
            sequence_length,
            batch_size as i32,
            cfg.hidden_size as i32,
        ).unwrap();

        layer.rnn_config = Some(Rc::from(
            config
        ));

        let mut weights_data = Vec::with_capacity(4);
        weights_data.push(SharedTensor::<f32>::new(&filter_dimensions));
        weights_data.push(SharedTensor::<f32>::new(&(1, cfg.hidden_size)));

        let mut weights_gradient = Vec::new();
        weights_gradient.push(SharedTensor::<f32>::new(&filter_dimensions));
        weights_gradient.push(SharedTensor::<f32>::new(&(1, cfg.hidden_size)));

        let filler = FillerType::Constant {
            value: 0.02,
        };

        filler.fill(&mut weights_data[0]);
        filler.fill(&mut weights_data[1]);

        layer.resize_shared_workspace(Rc::from(cuda_backend()), None);
        let mut workspace_forward = match layer.workspace.as_ref() {
            Some(workspace) => match workspace.write() {
                Ok(workspace) => workspace,
                Err(_) => panic!("Couldn't unwrap write for workspace"),
            },
            None => panic!("No workspace found"),
        };

        match backend.rnn_forward(
            &input_data,
            &mut output_data,
            match layer.rnn_config {
                Some(ref config) => &Rc::from(&config),
                None => panic!(""),
            },
            &weights_data[0],
            &mut workspace_forward,
        ) {
            Ok(_) => { dbg!("Completed Forward Pass"); }
            Err(e) => panic!("Couldn't complete RNN Forward"),
        };
    }
}
