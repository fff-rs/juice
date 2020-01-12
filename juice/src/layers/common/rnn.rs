//! Create a Recursive Layer
//!
//! TODO: Add Docs

use std::rc::Rc;
use std::sync::{Arc, RwLock};

use capnp::ErrorKind::Unimplemented;

use conn::{DirectionMode, RnnAlgorithm, RnnInputMode, RnnNetworkMode};
use util::native_backend;

use crate::capnp_util::*;
use crate::co::prelude::*;
use crate::conn;
use crate::conn::RnnConfig as connRnnConfig;
use crate::juice_capnp::rnn_config as capnp_config;
use crate::layer::*;
use crate::util::{ArcLock, cast_vec_usize_to_i32};
use crate::weight::FillerType;

#[derive(Clone, Copy)]
#[allow(non_camel_case_types)]
/// Types of Supported RNNs
pub enum RnnType {
    /// Long Short Term Memory
    LSTM,
    /// Gated Recurrent Unit
    GRU,
    /// ReLU Recursive Unit
    ReLU,
    /// Tanh Recursive Unit
    tanh
}

impl RnnType {
    fn to_text(&self) -> String {
        match self {
            RnnType::GRU => "GRU",
            RnnType::LSTM => "LSTM",
            RnnType::ReLU => "ReLU",
            RnnType::tanh => "tanh"
        }.to_string()
    }

    fn from_text(input: &str) -> Result<Self, &str> {
        match input {
            "GRU" => Ok(RnnType::GRU),
            "LSTM" => Ok(RnnType::LSTM),
            "ReLU" => Ok(RnnType::ReLU),
            "tanh" => Ok(RnnType::tanh),
            _ => Err("Unknown RnnType used - variants are GRU, LSTM, ReLU, and tanh")
        }
    }
}

impl std::fmt::Debug for RnnType {
    fn fmt(&self,f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_text())
    }
}


#[derive(Debug, Clone)]
///
pub struct Rnn<B: conn::Rnn<f32>> {
    num_output: usize,
    hidden_size: usize,
    num_layers: usize,
    // dropout_probability: f32,
    // dropout_seed: f32,
    workspace: Option<ArcLock<SharedTensor<u8>>>,
    rnn_config: Option<Rc<B::RC>>,
}

impl<B: conn::Rnn<f32>> Rnn<B> {
    /// Create a RNN from a RNNConfig
    pub fn from_config(config: &RnnConfig) -> Rnn<B> {
        Rnn {
            num_output: config.output_size,
            hidden_size: config.hidden_size,
            num_layers: config.num_layers,
            // dropout_probability: config.dropout_probability,
            // dropout_seed: config.dropout_seed,
            workspace: None,
            rnn_config: None,
        }
    }
}

impl<B: IBackend + conn::Rnn<f32>> ILayer<B> for Rnn<B> {
    impl_ilayer_common!();

    fn auto_weight_blobs(&self) -> bool { true }

    fn reshape(&mut self,
               backend: Rc<B>,
               input_data: &mut Vec<ArcLock<SharedTensor<f32>>>,
               input_gradient: &mut Vec<ArcLock<SharedTensor<f32>>>,
               weights_data: &mut Vec<ArcLock<SharedTensor<f32>>>,
               weights_gradient: &mut Vec<ArcLock<SharedTensor<f32>>>,
               output_data: &mut Vec<ArcLock<SharedTensor<f32>>>,
               output_gradient: &mut Vec<ArcLock<SharedTensor<f32>>>) {
        let input = input_data[0].read().unwrap();
        let input_shape = input.desc();
        // Input is Batch, Number of Inputs, Sequence Length
        let sequence_length = input_shape[2];
        let output_data = output_data[0].write().unwrap();
        //let mut output_gradient = output_gradient[0].write().unwrap();
        let stride = cast_vec_usize_to_i32(vec![sequence_length, 1, 1]);
        let config = backend.new_rnn_config(
            &input,
            // TODO: implement dropout options
            None,
            None,
            sequence_length as i32,
            RnnNetworkMode::LSTM,
            RnnInputMode::LinearInput,
            DirectionMode::UniDirectional,
            RnnAlgorithm::PersistStatic,
            self.hidden_size as i32,
            self.num_layers as i32,
            input_shape[0] as i32,
        ).unwrap();

        let x_desc = backend.rnn_sequence_descriptors(
            &input,
            sequence_length as i32,
            self.hidden_size as i32,
            input_shape[0] as i32,
        ).unwrap().x_desc;

        let filter_dimensions: TensorDesc = backend.generate_rnn_weight_description(
            &config,
            &x_desc,
        ).unwrap();
        weights_data[0].write().unwrap().resize(&filter_dimensions);
        let filler = FillerType::Glorot {
            input_size: filter_dimensions[1],
            output_size: self.num_output,
        };

        filler.fill(&mut weights_data[0].write().unwrap());
        self.rnn_config = Some(Rc::new(config));
    }

    fn resize_shared_workspace(&mut self,
                               backend: Rc<B>,
                               workspace: Option<ArcLock<SharedTensor<u8>>>)
                               -> Option<ArcLock<SharedTensor<u8>>> {
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
    fn compute_output(&self,
                      backend: &B,
                      weights: &[&SharedTensor<f32>],
                      input_data: &[&SharedTensor<f32>],
                      output_data: &mut [&mut SharedTensor<f32>]) {
        let rnn_config = self.rnn_config.as_ref().unwrap();
        let mut workspace = self.workspace.as_ref().unwrap().write().unwrap();
        backend.rnn_forward(
            input_data[0],
            output_data[0],
            rnn_config,
            weights[0],
            &mut workspace,
        );
        unimplemented!()
    }
}

impl<B: IBackend + conn::Rnn<f32>> ComputeInputGradient<f32, B> for Rnn<B> {
    fn compute_input_gradient(&self,
                              backend: &B,
                              weights_data: &[&SharedTensor<f32>],
                              _output_data: &[&SharedTensor<f32>],
                              output_gradients: &[&SharedTensor<f32>],
                              input_data: &[&SharedTensor<f32>],
                              input_gradients: &mut [&mut SharedTensor<f32>]) {
        unimplemented!()
    }
}

impl<B: IBackend + conn::Rnn<f32>> ComputeParametersGradient<f32, B> for Rnn<B> {
    fn compute_parameters_gradient(&self,
                                   backend: &B,
                                   _output_data: &[&SharedTensor<f32>],
                                   output_gradients: &[&SharedTensor<f32>],
                                   input_data: &[&SharedTensor<f32>],
                                   parameters_gradients: &mut [&mut SharedTensor<f32>]) {
        unimplemented!()
    }
}


#[derive(Debug, Clone, Copy)]
/// Specifies configuration parameters for a RNN Layer.
/// TODO: Update to RnnConfig in CUDA Layer
pub struct RnnConfig {
    /// Number of field outputs
    pub output_size: usize,
    /// Cell Size in LSTM
    pub cell_size: usize,
    /// Size of the Hidden Layer
    pub hidden_size: usize,
    /// Number of Hidden Layers
    pub num_layers: usize,
    /// Type of RNN
    pub rnn_type : RnnType
}

impl Into<LayerType> for RnnConfig {
    fn into(self) -> LayerType {
        LayerType::Rnn(self)
    }
}

impl<'a> CapnpWrite<'a> for RnnConfig{
    type Builder = capnp_config::Builder<'a>;

    /// Write the RnnConfig into a capnp message.
    fn write_capnp(&self, builder: &mut Self::Builder) {
        builder.reborrow().set_output_size(self.output_size as u64);
        builder.reborrow().set_cell_size(self.cell_size as u64);
        builder.reborrow().set_hidden_size(self.hidden_size as u64);
        builder.reborrow().set_num_layers(self.num_layers as u64);
        builder.reborrow().set_rnn_type(&self.rnn_type.to_text());
    }
}

impl<'a> CapnpRead<'a> for RnnConfig{
    type Reader = capnp_config::Reader<'a>;

    fn read_capnp(reader: Self::Reader) -> Self {
        let read_output_size = reader.get_output_size() as usize;
        let read_cell_size = reader.get_cell_size() as usize;
        let read_num_layers = reader.get_num_layers() as usize;
        let read_hidden_size = reader.get_hidden_size() as usize;
        let read_rnn_type = RnnType::from_text(
            reader.get_rnn_type()
                .unwrap())
            .unwrap();

        RnnConfig {
            output_size: read_output_size,
            cell_size : read_cell_size,
            hidden_size: read_hidden_size,
            num_layers: read_num_layers,
            rnn_type: read_rnn_type
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::co::*;

    use super::{Rnn, RnnConfig, RnnType};

    #[test]
    #[cfg(feature = "cuda")]
    fn correct_shapes() {
        let cfg = RnnConfig {
            output_size: 64,
            cell_size: 10,
            hidden_size: 10,
            num_layers: 10,
            rnn_type: RnnType::LSTM
        };
        let layer = Rnn::<Backend<Cuda>>::from_config(&cfg);
        unimplemented!()
    }
}
