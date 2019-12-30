//! Create a Recursive Layer
//!
//! TODO: Add Docs

use crate::capnp_util::*;
use crate::co::prelude::*;
use crate::conn;
use crate::conn::RnnConfig as connRnnConfig;
use crate::layer::*;
use crate::juice_capnp::rnn_config as capnp_config;
use std::rc::Rc;
use std::sync::{Arc, RwLock};
use crate::util::{ArcLock, cast_vec_usize_to_i32};
use crate::weight::FillerType;
use capnp::ErrorKind::Unimplemented;

#[derive(Debug, Clone)]
///
pub struct Rnn<B: conn::Rnn<f32>> {
    num_output: usize,
    hidden_size: usize,
    num_layers: usize,
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
            workspace: None,
            rnn_config : None
        }
    }

}

impl<B: IBackend + conn::Rnn<f32>> ILayer<B> for Rnn<B> {
    impl_ilayer_common!();

    fn reshape(&mut self,
               backend: Rc<B>,
               input_data: &mut Vec<ArcLock<SharedTensor<f32>>>,
               input_gradient: &mut Vec<ArcLock<SharedTensor<f32>>>,
               weights_data: &mut Vec<ArcLock<SharedTensor<f32>>>,
               weights_gradient: &mut Vec<ArcLock<SharedTensor<f32>>>,
               output_data: &mut Vec<ArcLock<SharedTensor<f32>>>,
               output_gradient: &mut Vec<ArcLock<SharedTensor<f32>>>) {
            unimplemented!()
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
        let filter_data = weights[0];
        let rnn_config = self.rnn_config.as_ref().unwrap();
        let mut workspace = self.workspace.as_ref().unwrap().write().unwrap();
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
// TODO: Add in LSTM Type
pub struct RnnConfig {
    /// Number of field outputs
    pub output_size: usize,
    /// Cell Size in LSTM
    pub cell_size: usize,
    /// Size of the Hidden Layer
    pub hidden_size: usize,
    /// Number of Hidden Layers
    pub num_layers: usize
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
    }
}

impl<'a> CapnpRead<'a> for RnnConfig{
    type Reader = capnp_config::Reader<'a>;

    fn read_capnp(reader: Self::Reader) -> Self {
        let read_output_size = reader.get_output_size() as usize;
        let read_cell_size = reader.get_cell_size() as usize;
        let read_num_layers = reader.get_num_layers() as usize;
        let read_hidden_size = reader.get_hidden_size() as usize;

        RnnConfig {
            output_size: read_output_size,
            cell_size : read_cell_size,
            hidden_size: read_hidden_size,
            num_layers: read_num_layers
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Rnn, RnnConfig};
    use crate::co::*;

    #[test]
    #[cfg(feature="cuda")]
    fn correct_shapes() {
        let cfg = RnnConfig {
            output_size: 64,
            cell_size: 10,
            hidden_size: 10,
            num_layers: 10
        };
        let layer = Rnn::<Backend<Cuda>>::from_config(&cfg);
        unimplemented!()
    }
}
