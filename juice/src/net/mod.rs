//! Representation of a neural network.
//!
//! Network consists of 2 parts: static and dynamic.
//!
//! ## Static network structure
//!
//! Network a graph of interconnected layers. Each layer has:
//! * Inputs.
//! * Outputs.
//! * (Optional) learnable weights.
//!
//! Inputs, outputs and weights are all tensors of a specific shape. Shapes of the inputs
//! and outputs represent a shape of one data unit; during runtime, the data flowing through
//! the network may be batched (see "Dynamic state" section). Number and shapes of all
//! inputs, outputs and weights are captures in the layer `Descriptor`.
//!
//! Layers are created and their configuration is determined by a waterflow process:
//! 1. A layer is created when its inputs shape is known, i.e. when upstream layers have
//!    been created.
//! 2. Layer construction function determine the shapes of the output and weights
//!    given the inputs shapes.
//! 3. Once the layer is created and its output shapes are known, downstream layers which
//!    take these outputs as inputs can be created.
//!
//! Layers can be combined into hierarchical structures via special container layers
//! (currently only one such layer exists, `Sequential`).
//!
//! ## Dynamic state
//!
//! Dynamic state represents everything that is related to the propagation of one specific input
//! in the network:
//! * The input and output data of each layer.
//! * Input/output gradients.
//! * Learnable weight gradients.
//!
//! Dynamic state is held in a `Context`, which contains all the data for a single invocation of
//! the network.
//!
//! During learning, data is typically passed through the network in batches. Batch size is fixed
//! for a single invocation and is stored in `Context`. It's typical to use batch size of 16-64
//! for learning and batch size of 1 for using network to produce output for a single input.

mod activation;
mod config;
mod container;
mod context;
mod descriptor;
mod layer;
mod network;

pub use config::*;
pub use context::*;
pub use descriptor::*;
pub use layer::*;
pub use network::*;
