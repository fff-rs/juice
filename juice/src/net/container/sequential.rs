//! A container layer the composes nested layers in a sequence.
//!
//! In the most simple case, all nested layers are executed one by one, with outputs of one layer
//! becoming inputs of the next one. Inputs of the Sequential layers are inputs to the first
//! nested layer, and outputs of the last nested layer are the outpus of the Sequential layer:
//!
//! ```
//! let mut cfg = SequentialConfig::default();
//! cfg.add_layer("linear", LayerConfig::Linear(LinearConfig{ output_size: 10}));
//! cfg.add_layer("softmax", LayerConfig::Softmax);
//! ```
//!
//! Sequential layer also supports complex flow graphs (as long as they are acyclic)
//! by allowing nested layers inputs and outputs to be mapped to named "buffers":
//!
//! ```
//! let mut cfg = SequentialConfig::default();
//! cfg.add_input("in");
//! cfg
//!   .add_layer("linear1", LayerConfig::Linear(LinearConfig{ output_size: 10}))
//!   .map_input("in")
//!   .map_output("linear1_out");
//! cfg
//!   .add_layer("linear2", LayerConfig::Linear(LinearConfig{ output_size: 10}))
//!   .map_input("in")
//!   .map_output("linear2_out");
//! cfg
//!   .add_layer(/*some 2-input 1-output layer*/)
//!   .map_input("linear1_out")
//!   .map_input("linear2_out")
//!   .map_output("out");
//! cfg.add_output("out");
//! ```
//!
//! Note that currently it is requires that layer i can only use outputs from
//! previous layers 0..i-1.
use std::collections::HashMap;
use std::fmt::{Debug, Formatter};

use crate::co::IBackend;
use crate::net::{layer_from_config, Context, Descriptor, Layer, LayerConfig};
use crate::util::LayerOps;

#[derive(Debug, Clone, Default)]
pub struct SequentialChildConfig {
    name: String,
    config: LayerConfig,
    inputs: Vec<String>,
    outputs: Vec<String>,
}

#[derive(Debug, Clone, Default)]
pub struct SequentialConfig {
    // List of named inputs. If not given, only a single input is assumed.
    inputs: Vec<String>,

    // Sequence of layers.
    layers: Vec<SequentialChildConfig>,

    // List of named outputs. If not given, Sequential layer outputs will match the outputs
    // of the last layer.
    outputs: Vec<String>,
}

pub struct Sequential<B: IBackend> {
    // Outward-facing inputs, outputs and params.
    descriptor: Descriptor,
    // Nested layers.
    children: Vec<Box<dyn Layer<B>>>,
}

impl SequentialChildConfig {
    pub fn map_input(mut self, name: &str) -> Self {
        self.inputs.push(name.to_string());
        self
    }
    pub fn map_output(mut self, name: &str) -> Self {
        self.outputs.push(name.to_string());
        self
    }
}

impl SequentialConfig {
    pub fn new() -> Self {
        SequentialConfig::default()
    }

    pub fn with_layer(mut self, name: &str, child_config: LayerConfig) -> Self {
        self.layers.push(SequentialChildConfig {
            name: name.to_string(),
            config: child_config,
            inputs: Vec::new(),
            outputs: Vec::new(),
        });
        self
    }

    pub fn with_input(mut self, name: &str) -> Self {
        self.inputs.push(name.to_string());
        self
    }
}

impl<B: IBackend + LayerOps<f32> + 'static> Sequential<B> {
    pub fn new(mut descriptor: Descriptor, config: &SequentialConfig) -> Self {
        // Create internal layers one by one and connect them.
        // For the purpose of connecting layers, all inputs and outputs have names,
        // which are either explicitly given in the config, or have implicit form of
        // io_{i}_{j} for layer inputs and io_{i+1}_{j} for outputs, where i is the layer index
        // (0-based) and j is the input/output index within the layer.
        // io_0_{j} are inputs to the Sequential layer itself.

        // All layer output Inouts known so far, keyed by their internal names.
        // This is initialized with shapes of the Sequential inputs below.
        let mut internal_outputs = HashMap::new();

        // Output names from the previous layer. These will be assumed to be the inputs of the next
        // layer unless it has explicit input names in the config.
        // Initialized with Sequenial input names below.
        let mut prev_layer_output_names = Vec::new();

        // Create an array of Sequential inputs where:
        // * internal names are either explicitly set in config or implicitly set to "io_0_{j}",
        // * shapes and data paths are taken from the descriptor.
        for (j, input) in descriptor.inputs().iter().enumerate() {
            let internal_name = if config.inputs.len() > j {
                config.inputs[j].clone()
            } else {
                format!("io_0_{}", j)
            };
            prev_layer_output_names.push(internal_name.clone());
            internal_outputs.insert(internal_name, input.clone());
        }

        // Create children layers.
        let mut children = Vec::new();
        for (i, child_config) in config.layers.iter().enumerate() {
            // Inputs for a child are either given explicitly in the config, or replicate the
            // outputs of the previous child.
            let child_input_names = if !child_config.inputs.is_empty() {
                child_config.inputs.clone()
            } else {
                prev_layer_output_names
            };

            let child_inputs = child_input_names
                .iter()
                .map(|name| {
                    internal_outputs
                        .get(name)
                        .unwrap_or_else(|| panic!("Unknown input/output name {}", name))
                        .clone()
                })
                .collect();

            let mut child_layer = layer_from_config(
                descriptor.sub(&child_config.name, child_inputs),
                &child_config.config,
            );

            // Create data buffer paths for child outpus and save the outputs for subsequent layers.
            let child_descriptor = child_layer.descriptor_mut();
            // Config cannot have more outputs that layer actually has.
            assert!(child_config.outputs.len() <= child_descriptor.outputs().len());
            prev_layer_output_names = Vec::with_capacity(child_descriptor.outputs().len());
            for (j, output) in child_descriptor.outputs_mut().iter_mut().enumerate() {
                let output_name = if j < child_config.outputs.len() {
                    child_config.outputs[j].clone()
                } else {
                    format!("io_{}_{}", i + 1, j)
                };

                // Assign data buffer path.
                output.set_path(&format!("{}.{}", descriptor.path(), &output_name));

                prev_layer_output_names.push(output_name.clone());
                internal_outputs.insert(output_name, output.clone());
            }

            // Copy layer learnable params links into Sequential descriptor.
            for params in child_layer.descriptor().params() {
                descriptor.add_params_copy(params);
            }

            children.push(child_layer);
        }

        // If outputs are given explicitly, use them. Otherwise take outputs of the last layer.
        if !config.outputs.is_empty() {
            for output_name in config.outputs.iter() {
                descriptor.add_output(
                    internal_outputs
                        .get(output_name)
                        .unwrap_or_else(|| panic!("Can't find output {}", output_name))
                        .unit_shape()
                        .clone(),
                );
            }
        } else {
            for output in children.last().unwrap().descriptor().outputs() {
                descriptor.add_output_copy(output)
            }
        }

        Sequential {
            descriptor: descriptor,
            children: children,
        }
    }
}

impl<B: IBackend + LayerOps<f32> + 'static> Layer<B> for Sequential<B> {
    fn compute_output(&self, backend: &B, context: &mut Context) {
        for child in self.children.iter() {
            child.compute_output(backend, context);
        }
    }

    fn compute_gradients(&self, backend: &B, context: &mut Context) {
        for child in self.children.iter().rev() {
            child.compute_gradients(backend, context);
        }
    }

    fn descriptor(&self) -> &Descriptor {
        &self.descriptor
    }

    fn descriptor_mut(&mut self) -> &mut Descriptor {
        &mut self.descriptor
    }
}

impl<B: IBackend> Debug for Sequential<B> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Sequential")
    }
}