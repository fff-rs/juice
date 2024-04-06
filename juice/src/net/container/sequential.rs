use std::collections::HashMap;
use std::fmt::{Debug, Formatter};

use crate::co::IBackend;
use crate::net::{layer_from_config, Context, Descriptor, Inout, Layer, LayerConfig, LayerError, LayerFromConfigError};
use crate::util::LayerOps;

// Config for a single child layer.
#[derive(Debug, Clone, Default)]
pub struct SequentialChildConfig {
    name: String,
    config: LayerConfig,
    inputs: Vec<String>,
    outputs: Vec<String>,
}

// Config for the sequential layer, listing child layers and (optionally) connections between them.
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

// A container layer that composes nested layers in a sequence.
//
// In the most simple case, all nested layers are executed one by one, with outputs of one layer
// becoming inputs of the next one. Inputs of the Sequential layers are inputs to the first
// nested layer, and outputs of the last nested layer are the outputs of the Sequential layer:
//
// ```
// let mut cfg = SequentialConfig::new();
// cfg.add_layer("linear", LinearConfig{ output_size: 10});
// cfg.add_layer("softmax", LayerConfig::Softmax);
// ```
//
// Sequential layer also supports complex flow graphs (as long as they are acyclic)
// by allowing nested layers inputs and outputs to be mapped to named "buffers":
//
// ```
// let mut cfg = SequentialConfig::new();
// cfg.map_input("in");
// cfg
//   .add_layer("linear1", LinearConfig{ output_size: 10})
//   .map_input("in")
//   .map_output("linear1_out");
// cfg
//   .add_layer("linear2", LinearConfig{ output_size: 10})
//   .map_input("in")
//   .map_output("linear2_out");
// cfg
//   .add_layer(/*some 2-input 1-output layer*/)
//   .map_input("linear1_out")
//   .map_input("linear2_out")
//   .map_output("out");
// cfg.add_output("out");
// ```
//
// Note that currently it is requires that layer i can only use outputs from
// previous layers 0..i-1.
pub struct Sequential<B: IBackend> {
    // Outward-facing inputs, outputs and params.
    descriptor: Descriptor,
    // Nested layers.
    children: Vec<Box<dyn Layer<B>>>,
}

impl SequentialChildConfig {
    pub fn map_input(&mut self, name: &str) -> &mut Self {
        self.inputs.push(name.to_string());
        self
    }
    pub fn map_output(&mut self, name: &str) -> &mut Self {
        self.outputs.push(name.to_string());
        self
    }
}

impl SequentialConfig {
    pub fn new() -> Self {
        SequentialConfig::default()
    }

    pub fn add_layer(&mut self, name: &str, child_config: impl Into<LayerConfig>) -> &mut SequentialChildConfig {
        self.layers.push(SequentialChildConfig {
            name: name.to_string(),
            config: child_config.into(),
            inputs: Vec::new(),
            outputs: Vec::new(),
        });

        self.layers.last_mut().unwrap()
    }

    pub fn map_input(&mut self, name: &str) {
        self.inputs.push(name.to_string());
    }

    pub fn map_output(&mut self, name: &str) {
        self.outputs.push(name.to_string());
    }
}

impl<B: IBackend + LayerOps<f32> + 'static> Sequential<B> {
    pub fn new(
        backend: &B,
        mut descriptor: Descriptor,
        config: &SequentialConfig,
    ) -> Result<Self, LayerFromConfigError> {
        // Create internal layers one by one and connect them.
        // For the purpose of connecting layers, all inputs and outputs have names,
        // which are either explicitly given in the config, or have implicit form of
        // io_{i}_{j} for layer inputs and io_{i+1}_{j} for outputs, where i is the layer index
        // (0-based) and j is the input/output index within the layer.
        // io_0_{j} are inputs to the Sequential layer itself.

        // All children layer output junction known so far, keyed by their internal names.
        // This is initialized with Sequential inputs below.
        let mut internal_junctions = HashMap::new();

        // Output names from the previous layer. These will be assumed to be the inputs of the next
        // layer unless it has explicit input names in the config.
        // Initialized with Sequential input names below.
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
            internal_junctions.insert(internal_name, input.junction.clone());
        }

        // Create children layers.
        let mut children = Vec::new();
        for (i, child_config) in config.layers.iter().enumerate() {
            // Inputs for a child are either given explicitly in the config, or replicate the
            // outputs of the previous child. Note that currently this doesn't support partial
            // name specification (if explicit names are given, they are assumed to be the only
            // inputs this layer has).
            let child_input_names = if !child_config.inputs.is_empty() {
                child_config.inputs.clone()
            } else {
                prev_layer_output_names
            };

            let child_inputs: Result<Vec<Inout>, LayerFromConfigError> = child_input_names
                .iter()
                .map(|name| -> Result<Inout, _> {
                    let junction = internal_junctions
                        .get(name)
                        .cloned()
                        .ok_or_else(|| LayerFromConfigError::NoSuchInternalOutput(name.to_owned()))?;
                    Ok(Inout::new_with_junction(junction))
                })
                .collect();

            let mut child_layer = layer_from_config(
                backend,
                descriptor.sub(&child_config.name, child_inputs?),
                &child_config.config,
            )?;

            // Create data buffer paths for child outputs and save the outputs for next layers.
            let child_descriptor = child_layer.descriptor_mut();
            // Config cannot have more outputs that layer actually has.
            assert!(child_config.outputs.len() <= child_descriptor.outputs().len());
            prev_layer_output_names = Vec::with_capacity(child_descriptor.outputs().len());
            for j in 0..child_descriptor.outputs().len() {
                let output = child_descriptor.output_mut(j);
                let output_name = if j < child_config.outputs.len() {
                    child_config.outputs[j].clone()
                } else {
                    format!("io_{}_{}", i + 1, j)
                };

                // Assign data buffer path.
                output
                    .junction
                    .path
                    .replace(format!("{}.{}", descriptor.path(), &output_name));

                prev_layer_output_names.push(output_name.clone());
                internal_junctions.insert(output_name, output.junction.clone());
            }

            // Copy layer learnable params links into Sequential descriptor.
            for params in child_layer.descriptor().params() {
                descriptor.add_params_copy(params);
            }

            children.push(child_layer);
        }

        // If outputs are given explicitly, use them. Otherwise take outputs of the last layer
        // (or inputs if there are no child layers).
        if !config.outputs.is_empty() {
            for output_name in config.outputs.iter() {
                let junction = internal_junctions
                    .get(output_name)
                    .cloned()
                    .ok_or_else(|| LayerFromConfigError::NoSuchInternalOutput(output_name.to_owned()))?;
                descriptor.add_output_with_junction(junction);
            }
        } else if !children.is_empty() {
            for output in children.last().unwrap().descriptor().outputs() {
                descriptor.add_output_with_junction(output.junction.clone());
            }
        } else {
            for j in 0..descriptor.inputs().len() {
                let junction = descriptor.input(j).junction.clone();
                descriptor.add_output_with_junction(junction);
            }
        }

        Ok(Sequential { descriptor, children })
    }
}

impl<B: IBackend + LayerOps<f32> + 'static> Layer<B> for Sequential<B> {
    fn compute_output(&self, backend: &B, context: &mut Context) -> Result<(), LayerError> {
        for child in self.children.iter() {
            child.compute_output(backend, context)?;
        }
        Ok(())
    }

    fn compute_gradients(&self, backend: &B, context: &mut Context) -> Result<(), LayerError> {
        for child in self.children.iter().rev() {
            child.compute_gradients(backend, context)?;
        }
        Ok(())
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
        f.debug_struct("Sequential")
            .field("descriptor", &self.descriptor)
            .field("children", &self.children)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use crate::net::container::SequentialConfig;
    use crate::net::{LayerConfig, LayerFromConfigError, Network, NetworkFromConfigError};
    use crate::util::{native_backend, write_batch_sample};
    use co::{Backend, Native, SharedTensor};

    // Passes a single f32 value through the network and returns result.
    fn transform(net: &Network<Backend<Native>>, backend: &Backend<Native>, value: f32) -> f32 {
        let mut input = SharedTensor::new(&[1]);
        write_batch_sample(&mut input, &[value], 0);

        let output = net.transform(backend, &input).unwrap();
        output.read(backend.device()).unwrap().as_slice::<f32>()[0]
    }

    // Tests that an empty Sequential layer passes data unchanged.
    #[test]
    fn empty() {
        let backend = native_backend();

        let cfg = SequentialConfig::new();
        let net = Network::from_config(&backend, cfg, &[vec![1]]).unwrap();

        assert_eq!(transform(&net, &backend, -1.0), -1.0);
    }

    // Tests a Sequential layer consisting of a single ReLU child layer.
    #[test]
    fn one_child() {
        let backend = native_backend();

        let mut cfg = SequentialConfig::new();
        cfg.add_layer("relu", LayerConfig::Relu);
        let net = Network::from_config(&backend, cfg, &[vec![1]]).unwrap();

        assert_eq!(transform(&net, &backend, -1.0), 0.0);
    }

    // Tests inputs and outputs mapping withing Sequential layer.
    #[test]
    fn mapping() {
        let backend = native_backend();

        let mut cfg = SequentialConfig::new();
        cfg.map_input("in");

        // ReLU that reads from "in" and writes to "out1".
        cfg.add_layer("relu", LayerConfig::Relu)
            .map_input("in")
            .map_output("out1");

        // Empty Sequential that reads from "in" and writes to "out2".
        cfg.add_layer("relu2", SequentialConfig::new())
            .map_input("in")
            .map_output("out2");

        // Take output of the Sequential layer.
        cfg.map_output("out2");

        let net = Network::from_config(&backend, cfg, &[vec![1]]).unwrap();

        assert_eq!(transform(&net, &backend, -1.0), -1.0);
    }

    #[test]
    fn bad_intermediate_input_name() {
        let backend = native_backend();

        let mut cfg = SequentialConfig::new();
        cfg.add_layer("relu", LayerConfig::Relu).map_output("out");
        cfg.add_layer("relu2", LayerConfig::Relu).map_input("out2");

        let result: Result<Network<Backend<Native>>, _> = Network::from_config(&backend, cfg, &[vec![1]]);

        match result {
            Err(NetworkFromConfigError::Layer(LayerFromConfigError::NoSuchInternalOutput(name))) => {
                assert_eq!(name, "out2");
            },
            Err(e) => panic!("Unexpected error {:?}", e),
            _ => panic!("Expected to fail but it didn't"),
        }
    }

    #[test]
    fn bad_output_name() {
        let backend = native_backend();

        let mut cfg = SequentialConfig::new();
        cfg.add_layer("relu", LayerConfig::Relu).map_output("out");
        cfg.map_output("out2");

        let result: Result<Network<Backend<Native>>, _> = Network::from_config(&backend, cfg, &[vec![1]]);

        match result {
            Err(NetworkFromConfigError::Layer(LayerFromConfigError::NoSuchInternalOutput(name))) => {
                assert_eq!(name, "out2");
            },
            Err(e) => panic!("Unexpected error {:?}", e),
            _ => panic!("Expected to fail but it didn't"),
        }
    }
}
