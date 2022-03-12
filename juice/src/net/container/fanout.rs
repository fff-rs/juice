use std::fmt::{Debug, Formatter};

use crate::co::IBackend;
use crate::net::{layer_from_config, Context, Descriptor, Inout, Layer, LayerConfig};
use crate::util::LayerOps;

#[derive(Debug, Clone)]
pub struct FanoutConfig {
    trunk: Box<LayerConfig>,
    branches: Vec<LayerConfig>,
}

pub struct Fanout<B: IBackend> {
    descriptor: Descriptor,
    trunk: Box<dyn Layer<B>>,
    branches: Vec<Box<dyn Layer<B>>>,
}

impl FanoutConfig {
    pub fn new(trunk: LayerConfig) -> Self {
        FanoutConfig {
            trunk: Box::new(trunk),
            branches: Vec::new(),
        }
    }

    pub fn with_branch(mut self, branch: LayerConfig) -> Self {
        self.branches.push(branch);
        self
    }
}

impl<B: IBackend + LayerOps<f32> + 'static> Fanout<B> {
    pub fn new(mut descriptor: Descriptor, config: &FanoutConfig) -> Self {
        // Create trunk.
        let trunk_inputs: Vec<Inout> = descriptor.inputs().iter().cloned().collect();
        let trunk = layer_from_config(
            descriptor.sub("trunk", trunk_inputs),
            &config.trunk,
        );

        // Create branches. Each branch will have same inputs which are trunk's outputs.
        let branch_inputs: Vec<Inout> = trunk.descriptor().outputs().iter().cloned().collect();
        let mut branches = Vec::new();
        for (i, branch_config) in config.branches.iter().enumerate() {
            let branch = layer_from_config(
                descriptor.sub(&format!("branch_{}", i), branch_inputs.clone()),
                branch_config,
            );

            // Expose branch outputs as this layer outputs.
            for output in branch.descriptor().outputs() {
                descriptor.add_output_copy(output);
            }

            // Expose branch learnable params as this layer params.
            for params in branch.descriptor().params() {
                descriptor.add_params_copy(params);
            }

            branches.push(branch);
        }

        Fanout {
            descriptor: descriptor,
            trunk: trunk,
            branches: branches,
        }
    }
}

impl<B: IBackend + LayerOps<f32> + 'static> Layer<B> for Fanout<B> {
    fn compute_output(&self, backend: &B, context: &mut Context) {
        self.trunk.compute_output(backend, context);
        for branch in self.branches.iter() {
            branch.compute_output(backend, context);
        }
    }

    fn compute_gradients(&self, backend: &B, context: &mut Context) {
        // Backpropagation process is only supported on a single branch.
        // Determine which branch is used (by finding output gradients on the context)
        // and compute gradients only along that branch.

        let mut branch_index = None;
        for i in 0..self.branches.len() {
            let branch_desc = self.branches[i].descriptor();
            for j in 0..branch_desc.outputs().len() {
                let has_output_gradient = context.has_data_gradient(&branch_desc.outputs()[j]);
                match (branch_index, has_output_gradient) {
                    // Unpopulated output gradient without a selected branch. Nothing to do.
                    (None, false) => (),
                    // This is the first populated output gradient we saw.
                    // Mark this branch as the one we'll be using for backprop.
                    (None, true) => {
                        assert_eq!(j, 0, "Branch {} has partial output gradients; should either have all or none", i);
                        branch_index = Some(i);
                    },
                    // We have a selected branch and this is an unpopulated output gradient.
                    (Some(i2), false) => {
                        assert_ne!(i, i2, "Branch {} has partial output gradients (missing {})", i, j);
                    },
                    // We have a selected branch and this is a populated output gradient.
                    (Some(i2), true) => {
                        assert_eq!(i, i2, 
                        "Seen output gradients on branches {} and {}; backprop only on one branch is supported", i, i2);
                    },
                }
            }
        }

        assert_ne!(branch_index, None, "No output gradiens on any branch");

        self.branches[branch_index.unwrap()].compute_gradients(backend, context);
        self.trunk.compute_gradients(backend, context);
    }

    fn descriptor(&self) -> &Descriptor {
        &self.descriptor
    }

    fn descriptor_mut(&mut self) -> &mut Descriptor {
        &mut self.descriptor
    }
}

impl<B: IBackend> Debug for Fanout<B> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Fanout")
    }
}