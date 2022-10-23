use std::cell::RefCell;

use std::rc::Rc;

use crate::co::{SharedTensor, TensorDesc};

/// A junction is a point connecting a layer output to another layer input (or several of them).
/// Junction describes the shape of the data which flows through it (except for the batch size
/// part of it, which is a property of specific invocation and is captured in a `Context`).
/// Junctions are used to locate data buffers in `Context`: each junction will have an associated
/// buffer in a `Context` (two of them, actually: one for data and another for gradient).
#[derive(Debug)]
pub struct Junction {
    /// Shape of one data unit in a batch.
    pub unit_shape: TensorDesc,
    /// Human-readable path of the junction, mostly for logging and debugging.
    /// Can change after creation to support layer waterfall construction/connection model.
    pub path: RefCell<String>,
}

/// A struct representing either an input or output of a layer.
/// Inouts that share same junction are "connected" in the sense that they use the same buffer
/// (written to by output Inout and read from by input Inout).
#[derive(Debug, Clone)]
pub struct Inout {
    // Junction used to read/write data.
    pub junction: Rc<Junction>,
}

/// A single blob of params that can be learned during network training.
/// Params are "owned" by the layers, but layers must expose them via LearnableParamsLink in
/// the Descriptor so they can be changed during learning. Container layers must expose all nested
/// layers params.
#[derive(Debug)]
pub struct LearnableParams {
    // Params data.
    pub data: SharedTensor<f32>,
    // 1.0 by default, can be changed to control how fast these weights are changed compared
    // to overall learning process.
    pub learning_rate: f32,
    /// Human-readable path which includes the owner layer path, mostly for logging and debugging.
    pub path: String,
}

/// A pointer to an instance of LearnableParams. Among other things is used to find associated
/// gradient buffer in the `Context`.
pub type LearnableParamsLink = Rc<RefCell<LearnableParams>>;

/// Descriptor of a layer, containing information about layer path, inputs, outputs and params.
/// Inputs and outputs are created using a waterfall model:
/// 1. Parent logic creates a Descriptor with a path and inputs (with shapes).
/// 2. Descriptor is passed to the layer constructor. Layer determines the number and shapes of
///    the outputs and adds them to the descriptor. Params are also initialized and added if
///    layer uses them.
/// 3. After layer is created, parent logic assigns human-readable paths to the outputs Junctions
///    and connects them appropriately to downstream layers.
#[derive(Debug, Clone)]
pub struct Descriptor {
    path: String,
    inputs: Vec<Inout>,
    outputs: Vec<Inout>,

    // All learnable params of the layer. For container layers, this must contain all params
    // of the nested layers.
    params: Vec<LearnableParamsLink>,
}

impl Inout {
    pub fn new(unit_shape: TensorDesc) -> Self {
        Inout {
            junction: Rc::new(Junction {
                unit_shape,
                path: RefCell::new("".to_owned()),
            }),
        }
    }

    pub fn new_with_junction(junction: Rc<Junction>) -> Self {
        Inout { junction }
    }

    // "Connects" this Inout to Inout `to` by making it point to the same Junction.
    pub fn connect(&mut self, to: &Inout) {
        self.junction = to.junction.clone();
    }

    // Unit shape of this Inout.
    pub fn unit_shape(&self) -> &TensorDesc {
        &self.junction.unit_shape
    }
}

impl Descriptor {
    // Create a top-level Descriptor.
    pub fn top(name: &str, inputs: Vec<Inout>) -> Self {
        Descriptor {
            path: name.to_owned(),
            inputs,
            outputs: Vec::new(),
            params: Vec::new(),
        }
    }

    // Create a Descriptor which is nested under this one.
    // In practice, "nested" only means the new descriptor path is constructed as
    // "<parent_path>.<name>".
    pub fn sub(&self, name: &str, inputs: Vec<Inout>) -> Self {
        Descriptor {
            path: format!("{}.{}", self.path, name),
            inputs,
            outputs: Vec::new(),
            params: Vec::new(),
        }
    }

    pub fn path(&self) -> &str {
        &self.path
    }

    pub fn inputs(&self) -> &[Inout] {
        &self.inputs
    }

    pub fn input(&self, index: usize) -> &Inout {
        &self.inputs[index]
    }

    pub fn outputs(&self) -> &[Inout] {
        &self.outputs
    }

    pub fn output(&self, index: usize) -> &Inout {
        &self.outputs[index]
    }

    pub fn output_mut(&mut self, index: usize) -> &mut Inout {
        &mut self.outputs[index]
    }

    pub fn params(&self) -> &[LearnableParamsLink] {
        &self.params
    }

    pub fn param(&self, index: usize) -> &LearnableParamsLink {
        &self.params[index]
    }

    pub fn add_output(&mut self, unit_shape: TensorDesc) -> &mut Inout {
        self.outputs.push(Inout::new(unit_shape));
        self.outputs.last_mut().unwrap()
    }

    pub fn add_output_with_junction(&mut self, junction: Rc<Junction>) -> &mut Inout {
        self.outputs.push(Inout::new_with_junction(junction));
        self.outputs.last_mut().unwrap()
    }

    // Creates a LearnableParams instances, stores it in the Descriptor and returns a
    // link to it (LearnableParamsLink).
    pub fn create_params(&mut self, name: &str, data: SharedTensor<f32>, learning_rate: f32) -> LearnableParamsLink {
        let params = LearnableParams {
            data,
            learning_rate,
            path: format!("{}.{}", self.path, name),
        };
        let params_rc = Rc::new(RefCell::new(params));
        self.params.push(params_rc.clone());
        params_rc
    }

    // Add a link to existing LearnableParams instance. Used by parent layers to collect
    // link to LearnableParams used by children layers.
    pub fn add_params_copy(&mut self, params: &LearnableParamsLink) {
        self.params.push(params.clone());
    }
}
