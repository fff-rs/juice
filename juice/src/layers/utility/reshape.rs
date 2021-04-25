//! Utility layer to give a tensor another shape.
//!
//! This layer should be used as in-place operation,
//! so the tensor that should be reshaped should be specified
//! as both input and output.
//!
//! Reshaping a tensor is required so that it becomes
//! usable for Layers that interpret meaning into the shape of
//! the tensor.
//!
//! A lot of layers interpret the last dimensions as NCHW,
//! where the letters stand for:
//!
//! - `N` : number of batch samples
//! - `C` : number of feature maps
//! - `H` : height
//! - `W` : width

use crate::capnp_util::*;
use crate::co::{IBackend, SharedTensor, TensorDesc};
use crate::juice_capnp::reshape_config as capnp_config;
use crate::layer::*;
use crate::util::ArcLock;
use anyhow::{anyhow, Result};

#[derive(Debug, Clone)]
/// Reshape Utility Layer
pub struct Reshape {
    shape: Vec<isize>,
}

impl Reshape {
    /// Create a Reshape layer from a ReshapeConfig.
    pub fn from_config(config: &ReshapeConfig) -> Reshape {
        Reshape {
            shape: config.shape.clone(),
        }
    }

    fn evaluate_shape(&self, input_shape: &TensorDesc) -> Result<Vec<usize>> {
        dbg!(&self.shape);
        dbg!(input_shape);
        let unknown_dimensions: usize = self.shape.iter().filter(|x| **x == -1).count();
        let invalid_dimensions: usize = self.shape.iter().filter(|x| **x < -1).count();
        if invalid_dimensions > 0 {
            return Err(anyhow!("Invalid elements provided to Reshape"))
        }
        return match unknown_dimensions {
            0 => Ok(self.shape.clone().into_iter().map(|x| x as usize).collect()),
            1 => {
                let total_prior_elements: usize = input_shape.iter().product();
                let known_elements: usize = self.shape.iter().filter(|x| **x > -1).product::<isize>() as usize;
                dbg!(total_prior_elements);
                dbg!(known_elements);
                if total_prior_elements != (total_prior_elements / known_elements * known_elements) {
                    Err(anyhow!(
                        "Dimensions {:?} do not cleanly reshape into {:?}",
                        input_shape, self.shape
                    ))
                } else {
                    let unknown_element: usize = total_prior_elements / known_elements;
                    Ok(self.shape
                        .iter()
                        .map(|x| if *x == -1 { unknown_element } else { *x as usize })
                        .collect())
                }
            }
            _ => Err(anyhow!("More than 2 unknown elements provided to Reshape")),
        }
    }
}

impl<B: IBackend> ILayer<B> for Reshape {
    fn compute_in_place(&self) -> bool {
        true
    }

    fn auto_output_blobs(&self) -> bool {
        false
    }

    fn reshape(
        &mut self,
        backend: ::std::rc::Rc<B>,
        input_data: &mut Vec<ArcLock<SharedTensor<f32>>>,
        input_gradient: &mut Vec<ArcLock<SharedTensor<f32>>>,
        weights_data: &mut Vec<ArcLock<SharedTensor<f32>>>,
        weights_gradient: &mut Vec<ArcLock<SharedTensor<f32>>>,
        output_data: &mut Vec<ArcLock<SharedTensor<f32>>>,
        output_gradient: &mut Vec<ArcLock<SharedTensor<f32>>>,
    ) {
        // Shape Evaluation has to be done at run-time.
        if !input_data.is_empty() {
            let output_shape : Vec<usize> = match input_data[0].read() {
                Ok(tensor) => self.evaluate_shape(tensor.desc()).unwrap(),
                Err(E) => panic!("")
            };
            output_data[0].write().unwrap().resize(&output_shape).unwrap();
            let output_grad_shape : Vec<usize> = match input_gradient[0].read() {
                Ok(tensor) => self.evaluate_shape(tensor.desc()).unwrap(),
                Err(E) => panic!("")
            };
            output_gradient[0].write().unwrap().resize(&output_grad_shape).unwrap();
        }
    }
}

impl<B: IBackend> ComputeOutput<f32, B> for Reshape {
    fn compute_output(
        &self,
        backend: &B,
        _weights: &[&SharedTensor<f32>],
        input_data: &[&SharedTensor<f32>],
        output_data: &mut [&mut SharedTensor<f32>],
    ) {
    }
}

impl<B: IBackend> ComputeInputGradient<f32, B> for Reshape {
    fn compute_input_gradient(
        &self,
        backend: &B,
        weights_data: &[&SharedTensor<f32>],
        output_data: &[&SharedTensor<f32>],
        output_gradients: &[&SharedTensor<f32>],
        input_data: &[&SharedTensor<f32>],
        input_gradients: &mut [&mut SharedTensor<f32>],
    ) {
    }
}

impl<B: IBackend> ComputeParametersGradient<f32, B> for Reshape {}

#[derive(Debug, Clone)]
/// Specifies configuration parameters for a Reshape Layer.
pub struct ReshapeConfig {
    /// The target shape that the input should assume.
    ///
    /// Preceding dimensions are treated as independent inputs. At most one value can be -1,
    /// indicating that the size of that element should be the remaining element dimensions, i.e.
    /// Input [2,8] -> Reshape [-1, 4] -> Output [4, 4]
    /// As the input has 16 elements, 16 / 4 is 4, so the output is [4, 4]
    ///
    /// Causes an error if the total elements are incompatible with the dimensions selected.
    ///
    /// Defaults to `1`
    pub shape: Vec<isize>,
}

impl ReshapeConfig {
    /// Create a ReshapeConfig that describes a Reshape layer with a provided shape.
    pub fn of_shape(shape: &[isize]) -> ReshapeConfig {
        ReshapeConfig {
            shape: shape.to_owned(),
        }
    }
}

impl<'a> CapnpWrite<'a> for ReshapeConfig {
    type Builder = capnp_config::Builder<'a>;

    /// Write the ReshapeConfig into a capnp message.
    fn write_capnp(&self, builder: &mut Self::Builder) {
        let mut shape = builder.reborrow().init_shape(self.shape.len() as u32);
        for (i, dim) in self.shape.iter().enumerate() {
            shape.set(i as u32, *dim as i64);
        }
    }
}

impl<'a> CapnpRead<'a> for ReshapeConfig {
    type Reader = capnp_config::Reader<'a>;

    fn read_capnp(reader: Self::Reader) -> Self {
        let read_shape = reader.get_shape().unwrap();
        let mut shape: Vec<isize> = Vec::new();
        for i in 0..read_shape.len() {
            shape.push(read_shape.get(i) as isize)
        }

        ReshapeConfig { shape: shape }
    }
}

impl Into<LayerType> for ReshapeConfig {
    fn into(self) -> LayerType {
        LayerType::Reshape(self)
    }
}
