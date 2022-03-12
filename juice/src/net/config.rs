use crate::net::activation::*;
use crate::net::common::*;
use crate::net::container::*;
use crate::net::loss::*;

// Reexport layer configs.
pub use crate::net::{
    common::LinearConfig, container::FanoutConfig, container::SequentialConfig,
    loss::NegativeLogLikelihoodConfig,
};

#[derive(Debug, Clone)]
pub enum LayerConfig {
    Sequential(SequentialConfig),
    Fanout(FanoutConfig),
    Linear(LinearConfig),
    LogSoftmax,
    Relu,
    Sigmoid,
    MeanSquaredError,
    NegativeLogLikelihood(NegativeLogLikelihoodConfig),
}

impl Default for LayerConfig {
    fn default() -> LayerConfig {
        LayerConfig::Sequential(SequentialConfig::default())
    }
}
