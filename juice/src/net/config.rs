// Reexport layer configs.
pub use crate::net::{
    common::LinearConfig, container::SequentialConfig, container::FanoutConfig,
    loss::NegativeLogLikelihoodConfig,
};

#[derive(Debug, Clone)]
pub enum LayerConfig {
    Fanout(FanoutConfig),
    Sequential(SequentialConfig),
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
