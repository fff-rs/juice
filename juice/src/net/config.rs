use super::{LinearConfig, SequentialConfig, NegativeLogLikelihoodConfig};

/// A configuration of the layer.
/// Determines the type of the layer and the (optional) layer settings.
#[derive(Debug, Clone)]
pub enum LayerConfig {
    Linear(LinearConfig),
    MeanSquaredError,
    NegativeLogLikelihood(NegativeLogLikelihoodConfig),
    Relu,
    Sequential(SequentialConfig),
    Sigmoid,
    // TODO: Add other layer configs.
}

impl Default for LayerConfig {
    fn default() -> LayerConfig {
        LayerConfig::Sequential(SequentialConfig::new())
    }
}

impl From<SequentialConfig> for LayerConfig {
    fn from(c: SequentialConfig) -> Self {
        LayerConfig::Sequential(c)
    }
}

impl From<LinearConfig> for LayerConfig {
    fn from(c: LinearConfig) -> Self {
        LayerConfig::Linear(c)
    }
}
