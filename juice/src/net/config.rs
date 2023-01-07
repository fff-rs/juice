use super::{
    ConvolutionConfig, DropoutConfig, LinearConfig, NegativeLogLikelihoodConfig, PoolingConfig, SequentialConfig,
};

/// A configuration of the layer.
/// Determines the type of the layer and the (optional) layer settings.
#[derive(Debug, Clone)]
pub enum LayerConfig {
    Convolution(ConvolutionConfig),
    Dropout(DropoutConfig),
    Linear(LinearConfig),
    LogSoftmax,
    MeanSquaredError,
    NegativeLogLikelihood(NegativeLogLikelihoodConfig),
    Pooling(PoolingConfig),
    Relu,
    Sequential(SequentialConfig),
    Sigmoid,
    Softmax,
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

impl From<ConvolutionConfig> for LayerConfig {
    fn from(c: ConvolutionConfig) -> Self {
        LayerConfig::Convolution(c)
    }
}

impl From<DropoutConfig> for LayerConfig {
    fn from(c: DropoutConfig) -> Self {
        LayerConfig::Dropout(c)
    }
}

impl From<PoolingConfig> for LayerConfig {
    fn from(c: PoolingConfig) -> Self {
        LayerConfig::Pooling(c)
    }
}

impl From<NegativeLogLikelihoodConfig> for LayerConfig {
    fn from(c: NegativeLogLikelihoodConfig) -> Self {
        LayerConfig::NegativeLogLikelihood(c)
    }
}
