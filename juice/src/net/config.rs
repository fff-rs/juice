use super::*;

#[derive(Debug, Clone)]
pub enum LayerConfig {
    Sequential(SequentialConfig),
    Linear(LinearConfig),
    LogSoftmax,
    NegativeLogLikelihood(NegativeLogLikelihoodConfig),
}

impl Default for LayerConfig {
    fn default() -> LayerConfig {
        LayerConfig::Sequential(SequentialConfig::default())
    }
}