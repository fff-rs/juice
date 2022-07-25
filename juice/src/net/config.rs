use crate::net::container::SequentialConfig;

/// A configuration of the layer.
/// Determines the type of the layer and the (optional) layer settings.
#[derive(Debug, Clone)]
pub enum LayerConfig {
    Relu,
    Sequential(SequentialConfig),
    // TODO: Add other layer configs.
}

impl Default for LayerConfig {
    fn default() -> LayerConfig {
        LayerConfig::Sequential(SequentialConfig::new())
    }
}
