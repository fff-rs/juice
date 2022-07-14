/// A configuration of the layer.
/// Determines the type of the layer and the (optional) layer settings.
#[derive(Debug, Clone)]
pub enum LayerConfig {
    Relu,
    // TODO: Add other layer configs.
}

impl Default for LayerConfig {
    fn default() -> LayerConfig {
        // TODO: Change to empty sequential when Sequential layer is added.
        LayerConfig::Relu
    }
}