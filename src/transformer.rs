pub trait Transformer {

    /// Transforms non-numeric data into a numeric Vector/Matrix
    ///
    /// The dimension attribute can be used to controll the numeric representation of the output.
    fn transform(&self, dimensions: u32) -> Option<Vec<u32>>;
}
