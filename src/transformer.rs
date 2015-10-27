/// The Transformer Trait
///
/// Gets implemented for all Transformable Data Types.
/// Allows all Transformable Data Types to get transformed into a `Blob`.
pub trait Transformer {

    /// Transforms non-numeric data into a numeric Vector/Matrix
    ///
    /// The dimension attribute can be used to controll the numeric representation of the output.
    fn transform(&self, dimensions: u32) -> Option<Vec<u32>>;
}
