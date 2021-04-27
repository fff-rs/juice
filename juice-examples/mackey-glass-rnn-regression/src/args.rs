use serde::Deserialize;
use std::path::PathBuf;

pub const MAIN_USAGE: &str = "
Demonstrate RNN caps of juice with the cuda backend.

Usage:
    mackey-glass-example train [--batch-size=<batch>] [--learning-rate=<lr>] [--momentum=<f>] <networkfile>
    mackey-glass-example test [--batch-size=<batch>] <networkfile>

Options:
    -b, --batch-size=<batch>    Network Batch Size.
    -l, --learning-rate=<lr>    Learning Rate.
    -m, --momentum=<f>         Momentum.
    -h, --help                 Show this screen.
";

#[allow(non_snake_case)]
#[derive(Deserialize, Debug, Default)]
pub struct Args {
    pub cmd_train: bool,
    pub cmd_test: bool,
    pub flag_batch_size: Option<usize>,
    pub flag_learning_rate: Option<f32>,
    pub flag_momentum: Option<f32>,
    /// Path to the stored network.
    pub arg_networkfile: PathBuf,
}

impl Args {
    pub(crate) fn data_mode(&self) -> DataMode {
        assert_ne!(self.cmd_train, self.cmd_test);
        if self.cmd_train {
            return DataMode::Train;
        }
        if self.cmd_test {
            return DataMode::Test;
        }
        unreachable!("nope");
    }
}

pub const fn default_learning_rate() -> f32 {
    0.10_f32
}

pub const fn default_momentum() -> f32 {
    0.00
}

pub const fn default_batch_size() -> usize {
    10
}

impl std::cmp::PartialEq for Args {
    fn eq(&self, other: &Self) -> bool {
        match (self.flag_learning_rate, other.flag_learning_rate) {
            (Some(lhs), Some(rhs)) if (rhs - lhs).abs() < 1e6 => {}
            (None, None) => {}
            _ => return false,
        }
        match (self.flag_momentum, other.flag_momentum) {
            (Some(lhs), Some(rhs)) if (rhs - lhs).abs() < 1e6 => {}
            (None, None) => {}
            _ => return false,
        }
        self.cmd_test == other.cmd_test
            && self.cmd_train == other.cmd_train
            && self.arg_networkfile == other.arg_networkfile
            && self.flag_batch_size == other.flag_batch_size
    }
}

impl std::cmp::Eq for Args {}

pub enum DataMode {
    Train,
    Test,
}

impl DataMode {
    pub fn as_path(&self) -> &'static str {
        match self {
            DataMode::Train => "assets/norm_mackeyglass_train.csv",
            DataMode::Test => "assets/norm_mackeyglass_test.csv",
        }
    }
}
