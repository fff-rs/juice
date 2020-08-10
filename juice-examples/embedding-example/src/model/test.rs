use coaster::frameworks::cuda::get_cuda_backend;
use coaster::prelude::*;
use coaster_nn::{DirectionMode, RnnInputMode, RnnNetworkMode};
use juice::layer::*;
use juice::layers::*;
use juice::solver::*;
use juice::util::*;
use std::rc::Rc;
use std::sync::{Arc, RwLock};

use crate::model::params::PHRASE_LENGTH;

pub(crate) fn evaluate_batch(
    network: &mut Solver<Backend<Cuda>, Backend<Cuda>>,
    batch_size: usize,
    mut data_loader: impl Iterator,
) {
    unimplemented!()
}
