use std::cell::RefCell;
use std::collections::HashMap;
use std::iter;
use std::rc::Rc;

use crate::co::{SharedTensor, TensorDesc};
use crate::net::{Inout, LearnableParamsLink};

/// Context stores data for a single invocation of a network (forward and optionally backward),
/// which includes data passed between layers (at Junctions), loss function gradients with respect
/// to this data (again, at Junctions) and also loss function gradients with respect to the
/// learnable parameters.
#[derive(Debug)]
pub struct Context {
    batch_size: usize,

    // Data at junctions.
    data: HashMap<usize, Rc<RefCell<SharedTensor<f32>>>>,

    // Loss function gradients with respect to data at junctions.
    data_gradient: HashMap<usize, Rc<RefCell<SharedTensor<f32>>>>,

    // Loss function gradients with respect to learnable params.
    param_gradient: HashMap<usize, Rc<RefCell<SharedTensor<f32>>>>,
}

impl Context {
    pub fn new(batch_size: usize) -> Self {
        Context {
            batch_size: batch_size,
            data: HashMap::new(),
            data_gradient: HashMap::new(),
            param_gradient: HashMap::new(),
        }
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    pub fn has_data_gradient(&self, inout: &Inout) -> bool {
        Self::has_inout_buffer(&self.data_gradient, inout)
    }

    // Returns data buffer for the given Inout. Panics if this data buffer doesn't exist.
    // Typically used by layers to get input data for the requested computation.
    pub fn get_data(&mut self, inout: &Inout) -> Rc<RefCell<SharedTensor<f32>>> {
        Self::get_inout_buffer(&self.data, inout, "data")
    }

    // Returns data gradient buffer for the given Inout. Panics if this data buffer doesn't exist.
    // Typically used by layers to get input data for the requested computation.
    pub fn get_data_gradient(&mut self, inout: &Inout) -> Rc<RefCell<SharedTensor<f32>>> {
        Self::get_inout_buffer(&self.data_gradient, inout, "data gradient")
    }

    // Same as `get_data()` but creates the buffer on the fly if it doesn't exist.
    // Typically used by layers to get buffers to store outputs of the requested computation.
    pub fn acquire_data(&mut self, inout: &Inout) -> Rc<RefCell<SharedTensor<f32>>> {
        Self::acquire_inout_buffer(&mut self.data, inout, self.batch_size, "data")
    }

    // Same as `get_data_gradient()` but creates the buffer on the fly if it doesn't exist.
    // Typically used by layers to get buffers to store outputs of the requested computation.
    pub fn acquire_data_gradient(&mut self, inout: &Inout) -> Rc<RefCell<SharedTensor<f32>>> {
        Self::acquire_inout_buffer(&mut self.data_gradient, inout, self.batch_size, "data gradient")
    }

    // Takes the tensor out of the context. Panics if no such tensor.
    pub fn take_data(&mut self, inout: &Inout) -> SharedTensor<f32> {
        Self::take_inout_buffer(&mut self.data, inout, "data")
    }

    // Returns params gradient buffer for the given learnable params link.
    // Panics if this data buffer doesn't exist.
    pub fn get_params_gradient(&mut self, params: &LearnableParamsLink) -> Rc<RefCell<SharedTensor<f32>>> {
        let key = Rc::as_ptr(&params) as usize;
        match self.param_gradient.get(&key) {
            Some(data) => data.clone(),
            None => panic!("No params gradient for {:?}", params.borrow()),
        }
    }

    // Same as `get_params_gradient()` but creates the buffer on the fly if it doesn't exist.
    pub fn acquire_params_gradient(&mut self, params: &LearnableParamsLink) -> Rc<RefCell<SharedTensor<f32>>> {
        let key = Rc::as_ptr(&params) as usize;
        match self.param_gradient.get(&key) {
            Some(data) => return data.clone(),
            None => (),
        }

        let shape = params.borrow().data.desc().clone();
        let buffer = SharedTensor::<f32>::new(&shape);
        let buffer_rc = Rc::new(RefCell::new(buffer));
        self.param_gradient.insert(key, buffer_rc.clone());
        buffer_rc
    }

    fn has_inout_buffer(storage: &HashMap<usize, Rc<RefCell<SharedTensor<f32>>>>, inout: &Inout) -> bool {
        let key = Rc::as_ptr(&inout.junction) as usize;
        storage.get(&key).is_some()
    }

    fn get_inout_buffer(
        storage: &HashMap<usize, Rc<RefCell<SharedTensor<f32>>>>,
        inout: &Inout,
        purpose: &str,
    ) -> Rc<RefCell<SharedTensor<f32>>> {
        let key = Rc::as_ptr(&inout.junction) as usize;
        match storage.get(&key) {
            Some(data) => data.clone(),
            None => panic!("No {} for {:?}", purpose, inout),
        }
    }

    fn acquire_inout_buffer(
        storage: &mut HashMap<usize, Rc<RefCell<SharedTensor<f32>>>>,
        inout: &Inout,
        batch_size: usize,
        purpose: &str,
    ) -> Rc<RefCell<SharedTensor<f32>>> {
        let key = Rc::as_ptr(&inout.junction) as usize;
        match storage.get(&key) {
            Some(data) => return data.clone(),
            None => (),
        };

        let shape: TensorDesc = iter::once(batch_size)
            .chain(inout.junction.unit_shape.iter().map(|i| *i))
            .collect();
        let buffer = SharedTensor::<f32>::new(&shape);
        let buffer_rc = Rc::new(RefCell::new(buffer));
        storage.insert(key, buffer_rc.clone());
        buffer_rc
    }

    fn take_inout_buffer(
        storage: &mut HashMap<usize, Rc<RefCell<SharedTensor<f32>>>>,
        inout: &Inout,
        purpose: &str,
    ) -> SharedTensor<f32> {
        let key = Rc::as_ptr(&inout.junction) as usize;
        match storage.remove(&key) {
            Some(data) => Rc::try_unwrap(data).unwrap().into_inner(),
            None => panic!("No {} for {:?}", purpose, inout),
        }
    }
}
