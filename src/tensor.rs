//! Provides the functionality for memory management across devices.
//!
//! A Tensor is a potentially multi-dimensional matrix containing information about the actual data and its structure.
//! A Collenchyma Tensor tracks the memory copies of the numeric data of an Tensor across the devices of the Backend
//! and manages
//!
//! * the location of these memory copies
//! * the location of the latest memory copy and
//! * the synchronisation of memory copies between devices
//!
//! This is important, as this provides a unified data interface for exectuing Tensor operations on CUDA, OpenCL and
//! common host CPU.
//!
//! A [memory copy][mem] represents one logical unit of data, which might be located at the host. The
//! Tensor, tracks the location of the data blob across the various devices that the backend might
//! consist of. This allows us to run operations on various backends with the same data blob.
//!
//! ## Terminology
//!
//! A Tensor is a homogeneous multi-dimensional array - a table of elements (usually numeric elements) of the same type,
//! indexed by tuples of positive integers. In Collenchyma, `dimensions` of a Tensor describe the axis for a
//! coordinate system. The numbers of dimensions is known as the `rank`. A scalar value like `3` has the rank 0, and a Rust array
//! like `[1, 2, 3]` has a rank of 1 as it has one dimension. A array of arrays like `[[1, 2, 3], [2, 3]]` has a rank
//! of 2 as it has two dimensions. The number of elements for a dimension is called `length`.
//! And the number of all elements for each dimension summed up is the `size`. These meta data about a Tensor is called
//! the `descriptor` of the Tensor.
//!
//! [frameworks]: ../frameworks/index.html
//! [mem]: ../memory/index.html
//! ## Examples
//!
//! Create a SharedTensor and fill it with some numbers:
//!
//! ```
//! # extern crate collenchyma;
//! use collenchyma::framework::IFramework;
//! use collenchyma::frameworks::Native;
//! use collenchyma::tensor::SharedTensor;
//! # fn main() {
//! // allocate memory
//! let native = Native::new();
//! let device = native.new_device(native.hardwares()).unwrap();
//! let shared_data = &mut SharedTensor::<i32>::new(&device, &5).unwrap();
//! // fill memory with some numbers
//! let local_data = [0, 1, 2, 3, 4];
//! let data = shared_data.get_mut(&device).unwrap().as_mut_native().unwrap();
//! # }
//! ```

use linear_map::LinearMap;
use device::{IDevice, DeviceType};
use memory::MemoryType;
use std::marker::PhantomData;
use std::{fmt, mem, error};

/// Describes the Descriptor of a SharedTensor.
pub type TensorDesc = Vec<usize>;

#[derive(Debug)]
/// Container that handles synchronization of [Memory][1] of type `T`.
/// [1]: ../memory/index.html
pub struct SharedTensor<T> {
    desc: TensorDesc,
    latest_location: DeviceType,
    latest_copy: MemoryType,
    copies: LinearMap<DeviceType, MemoryType>,
    phantom: PhantomData<T>,
}

/// Describes the Descriptor of a Tensor.
pub trait ITensorDesc {
    /// Returns the rank of the Tensor.
    ///
    /// The rank of the Tensor is the number of its dimensions.
    fn rank(&self) -> usize;

    /// Returns the summed up length of all dimensions of the Tensor.
    ///
    /// A Tensor of rank 2 with the following dimesion specification [5, 5] would have a size of 25.
    fn size(&self) -> usize;

    /// Returns the dimensions of the Tensor.
    ///
    /// To return the length of one dimensions of the Tensor, you would call
    /// tensor_desc.dims()[0] // e.g. 64
    fn dims(&self) -> &Vec<usize>;

    /// Returns the dimensions of the Tensor as Vec<i32>.
    fn dims_i32(&self) -> Vec<i32>;

    /// Returns the default stride for an Rust allocated Tensor.
    ///
    /// A rank 2 Tensor with dimensions [a, b] has a default stride of [b, 1]
    /// A rank 3 Tensor with dimensions [a, b, c] has a default stride of [b * c, c, 1]
    /// A rank 4 Tensor with dimensions [a, b, c, d] has a default stride of [b * c * d, c * d, d, 1]
    /// and so on.
    fn default_stride(&self) -> Vec<usize> {
        let mut strides: Vec<usize> = Vec::with_capacity(self.rank());
        let dim_length = self.dims().len();
        match dim_length {
            0 => strides,
            1 => {
                strides.push(1);
                strides
            },
            _ => {
                let imp_dims = &self.dims()[1..dim_length];
                for (i, _) in imp_dims.iter().enumerate() {
                    strides.push(imp_dims[i..imp_dims.len()].iter().fold(1, |prod, &x| prod * x))
                }
                strides.push(1);
                strides
            }
        }
    }

    /// Returns the default stride for a Rust allocated Tensor as i32.
    fn default_stride_i32(&self) -> Vec<i32> {
        self.default_stride().iter().map(|&e| e as i32).collect()
    }
}

/// Describes a conversion into a Tensor Descriptor.
///
/// This allows for convenient creation of a new SharedTensor.
/// e.g. (2, 4) -> [2,4] or () -> [] or 2 -> [2]
pub trait IntoTensorDesc {
    /// Converts the implemented type into a TensorDesc.
    fn into(&self) -> TensorDesc;
}

impl IntoTensorDesc for () {
    fn into(&self) -> TensorDesc {
        Vec::with_capacity(1)
    }
}

impl IntoTensorDesc for usize {
    fn into(&self) -> TensorDesc {
        vec![*self]
    }
}

impl IntoTensorDesc for u32 {
    fn into(&self) -> TensorDesc {
        vec![*self as usize]
    }
}

impl IntoTensorDesc for isize {
    fn into(&self) -> TensorDesc {
        vec![*self as usize]
    }
}

impl IntoTensorDesc for i32 {
    fn into(&self) -> TensorDesc {
        vec![*self as usize]
    }
}

impl IntoTensorDesc for Vec<usize> {
    fn into(&self) -> TensorDesc {
        self.clone()
    }
}

impl<'a> IntoTensorDesc for &'a [usize] {
    fn into(&self) -> TensorDesc {
        From::from(self.to_owned())
    }
}

impl IntoTensorDesc for (usize, usize) {
    fn into(&self) -> TensorDesc {
        vec![self.0, self.1]
    }
}

impl IntoTensorDesc for (usize, usize, usize) {
    fn into(&self) -> TensorDesc {
        vec![self.0, self.1, self.2]
    }
}

impl IntoTensorDesc for (usize, usize, usize, usize) {
    fn into(&self) -> TensorDesc {
        vec![self.0, self.1, self.2, self.3]
    }
}

impl IntoTensorDesc for (usize, usize, usize, usize, usize) {
    fn into(&self) -> TensorDesc {
        vec![self.0, self.1, self.2, self.3, self.4]
    }
}

impl IntoTensorDesc for (usize, usize, usize, usize, usize, usize) {
    fn into(&self) -> TensorDesc {
        vec![self.0, self.1, self.2, self.3, self.4, self.5]
    }
}

macro_rules! impl_array_into_tensor_desc {
    ($($N:expr)+) => {
        $(
            impl IntoTensorDesc for [usize; $N] {
                fn into(&self) -> TensorDesc {
                    let slice: &[_] = self;
                    From::from(slice)
                }
            }
        )+
    }
}
impl_array_into_tensor_desc!(1 2 3 4 5 6);

impl ITensorDesc for TensorDesc {
    fn rank(&self) -> usize {
        self.len()
    }

    fn size(&self) -> usize {
        match self.rank() {
            0 => 1,
            _ => self.iter().fold(1, |s, &a| s * a)
        }
    }

    fn dims(&self) -> &Vec<usize> {
        self
    }

    fn dims_i32(&self) -> Vec<i32> {
        self.iter().map(|&e| e as i32).collect()
    }
}

impl<T> SharedTensor<T> {
    /// Create new Tensor by allocating [Memory][1] on a Device.
    /// [1]: ../memory/index.html
    pub fn new<D: IntoTensorDesc>(dev: &DeviceType, desc: &D) -> Result<SharedTensor<T>, Error> {
        let copies = LinearMap::<DeviceType, MemoryType>::new();
        let copy = try!(Self::alloc_on_device(dev, desc));
        let tensor_desc: TensorDesc = desc.into();
        Ok(SharedTensor {
            desc: tensor_desc,
            latest_location: dev.clone(),
            latest_copy: copy,
            copies: copies,
            phantom: PhantomData,
        })
    }

    /// Change the shape of the Tensor.
    ///
    /// Will return an Error if size of new shape is not equal to the old shape.
    /// If you want to change the shape to one of a different size, use `resize`.
    pub fn reshape<D: IntoTensorDesc>(&mut self, desc: &D) -> Result<(), Error> {
        let new_desc: TensorDesc = desc.into();
        if new_desc.size() == self.desc().size() {
            self.desc = new_desc;
            Ok(())
        } else {
            Err(Error::InvalidShape("Size of the provided shape is not equal to the old shape."))
        }
    }

    /// Change the size and shape of the Tensor.
    ///
    /// **Caution**: Drops all copies which are not on the current device.
    ///
    /// 'reshape' is preffered over this method if the size of the old and new shape
    /// are identical because it will not reallocate memory.
    pub fn resize<D: IntoTensorDesc>(&mut self, desc: &D) -> Result<(), Error> {
        self.copies.clear();
        self.latest_copy = try!(Self::alloc_on_device(self.latest_device(), desc));
        let new_desc: TensorDesc = desc.into();
        self.desc = new_desc;
        Ok(())
    }

    /// Allocate memory on the provided DeviceType.
    fn alloc_on_device<D: IntoTensorDesc>(dev: &DeviceType, desc: &D) -> Result<MemoryType, Error> {
        let tensor_desc: TensorDesc = desc.into();
        let alloc_size = Self::mem_size(tensor_desc.size());
        let copy = match *dev {
            #[cfg(feature = "native")]
            DeviceType::Native(ref cpu) => MemoryType::Native(try!(cpu.alloc_memory(alloc_size))),
            #[cfg(feature = "opencl")]
            DeviceType::OpenCL(ref context) => MemoryType::OpenCL(try!(context.alloc_memory(alloc_size))),
            #[cfg(feature = "cuda")]
            DeviceType::Cuda(ref context) => MemoryType::Cuda(try!(context.alloc_memory(alloc_size))),
        };
        Ok(copy)
    }

    /// Synchronize memory from latest location to `destination`.
    pub fn sync(&mut self, destination: &DeviceType) -> Result<(), Error> {
        if &self.latest_location != destination {
            let latest = self.latest_location.clone();
            try!(self.sync_from_to(&latest, &destination));

            let mut swap_location = destination.clone();
            let mut swap_copy = try!(self.copies.remove(destination).ok_or(Error::MissingDestination("Tensor does not hold a copy on destination device.")));
            mem::swap(&mut self.latest_location, &mut swap_location);
            mem::swap(&mut self.latest_copy, &mut swap_copy);
            self.copies.insert(swap_location, swap_copy);
        }
        Ok(())
    }

    /// Get a reference to the memory copy on the provided `device`.
    ///
    /// Returns `None` if there is no memory copy on the device.
    pub fn get(&self, device: &DeviceType) -> Option<&MemoryType> {
        // first check if device is not current location. This is cheaper than a lookup in `copies`.
        if &self.latest_location == device {
            return Some(&self.latest_copy)
        }
        self.copies.get(device)
    }

    /// Get a mutable reference to the memory copy on the provided `device`.
    ///
    /// Returns `None` if there is no memory copy on the device.
    pub fn get_mut(&mut self, device: &DeviceType) -> Option<&mut MemoryType> {
        // first check if device is not current location. This is cheaper than a lookup in `copies`.
        if &self.latest_location == device {
            return Some(&mut self.latest_copy)
        }
        self.copies.get_mut(device)
    }

    /// Synchronize memory from `source` device to `destination` device.
    fn sync_from_to(&mut self, source: &DeviceType, destination: &DeviceType) -> Result<(), Error> {
        if source != destination {
            match self.copies.get_mut(destination) {
                Some(mut destination_copy) => {
                    match destination {
                        #[cfg(feature = "native")]
                        &DeviceType::Native(ref cpu) => {
                            match destination_copy.as_mut_native() {
                                Some(ref mut mem) => try!(cpu.sync_in(&self.latest_location, &self.latest_copy, mem)),
                                None => return Err(Error::InvalidMemory("Expected Native Memory (FlatBox)"))
                            }
                        },
                        #[cfg(feature = "cuda")]
                        &DeviceType::Cuda(ref context) => {
                            match destination_copy.as_mut_cuda() {
                                Some(ref mut mem) => try!(context.sync_in(&self.latest_location, &self.latest_copy, mem)),
                                None => return Err(Error::InvalidMemory("Expected CUDA Memory."))
                            }
                        },
                        #[cfg(feature = "opencl")]
                        &DeviceType::OpenCL(ref context) => {
                            match destination_copy.as_mut_opencl() {
                                Some(ref mut mem) => try!(context.sync_in(&self.latest_location, &self.latest_copy, mem)),
                                None => return Err(Error::InvalidMemory("Expected OpenCL Memory."))
                            }
                        }
                    }
                    Ok(())
                },
                None => Err(Error::MissingDestination("Tensor does not hold a copy on destination device."))
            }
        } else {
            Ok(())
        }
    }

    /// Removes Copy from SharedTensor and therefore aquires ownership over the removed memory copy for synchronizing.
    pub fn remove_copy(&mut self, destination: &DeviceType) -> Result<(MemoryType), Error> {
        // If `destination` holds the latest data, sync to another memory first, before removing it.
        if &self.latest_location == destination {
            let first = self.copies.keys().nth(0).unwrap().clone();
            try!(self.sync(&first));
        }
        match self.copies.remove(destination) {
            Some(destination_cpy) => Ok(destination_cpy),
            None => Err(Error::MissingDestination("Tensor does not hold a copy on destination device."))
        }
    }

    /// Return ownership over a memory copy after synchronizing.
    fn return_copy(&mut self, dest: &DeviceType, dest_mem: MemoryType) {
        self.copies.insert(dest.clone(), dest_mem);
    }

    /// Track a new `device` and allocate memory on it.
    ///
    /// Returns an error if the Tensor is already tracking the `device`.
    pub fn add_device(&mut self, device: &DeviceType) -> Result<&mut Self, Error> {
        // first check if device is not current location. This is cheaper than a lookup in `copies`.
        if &self.latest_location == device {
            return Err(Error::InvalidMemoryAllocation("Tensor already tracks memory for this device. No memory allocation."))
        }
        match self.copies.get(device) {
            Some(_) => Err(Error::InvalidMemoryAllocation("Tensor already tracks memory for this device. No memory allocation.")),
            None => {
                let copy: MemoryType;
                match *device {
                    #[cfg(feature = "native")]
                    DeviceType::Native(ref cpu) => copy = MemoryType::Native(try!(cpu.alloc_memory(Self::mem_size(self.capacity())))),
                    #[cfg(feature = "opencl")]
                    DeviceType::OpenCL(ref context) => copy = MemoryType::OpenCL(try!(context.alloc_memory(Self::mem_size(self.capacity())))),
                    #[cfg(feature = "cuda")]
                    DeviceType::Cuda(ref context) => copy = MemoryType::Cuda(try!(context.alloc_memory(Self::mem_size(self.capacity())))),
                };
                self.copies.insert(device.clone(), copy);
                Ok(self)
            }
        }
    }

    /// Returns the device that contains the up-to-date memory copy.
    pub fn latest_device(&self) -> &DeviceType {
        &self.latest_location
    }

    /// Returns the number of elements for which the Tensor has been allocated.
    pub fn capacity(&self) -> usize {
        self.desc.size()
    }

    /// Returns the descriptor of the Tensor.
    pub fn desc(&self) -> &TensorDesc {
        &self.desc
    }

    /// Returns the allocated Memory size in bytes.
    pub fn mem_size(capacity: usize) -> usize {
        mem::size_of::<T>() * capacity
    }
}

/// Errors than can occur when synchronizing memory.
#[derive(Debug, Copy, Clone)]
pub enum Error {
    /// No copy on source device.
    MissingSource(&'static str),
    /// No copy on destination device.
    MissingDestination(&'static str),
    /// No valid MemoryType provided. Other than expected.
    InvalidMemory(&'static str),
    /// No memory allocation on specified device happened.
    InvalidMemoryAllocation(&'static str),
    /// Unable to remove Memory copy from SharedTensor.
    InvalidRemove(&'static str),
    /// Framework error at memory allocation.
    MemoryAllocationError(::device::Error),
    /// Framework error at memory synchronization.
    MemorySynchronizationError(::device::Error),
    /// Shape provided for reshaping is not compatible with old shape.
    InvalidShape(&'static str)
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Error::MissingSource(ref err) => write!(f, "{:?}", err),
            Error::MissingDestination(ref err) => write!(f, "{:?}", err),
            Error::InvalidMemory(ref err) => write!(f, "{:?}", err),
            Error::InvalidMemoryAllocation(ref err) => write!(f, "{:?}", err),
            Error::InvalidRemove(ref err) => write!(f, "{:?}", err),
            Error::MemoryAllocationError(ref err) => write!(f, "{}", err),
            Error::MemorySynchronizationError(ref err) => write!(f, "{}", err),
            Error::InvalidShape(ref err) => write!(f, "{}", err),
        }
    }
}

impl error::Error for Error {
    fn description(&self) -> &str {
        match *self {
            Error::MissingSource(ref err) => err,
            Error::MissingDestination(ref err) => err,
            Error::InvalidMemory(ref err) => err,
            Error::InvalidMemoryAllocation(ref err) => err,
            Error::InvalidRemove(ref err) => err,
            Error::MemoryAllocationError(ref err) => err.description(),
            Error::MemorySynchronizationError(ref err) => err.description(),
            Error::InvalidShape(ref err) => err,
        }
    }

    fn cause(&self) -> Option<&error::Error> {
        match *self {
            Error::MissingSource(_) => None,
            Error::MissingDestination(_) => None,
            Error::InvalidMemory(_) => None,
            Error::InvalidMemoryAllocation(_) => None,
            Error::InvalidRemove(_) => None,
            Error::MemoryAllocationError(ref err) => Some(err),
            Error::MemorySynchronizationError(ref err) => Some(err),
            Error::InvalidShape(_) => None,
        }
    }
}

impl From<Error> for ::error::Error {
    fn from(err: Error) -> ::error::Error {
        ::error::Error::Tensor(err)
    }
}
