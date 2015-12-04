//! Provides the functionality for memory management across devices.
//!
//! A SharedMemory tracks the memory copies across the devices of the Backend and manages
//!
//! * the location of these memory copies
//! * the location of the latest memory copy and
//! * the synchronisation of memory copies between devices
//!
//! A [memory copy][mem] represents one logical unit of data, which might me located at the host. The
//! SharedMemory, tracks the location of the data blob across the various devices that the backend might
//! consist of. This allows us to run operations on various backends with the same data blob.
//!
//! [frameworks]: ../frameworks/index.html
//! [mem]: ../memory/index.html
//!
//! ## Examples
//!
//! Create SharedMemory and fill it with some numbers:
//!
//! ```
//! #![feature(clone_from_slice)]
//! # extern crate collenchyma;
//! use collenchyma::framework::IFramework;
//! use collenchyma::frameworks::Native;
//! use collenchyma::shared_memory::{SharedMemory, ITensor, TensorR1};
//! # fn main() {
//! // allocate memory
//! let native = Native::new();
//! let device = native.new_device(native.hardwares()).unwrap();
//! let shared_data = &mut SharedMemory::<i32, TensorR1>::new(&device, TensorR1::new([5])).unwrap();
//! // fill memory with some numbers
//! let local_data = [0, 1, 2, 3, 4];
//! let data = shared_data.get_mut(&device).unwrap().as_mut_native().unwrap();
//! data.as_mut_slice().clone_from_slice(&local_data);
//! # }
//! ```

use linear_map::LinearMap;
use device::{IDevice, DeviceType};
use memory::MemoryType;
use std::marker::PhantomData;
use std::{fmt, mem, error};

// #[derive(Debug)]
/// Container that handles synchronization of [Memory][1] of type `T`.
/// [1]: ../memory/index.html
#[allow(missing_debug_implementations)] // due to LinearMap
pub struct SharedMemory<T, D: ITensor> {
    latest_location: DeviceType,
    latest_copy: MemoryType,
    copies: LinearMap<DeviceType, MemoryType>,
    dim: D,
    phantom: PhantomData<T>,
}

/// Describes the dimensionality of a slice.
///
/// Is used for implementation of the exact Tensor ranks.
pub trait ITensor {
    /// Returns the dimensionality of the Tensor.
    #[allow(non_snake_case)]
    fn D() -> usize;

    /// Returns the number of elements represented by a Tensor - its capacity.
    ///
    /// A TensorR2[5, 5] would contain 25 elements.
    fn elements(&self) -> usize;

    /// Returns the dimensions as a slice.
    fn dims(&self) -> Vec<usize>;
}

#[derive(Debug, Copy, Clone)]
/// Describes a scala value.
pub struct TensorR0;
impl TensorR0 {
    /// Initializes a new TensorR0
    pub fn new() -> TensorR0 {
        TensorR0
    }
}
impl ITensor for TensorR0 {
    fn D() -> usize { 0 }

    fn elements(&self) -> usize { 1 }

    fn dims(&self) -> Vec<usize> { vec!() }
}

#[derive(Debug, Copy, Clone)]
/// Describes a vector value.
pub struct TensorR1 {
    dims: [usize; 1],
}
impl TensorR1 {
    /// Initializes a new TensorR1 with the cardinality of it dimensions.
    pub fn new(dims: [usize; 1]) -> TensorR1 {
        TensorR1 { dims: dims }
    }
}
impl ITensor for TensorR1 {
    fn D() -> usize { 1 }

    fn elements(&self) -> usize {
        self.dims[0]
    }

    fn dims(&self) -> Vec<usize> {
        self.dims.to_vec()
    }
}

#[derive(Debug, Copy, Clone)]
/// Describes a matrix value.
pub struct TensorR2 {
    dims: [usize; 2],
}
impl TensorR2 {
    /// Initializes a new TensorR2 with the cardinality of it dimensions.
    pub fn new(dims: [usize; 2]) -> TensorR2 {
        TensorR2 { dims: dims }
    }
}
impl ITensor for TensorR2 {
    fn D() -> usize { 2 }

    fn elements(&self) -> usize {
        self.dims[0] * self.dims[1]
    }

    fn dims(&self) -> Vec<usize> {
        self.dims.to_vec()
    }
}

#[derive(Debug, Copy, Clone)]
/// Describes a rank 3 Tensor value.
pub struct TensorR3 {
    dims: [usize; 3],
}
impl TensorR3 {
    /// Initializes a new TensorR3 with the cardinality of it dimensions.
    pub fn new(dims: [usize; 3]) -> TensorR3 {
        TensorR3 { dims: dims }
    }
}
impl ITensor for TensorR3 {
    fn D() -> usize { 3 }

    fn elements(&self) -> usize {
        self.dims[0] * self.dims[1] * self.dims[2]
    }

    fn dims(&self) -> Vec<usize> {
        self.dims.to_vec()
    }
}

#[derive(Debug, Copy, Clone)]
/// Describes a rank 4 Tensor value.
pub struct TensorR4 {
    dims: [usize; 4],
}
impl TensorR4 {
    /// Initializes a new TensorR4 with the cardinality of it dimensions.
    pub fn new(dims: [usize; 4]) -> TensorR4 {
        TensorR4 { dims: dims }
    }
}
impl ITensor for TensorR4 {
    fn D() -> usize { 4 }

    fn elements(&self) -> usize {
        self.dims[0] * self.dims[1] * self.dims[2] * self.dims[3]
    }

    fn dims(&self) -> Vec<usize> {
        self.dims.to_vec()
    }
}

#[derive(Debug, Copy, Clone)]
/// Describes a rank 5 Tensor value.
pub struct TensorR5 {
    dims: [usize; 5],
}
impl TensorR5 {
    /// Initializes a new TensorR5 with the cardinality of it dimensions.
    pub fn new(dims: [usize; 5]) -> TensorR5 {
        TensorR5 { dims: dims }
    }
}
impl ITensor for TensorR5 {
    fn D() -> usize { 5 }

    fn elements(&self) -> usize {
        self.dims[0] * self.dims[1] * self.dims[2] * self.dims[3] * self.dims[4]
    }

    fn dims(&self) -> Vec<usize> {
        self.dims.to_vec()
    }
}

#[derive(Debug, Copy, Clone)]
/// Describes a rank 6 Tensor value.
pub struct TensorR6 {
    dims: [usize; 6],
}
impl TensorR6 {
    /// Initializes a new TensorR6 with the cardinality of it dimensions.
    pub fn new(dims: [usize; 6]) -> TensorR6 {
        TensorR6 { dims: dims }
    }
}
impl ITensor for TensorR6 {
    fn D() -> usize { 6 }

    fn elements(&self) -> usize {
        self.dims[0] * self.dims[1] * self.dims[2] * self.dims[3] * self.dims[4] * self.dims[5]
    }

    fn dims(&self) -> Vec<usize> {
        self.dims.to_vec()
    }
}

impl<T, D: ITensor> SharedMemory<T, D> {
    /// Create new SharedMemory by allocating [Memory][1] on a Device.
    /// [1]: ../memory/index.html
    pub fn new(dev: &DeviceType, dim: D) -> Result<SharedMemory<T, D>, Error> {
        let copies = LinearMap::<DeviceType, MemoryType>::new();
        let copy: MemoryType;
        let alloc_size = Self::mem_size(dim.elements());
        match *dev {
            DeviceType::Native(ref cpu) => copy = MemoryType::Native(try!(cpu.alloc_memory(alloc_size as u64))),
            DeviceType::OpenCL(ref context) => copy = MemoryType::OpenCL(try!(context.alloc_memory(alloc_size as u64))),
            #[cfg(feature = "cuda")]
            DeviceType::Cuda(ref context) => copy = MemoryType::Cuda(try!(context.alloc_memory(alloc_size as u64))),
        }
        Ok(SharedMemory {
            latest_location: dev.clone(),
            latest_copy: copy,
            copies: copies,
            dim: dim,
            phantom: PhantomData,
        })
    }

    /// Synchronize memory from latest location to `destination`.
    pub fn sync(&mut self, destination: &DeviceType) -> Result<(), Error> {
        if &self.latest_location != destination {
            let latest = self.latest_location.clone();
            try!(self.sync_from_to(&latest, &destination));

            let mut swap_location = destination.clone();
            let mut swap_copy = try!(self.copies.remove(destination).ok_or(Error::MissingDestination("SharedMemory does not hold a copy on destination device.")));
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
            match self.aquire_copy(destination) {
                Ok(mut destination_copy) => {
                    match destination {
                        &DeviceType::Native(ref cpu) => {
                            match destination_copy.as_mut_native() {
                                Some(ref mut mem) => try!(cpu.sync_in(&self.latest_location, &self.latest_copy, mem)),
                                None => return Err(Error::InvalidMemory("Expected Native Memory (FlatBox)"))
                            }
                        },
                        &DeviceType::OpenCL(ref context) => unimplemented!(),
                        #[cfg(feature = "cuda")]
                        &DeviceType::Cuda(ref context) => {
                            match destination_copy.as_mut_cuda() {
                                Some(ref mut mem) => try!(context.sync_in(&self.latest_location, &self.latest_copy, mem)),
                                None => return Err(Error::InvalidMemory("Expected CUDA Memory."))
                            }
                        }
                    }
                    self.return_copy(destination, destination_copy);
                    Ok(())
                },
                Err(err) => Err(err),
            }
        } else {
            Ok(())
        }
    }

    /// Aquire ownership over a memory copy for synchronizing.
    fn aquire_copy(&mut self, destination: &DeviceType) -> Result<(MemoryType), Error> {
        let destination_copy: MemoryType;
        match self.copies.remove(destination) {
            Some(destination_cpy) => destination_copy = destination_cpy,
            None => return Err(Error::MissingDestination("SharedMemory does not hold a copy on destination device."))
        }

        Ok(destination_copy)
    }

    /// Return ownership over a memory copy after synchronizing.
    fn return_copy(&mut self, dest: &DeviceType, dest_mem: MemoryType) {
        self.copies.insert(dest.clone(), dest_mem);
    }

    /// Track a new `device` and allocate memory on it.
    ///
    /// Returns an error if the SharedMemory is already tracking the `device`.
    pub fn add_device(&mut self, device: &DeviceType) -> Result<&mut Self, Error> {
        // first check if device is not current location. This is cheaper than a lookup in `copies`.
        if &self.latest_location == device {
            return Err(Error::InvalidMemoryAllocation("SharedMemory already tracks memory for this device. No memory allocation."))
        }
        match self.copies.get(device) {
            Some(_) => Err(Error::InvalidMemoryAllocation("SharedMemory already tracks memory for this device. No memory allocation.")),
            None => {
                let copy: MemoryType;
                match *device {
                    DeviceType::Native(ref cpu) => copy = MemoryType::Native(try!(cpu.alloc_memory(Self::mem_size(self.capacity()) as u64))),
                    DeviceType::OpenCL(ref context) => copy = MemoryType::OpenCL(try!(context.alloc_memory(Self::mem_size(self.capacity()) as u64))),
                    #[cfg(feature = "cuda")]
                    DeviceType::Cuda(ref context) => copy = MemoryType::Cuda(try!(context.alloc_memory(Self::mem_size(self.capacity()) as u64))),
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

    /// Returns the number of elements for which the SharedMemory has been allocated.
    pub fn capacity(&self) -> usize {
        self.dim.elements()
    }

    fn mem_size(capacity: usize) -> usize {
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
    /// Framework error at memory allocation.
    MemoryAllocationError(::device::Error),
    /// Framework error at memory synchronization.
    MemorySynchronizationError(::device::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Error::MissingSource(ref err) => write!(f, "{:?}", err),
            Error::MissingDestination(ref err) => write!(f, "{:?}", err),
            Error::InvalidMemory(ref err) => write!(f, "{:?}", err),
            Error::InvalidMemoryAllocation(ref err) => write!(f, "{:?}", err),
            Error::MemoryAllocationError(ref err) => write!(f, "{}", err),
            Error::MemorySynchronizationError(ref err) => write!(f, "{}", err),
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
            Error::MemoryAllocationError(ref err) => err.description(),
            Error::MemorySynchronizationError(ref err) => err.description(),
        }
    }

    fn cause(&self) -> Option<&error::Error> {
        match *self {
            Error::MissingSource(_) => None,
            Error::MissingDestination(_) => None,
            Error::InvalidMemory(_) => None,
            Error::InvalidMemoryAllocation(_) => None,
            Error::MemoryAllocationError(ref err) => Some(err),
            Error::MemorySynchronizationError(ref err) => Some(err),
        }
    }
}

impl From<Error> for ::error::Error {
    fn from(err: Error) -> ::error::Error {
        ::error::Error::SharedMemory(err)
    }
}
