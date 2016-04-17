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
//! let shared_data = &mut SharedTensor::<i32>::new(&5).unwrap();
//! // fill memory with some numbers
//! let mut mem = shared_data.write_only(&device).unwrap().as_mut_native().unwrap();
//! mem.as_mut_slice::<i32>().clone_from_slice(&[0, 1, 2, 3, 4]);
//! # }
//! ```

use device::{IDevice, DeviceType};
use memory::MemoryType;
use std::cell::{Cell, RefCell};
use std::marker::PhantomData;
use std::{fmt, mem, error};
use std::error::Error as StdError;

/// Describes the Descriptor of a SharedTensor.
pub type TensorDesc = Vec<usize>;

/// BitMap type for keeping track of up-to-date locations. If number of
/// locations provided by the integer isn't enough, this type can be easily
/// replaced with BitSet at cost of a heap allocation and extra inderection
/// on access.
type BitMap = u64;

/// Number of bits in `BitMap`. It's currently no possible to get this
/// information from `BitMap` cleanly. Though there are plans to add a
/// static method or associated constant.
const BIT_MAP_SIZE: usize = 64;

struct TensorLocation {
    device: DeviceType,

    // Box is required to keep references to MemoryType alive if
    // SharedTensor::locations vec reallocates storage and moves elements.
    // See also comment on `unsafe` near `SharedTensor::read()` impl.
    mem: Box<MemoryType>,
}

/// Container that handles synchronization of [Memory][1] of type `T`.
/// [1]: ../memory/index.html
pub struct SharedTensor<T> {
    desc: TensorDesc,
    locations: RefCell<Vec<TensorLocation>>,
    up_to_date: Cell<BitMap>,

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


impl <T> fmt::Debug for SharedTensor<T> {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        write!(f, "SharedTensor desc={:?}", self.desc)
    }
}

impl<T> SharedTensor<T> {
    /// Create new Tensor by allocating [Memory][1] on a Device.
    /// [1]: ../memory/index.html
    pub fn new<D: IntoTensorDesc>(desc: &D) -> Result<SharedTensor<T>, Error> {
        Ok(SharedTensor {
            desc: desc.into(),
            locations: RefCell::new(Vec::new()),
            up_to_date: Cell::new(0),
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
        self.locations.borrow_mut().clear();
        self.up_to_date.set(0);
        self.desc = desc.into();
        Ok(())
    }

    fn get_location_index(&self, device: &DeviceType) -> Option<usize> {
        for (i, loc) in self.locations.borrow().iter().enumerate() {
            if loc.device == *device {
                return Some(i);
            }
        }
        None
    }

    /// Looks up `device` in self.locations and returns its index. If lookup
    /// fails then new location is created and its index is returned.
    fn get_or_create_location_index(&self, device: &DeviceType)
                                    -> Result<usize, Error> {
        if let Some(i) = self.get_location_index(device) {
            return Ok(i);
        }

        if self.locations.borrow().len() == BIT_MAP_SIZE {
            return Err(Error::CapacityExceeded);
        }

        let mem = try!(Self::alloc_on_device(device, self.desc().size()));
        self.locations.borrow_mut().push(TensorLocation {
            device: device.clone(),
            mem: Box::new(mem),
        });
        Ok(self.locations.borrow().len() - 1)
    }

    // TODO: chose the best source to copy data from.
    // That would require some additional traits that return costs for
    // transferring data between different backends.
    // Actually I think that there would be only transfers between
    // `Native` <-> `Cuda` and `Native` <-> `OpenCL` in foreseeable future,
    // so it's best to not overengineer here.
    fn sync_if_needed(&self, dst_i: usize) -> Result<(), Error> {
        if self.up_to_date.get() & (1 << dst_i) != 0 {
            return Ok(());
        }

        let src_i = self.up_to_date.get().trailing_zeros() as usize;
        assert!(src_i != BIT_MAP_SIZE);

        // We need to borrow two different Vec elements: src and mut dst.
        // Borrowck doesn't allow to do it in a straightforward way, so
        // here is workaround.
        assert!(src_i != dst_i);
        let mut locs = self.locations.borrow_mut();
        let (src_loc, mut dst_loc) = if src_i < dst_i {
            let (left, right) = locs.split_at_mut(dst_i);
            (&left[src_i], &mut right[0])
        } else {
            let (left, right) = locs.split_at_mut(src_i);
            (&right[0], &mut left[dst_i])
        };

        match &dst_loc.device {
            #[cfg(feature = "native")]
            &DeviceType::Native(ref cpu) => {
                let mem = dst_loc.mem.as_mut_native()
                    .expect("Broken invariant: expected Native Memory");
                try!(cpu.sync_in(&src_loc.device, &src_loc.mem, mem));
            },
            #[cfg(feature = "cuda")]
            &DeviceType::Cuda(ref context) => {
                let mem = dst_loc.mem.as_mut_cuda()
                    .expect("Broken invariant: expected Cuda Memory");
                try!(context.sync_in(&src_loc.device, &src_loc.mem, mem));
            },
            #[cfg(feature = "opencl")]
            &DeviceType::OpenCL(ref context) => {
                let mem = dst_loc.mem.as_mut_opencl()
                    .expect("Broken invariant: expected OpenCL Memory");
                try!(context.sync_in(&src_loc.device, &src_loc.mem, mem));
            }
        }

        Ok(())
    }

    /// Allocate memory on the provided DeviceType for `n` elements.
    fn alloc_on_device(dev: &DeviceType, n: usize) -> Result<MemoryType, Error> {
        let alloc_size = Self::mem_size(n);
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

    // Functions `read()`, `read_write()`, `write_only()` use `unsafe` to
    // extend lifetime of retured reference to internally owned memory chunk.
    // Borrowck guarantees that SharedTensor outlives all of its Tensors, and
    // there is only one mutable borrow. So we only need to make sure that
    // memory locations won't be dropped or moved while there are live Tensors.
    // It's quite easy to do: by convention we only allow to remove elements from
    // `self.locations` in methods with `&mut self`. Since we store `MemoryType`
    // inside `Vec` in a `Box`, reference to it won't change during Vec
    // reallocations.

    /// Get memory for reading on the specified `device`.
    /// Can fail if memory allocation fails, or if tensor wasn't initialized yet.
    pub fn read<'a>(&'a self, device: &DeviceType) -> Result<&'a MemoryType, Error> {
        if self.up_to_date.get() == 0 {
            return Err(Error::UninitializedMemory);
        }
        let i = try!(self.get_or_create_location_index(device));
        try!(self.sync_if_needed(i));
        self.up_to_date.set(self.up_to_date.get() | (1 << i));

        let locs = self.locations.borrow();
        let mem: &MemoryType = &locs[i].mem;
        let mem_a: &'a MemoryType = unsafe { ::std::mem::transmute(mem) };
        Ok(mem_a)
    }

    /// Get memory for reading and writing on the specified `device`.
    /// Can fail if memory allocation fails, or if tensor wasn't initialized yet.
    pub fn read_write<'a>(&'a mut self, device: &DeviceType)
                          -> Result<&'a mut MemoryType, Error> {
        if self.up_to_date.get() == 0 {
            return Err(Error::UninitializedMemory);
        }
        let i = try!(self.get_or_create_location_index(device));
        try!(self.sync_if_needed(i));
        self.up_to_date.set(1 << i);

        let mut locs = self.locations.borrow_mut();
        let mem: &mut MemoryType = &mut locs[i].mem;
        let mem_a: &'a mut  MemoryType = unsafe { ::std::mem::transmute(mem) };
        Ok(mem_a)
    }

    /// Get memory for writing only.
    /// This function skips synchronization and initialization checks, since
    /// contents will be overwritten anyway. By convention caller must fully
    /// initialize returned memory. Failure to do so may result in use of
    /// uninitialized data later. If caller has failed to overwrite memory,
    /// for some reason, it must call `invalidate()` to return vector to
    /// uninitialized state.
    pub fn write_only<'a>(&'a mut self, device: &DeviceType)
                          -> Result<&'a mut MemoryType, Error> {
        let i = try!(self.get_or_create_location_index(device));
        self.up_to_date.set(1 << i);

        let mut locs = self.locations.borrow_mut();
        let mem: &mut MemoryType = &mut locs[i].mem;
        let mem_a: &'a mut  MemoryType = unsafe { ::std::mem::transmute(mem) };
        Ok(mem_a)
    }

    /// Drops memory allocation on the specified device. Returns error if
    /// no memory has been allocated on this device.
    pub fn drop_device(&mut self, device: &DeviceType) -> Result<(), Error> {
        match self.get_location_index(device) {
            Some(i) => {
                self.locations.borrow_mut().remove(i);

                let up_to_date = self.up_to_date.get();
                let mask = (1 << i) - 1;
                let lower = up_to_date & mask;
                let upper = (up_to_date >> 1) & (!mask);
                self.up_to_date.set(lower | upper);
                Ok(())
            },
            None =>
                Err(Error::InvalidRemove("Memory isn't allocated on this device"))
        }
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
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
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
    InvalidShape(&'static str),
    /// Maximal number of backing memories has been reached.
    CapacityExceeded,
    /// Memory is requested for reading, but it hasn't been initialized.
    UninitializedMemory,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.description())
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
            Error::CapacityExceeded =>
                "Max number of backing memories has been reached",
            Error::UninitializedMemory =>
                "Uninitialized memory is requested for reading",
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
            Error::CapacityExceeded => None,
            Error::UninitializedMemory => None,
        }
    }
}

impl From<Error> for ::error::Error {
    fn from(err: Error) -> ::error::Error {
        ::error::Error::Tensor(err)
    }
}
