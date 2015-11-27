//! Provides the functionality for memory management accross devices.
//!
//! A SharedMemory tracks the memory copies accross the devices of the Backend and manages
//!
//! * the location of these memory copies
//! * the location of the latest memory copy and
//! * the synchronisation of memory copies between devices
//!
//! A [memory copy][mem] represents one logical unit of data, which might me located at the host. The
//! SharedMemory, tracks the location of the data blob accross the various devices that the backend might
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
//! use collenchyma::shared_memory::SharedMemory;
//! # fn main() {
//! // allocate memory
//! let native = Native::new();
//! let device = native.new_device(native.hardwares()).unwrap();
//! let shared_data = &mut SharedMemory::<i32>::new(&device, 5);
//! // fill memory with some numbers
//! let local_data = [0, 1, 2, 3, 4];
//! let data = shared_data.get_mut(&device).unwrap().as_mut_native().unwrap();
//! data.as_mut_slice().clone_from_slice(&local_data);
//! # }
//! ```

use std::collections::HashMap;
use device::{IDevice, DeviceType};
use memory::MemoryType;
use std::marker::PhantomData;
use std::{fmt, mem, error};

#[derive(Debug)]
/// Container that handles synchronization of [Memory][1] of type `T`.
/// [1]: ../memory/index.html
pub struct SharedMemory<T> {
    latest_location: DeviceType,
    copies: HashMap<DeviceType, MemoryType>,
    cap: usize,
    phantom: PhantomData<T>,
}

impl<T> SharedMemory<T> {
    /// Create new SharedMemory by allocating [Memory][1] on a Device.
    /// [1]: ../memory/index.html
    pub fn new(dev: &DeviceType, capacity: usize) -> SharedMemory<T> {
        let mut copies = HashMap::<DeviceType, MemoryType>::new();
        let copy: MemoryType;
        let alloc_size = mem::size_of::<T>() * capacity;
        match *dev {
            DeviceType::Native(ref cpu) => copy = MemoryType::Native(cpu.alloc_memory(alloc_size)),
            DeviceType::OpenCL(ref context) => copy = MemoryType::OpenCL(context.alloc_memory(alloc_size)),
        }
        copies.insert(dev.clone(), copy);
        SharedMemory {
            latest_location: dev.clone(),
            copies: copies,
            cap: capacity,
            phantom: PhantomData,
        }
    }

    /// Synchronize memory from latest location to `destination`.
    pub fn sync(&mut self, destination: &DeviceType) -> Result<(), Error> {
        if &self.latest_location != destination {
            let latest = self.latest_location.clone();
            try!(self.sync_from_to(&latest, &destination));
            self.latest_location = destination.clone();
        }
        Ok(())
    }

    /// Get a reference to the memory copy on the provided `device`.
    ///
    /// Returns `None` if there is no memory copy on the device.
    pub fn get(&self, device: &DeviceType) -> Option<&MemoryType> {
        self.copies.get(device)
    }

    /// Get a mutable reference to the memory copy on the provided `device`.
    ///
    /// Returns `None` if there is no memory copy on the device.
    pub fn get_mut(&mut self, device: &DeviceType) -> Option<&mut MemoryType> {
        self.copies.get_mut(device)
    }

    /// Synchronize memory from `source` device to `destination` device.
    fn sync_from_to(&mut self, source: &DeviceType, destination: &DeviceType) -> Result<(), Error> {
        if source != destination {
            match self.aquire_copies(source, destination) {
                Ok((source_copy, mut destination_copy)) => {
                    match source.clone() {
                        DeviceType::Native(cpu) => {
                            if let MemoryType::Native(ref src) = source_copy {
                                cpu.sync_memory_to(&src, &mut destination_copy, destination)
                            }
                        },
                        DeviceType::OpenCL(context) => {
                            if let MemoryType::OpenCL(ref src) = source_copy {
                                context.sync_memory_to(&src,&mut destination_copy, destination)
                            }
                        },
                    }
                    self.return_copies(source, source_copy, destination, destination_copy);
                    Ok(())
                },
                Err(err) => Err(err),
            }
        } else {
            Ok(())
        }
    }

    /// Aquire ownership over the copies for synchronizing.
    fn aquire_copies(&mut self, source: &DeviceType, destination: &DeviceType) -> Result<(MemoryType, MemoryType), Error> {
        let source_copy: MemoryType;
        let destination_copy: MemoryType;
        match self.copies.remove(source) {
            Some(source_cpy) => source_copy = source_cpy,
            None => return Err(Error::MissingSource(format!("SharedMemory does not hold a copy on source device {:?}.", source)))
        }
        match self.copies.remove(destination) {
            Some(destination_cpy) => destination_copy = destination_cpy,
            None => return Err(Error::MissingDestination(format!("SharedMemory does not hold a copy on destination device {:?}.", destination)))
        }

        Ok((source_copy, destination_copy))
    }

    /// Return ownership over the copies after synchronizing.
    fn return_copies(&mut self, src: &DeviceType, src_mem: MemoryType, dest: &DeviceType, dest_mem: MemoryType) {
        self.copies.insert(src.clone(), src_mem);
        self.copies.insert(dest.clone(), dest_mem);
    }

    /// Track a new `device` and allocate memory on it.
    ///
    /// Returns an error if the SharedMemory is already tracking the `device`.
    pub fn add_device(&mut self, device: &DeviceType) -> Result<&mut Self, Error> {
        match self.copies.get(device) {
            Some(_) => Err(Error::InvalidMemoryAllocation(format!("SharedMemory already tracks memory for this device. No memory allocation."))),
            None => {
                let copy: MemoryType;
                match *device {
                    DeviceType::Native(ref cpu) => copy = MemoryType::Native(cpu.alloc_memory(mem::size_of::<T>())),
                    DeviceType::OpenCL(ref context) => copy = MemoryType::OpenCL(context.alloc_memory(mem::size_of::<T>())),
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
        self.cap
    }
}

/// Errors than can occur when synchronizing memory.
#[derive(Debug)]
pub enum Error {
    /// No copy on source device.
    MissingSource(String),
    /// No copy on destination device.
    MissingDestination(String),
    /// No memory allocation on specified device happened.
    InvalidMemoryAllocation(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Error::MissingSource(ref err) => write!(f, "{:?}", err),
            Error::MissingDestination(ref err) => write!(f, "{:?}", err),
            Error::InvalidMemoryAllocation(ref err) => write!(f, "{:?}", err),
        }
    }
}

impl error::Error for Error {
    fn description(&self) -> &str {
        match *self {
            Error::MissingSource(ref err) => err,
            Error::MissingDestination(ref err) => err,
            Error::InvalidMemoryAllocation(ref err) => err,
        }
    }

    fn cause(&self) -> Option<&error::Error> {
        match *self {
            Error::MissingSource(_) => None,
            Error::MissingDestination(_) => None,
            Error::InvalidMemoryAllocation(_) => None,
        }
    }
}

impl From<Error> for ::error::Error {
    fn from(err: Error) -> ::error::Error {
        ::error::Error::SharedMemory(err)
    }
}
