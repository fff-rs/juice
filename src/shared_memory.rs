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

use std::collections::HashMap;
use device::{IDevice, DeviceType};
use memory::MemoryType;
use std::marker::PhantomData;

#[derive(Debug)]
/// Container that handles synchronization of [Memory][1] of type `T`.
/// [1]: ../memory/index.html
pub struct SharedMemory<T> {
    latest_location: DeviceType,
    copies: HashMap<DeviceType, MemoryType>,
    phantom: PhantomData<T>,
}

impl<T> SharedMemory<T> {
    /// Create new SharedMemory from allocated [Memory][1].
    /// [1]: ../memory/index.html
    pub fn new(dev: &DeviceType, copy: MemoryType) -> SharedMemory<T> {
        let mut copies = HashMap::<DeviceType, MemoryType>::new();
        copies.insert(dev.clone(), copy);
        SharedMemory {
            latest_location: dev.clone(),
            copies: copies,
            phantom: PhantomData,
        }
    }

    /// Synchronize memory from latest location to `destination`.
    pub fn sync(&mut self, destination: &DeviceType) -> Result<(), SharedMemoryError> {
        if &self.latest_location != destination {
            let latest = self.latest_location.clone();
            try!(self.sync_from_to(&latest, &destination));
            self.latest_location = destination.clone();
        }
        Ok(())
    }

    /// Synchronize memory from `source` device to `destination` device.
    fn sync_from_to(&mut self, source: &DeviceType, destination: &DeviceType) -> Result<(), SharedMemoryError> {
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
                    self.add_copy(source, source_copy);
                    self.add_copy(destination, destination_copy);
                    Ok(())
                },
                Err(err) => Err(err),
            }
        } else {
            Ok(())
        }
    }

    /// Register a memory copy for a device.
    pub fn add_copy(&mut self, dev: &DeviceType, copy: MemoryType) {
        self.copies.insert(dev.clone(), copy);
    }

    fn aquire_copies(&mut self, source: &DeviceType, destination: &DeviceType) -> Result<(MemoryType, MemoryType), SharedMemoryError> {
        let source_copy: MemoryType;
        let destination_copy: MemoryType;
        match self.copies.remove(source) {
            Some(source_cpy) => source_copy = source_cpy,
            None => return Err(SharedMemoryError::MissingSource(format!("SharedMemory does not hold a copy on source device {:?}.", source)))
        }
        match self.copies.remove(destination) {
            Some(destination_cpy) => destination_copy = destination_cpy,
            None => return Err(SharedMemoryError::MissingDestination(format!("SharedMemory does not hold a copy on destination device {:?}.", destination)))
        }

        Ok((source_copy, destination_copy))
    }

}

/// Errors than can occur when synchronizing memory.
#[derive(Debug)]
pub enum SharedMemoryError {
    /// No copy on source device.
    MissingSource(String),
    /// No copy on destination device.
    MissingDestination(String),
}
