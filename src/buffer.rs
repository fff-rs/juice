//! Provides the functionality for memory management accross devices.
//!
//! A Buffer contains the data blobs accross the devices of the backend and manages
//!
//! * the location of these data blobs
//! * the location of the latest data blob and
//! * the synchronisation of data blobs between devices
//!
//! A data blob represents one logical unit of data, which might me located at the host. The
//! Buffer, tracks the location of the data blob accross the various devices that the backend might
//! consist of. This allows us to run operations on various backends with the same data blob.
//!
//! [frameworks]: ../frameworks/index.html

use framework::FrameworkError;

#[derive(Debug, Copy, Clone)]
/// Defines the main and highest struct of Collenchyma.
pub struct Buffer;

/// Defines the functionality of a Backend.
pub trait IBuffer {

}
