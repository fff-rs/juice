//! Provides Rust Errors for OpenCL's status.

use std::{fmt, error};

#[derive(Debug)]
/// Defines OpenCL errors.
pub enum Error {
    /// Failure with provided value.
    InvalidValue(String),
    /// Failure with memory allocation.
    OutOfMemory(String),
    /// Failure with Cuda initialization.
    NotInitialized(String),
    /// Failure with Cuda initialization.
    Deinitialized(String),
    /// Failure with Profiler.
    ProfilerDisabled(String),
    /// Failure with Profiler.
    ProfilerNotInitialized(String),
    /// Failure with Profiler.
    ProfilerAlreadyStarted(String),
    /// Failure with Profiler.
    ProfilerAlreadyStopped(String),
    /// Failure with Cuda devices.
    NoDevice(String),
    /// Failure with provided Cuda device.
    InvalidDevice(String),
    /// Failure with provided Cuda image.
    InvalidImage(String),
    /// Failure with provided Cuda context.
    InvalidContext(String),
    /// Failure with provided Cuda context.
    ContextAlreadyCurrent(String),
    /// Failure
    MapFailed(String),
    /// Failure
    UnmapFailed(String),
    /// Failure
    ArrayIsMapped(String),
    /// Failure
    AlreadyMapped(String),
    /// Failure with binary.
    NoBinaryForGpu(String),
    /// Failure
    AlreadyAquired(String),
    /// Failure
    NotMapped(String),
    /// Failure
    NotMappedAsArray(String),
    /// Failure
    NotMappedAsPointer(String),
    /// Failure
    EccUncorrectable(String),
    /// Failure
    UnsupportedLimit(String),
    /// Failure with context.
    ContextAlreadyInUse(String),
    /// Failure
    PeerAccessUnsupported(String),
    /// Failure with provided PTX.
    InvalidPtx(String),
    /// Failure
    InvalidGraphicsContent(String),
    /// Failure
    InvalidSource(String),
    /// Failure
    FileNotFound(String),
    /// Failure
    SharedObjectSymbolNotFound(String),
    /// Failure
    SharedObjectInitFailed(String),
    /// Failure
    OperatingSystem(String),
    /// Failure
    InvalidHandle(String),
    /// Failure
    NotFound(String),
    /// Failure
    NotReady(String),
    /// Failure
    IllegalAddress(String),
    /// Failure
    LaunchOutOfResources(String),
    /// Failure
    LaunchTimeout(String),
    /// Failure
    LauncIncompatibleTexturing(String),
    /// Failure
    PeerAccessAlreadyEnabled(String),
    /// Failure
    PeerAccessNotEnabled(String),
    /// Failure
    PrimaryContextActive(String),
    /// Failure
    ContextIsDestroyed(String),
    /// Failure
    Assert(String),
    /// Failure
    TooManyPeers(String),
    /// Failure
    HostMemoryAlreadyRegistered(String),
    /// Failure
    HostMemoryNotRegistered(String),
    /// Failure
    HardwareStackError(String),
    /// Failure
    IllegalInstruction(String),
    /// Failure
    MisalignedAddress(String),
    /// Failure
    InvalidAddressSpace(String),
    /// Failure
    InvalidPc(String),
    /// Failure
    LaunchFailed(String),
    /// Failure
    NotPermitted(String),
    /// Failure
    NotSupported(String),
    /// Failure
    Unknown(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Error::ProfilerNotInitialized(ref err) => write!(f, "{:?}", err),
            Error::ProfilerDisabled(ref err) => write!(f, "{:?}", err),
            Error::Deinitialized(ref err) => write!(f, "{:?}", err),
            Error::NotInitialized(ref err) => write!(f, "{:?}", err),
            Error::OutOfMemory(ref err) => write!(f, "{:?}", err),
            Error::InvalidValue(ref err) => write!(f, "{:?}", err),
            Error::NoBinaryForGpu(ref err) => write!(f, "{:?}", err),
            Error::AlreadyMapped(ref err) => write!(f, "{:?}", err),
            Error::ArrayIsMapped(ref err) => write!(f, "{:?}", err),
            Error::UnmapFailed(ref err) => write!(f, "{:?}", err),
            Error::MapFailed(ref err) => write!(f, "{:?}", err),
            Error::ContextAlreadyCurrent(ref err) => write!(f, "{:?}", err),
            Error::InvalidContext(ref err) => write!(f, "{:?}", err),
            Error::InvalidImage(ref err) => write!(f, "{:?}", err),
            Error::InvalidDevice(ref err) => write!(f, "{:?}", err),
            Error::NoDevice(ref err) => write!(f, "{:?}", err),
            Error::ProfilerAlreadyStopped(ref err) => write!(f, "{:?}", err),
            Error::ProfilerAlreadyStarted(ref err) => write!(f, "{:?}", err),
            Error::IllegalAddress(ref err) => write!(f, "{:?}", err),
            Error::NotReady(ref err) => write!(f, "{:?}", err),
            Error::NotFound(ref err) => write!(f, "{:?}", err),
            Error::InvalidHandle(ref err) => write!(f, "{:?}", err),
            Error::OperatingSystem(ref err) => write!(f, "{:?}", err),
            Error::SharedObjectInitFailed(ref err) => write!(f, "{:?}", err),
            Error::SharedObjectSymbolNotFound(ref err) => write!(f, "{:?}", err),
            Error::FileNotFound(ref err) => write!(f, "{:?}", err),
            Error::InvalidSource(ref err) => write!(f, "{:?}", err),
            Error::InvalidGraphicsContent(ref err) => write!(f, "{:?}", err),
            Error::InvalidPtx(ref err) => write!(f, "{:?}", err),
            Error::PeerAccessUnsupported(ref err) => write!(f, "{:?}", err),
            Error::ContextAlreadyInUse(ref err) => write!(f, "{:?}", err),
            Error::UnsupportedLimit(ref err) => write!(f, "{:?}", err),
            Error::EccUncorrectable(ref err) => write!(f, "{:?}", err),
            Error::NotMappedAsPointer(ref err) => write!(f, "{:?}", err),
            Error::NotMappedAsArray(ref err) => write!(f, "{:?}", err),
            Error::NotMapped(ref err) => write!(f, "{:?}", err),
            Error::AlreadyAquired(ref err) => write!(f, "{:?}", err),
            Error::Unknown(ref err) => write!(f, "{:?}", err),
            Error::NotSupported(ref err) => write!(f, "{:?}", err),
            Error::NotPermitted(ref err) => write!(f, "{:?}", err),
            Error::LaunchFailed(ref err) => write!(f, "{:?}", err),
            Error::InvalidPc(ref err) => write!(f, "{:?}", err),
            Error::InvalidAddressSpace(ref err) => write!(f, "{:?}", err),
            Error::MisalignedAddress(ref err) => write!(f, "{:?}", err),
            Error::IllegalInstruction(ref err) => write!(f, "{:?}", err),
            Error::HardwareStackError(ref err) => write!(f, "{:?}", err),
            Error::HostMemoryNotRegistered(ref err) => write!(f, "{:?}", err),
            Error::HostMemoryAlreadyRegistered(ref err) => write!(f, "{:?}", err),
            Error::TooManyPeers(ref err) => write!(f, "{:?}", err),
            Error::Assert(ref err) => write!(f, "{:?}", err),
            Error::ContextIsDestroyed(ref err) => write!(f, "{:?}", err),
            Error::PrimaryContextActive(ref err) => write!(f, "{:?}", err),
            Error::PeerAccessNotEnabled(ref err) => write!(f, "{:?}", err),
            Error::PeerAccessAlreadyEnabled(ref err) => write!(f, "{:?}", err),
            Error::LauncIncompatibleTexturing(ref err) => write!(f, "{:?}", err),
            Error::LaunchTimeout(ref err) => write!(f, "{:?}", err),
            Error::LaunchOutOfResources(ref err) => write!(f, "{:?}", err),
        }
    }
}

impl error::Error for Error {
    fn description(&self) -> &str {
        match *self {
            Error::ProfilerNotInitialized(ref err) => err,
            Error::ProfilerDisabled(ref err) => err,
            Error::Deinitialized(ref err) => err,
            Error::NotInitialized(ref err) => err,
            Error::OutOfMemory(ref err) => err,
            Error::InvalidValue(ref err) => err,
            Error::NoBinaryForGpu(ref err) => err,
            Error::AlreadyMapped(ref err) => err,
            Error::ArrayIsMapped(ref err) => err,
            Error::UnmapFailed(ref err) => err,
            Error::MapFailed(ref err) => err,
            Error::ContextAlreadyCurrent(ref err) => err,
            Error::InvalidContext(ref err) => err,
            Error::InvalidImage(ref err) => err,
            Error::InvalidDevice(ref err) => err,
            Error::NoDevice(ref err) => err,
            Error::ProfilerAlreadyStopped(ref err) => err,
            Error::ProfilerAlreadyStarted(ref err) => err,
            Error::IllegalAddress(ref err) => err,
            Error::NotReady(ref err) => err,
            Error::NotFound(ref err) => err,
            Error::InvalidHandle(ref err) => err,
            Error::OperatingSystem(ref err) => err,
            Error::SharedObjectInitFailed(ref err) => err,
            Error::SharedObjectSymbolNotFound(ref err) => err,
            Error::FileNotFound(ref err) => err,
            Error::InvalidSource(ref err) => err,
            Error::InvalidGraphicsContent(ref err) => err,
            Error::InvalidPtx(ref err) => err,
            Error::PeerAccessUnsupported(ref err) => err,
            Error::ContextAlreadyInUse(ref err) => err,
            Error::UnsupportedLimit(ref err) => err,
            Error::EccUncorrectable(ref err) => err,
            Error::NotMappedAsPointer(ref err) => err,
            Error::NotMappedAsArray(ref err) => err,
            Error::NotMapped(ref err) => err,
            Error::AlreadyAquired(ref err) => err,
            Error::Unknown(ref err) => err,
            Error::NotSupported(ref err) => err,
            Error::NotPermitted(ref err) => err,
            Error::LaunchFailed(ref err) => err,
            Error::InvalidPc(ref err) => err,
            Error::InvalidAddressSpace(ref err) => err,
            Error::MisalignedAddress(ref err) => err,
            Error::IllegalInstruction(ref err) => err,
            Error::HardwareStackError(ref err) => err,
            Error::HostMemoryNotRegistered(ref err) => err,
            Error::HostMemoryAlreadyRegistered(ref err) => err,
            Error::TooManyPeers(ref err) => err,
            Error::Assert(ref err) => err,
            Error::ContextIsDestroyed(ref err) => err,
            Error::PrimaryContextActive(ref err) => err,
            Error::PeerAccessNotEnabled(ref err) => err,
            Error::PeerAccessAlreadyEnabled(ref err) => err,
            Error::LauncIncompatibleTexturing(ref err) => err,
            Error::LaunchTimeout(ref err) => err,
            Error::LaunchOutOfResources(ref err) => err,
        }
    }

    fn cause(&self) -> Option<&error::Error> {
        match *self {
            Error::ProfilerNotInitialized(_) => None,
            Error::ProfilerDisabled(_) => None,
            Error::Deinitialized(_) => None,
            Error::NotInitialized(_) => None,
            Error::OutOfMemory(_) => None,
            Error::InvalidValue(_) => None,
            Error::NoBinaryForGpu(_) => None,
            Error::AlreadyMapped(_) => None,
            Error::ArrayIsMapped(_) => None,
            Error::UnmapFailed(_) => None,
            Error::MapFailed(_) => None,
            Error::ContextAlreadyCurrent(_) => None,
            Error::InvalidContext(_) => None,
            Error::InvalidImage(_) => None,
            Error::InvalidDevice(_) => None,
            Error::NoDevice(_) => None,
            Error::ProfilerAlreadyStopped(_) => None,
            Error::ProfilerAlreadyStarted(_) => None,
            Error::IllegalAddress(_) => None,
            Error::NotReady(_) => None,
            Error::NotFound(_) => None,
            Error::InvalidHandle(_) => None,
            Error::OperatingSystem(_) => None,
            Error::SharedObjectInitFailed(_) => None,
            Error::SharedObjectSymbolNotFound(_) => None,
            Error::FileNotFound(_) => None,
            Error::InvalidSource(_) => None,
            Error::InvalidGraphicsContent(_) => None,
            Error::InvalidPtx(_) => None,
            Error::PeerAccessUnsupported(_) => None,
            Error::ContextAlreadyInUse(_) => None,
            Error::UnsupportedLimit(_) => None,
            Error::EccUncorrectable(_) => None,
            Error::NotMappedAsPointer(_) => None,
            Error::NotMappedAsArray(_) => None,
            Error::NotMapped(_) => None,
            Error::AlreadyAquired(_) => None,
            Error::Unknown(_) => None,
            Error::NotSupported(_) => None,
            Error::NotPermitted(_) => None,
            Error::LaunchFailed(_) => None,
            Error::InvalidPc(_) => None,
            Error::InvalidAddressSpace(_) => None,
            Error::MisalignedAddress(_) => None,
            Error::IllegalInstruction(_) => None,
            Error::HardwareStackError(_) => None,
            Error::HostMemoryNotRegistered(_) => None,
            Error::HostMemoryAlreadyRegistered(_) => None,
            Error::TooManyPeers(_) => None,
            Error::Assert(_) => None,
            Error::ContextIsDestroyed(_) => None,
            Error::PrimaryContextActive(_) => None,
            Error::PeerAccessNotEnabled(_) => None,
            Error::PeerAccessAlreadyEnabled(_) => None,
            Error::LauncIncompatibleTexturing(_) => None,
            Error::LaunchTimeout(_) => None,
            Error::LaunchOutOfResources(_) => None,
        }
    }
}
