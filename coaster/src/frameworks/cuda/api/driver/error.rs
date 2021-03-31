//! Provides Rust Errors for OpenCL's status.

#[allow(missing_docs)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, thiserror::Error)]
/// OpenCL device errors
pub enum Error {
    #[error("{0}")]
    InvalidValue(&'static str),
    #[error("{0}")]
    OutOfMemory(&'static str),
    #[error("{0}")]
    NotInitialized(&'static str),
    #[error("{0}")]
    Deinitialized(&'static str),
    #[error("{0}")]
    ProfilerDisabled(&'static str),
    #[error("{0}")]
    ProfilerNotInitialized(&'static str),
    #[error("{0}")]
    ProfilerAlreadyStarted(&'static str),
    #[error("{0}")]
    ProfilerAlreadyStopped(&'static str),
    #[error("{0}")]
    NoDevice(&'static str),
    #[error("{0}")]
    InvalidDevice(&'static str),
    #[error("{0}")]
    InvalidImage(&'static str),
    #[error("{0}")]
    InvalidContext(&'static str),
    #[error("{0}")]
    ContextAlreadyCurrent(&'static str),
    #[error("{0}")]
    MapFailed(&'static str),
    #[error("{0}")]
    UnmapFailed(&'static str),
    #[error("{0}")]
    ArrayIsMapped(&'static str),
    #[error("{0}")]
    AlreadyMapped(&'static str),
    #[error("{0}")]
    NoBinaryForGpu(&'static str),
    #[error("{0}")]
    AlreadyAquired(&'static str),
    #[error("{0}")]
    NotMapped(&'static str),
    #[error("{0}")]
    NotMappedAsArray(&'static str),
    #[error("{0}")]
    NotMappedAsPointer(&'static str),
    #[error("{0}")]
    EccUncorrectable(&'static str),
    #[error("{0}")]
    UnsupportedLimit(&'static str),
    #[error("{0}")]
    ContextAlreadyInUse(&'static str),
    #[error("{0}")]
    PeerAccessUnsupported(&'static str),
    #[error("{0}")]
    InvalidPtx(&'static str),
    #[error("{0}")]
    InvalidGraphicsContent(&'static str),
    #[error("{0}")]
    InvalidSource(&'static str),
    #[error("{0}")]
    FileNotFound(&'static str),
    #[error("{0}")]
    SharedObjectSymbolNotFound(&'static str),
    #[error("{0}")]
    SharedObjectInitFailed(&'static str),
    #[error("{0}")]
    OperatingSystem(&'static str),
    #[error("{0}")]
    InvalidHandle(&'static str),
    #[error("{0}")]
    NotFound(&'static str),
    #[error("{0}")]
    NotReady(&'static str),
    #[error("{0}")]
    IllegalAddress(&'static str),
    #[error("{0}")]
    LaunchOutOfResources(&'static str),
    #[error("{0}")]
    LaunchTimeout(&'static str),
    #[error("{0}")]
    LauncIncompatibleTexturing(&'static str),
    #[error("{0}")]
    PeerAccessAlreadyEnabled(&'static str),
    #[error("{0}")]
    PeerAccessNotEnabled(&'static str),
    #[error("{0}")]
    PrimaryContextActive(&'static str),
    #[error("{0}")]
    ContextIsDestroyed(&'static str),
    #[error("{0}")]
    Assert(&'static str),
    #[error("{0}")]
    TooManyPeers(&'static str),
    #[error("{0}")]
    HostMemoryAlreadyRegistered(&'static str),
    #[error("{0}")]
    HostMemoryNotRegistered(&'static str),
    #[error("{0}")]
    HardwareStackError(&'static str),
    #[error("{0}")]
    IllegalInstruction(&'static str),
    #[error("{0}")]
    MisalignedAddress(&'static str),
    #[error("{0}")]
    InvalidAddressSpace(&'static str),
    #[error("{0}")]
    InvalidPc(&'static str),
    #[error("{0}")]
    LaunchFailed(&'static str),
    #[error("{0}")]
    NotPermitted(&'static str),
    #[error("{0}")]
    NotSupported(&'static str),
    #[error("{0}")]
    Unknown(&'static str, u64),
}
