use log::debug;
use std::clone::Clone;
use std::cmp::{Eq, PartialEq};
use std::convert::TryFrom;
use std::fmt;
use std::marker::Send;

use std::hash::Hash;

#[macro_use]
extern crate mashup;

#[derive(Debug, PartialEq, Eq)]
pub enum Error<T>
where
    T: fmt::Debug,
{
    NullPointer,
    UnknownPointer(*mut T),
    DuplicateTracking(*mut T),
}

impl<T> fmt::Display for Error<T>
where
    T: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Nullptr detected")
    }
}

impl<T> std::error::Error for Error<T> where T: fmt::Debug {}

pub trait Trackable {
    type Tracker;

    fn tracker() -> Self::Tracker;
}

#[macro_export]
macro_rules! setup_tracker {
    ($t:ident) => {{
        mashup! {
            m["__NAME__"] = $t _PARENT_TYPE;
        }

        m! {
            lazy_static::lazy_static! {
                static ref "__NAME__" : $crate::Tracker<$t> = $crate::Tracker::<$t>::new();
            }

            impl Trackable for $t
            {
                type Tracker = $crate::Tracker<$t>;
                fn tracker() -> Self::Tracker {
                    (*"__NAME__").clone()
                }
            }
        }
    }};
}

#[derive(PartialEq, Eq, Hash, Debug)]
pub struct Cookie<T>(*mut T)
where
    T: fmt::Debug;

impl<T> Cookie<T>
where
    T: fmt::Debug,
{
    pub fn as_ptr(&self) -> *mut T {
        self.0
    }

    pub fn try_from(ptr: *mut T) -> std::result::Result<Self, Error<T>> {
        Ok(Cookie::<T>(ptr))
    }
}

unsafe impl<T> Send for Cookie<T> where T: fmt::Debug {}

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::Mutex;

#[derive(Debug)]
pub struct Tracker<T>
where
    T: fmt::Debug,
    Cookie<T>: Hash + Eq,
    T: Trackable<Tracker = Self>,
{
    inner: Arc<Mutex<HashSet<Cookie<T>>>>,
}

impl<T> Clone for Tracker<T>
where
    T: fmt::Debug,
    Cookie<T>: Hash + Eq,
    T: Trackable<Tracker = Self>,
{
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<T> Tracker<T>
where
    Cookie<T>: Hash + Eq,
    T: fmt::Debug,
    T: Trackable<Tracker = Self>,
{
    #[allow(unused)]
    fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(HashSet::with_capacity(128))),
        }
    }

    fn tracker() -> Option<Tracker<T>> {
        Some(<T as Trackable>::tracker())
    }

    pub fn track(handle: *mut T) -> Result<(), Error<T>> {
        if let Some(tracker) = Self::tracker() {
            let mut guard = tracker.inner.lock().unwrap();
            let cookie = Cookie::try_from(handle)?;
            if guard.contains(&cookie) {
                return Err(Error::DuplicateTracking(handle));
            } else {
                let _ = guard.insert(cookie);
                debug!("Added handle {:?}, total of {}", handle, guard.len());
                return Ok(());
            }
        }
        unreachable!("Forgot to setup a registry");
    }

    pub fn contains(handle: *mut T) -> bool {
        if let Some(tracker) = Self::tracker() {
            let guard = tracker.inner.lock().unwrap();
            debug!("Removed handle {:?}, total of {}", handle, guard.len());
            let k = Cookie::try_from(handle).unwrap();
            return guard.contains(&k);
        }
        unreachable!("Forgot to setup a registry");
    }

    pub fn untrack(handle: *mut T) -> Result<(), Error<T>> {
        if let Some(tracker) = Self::tracker() {
            let mut guard = tracker.inner.lock().unwrap();
            debug!("Removed handle {:?}, total of {}", handle, guard.len());
            let k = Cookie::try_from(handle).unwrap();
            let _ = guard.remove(&k);
            return Ok(());
        }
        unreachable!("Forgot to setup a registry");
    }

    pub fn cleanup<F>(f: F)
    where
        F: Fn(*mut T) -> (),
    {
        if let Some(tracker) = Self::tracker() {
            let guard = tracker.inner.lock().unwrap();
            for handle in guard.iter() {
                f(handle.as_ptr())
            }
        }
        unreachable!("Forgot to setup a registry");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() {}

    // helper demo struct
    // could be anything that is send
    #[derive(Debug, Default, PartialEq, Eq, Hash)]
    pub struct X {
        a: u8,
        b: u32,
    }

    // commonly we track pointers
    // which are send
    type Y = *mut X;

    #[test]
    fn tracky() {
        setup_tracker!(X);

        let mut x = X::default();
        let ptr = &mut x as *mut X;
        assert_eq!(Tracker::<X>::track(ptr), Ok(()));
        assert!(Tracker::<X>::contains(ptr));
        assert_eq!(Tracker::<X>::untrack(ptr), Ok(()));
    }
}
