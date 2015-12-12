//! A module providing a map implementation `LinearMap` backed by a vector.

#![warn(missing_docs)]
#![cfg_attr(all(test, feature = "nightly"), feature(test))]

use std::borrow::Borrow;
use std::cmp::Ordering;
use std::fmt::{self, Debug};
use std::hash::{self, Hash};
use std::iter::{self, Map};
use std::mem;
use std::ops;
use std::slice;

use self::Entry::{Occupied, Vacant};

// TODO: Unzip the vectors?
// Consideration: When unzipped, the compiler will not be able to understand
// that both of the `Vec`s have the same length, thus stuff like `iter` and so
// on should probably be implemented in unsafe code.

/// A very simple map implementation backed by a vector.
///
/// Use it like any map, as long as the number of elements that it stores is
/// very small.
///
/// # Example (like std's HashMap)
///
/// ```
/// use linear_map::LinearMap;
///
/// // type inference lets us omit an explicit type signature (which
/// // would be `LinearMap<&str, &str>` in this example).
/// let mut book_reviews = LinearMap::new();
///
/// // review some books.
/// book_reviews.insert("Adventures of Huckleberry Finn",    "My favorite book.");
/// book_reviews.insert("Grimms' Fairy Tales",               "Masterpiece.");
/// book_reviews.insert("Pride and Prejudice",               "Very enjoyable.");
/// book_reviews.insert("The Adventures of Sherlock Holmes", "Eye lyked it alot.");
///
/// // check for a specific one.
/// if !book_reviews.contains_key("Les Misérables") {
///     println!("We've got {} reviews, but Les Misérables ain't one.",
///              book_reviews.len());
/// }
///
/// // oops, this review has a lot of spelling mistakes, let's delete it.
/// book_reviews.remove("The Adventures of Sherlock Holmes");
///
/// // look up the values associated with some keys.
/// let to_find = ["Pride and Prejudice", "Alice's Adventure in Wonderland"];
/// for book in to_find.iter() {
///     match book_reviews.get(book) {
///         Some(review) => println!("{}: {}", *book, *review),
///         None => println!("{} is unreviewed.", *book)
///     }
/// }
///
/// // iterate over everything.
/// for (book, review) in book_reviews.iter() {
///     println!("{}: \"{}\"", *book, *review);
/// }
/// ```
#[derive(Clone)]
pub struct LinearMap<K,V> {
    storage: Vec<(K,V)>,
}

impl<K:Eq,V> LinearMap<K,V> {
    /// Creates an empty map. This method does not allocate.
    pub fn new() -> LinearMap<K,V> {
        LinearMap {
            storage: Vec::new(),
        }
    }

    /// Creates an empty map with the given initial capacity.
    pub fn with_capacity(capacity: usize) -> LinearMap<K,V> {
        LinearMap {
            storage: Vec::with_capacity(capacity),
        }
    }

    /// Returns the number of elements the map can hold without reallocating.
    pub fn capacity(&self) -> usize {
        self.storage.capacity()
    }

    /// Reserves capacity for at least `additional` more to be inserted in the
    /// map. The collection may reserve more space to avoid frequent
    /// reallocations.
    ///
    /// # Panics
    ///
    /// Panics if the new allocation size overflows `usize`.
    pub fn reserve(&mut self, additional: usize) {
        self.storage.reserve(additional);
    }

    /// Reserves the minimum capacity for exactly `additional` more elemnnts to
    /// be inserted in the map.
    ///
    /// Note that the allocator may give the collection more space than it
    /// requests. Therefore capacity cannot be relied upon to be precisely
    /// minimal. Prefer `reserve` if future insertions are expected.
    ///
    /// # Panics
    ///
    /// Panics if the new capacity overflows `usize`.
    pub fn reserve_exact(&mut self, additional: usize) {
        self.storage.reserve_exact(additional);
    }

    /// Shrinks the capacity of the map as much as possible.
    ///
    /// It will drop down as close as possible to the current length but the
    /// allocator may still inform the map that there is more space than
    /// necessary. Therefore capacity cannot be relid upon to be minimal.
    pub fn shrink_to_fit(&mut self) {
        self.storage.shrink_to_fit();
    }

    /// Returns the number of elements in the map.
    pub fn len(&self) -> usize {
        self.storage.len()
    }

    /// Returns true if the map contains no elements.
    pub fn is_empty(&self) -> bool {
        self.storage.is_empty()
    }

    /// Clears the map, removing all elements. Keeps the allocated memory for
    /// reuse.
    pub fn clear(&mut self) {
        self.storage.clear();
    }

    /// An iterator visiting all key-value pairs in arbitrary order. Iterator
    /// element type is `(&'a K, &'a V)`.
    pub fn iter<'a>(&'a self) -> Iter<'a, K, V> {
        fn ref_<A,B>(&(ref v1, ref v2): &(A, B)) -> (&A, &B) { (v1, v2) }
        Iter { iter: self.storage.iter().map(ref_::<K, V> as fn(&'a (K, V)) -> (&'a K, &'a V)) }
    }

    /// An iterator visiting all key-value pairs in arbitrary order with
    /// mutable references to the values. Iterator element type is `(&'a K, &'a
    /// mut V)`.
    pub fn iter_mut<'a>(&'a mut self) -> IterMut<'a, K, V> {
        fn ref_<A,B>(&mut (ref v1, ref mut v2): &mut (A, B)) -> (&A, &mut B) { (v1, v2) }
        IterMut { iter: self.storage.iter_mut().map(ref_::<K, V> as fn(&'a mut (K, V)) -> (&'a K, &'a mut V)) }
    }

    /// An iterator visiting all keys in arbitrary order. Iterator element type
    /// is `&'a K`.
    pub fn keys<'a>(&'a self) -> Keys<'a, K, V> {
        fn first<A,B>((v, _): (A, B)) -> A { v }
        Keys { iter: self.iter().map(first::<&'a K, &'a V> as fn((&'a K, &'a V)) -> &'a K) }
    }

    /// An iterator visiting all values in arbitrary order. Iterator element
    /// type is `&'a V`.
    pub fn values<'a>(&'a self) -> Values<'a, K, V> {
        fn second<A,B>((_, v): (A, B)) -> B { v }
        Values { iter: self.iter().map(second::<&'a K, &'a V> as fn((&'a K, &'a V)) -> &'a V) }
    }

    /// Returns a reference to the value corresponding to the key.
    pub fn get<Q: ?Sized>(&self, key: &Q) -> Option<&V> where K: Borrow<Q>, Q: Eq {
        for (k, v) in self.iter() {
            if key == k.borrow() {
                return Some(v);
            }
        }
        None
    }

    /// Returns a mutable reference to the value corresponding to the key.
    pub fn get_mut<Q: ?Sized>(&mut self, key: &Q) -> Option<&mut V> where K: Borrow<Q>, Q: Eq {
        for (k, v) in self.iter_mut() {
            if key == k.borrow() {
                return Some(v);
            }
        }
        None
    }

    /// Returns true if the map contains a value to the specified key.
    pub fn contains_key<Q: ?Sized>(&self, key: &Q) -> bool where K: Borrow<Q>, Q: Eq {
        self.get(key).is_some()
    }

    /// Inserts a key-value pair into the map. If the key already had a value
    /// present in the map, it is returned. Otherwise, `None` is returned.
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        for kv in self.storage.iter_mut() {
            let found;
            {
                let &mut (ref k, _) = kv;
                found = key == *k;
            }
            if found {
                let (_, v) = mem::replace(kv, (key, value));
                return Some(v);
            }
        }
        self.storage.push((key, value));
        None
    }

    /// Removes a key-value pair from the map. If the key had a value present
    /// in the map, it is returned. Otherwise, `None` is returned.
    pub fn remove<Q: ?Sized>(&mut self, key: &Q) -> Option<V> where K: Borrow<Q>, Q: Eq {
        for i in 0..self.storage.len() {
            let found;
            {
                let (ref k, _) = self.storage[i];
                found = key == k.borrow();
            }
            if found {
                let (_, v) = self.storage.swap_remove(i);
                return Some(v);
            }
        }
        None
    }

    /// Gets the given key's corresponding entry in the map for in-place manipulation.
    pub fn entry(&mut self, key: K) -> Entry<K, V> {
        match self.storage.iter().position(|&(ref k, _)| key == *k) {
            None => Vacant(VacantEntry {
                map: self,
                key: key
            }),
            Some(index) => Occupied(OccupiedEntry {
                map: self,
                index: index
            })
        }
    }
}

impl<K, V> Debug for LinearMap<K, V> where K: Eq + Debug, V: Debug {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

impl<K, V> Default for LinearMap<K, V> where K: Eq {
    fn default() -> Self { LinearMap::new() }
}

impl<K, V> Extend<(K, V)> for LinearMap<K, V> where K: Eq {
    fn extend<I: IntoIterator<Item = (K, V)>>(&mut self, key_values: I) {
        for (key, value) in key_values { self.insert(key, value); }
    }
}

impl<K, V> iter::FromIterator<(K, V)> for LinearMap<K, V> where K: Eq {
    fn from_iter<I: IntoIterator<Item = (K, V)>>(key_values: I) -> Self {
        let mut map = Self::new();
        map.extend(key_values);
        map
    }
}

impl<K, V> Hash for LinearMap<K, V> where K: Eq + Hash, V: Hash {
    fn hash<H: hash::Hasher>(&self, h: &mut H) {
        for e in self { e.hash(h); }
    }
}

impl<'a, K, V, Q: ?Sized> ops::Index<&'a Q> for LinearMap<K, V> where K: Eq + Borrow<Q>, Q: Eq {
    type Output = V;
    fn index(&self, key: &'a Q) -> &V { self.get(key).expect("key not found") }
}

impl<K, V> PartialEq for LinearMap<K, V> where K: Eq, V: PartialEq {
    fn eq(&self, other: &Self) -> bool { self.iter().eq(other.iter()) }
}

impl<K, V> Eq for LinearMap<K, V> where K: Eq, V: Eq {}

impl<K, V> PartialOrd for LinearMap<K, V> where K: Eq + PartialOrd, V: PartialOrd {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.iter().partial_cmp(other.iter())
    }
}

impl<K, V> Ord for LinearMap<K, V> where K: Ord, V: Ord {
    fn cmp(&self, other: &Self) -> Ordering { self.iter().cmp(other.iter()) }
}

/// A view into a single occupied location in a LinearMap.
pub struct OccupiedEntry<'a, K: 'a, V: 'a> {
    map: &'a mut LinearMap<K, V>,
    index: usize
}

/// A view into a single empty location in a LinearMap.
pub struct VacantEntry<'a, K: 'a, V: 'a> {
    map: &'a mut LinearMap<K, V>,
    key: K
}

/// A view into a single location in a map, which may be vacant or occupied.
pub enum Entry<'a, K: 'a, V: 'a> {
    /// An occupied Entry.
    Occupied(OccupiedEntry<'a, K, V>),

    /// A vacant Entry.
    Vacant(VacantEntry<'a, K, V>)
}

impl<'a, K, V> Entry<'a, K, V> {
    /// Ensures a value is in the entry by inserting the default if empty, and returns
    /// a mutable reference to the value in the entry.
    pub fn or_insert(self, default: V) -> &'a mut V {
        match self {
            Occupied(entry) => entry.into_mut(),
            Vacant(entry) => entry.insert(default)
        }
    }

    /// Ensures a value is in the entry by inserting the result of the default function if empty,
    /// and returns a mutable reference to the value in the entry.
    pub fn or_insert_with<F: FnOnce() -> V>(self, default: F) -> &'a mut V {
        match self {
            Occupied(entry) => entry.into_mut(),
            Vacant(entry) => entry.insert(default())
        }
    }
}

impl<'a, K, V> OccupiedEntry<'a, K, V> {
    /// Gets a reference to the value in the entry.
    pub fn get(&self) -> &V {
        &self.map.storage[self.index].1
    }

    /// Gets a mutable reference to the value in the entry.
    pub fn get_mut(&mut self) -> &mut V {
        &mut self.map.storage[self.index].1
    }

    /// Converts the OccupiedEntry into a mutable reference to the value in the entry
    /// with a lifetime bound to the map itself
    pub fn into_mut(self) -> &'a mut V {
        &mut self.map.storage[self.index].1
    }

    /// Sets the value of the entry, and returns the entry's old value
    pub fn insert(&mut self, mut value: V) -> V {
        let old_value = self.get_mut();
        mem::swap(&mut value, old_value);
        value
    }

    /// Takes the value out of the entry, and returns it
    pub fn remove(self) -> V {
        self.map.storage.swap_remove(self.index).1
    }
}

impl<'a, K, V> VacantEntry<'a, K, V> {
    /// Sets the value of the entry with the VacantEntry's key,
    /// and returns a mutable reference to it
    pub fn insert(self, value: V) -> &'a mut V {
        self.map.storage.push((self.key, value));
        &mut self.map.storage.last_mut().unwrap().1
    }
}

/// A consuming iterator over a map.
pub struct IntoIter<K, V> {
    iter: ::std::vec::IntoIter<(K, V)>,
}

impl<K, V> Iterator for IntoIter<K, V> {
    type Item = (K, V);
    fn next(&mut self) -> Option<Self::Item> { self.iter.next() }
    fn size_hint(&self) -> (usize, Option<usize>) { self.iter.size_hint() }
}

impl<K, V> DoubleEndedIterator for IntoIter<K, V> {
    fn next_back(&mut self) -> Option<Self::Item> { self.iter.next_back() }
}

impl<K, V> ExactSizeIterator for IntoIter<K, V> {
    fn len(&self) -> usize { self.iter.len() }
}

/// The iterator returned by `LinearMap::iter`.
pub struct Iter<'a, K:'a, V:'a> {
    iter: Map<slice::Iter<'a, (K, V)>, fn(&'a (K, V)) -> (&'a K, &'a V)>,
}

/// The iterator returned by `LinearMap::iter_mut`.
pub struct IterMut<'a, K:'a, V:'a> {
    iter: Map<slice::IterMut<'a, (K, V)>, fn(&'a mut (K, V)) -> (&'a K, &'a mut V)>,
}

/// The iterator returned by `LinearMap::keys`.
pub struct Keys<'a, K:'a, V:'a> {
    iter: Map<Iter<'a, K, V>, fn((&'a K, &'a V)) -> &'a K>,
}

/// The iterator returned by `LinearMap::values`.
pub struct Values<'a, K:'a, V:'a> {
    iter: Map<Iter<'a, K, V>, fn((&'a K, &'a V)) -> &'a V>,
}

impl<'a, K, V> Iterator for Iter<'a, K, V> {
    type Item = (&'a K, &'a V);
    fn next(&mut self) -> Option<(&'a K, &'a V)> { self.iter.next() }
    fn size_hint(&self) -> (usize, Option<usize>) { self.iter.size_hint() }
}

impl<'a, K, V> Iterator for IterMut<'a, K, V> {
    type Item = (&'a K, &'a mut V);
    fn next(&mut self) -> Option<(&'a K, &'a mut V)> { self.iter.next() }
    fn size_hint(&self) -> (usize, Option<usize>) { self.iter.size_hint() }
}

impl<'a, K, V> Iterator for Keys<'a, K, V> {
    type Item = &'a K;
    fn next(&mut self) -> Option<&'a K> { self.iter.next() }
    fn size_hint(&self) -> (usize, Option<usize>) { self.iter.size_hint() }
}

impl<'a, K, V> Iterator for Values<'a, K, V> {
    type Item = &'a V;
    fn next(&mut self) -> Option<&'a V> { self.iter.next() }
    fn size_hint(&self) -> (usize, Option<usize>) { self.iter.size_hint() }
}

impl<'a, K, V> Clone for Iter<'a, K, V> {
    fn clone(&self) -> Iter<'a, K, V> { Iter { iter: self.iter.clone() } }
}

impl<'a, K, V> Clone for Keys<'a, K, V> {
    fn clone(&self) -> Keys<'a, K, V> { Keys { iter: self.iter.clone() } }
}

impl<'a, K, V> Clone for Values<'a, K, V> {
    fn clone(&self) -> Values<'a, K, V> { Values { iter: self.iter.clone() } }
}

impl<'a, K, V> DoubleEndedIterator for Iter<'a, K, V> {
    fn next_back(&mut self) -> Option<(&'a K, &'a V)> { self.iter.next_back() }
}

impl<'a, K, V> DoubleEndedIterator for IterMut<'a, K, V> {
    fn next_back(&mut self) -> Option<(&'a K, &'a mut V)> { self.iter.next_back() }
}

impl<'a, K, V> DoubleEndedIterator for Keys<'a, K, V> {
    fn next_back(&mut self) -> Option<&'a K> { self.iter.next_back() }
}

impl<'a, K, V> DoubleEndedIterator for Values<'a, K, V> {
    fn next_back(&mut self) -> Option<&'a V> { self.iter.next_back() }
}

impl<'a, K, V> ExactSizeIterator for Iter   <'a, K, V> { }
impl<'a, K, V> ExactSizeIterator for IterMut<'a, K, V> { }
impl<'a, K, V> ExactSizeIterator for Keys   <'a, K, V> { }
impl<'a, K, V> ExactSizeIterator for Values <'a, K, V> { }

impl<K, V> IntoIterator for LinearMap<K, V> where K: Eq {
    type Item = (K, V);
    type IntoIter = IntoIter<K, V>;
    fn into_iter(self) -> IntoIter<K, V> { IntoIter { iter: self.storage.into_iter() } }
}

impl<'a, K, V> IntoIterator for &'a LinearMap<K, V> where K: Eq {
    type Item = (&'a K, &'a V);
    type IntoIter = Iter<'a, K, V>;
    fn into_iter(self) -> Iter<'a, K, V> { self.iter() }
}

impl<'a, K, V> IntoIterator for &'a mut LinearMap<K, V> where K: Eq {
    type Item = (&'a K, &'a mut V);
    type IntoIter = IterMut<'a, K, V>;
    fn into_iter(self) -> IterMut<'a, K, V> { self.iter_mut() }
}

#[cfg(test)]
mod test {
    use super::LinearMap;
    use super::Entry::{Occupied, Vacant};

    const TEST_CAPACITY: usize = 10;

    #[test]
    fn test_new() {
        let map: LinearMap<i32, i32> = LinearMap::new();
        assert_eq!(map.capacity(), 0);
        assert_eq!(map.len(), 0);
        assert!(map.is_empty());
    }

    #[test]
    fn test_with_capacity() {
        let map: LinearMap<i32, i32> = LinearMap::with_capacity(TEST_CAPACITY);
        assert!(map.capacity() >= TEST_CAPACITY);
    }

    #[test]
    fn test_capacity() {
        let mut map = LinearMap::new();
        map.insert(1, 2);
        assert!(map.capacity() >= 1);
        map.remove(&1);
        assert!(map.capacity() >= 1);
        map.reserve(TEST_CAPACITY);
        let capacity = map.capacity();
        assert!(capacity >= TEST_CAPACITY);
        for i in 0..TEST_CAPACITY as i32 {
            assert!(map.insert(i, i).is_none());
        }
        assert_eq!(capacity, map.capacity());
    }

    #[test]
    fn test_reserve() {
        let mut map = LinearMap::new();
        map.reserve(TEST_CAPACITY);
        assert!(map.capacity() >= TEST_CAPACITY);
        for i in 0..TEST_CAPACITY as i32 {
            assert!(map.insert(i, i).is_none());
        }
        map.reserve(TEST_CAPACITY);
        assert!(map.capacity() >= 2 * TEST_CAPACITY);

        let mut map = LinearMap::new();
        map.reserve(TEST_CAPACITY);
        assert!(map.capacity() >= TEST_CAPACITY);
        for i in 0..TEST_CAPACITY as i32 {
            assert!(map.insert(i, i).is_none());
        }
        map.reserve(TEST_CAPACITY);
        assert!(map.capacity() >= 2 * TEST_CAPACITY);
    }

    #[test]
    fn test_shrink_to_fit() {
        let mut map = LinearMap::new();
        map.shrink_to_fit();
        assert_eq!(map.capacity(), 0);
        map.reserve(TEST_CAPACITY);
        map.shrink_to_fit();
        assert_eq!(map.capacity(), 0);
        for i in 0..TEST_CAPACITY as i32 {
            assert!(map.insert(i, i).is_none());
        }
        map.shrink_to_fit();
        assert_eq!(map.len(), TEST_CAPACITY);
        assert!(map.capacity() >= TEST_CAPACITY);
    }

    #[test]
    fn test_len_and_is_empty() {
        let mut map = LinearMap::new();
        assert_eq!(map.len(), 0);
        assert!(map.is_empty());
        map.insert(100, 100);
        assert_eq!(map.len(), 1);
        assert!(!map.is_empty());
        for i in 0..TEST_CAPACITY as i32 {
            assert!(map.insert(i, i).is_none());
        }
        assert_eq!(map.len(), 1 + TEST_CAPACITY);
        assert!(!map.is_empty());
        assert!(map.remove(&100).is_some());
        assert_eq!(map.len(), TEST_CAPACITY);
        assert!(!map.is_empty());
    }

    #[test]
    fn test_clear() {
        let mut map = LinearMap::new();
        map.clear();
        assert_eq!(map.len(), 0);
        for i in 0..TEST_CAPACITY as i32 {
            assert!(map.insert(i, i).is_none());
        }
        map.clear();
        assert_eq!(map.len(), 0);
        assert!(map.capacity() > 0);
    }

    #[test]
    fn test_iterators() {
        const ONE:   i32 = 0b0001;
        const TWO:   i32 = 0b0010;
        const THREE: i32 = 0b0100;
        const FOUR:  i32 = 0b1000;
        const ALL:   i32 = 0b1111;
        let mut map = LinearMap::new();
        assert!(map.insert(ONE, TWO).is_none());
        assert!(map.insert(TWO, THREE).is_none());
        assert!(map.insert(THREE, FOUR).is_none());
        assert!(map.insert(FOUR, ONE).is_none());

        {
            let mut result_k = 0;
            let mut result_v = 0;
            for (&k, &v) in map.iter() {
                result_k ^= k;
                result_v ^= v;
                assert_eq!(((k << 1) & ALL) | ((k >> 3) & ALL), v);
            }
            assert_eq!(result_k, ALL);
            assert_eq!(result_v, ALL);
        }
        {
            let mut result_k = 0;
            let mut result_v = 0;
            for (&k, &mut v) in map.iter_mut() {
                result_k ^= k;
                result_v ^= v;
                assert_eq!(((k << 1) & ALL) | ((k >> 3) & ALL), v);
            }
            assert_eq!(result_k, ALL);
            assert_eq!(result_v, ALL);
        }
        {
            let mut result = 0;
            for &k in map.keys() {
                result ^= k;
            }
            assert_eq!(result, ALL);
        }
        {
            let mut result = 0;
            for &v in map.values() {
                result ^= v;
            }
            assert_eq!(result, ALL);
        }
    }

    #[test]
    fn test_insert_remove_get() {
        let mut map = LinearMap::new();
        assert!(map.insert(100, 101).is_none());
        assert!(map.contains_key(&100));
        assert_eq!(map.get(&100), Some(&101));
        assert_eq!(map.get_mut(&100), Some(&mut 101));
        for i in 0..TEST_CAPACITY as i32 {
            assert!(map.insert(i, i).is_none());
        }
        assert_eq!(map.insert(100, 102), Some(101));
        assert_eq!(map.remove(&100), Some(102));
        assert_eq!(map.remove(&100), None);
        assert_eq!(map.remove(&1000), None);
    }

    #[test]
    fn test_entry() {
        let xs = [(1, 10), (2, 20), (3, 30), (4, 40), (5, 50), (6, 60)];

        let mut map = LinearMap::new();

        for &(k, v) in &xs {
            map.insert(k, v);
        }

        // Existing key (insert)
        match map.entry(1) {
            Vacant(_) => unreachable!(),
            Occupied(mut view) => {
                assert_eq!(view.get(), &10);
                assert_eq!(view.insert(100), 10);
            }
        }
        assert_eq!(map.get(&1).unwrap(), &100);
        assert_eq!(map.len(), 6);


        // Existing key (update)
        match map.entry(2) {
            Vacant(_) => unreachable!(),
            Occupied(mut view) => {
                let v = view.get_mut();
                let new_v = (*v) * 10;
                *v = new_v;
            }
        }
        assert_eq!(map.get(&2).unwrap(), &200);
        assert_eq!(map.len(), 6);

        // Existing key (take)
        match map.entry(3) {
            Vacant(_) => unreachable!(),
            Occupied(view) => {
                assert_eq!(view.remove(), 30);
            }
        }
        assert_eq!(map.get(&3), None);
        assert_eq!(map.len(), 5);


        // Inexistent key (insert)
        match map.entry(10) {
            Occupied(_) => unreachable!(),
            Vacant(view) => {
                assert_eq!(*view.insert(1000), 1000);
            }
        }
        assert_eq!(map.get(&10).unwrap(), &1000);
        assert_eq!(map.len(), 6);
    }
}

#[cfg(all(test, feature = "nightly"))]
mod bench {
    use super::LinearMap;

    extern crate test;

    const SMALL:  u32 =   10;
    const MEDIUM: u32 =  100;
    const BIG:    u32 = 1000;

    fn insert(b: &mut test::Bencher, num: u32) {
        b.iter(|| {
            let mut map = LinearMap::new();
            for i in 0..num {
                map.insert(i, i);
            }
        })
    }

    fn remove_insert(b: &mut test::Bencher, num: u32) {
        b.iter(|| {
            let mut map = LinearMap::new();
            for i in 0..num {
                map.insert(i, i);
            }
            for i in 0..num {
                map.remove(&i);
            }
        })
    }

    fn remove_rev_insert(b: &mut test::Bencher, num: u32) {
        b.iter(|| {
            let mut map = LinearMap::new();
            for i in 0..num {
                map.insert(i, i);
            }
            for i in 0..num {
                map.remove(&(num - i - 1));
            }
        })
    }

    fn get_middle(b: &mut test::Bencher, num: u32) {
        let mut map = LinearMap::new();
        for i in 0..num {
            map.insert(i, i);
        }
        let middle = num / 2;
        b.iter(|| {
            test::black_box(map.get(&middle));
            test::black_box(map.get_mut(&middle));
        })
    }

    fn get_none(b: &mut test::Bencher, num: u32) {
        let mut map = LinearMap::new();
        for i in 0..num {
            map.insert(i, i);
        }
        let none = num + 1;
        b.iter(|| {
            test::black_box(map.get(&none));
            test::black_box(map.get_mut(&none));
        })
    }

    #[bench] fn bench_insert_small (b: &mut test::Bencher) { insert(b, SMALL);  }
    #[bench] fn bench_insert_medium(b: &mut test::Bencher) { insert(b, MEDIUM); }
    #[bench] fn bench_insert_big   (b: &mut test::Bencher) { insert(b, BIG);    }

    #[bench] fn bench_remove_insert_small (b: &mut test::Bencher) { remove_insert(b, SMALL);  }
    #[bench] fn bench_remove_insert_medium(b: &mut test::Bencher) { remove_insert(b, MEDIUM); }
    #[bench] fn bench_remove_insert_big   (b: &mut test::Bencher) { remove_insert(b, BIG);    }

    #[bench] fn bench_remove_rev_insert_small (b: &mut test::Bencher) { remove_rev_insert(b, SMALL);  }
    #[bench] fn bench_remove_rev_insert_medium(b: &mut test::Bencher) { remove_rev_insert(b, MEDIUM); }
    #[bench] fn bench_remove_rev_insert_big   (b: &mut test::Bencher) { remove_rev_insert(b, BIG);    }

    #[bench] fn bench_get_middle_small (b: &mut test::Bencher) { get_middle(b, SMALL);  }
    #[bench] fn bench_get_middle_medium(b: &mut test::Bencher) { get_middle(b, MEDIUM); }
    #[bench] fn bench_get_middle_big   (b: &mut test::Bencher) { get_middle(b, BIG);    }

    #[bench] fn bench_get_none_small (b: &mut test::Bencher) { get_none(b, SMALL);  }
    #[bench] fn bench_get_none_medium(b: &mut test::Bencher) { get_none(b, MEDIUM); }
    #[bench] fn bench_get_none_big   (b: &mut test::Bencher) { get_none(b, BIG);    }
}
