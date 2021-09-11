#![feature(simd_x86_bittest)]
// bench.rs
#![feature(test)]

extern crate test;

use rand::prelude::*;
use std::fmt;
use std::mem;
use test::Bencher;

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    _bittest64, _bittestandcomplement64, _bittestandreset64, _bittestandset64,
};

type WORD = i64;
pub const WORD_SIZE: usize = mem::size_of::<WORD>() * 8;
const MASK: usize = WORD_SIZE - 1;

pub struct BitVec {
    store: Vec<i64>,
}

impl fmt::Debug for BitVec {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let bits: Vec<String> = self
            .store
            .iter()
            .map(|&v| {
                let a7 = v as u8;
                let a6 = (v >> 8) as u8;
                let a5 = (v >> 16) as u8;
                let a4 = (v >> 24) as u8;
                let a3 = (v >> 32) as u8;
                let a2 = (v >> 40) as u8;
                let a1 = (v >> 48) as u8;
                let a0 = (v >> 56) as u8;
                format!(
                    "{:08b}_{:08b}_{:08b}_{:08b}_{:08b}_{:08b}_{:08b}_{:08b}",
                    a0, a1, a2, a3, a4, a5, a6, a7
                )
            })
            .collect();
        write!(f, "{}", bits.join("_"))
    }
}

impl BitVec {
    pub fn new() -> Self {
        BitVec { store: vec![0_i64] }
    }

    // capacity in bits
    pub fn with_capacity(cap: usize) -> Self {
        let ncap = cap / WORD_SIZE + ((cap & MASK != 0) as usize);
        BitVec {
            store: std::iter::repeat(0_i64).take(ncap).collect(),
        }
    }

    pub fn capacity(&self) -> usize {
        self.store.len() * WORD_SIZE
    }

    pub fn chunks_count(&self) -> usize {
        self.store.len()
    }

    pub fn extend_to_fit(&mut self, idx: usize) {
        let cap = idx / WORD_SIZE;
        while self.store.len() <= cap {
            self.store.push(0_i64);
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.store
            .iter()
            .map(|word| word.count_ones() as usize)
            .sum()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.store.iter().sum::<i64>() == 0
    }

    #[inline]
    #[cfg(target_arch = "x86_64")]
    pub fn toggle_bit(&mut self, idx: usize) {
        unsafe { _bittestandcomplement64(self.store.as_mut_ptr(), idx as i64) };
    }

    #[inline]
    #[cfg(not(target_arch = "x86_64"))]
    pub fn toggle_bit(&mut self, idx: usize) {
        let word = idx / WORD_SIZE;
        debug_assert!(
            word < self.store.len(),
            "word: {} store len: {}",
            word,
            self.store.len()
        );
        let bit = idx & MASK;
        self.store[word] ^= 1 << bit;
    }

    #[inline]
    #[cfg(target_arch = "x86_64")]
    pub fn is_set(&self, idx: usize) -> bool {
        let bit = unsafe { _bittest64(self.store.as_ptr(), idx as i64) };
        bit != 0
    }

    #[inline]
    // #[cfg(not(target_arch = "x86_64"))]
    pub fn is_set_scalar(&self, idx: usize) -> bool {
        let word = idx / WORD_SIZE;
        debug_assert!(
            word < self.store.len(),
            "word: {} store len: {}",
            word,
            self.store.len()
        );
        let bit = idx & MASK;
        (self.store[word] & (1 << bit)) != 0
    }

    #[inline]
    #[cfg(target_arch = "x86_64")]
    pub fn set_bit(&mut self, idx: usize) {
        unsafe { _bittestandset64(self.store.as_mut_ptr(), idx as i64) };
    }

    #[inline]
    #[cfg(not(target_arch = "x86_64"))]
    pub fn set_bit(&mut self, idx: usize) {
        let word = idx / WORD_SIZE;
        debug_assert!(
            word < self.store.len(),
            "word: {} store len: {}",
            word,
            self.store.len()
        );
        let bit = idx & MASK;
        self.store[word] |= 1 << bit;
    }

    #[inline]
    #[cfg(target_arch = "x86_64")]
    pub fn reset_bit(&mut self, idx: usize) {
        unsafe { _bittestandreset64(self.store.as_mut_ptr(), idx as i64) };
    }

    #[inline]
    #[cfg(not(target_arch = "x86_64"))]
    pub fn reset_bit(&mut self, idx: usize) {
        let word = idx / WORD_SIZE;
        debug_assert!(
            word < self.store.len(),
            "word: {} store len: {}",
            word,
            self.store.len()
        );
        let bit = idx & MASK;
        self.store[word] &= !(1 << bit);
    }

    #[inline]
    pub fn contains(&self, other: &Self) -> bool {
        for (i, word) in other.store.iter().map(|v| *v).enumerate() {
            if word != (self.store[i] & word) {
                return false;
            }
        }
        true
    }

    #[inline]
    pub fn matches(&self, other: &Self) -> bool {
        self.store
            .iter()
            .zip(other.store.iter())
            .all(|(l, r)| l == r)
    }

    #[inline]
    pub fn iter<'a>(&'a self) -> BitIterator<'a> {
        let inner = self.store[0];
        BitIterator {
            bufer: self.store.as_ref(),
            index: 0,
            inner: inner,
        }
    }

    #[inline]
    pub fn drain<'a>(&'a mut self) -> Drain<'a> {
        Drain {
            bufer: self.store.as_mut(),
            index: 0,
        }
    }

    #[inline]
    pub fn inner_mut(&mut self) -> &mut [i64] {
        self.store.as_mut()
    }

    #[inline]
    pub fn next_zero(&self) -> Option<usize> {
        self.store.iter().enumerate().find_map(|(i, word)| {
            let bit = word.trailing_ones() as usize;
            if bit == WORD_SIZE {
                None
            } else {
                Some(bit + i * WORD_SIZE)
            }
        })
    }

    #[inline]
    pub fn next_one(&self) -> Option<usize> {
        self.store.iter().enumerate().find_map(|(i, word)| {
            let bit = word.trailing_zeros() as usize;
            if bit == WORD_SIZE {
                None
            } else {
                Some(bit + i * WORD_SIZE)
            }
        })
    }

    // #[inline]
    // pub fn drain_map<'a, F: 'static, T>(&'a mut self, f: F) -> DrainMap<'a, F, T>
    // where
    //     F: FnMut(usize) -> Option<T>,
    // {
    //     let inner = self.store[0];
    //     DrainMap {
    //         bufer: self.store.as_mut(),
    //         index: 0,
    //         inner: inner,
    //         filtr: f,
    //     }
    // }

    #[inline]
    pub fn zip_drain_map<'a, F: 'static, T>(
        &'a mut self,
        other: &'a mut Self,
        f: F,
    ) -> ZipDrainMap<'a, F, T>
    where
        F: FnMut(usize) -> Option<T>,
    {
        ZipDrainMap::new(self.inner_mut(), other.inner_mut(), f)
    }

    #[inline]
    pub fn swap(&mut self, other: &mut Self) {
        std::mem::swap(&mut self.store, &mut other.store)
    }
}

pub struct BitIterator<'a> {
    bufer: &'a [i64],
    index: usize,
    inner: i64,
}

impl<'a> Iterator for BitIterator<'a> {
    type Item = usize;
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // process bit word
            while self.inner != 0 {
                let bit = self.inner.trailing_zeros();
                self.inner &= self.inner.wrapping_sub(1);
                return Some(bit as usize + self.index * WORD_SIZE);
            }

            // step to next word if any
            if self.index + 1 == self.bufer.len() {
                return None;
            }

            self.index += 1;
            self.inner = self.bufer[self.index];
        }
    }
}

pub struct Drain<'a> {
    bufer: &'a mut [i64],
    index: usize,
}

impl<'a> Iterator for Drain<'a> {
    type Item = usize;
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // process bit word
            let inner = &mut self.bufer[self.index];
            while *inner != 0 {
                let bit = inner.trailing_zeros();
                *inner &= inner.wrapping_sub(1);
                return Some(bit as usize + self.index * WORD_SIZE);
            }

            // step to next word if any
            if self.index + 1 == self.bufer.len() {
                return None;
            }

            self.index += 1;
        }
    }
}

pub struct DrainMap<'a, F, T>
where
    F: FnMut(usize) -> Option<T>,
{
    bufer: &'a mut [i64],
    index: usize,
    inner: i64,
    filtr: F,
}

impl<'a, F, T> Iterator for DrainMap<'a, F, T>
where
    F: FnMut(usize) -> Option<T>,
{
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let word = &mut self.bufer[self.index];
            // process bit word
            while self.inner != 0 {
                let bit = self.inner.trailing_zeros();
                self.inner &= self.inner.wrapping_sub(1);
                match (self.filtr)(bit as usize + self.index * WORD_SIZE) {
                    Some(v) => {
                        *word &= !(1 << bit);
                        return Some(v);
                    }
                    None => (),
                }
            }

            // step to next word if any
            if self.index + 1 == self.bufer.len() {
                return None;
            }

            self.index += 1;
            self.inner = self.bufer[self.index];
        }
    }
}

pub struct ZipDrainMap<'a, F, T>
where
    F: FnMut(usize) -> Option<T>,
{
    buff1: &'a mut [i64],
    buff2: &'a mut [i64],
    limit: usize,
    index: usize,
    inner: i64,
    maper: F,
}

impl<'a, F, T> ZipDrainMap<'a, F, T>
where
    F: FnMut(usize) -> Option<T>,
{
    fn new(buff1: &'a mut [i64], buff2: &'a mut [i64], maper: F) -> Self {
        let limit = std::cmp::min(buff1.len(), buff2.len());
        let index = 0;
        let inner = buff1[0] & buff2[0];
        ZipDrainMap {
            buff1,
            buff2,
            limit,
            index,
            inner,
            maper,
        }
    }
}

impl<'a, F, T> Iterator for ZipDrainMap<'a, F, T>
where
    F: FnMut(usize) -> Option<T>,
{
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // process bit word
            while self.inner != 0 {
                let bit = self.inner.trailing_zeros();
                self.inner &= self.inner.wrapping_sub(1);
                if let Some(v) = (self.maper)(bit as usize + self.index * WORD_SIZE) {
                    let word1 = &mut self.buff1[self.index];
                    let word2 = &mut self.buff2[self.index];
                    *word1 &= !(1 << bit as i64);
                    *word2 &= !(1 << bit as i64);
                    return Some(v);
                }
            }

            // step to next word if any
            if self.index + 1 == self.limit {
                return None;
            }

            self.index += 1;

            let word1 = &mut self.buff1[self.index];
            let word2 = &mut self.buff2[self.index];

            self.inner = *word1 & *word2;
        }
    }
}

fn main() {
    let mut bits = BitVec::with_capacity(1024);
    bits.set_bit(657);
    println!("{}", bits.is_set(657));
}

const CAP: usize = 1000000;

#[bench]
fn bitvec_avx(b: &mut Bencher) {
    let mut bitvec = BitVec::with_capacity(CAP);
    let mut rng = rand::thread_rng();

    for _ in 0..CAP {
        let y: usize = rng.gen::<usize>() % CAP;
        bitvec.set_bit(y);
    }

    b.iter(|| {
        for i in 0..CAP {
            bitvec.is_set(i);
        }
    })
}

#[bench]
fn bitvec_scalar(b: &mut Bencher) {
    let mut bitvec = BitVec::with_capacity(CAP);
    let mut rng = rand::thread_rng();

    for _ in 0..CAP {
        let y: usize = rng.gen::<usize>() % CAP;
        bitvec.set_bit(y);
    }

    b.iter(|| {
        for i in 0..CAP {
            bitvec.is_set_scalar(i);
        }
    })
}
