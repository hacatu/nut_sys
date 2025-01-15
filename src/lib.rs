#![feature(allocator_api, ptr_as_ref_unchecked)]
use std::{borrow::{Borrow, BorrowMut}, marker::PhantomData, mem::MaybeUninit, ops::{Index, IndexMut}};
use raw::{_Diri, _Factors};

pub mod raw {
    use crate::{FactorConf, PrimePower};

	#[repr(C)]
	pub struct _Factors {
		pub(crate) num_primes: u64,
		pub(crate) factors: [PrimePower; 0]
	}

	#[repr(C)]
	pub struct _Diri {
		pub(crate) x: i64,
		pub(crate) y: i64,
		pub(crate) yinv: i64,
		pub(crate) buf: *mut i64
	}
	
	#[link(name = "nut")]
	extern {
		pub fn nut_sieve_phi(max: u64) -> *mut u64;
		pub fn nut_sieve_sigma_0(max: u64) -> *mut u64;
		pub fn nut_sieve_omega(max: u64) -> *mut u8;
		pub fn nut_sieve_mobius(max: u64) -> *mut u8;
		pub fn nut_sieve_largest_factors(max: u64) -> *mut u64;
		pub fn nut_sieve_carmichael(max: u64) -> *mut u64;
		pub fn nut_make_Factors_w(max_primes: u64) -> *mut _Factors;
		pub fn nut_fill_factors_from_largest(out: *mut _Factors, n: u64, largest_factors: *const u64);
		pub fn nut_Factor_fprint(file: *mut libc::FILE, factors: *const _Factors);
		pub fn nut_Factor_append(factors: *mut _Factors, m: u64, k: u64);
		pub fn nut_Factor_combine(factors: *mut _Factors, factors2: *const _Factors, k: u64);
		pub fn nut_Factor_ipow(factors: *mut _Factors, power: u64);
		pub fn nut_Factor_forall_divs_tmptmp(factors: *const _Factors, f: extern fn(*const _Factors, u64, *mut libc::c_void) -> libc::c_int, data: *mut libc::c_void) -> libc::c_int;
		pub fn nut_Factor_forall_divs_le_tmptmp(factors: *const _Factors, d_max: u64, f: extern fn(*const _Factors, u64, *mut libc::c_void) -> libc::c_int, data: *mut libc::c_void) -> libc::c_int;
		pub fn nut_u64_factor_heuristic(n: u64, num_primes: u64, primes: *const u64, conf: *const FactorConf, factors: *mut _Factors) -> u64;
		pub fn nut_u64_order_mod(a: u64, n: u64, cn: u64, cn_factors: *mut _Factors) -> u64;
		pub fn nut_max_prime_divs(max: u64) -> u64;
		pub fn nut_u64_is_prime_dmr(n: u64) -> bool;
		pub fn nut_Diri_init(_self: *mut _Diri, x: i64, y: i64) -> bool;
		pub fn nut_Diri_destroy(_self: *mut _Diri);
		pub fn nut_Diri_compute_pi(_self: *mut _Diri) -> bool;
		pub static nut_small_primes: *const u64;
		pub static nut_default_factor_conf: FactorConf;
	}
}

#[repr(C)]
pub struct FactorConf {
	pollard_max: u64,
 	pollard_stride: u64,
 	lenstra_max: u64,
 	lenstra_bfac: u64,
 	qsieve_max: u64
}

#[repr(C)]
pub struct PrimePower {
	pub prime: u64,
	pub power: u64
}

pub struct Factors {
	_inner: *mut raw::_Factors
}

impl Factors {
	pub fn make_ub(max: u64) -> Factors {
		Self::make_w(max_prime_divs(max))
	}

	pub fn make_w(max_primes: u64) -> Self {
		Self { _inner: unsafe { raw::nut_make_Factors_w(max_primes) } }
	}

	pub fn fill_from_largest_factors(&mut self, n: u64, largest_factors: &[u64]) {
		unsafe { raw::nut_fill_factors_from_largest(self.borrow_mut(), n, largest_factors.as_ptr()); }
	}

	pub fn factor_heuristic(&mut self, n: u64) -> u64 {
		unsafe { raw::nut_u64_factor_heuristic(n, 25, raw::nut_small_primes, &raw::nut_default_factor_conf, self.borrow_mut()) }
	}

	pub fn append(&mut self, p: u64, e: u64) {
		unsafe { raw::nut_Factor_append(self.borrow_mut(), p, e); }
	}

	pub fn combine(&mut self, other: &Self, e: u64) {
		unsafe { raw::nut_Factor_combine(self.borrow_mut(), other.borrow(), e); }
	}

	pub fn pow(&mut self, power: u64) {
		unsafe { raw::nut_Factor_ipow(self.borrow_mut(), power) }
	}

	pub fn num_primes(&self) -> usize {
		unsafe { &*self._inner } .num_primes as _
	}

	pub fn iter_divs(&self) -> DivisorIterator<'_> {
		let mut dfactors = Self::make_w(self.num_primes()as _);
		unsafe { (*dfactors._inner).num_primes = self.num_primes()as _ };
		for i in 0..self.num_primes() {
			dfactors[i] = PrimePower { prime: 1, power: 0 };
		}
		DivisorIterator { factors: self, dfactors, d: 1 }
	}

	pub fn iter_divs_le(&self, d_max: u64) -> DivisorLeIterator<'_> {
		let mut dfactors = Self::make_w(self.num_primes()as _);
		unsafe { (*dfactors._inner).num_primes = self.num_primes()as _ };
		for i in 0..self.num_primes() {
			dfactors[i] = PrimePower { prime:1, power: 0 };
		}
		DivisorLeIterator { factors: self, dfactors, d_max, d: 1 }
	}
}

impl Borrow<_Factors> for Factors {
	fn borrow(&self) -> &_Factors {
		unsafe { self._inner.as_ref() }.unwrap()
	}
}

impl BorrowMut<_Factors> for Factors {
	fn borrow_mut(&mut self) -> &mut _Factors {
		unsafe { self._inner.as_mut() }.unwrap()
	}
}

impl Drop for Factors {
	fn drop(&mut self) {
		unsafe { libc::free(self._inner.cast()) }
	}
}

impl Index<usize> for Factors {
	type Output = PrimePower;
	fn index(&self, index: usize) -> &Self::Output {
		assert!(index < self.num_primes(), "Index out of bound!");
		unsafe { (*self._inner).factors.as_ptr().offset(index as _).as_ref_unchecked() }
	}
}

impl IndexMut<usize> for Factors {
	fn index_mut(&mut self, index: usize) -> &mut Self::Output {
		assert!(index < self.num_primes(), "Index out of bounds!");
		unsafe { (*self._inner).factors.as_mut_ptr().offset(index as _).as_mut_unchecked() }
	}
}

pub struct DivisorIterator<'a> {
	factors: &'a Factors,
	dfactors: Factors,
	d: u64
}

impl<'a> Iterator for DivisorIterator<'a> {
	type Item = u64;
	fn next(&mut self) -> Option<Self::Item> {
		if self.d == 0 {
			return None
		}
		for i in 0..self.dfactors.num_primes() {
			let cur_ppow = &mut self.dfactors[i];
			let max_ppow = &self.factors[i];
			if cur_ppow.power < max_ppow.power {
				cur_ppow.power += 1;
				cur_ppow.prime *= max_ppow.prime;
				self.d *= max_ppow.prime;
				return Some(self.d);
			}
			cur_ppow.power = 0;
			self.d /= cur_ppow.prime;
			cur_ppow.prime = 1;
		}
		self.d = 0;
		Some(1)
	}
}

pub struct DivisorLeIterator<'a> {
	factors: &'a Factors,
	dfactors: Factors,
	d_max: u64,
	d: u64
}

impl<'a> Iterator for DivisorLeIterator<'a> {
	type Item = u64;
	fn next(&mut self) -> Option<Self::Item> {
		if self.d == 0 {
			return None
		}
		if self.d_max <= 1 {
			self.d = 0;
			return if self.d_max == 1 { Some(1) } else { None }
		}
		for i in 0..self.dfactors.num_primes() {
			let cur_ppow = &mut self.dfactors[i];
			let max_ppow = &self.factors[i];
			if cur_ppow.power < max_ppow.power {
				cur_ppow.power += 1;
				cur_ppow.prime *= max_ppow.prime;
				self.d *= max_ppow.prime;
				if self.d <= self.d_max {
					return Some(self.d);
				}
			}
			cur_ppow.power = 0;
			self.d /= cur_ppow.prime;
			cur_ppow.prime = 1;
		}
		self.d = 0;
		Some(1)
	}
}

pub struct Diri {
	_inner: raw::_Diri
}

impl Diri {
	pub fn try_new(x: i64, y: i64) -> Option<Self> {
		let mut inner = MaybeUninit::uninit();
		unsafe {
			if !raw::nut_Diri_init(inner.as_mut_ptr(), x, y) {
				None
			} else {
				Some(Self { _inner: inner.assume_init() })
			}
		}
	}

	pub fn compute_pi(&mut self) -> bool {
		unsafe { raw::nut_Diri_compute_pi(self.borrow_mut()) }
	}
}

unsafe impl Sync for Diri {}

impl Borrow<_Diri> for Diri {
	fn borrow(&self) -> &_Diri {
		&self._inner
	}
}

impl BorrowMut<_Diri> for Diri {
	fn borrow_mut(&mut self) -> &mut _Diri {
		&mut self._inner
	}
}

impl Index<usize> for Diri {
	type Output = i64;
	fn index(&self, index: usize) -> &Self::Output {
		let index = if index > self._inner.y as usize {
			self._inner.y as usize + self._inner.x as usize / index
		} else { index };
		unsafe { self._inner.buf.offset(index as _).as_ref().unwrap() }
	}
}

impl Drop for Diri {
	fn drop(&mut self) {
		unsafe { raw::nut_Diri_destroy(&mut self._inner) }
	}
}

pub struct MallocArray<T> {
	_buf: *mut libc::c_void,
	_len: usize,
	_elem: PhantomData<T>
}

unsafe impl<T> Send for MallocArray<T> {}

impl<T> Borrow<[T]> for MallocArray<T> {
	fn borrow<'a>(&'a self) -> &'a [T] {
		unsafe { std::slice::from_raw_parts(self._buf.cast(), self._len) }
	}
}

impl<T> BorrowMut<[T]> for MallocArray<T> {
	fn borrow_mut<'a>(&'a mut self) -> &'a mut [T] {
		unsafe { std::slice::from_raw_parts_mut(self._buf.cast(), self._len) }
	}
}

impl<T> Drop for MallocArray<T> {
	fn drop(&mut self) {
		unsafe { libc::free(self._buf) }
	}
}


pub fn sieve_phi(max: u64) -> MallocArray<u64> {
	unsafe {
		let buf = raw::nut_sieve_phi(max);
		MallocArray {
			_buf: buf.cast(),
			_len: (max + 1) as usize,
			_elem: PhantomData
		}
	}
}

pub fn sieve_sigma_0(max: u64) -> MallocArray<u64> {
	unsafe {
		let buf = raw::nut_sieve_sigma_0(max);
		MallocArray {
			_buf: buf.cast(),
			_len: (max + 1) as usize,
			_elem: PhantomData
		}
	}
}

pub fn sieve_omega(max: u64) -> MallocArray<u8> {
	unsafe {
		let buf = raw::nut_sieve_omega(max);
		MallocArray {
			_buf: buf.cast(),
			_len: (max + 1) as usize,
			_elem: PhantomData
		}
	}
}

pub fn sieve_mobius(max: u64) -> MallocArray<u8> {
	unsafe {
		let buf = raw::nut_sieve_mobius(max);
		MallocArray {
			_buf: buf.cast(),
			_len: (max + 1) as usize,
			_elem: PhantomData
		}
	}
}

pub fn sieve_largest_factors(max: u64) -> MallocArray<u64> {
	unsafe {
		let buf = raw::nut_sieve_largest_factors(max);
		MallocArray {
			_buf: buf.cast(),
			_len: (max + 1) as usize,
			_elem: PhantomData
		}
	}
}

pub fn sieve_carmichael(max: u64) -> MallocArray<u64> {
	unsafe {
		let buf = raw::nut_sieve_carmichael(max);
		MallocArray {
			_buf: buf.cast(),
			_len: (max + 1) as usize,
			_elem: PhantomData
		}
	}
}

pub fn get_mobius_val(mobius_tbl: &[u8], idx: usize) -> i64 {
	match (mobius_tbl[idx/4] >> (idx%4*2))%4 {
		0 => 0, 1 => 1, 3 => -1, _ => unreachable!()
	}
}

pub fn order_mod(a: u64, n: u64, cn: u64, cn_factors: &mut Factors) -> u64 {
	unsafe { raw::nut_u64_order_mod(a, n, cn, cn_factors.borrow_mut()) }
}

pub fn max_prime_divs(max: u64) -> u64 {
	unsafe { raw::nut_max_prime_divs(max) }
}

pub fn is_prime_dmr(n: u64) -> bool {
	unsafe { raw::nut_u64_is_prime_dmr(n) }
}

