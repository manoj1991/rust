// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::cmp::Ordering;
use std::num::ParseFloatError;

use syntax::ast;

use super::apfloat::{Float, IeeeSingle, IeeeDouble, OpStatus, Round};
use super::err::*;

// Note that equality for `ConstFloat` means that the it is the same
// constant, not that the rust values are equal. In particular, `NaN
// == NaN` (at least if it's the same NaN; distinct encodings for NaN
// are considering unequal).
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash, RustcEncodable, RustcDecodable)]
pub enum ConstFloat {
    F32(u32),
    F64(u64)
}
use self::ConstFloat::*;

impl ConstFloat {
    /// Description of the type, not the value
    pub fn description(&self) -> &'static str {
        match *self {
            F32(_) => "f32",
            F64(_) => "f64",
        }
    }

    pub fn is_nan(&self) -> bool {
        match *self {
            F32(f) => IeeeSingle::from_bits(f as u128).is_nan(),
            F64(f) => IeeeDouble::from_bits(f as u128).is_nan(),
        }
    }

    /// Compares the values if they are of the same type
    pub fn try_cmp(self, rhs: Self) -> Result<Ordering, ConstMathErr> {
        match (self, rhs) {
            (F64(a), F64(b))  => {
                let a = IeeeDouble::from_bits(a as u128);
                let b = IeeeDouble::from_bits(b as u128);
                // This is pretty bad but it is the existing behavior.
                Ok(a.partial_cmp(&b).unwrap_or(Ordering::Greater))
            }

            (F32(a), F32(b)) => {
                let a = IeeeSingle::from_bits(a as u128);
                let b = IeeeSingle::from_bits(b as u128);
                Ok(a.partial_cmp(&b).unwrap_or(Ordering::Greater))
            }

            _ => Err(CmpBetweenUnequalTypes),
        }
    }

    pub fn from_i128(input: i128, fty: ast::FloatTy) -> Self {
        match fty {
            ast::FloatTy::F32 => {
                let (r, _) = IeeeSingle::from_i128(input, Round::NearestTiesToEven);
                F32(r.to_bits() as u32)
            }
            ast::FloatTy::F64 => {
                let (r, _) = IeeeDouble::from_i128(input, Round::NearestTiesToEven);
                F64(r.to_bits() as u64)
            }
        }
    }

    pub fn from_u128(input: u128, fty: ast::FloatTy) -> Self {
        match fty {
            ast::FloatTy::F32 => {
                let (r, _) = IeeeSingle::from_u128(input, Round::NearestTiesToEven);
                F32(r.to_bits() as u32)
            }
            ast::FloatTy::F64 => {
                let (r, _) = IeeeDouble::from_u128(input, Round::NearestTiesToEven);
                F64(r.to_bits() as u64)
            }
        }
    }

    pub fn from_str(num: &str, fty: ast::FloatTy) -> Result<Self, ParseFloatError> {
        match fty {
            ast::FloatTy::F32 => {
                let rust_bits = num.parse::<f32>()?.to_bits();
                let apfloat = num.parse::<IeeeSingle>().unwrap_or_else(|e| {
                    panic!("apfloat::IeeeSingle failed to parse `{}`: {:?}", num, e);
                });
                let apfloat_bits = apfloat.to_bits() as u32;
                assert!(rust_bits == apfloat_bits,
                        "apfloat::IeeeSingle gave different result for `{}`: \
                         {}({:#x}) vs Rust's {}({:#x})", num,
                        F32(apfloat_bits), apfloat_bits, F32(rust_bits), rust_bits);
                Ok(F32(apfloat_bits))
            }
            ast::FloatTy::F64 => {
                let rust_bits = num.parse::<f64>()?.to_bits();
                let apfloat = num.parse::<IeeeDouble>().unwrap_or_else(|e| {
                    panic!("apfloat::IeeeDouble failed to parse `{}`: {:?}", num, e);
                });
                let apfloat_bits = apfloat.to_bits() as u64;
                assert!(rust_bits == apfloat_bits,
                        "apfloat::IeeeDouble gave different result for `{}`: \
                         {}({:#x}) vs Rust's {}({:#x})", num,
                        F64(apfloat_bits), apfloat_bits, F64(rust_bits), rust_bits);
                Ok(F64(apfloat_bits))
            }
        }
    }

    pub fn to_i128(self, width: usize) -> Option<i128> {
        assert!(width <= 128);
        let (r, fs) = match self {
            F32(f) => {
                IeeeSingle::from_bits(f as u128).to_i128(width, Round::TowardZero, &mut true)
            }
            F64(f) => {
                IeeeDouble::from_bits(f as u128).to_i128(width, Round::TowardZero, &mut true)
            }
        };
        if fs.intersects(OpStatus::INVALID_OP) {
            None
        } else {
            Some(r)
        }
    }

    pub fn to_u128(self, width: usize) -> Option<u128> {
        assert!(width <= 128);
        let (r, fs) = match self {
            F32(f) => {
                IeeeSingle::from_bits(f as u128).to_u128(width, Round::TowardZero, &mut true)
            }
            F64(f) => {
                IeeeDouble::from_bits(f as u128).to_u128(width, Round::TowardZero, &mut true)
            }
        };
        if fs.intersects(OpStatus::INVALID_OP) {
            None
        } else {
            Some(r)
        }
    }

    pub fn convert(self, fty: ast::FloatTy) -> Self {
        match (self, fty) {
            (F32(f), ast::FloatTy::F32) => F32(f),
            (F64(f), ast::FloatTy::F64) => F64(f),
            (F32(f), ast::FloatTy::F64) => {
                let from = IeeeSingle::from_bits(f as u128);
                let (to, _) = from.convert(Round::NearestTiesToEven, &mut false);
                F64(IeeeDouble::to_bits(to) as u64)
            }
            (F64(f), ast::FloatTy::F32) => {
                let from = IeeeDouble::from_bits(f as u128);
                let (to, _) = from.convert(Round::NearestTiesToEven, &mut false);
                F32(IeeeSingle::to_bits(to) as u32)
            }
        }
    }
}

impl ::std::fmt::Display for ConstFloat {
    fn fmt(&self, fmt: &mut ::std::fmt::Formatter) -> Result<(), ::std::fmt::Error> {
        match *self {
            F32(f) => write!(fmt, "{}f32", IeeeSingle::from_bits(f as u128)),
            F64(f) => write!(fmt, "{}f64", IeeeDouble::from_bits(f as u128)),
        }
    }
}

macro_rules! derive_binop {
    ($op:ident, $func:ident) => {
        impl ::std::ops::$op for ConstFloat {
            type Output = Result<Self, ConstMathErr>;
            fn $func(self, rhs: Self) -> Result<Self, ConstMathErr> {
                match (self, rhs) {
                    (F32(a), F32(b)) =>{
                        let a = IeeeSingle::from_bits(a as u128);
                        let b = IeeeSingle::from_bits(b as u128);
                        Ok(F32(a.$func(b).to_bits() as u32))
                    }
                    (F64(a), F64(b)) => {
                        let a = IeeeDouble::from_bits(a as u128);
                        let b = IeeeDouble::from_bits(b as u128);
                        Ok(F64(a.$func(b).to_bits() as u64))
                    }
                    _ => Err(UnequalTypes(Op::$op)),
                }
            }
        }
    }
}

derive_binop!(Add, add);
derive_binop!(Sub, sub);
derive_binop!(Mul, mul);
derive_binop!(Div, div);
derive_binop!(Rem, rem);

impl ::std::ops::Neg for ConstFloat {
    type Output = Self;
    fn neg(self) -> Self {
        match self {
            F32(f) => F32((-IeeeSingle::from_bits(f as u128)).to_bits() as u32),
            F64(f) => F64((-IeeeDouble::from_bits(f as u128)).to_bits() as u64),
        }
    }
}
