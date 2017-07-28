// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![forbid(unsafe_code)]

use std::cmp::{self, Ordering};
use std::convert::TryFrom;
use std::fmt::{self, Write};
use std::i32;
use std::marker::PhantomData;
use std::mem;
use std::ops::{Neg, Add, Sub, Mul, Div, Rem};
use std::ops::{AddAssign, SubAssign, MulAssign, DivAssign, RemAssign, BitOrAssign};
use std::str::FromStr;

// Translated from LLVM (C++; FIXME LICENSE) by eddyb, exact sources:
// https://github.com/llvm-mirror/llvm/tree/23efab2bbd424ed13495a420ad8641cb2c6c28f9
//   include/llvm/ADT/APFloat.h
//   lib/Support/APFloat.cpp
//   unittests/ADT/APFloatTest.cpp

/// IEEE-754R 4.3: Rounding-direction attributes.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Round {
    NearestTiesToEven,
    TowardPositive,
    TowardNegative,
    TowardZero,
    NearestTiesToAway,
}

impl Neg for Round {
    type Output = Round;
    fn neg(self) -> Round {
        match self {
            Round::TowardPositive => Round::TowardNegative,
            Round::TowardNegative => Round::TowardPositive,
            Round::NearestTiesToEven | Round::TowardZero | Round::NearestTiesToAway => self,
        }
    }
}

/// Category of internally-represented number.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Category {
    Infinity,
    NaN,
    Normal,
    Zero,
}

/// IEEE-754R 7: Default exception handling.
///
/// UNDERFLOW or OVERFLOW are always returned or-ed with INEXACT.
bitflags! {
    #[must_use]
    #[derive(Debug)]
    flags OpStatus: u8 {
        const OK = 0x00,
        const INVALID_OP = 0x01,
        const DIV_BY_ZERO = 0x02,
        const OVERFLOW = 0x04,
        const UNDERFLOW = 0x08,
        const INEXACT = 0x10
    }
}

impl BitOrAssign for OpStatus {
    fn bitor_assign(&mut self, rhs: Self) {
        *self = *self | rhs;
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct ParseError(&'static str);

// \c ilogb error results.
#[allow(unused)]
const IEK_ZERO: i32 = i32::MIN + 1;
#[allow(unused)]
const IEK_NAN: i32 = i32::MIN;
#[allow(unused)]
const IEK_INF: i32 = i32::MAX;

/// A self-contained host- and target-independent arbitrary-precision
/// floating-point software implementation.
///
/// APFloat uses bignum integer arithmetic as provided by static functions in
/// the APInt class. The library will work with bignum integers whose limbs are
/// any unsigned type at least 16 bits wide, but 64 bits is recommended.
///
/// Written for clarity rather than speed, in particular with a view to use in
/// the front-end of a cross compiler so that target arithmetic can be correctly
/// performed on the host. Performance should nonetheless be reasonable,
/// particularly for its intended use. It may be useful as a base
/// implementation for a run-time library during development of a faster
/// target-specific one.
///
/// All 5 rounding modes in the IEEE-754R draft are handled correctly for all
/// implemented operations. Currently implemented operations are add, subtract,
/// multiply, divide, fused-multiply-add, conversion-to-float,
/// conversion-to-integer and conversion-from-integer. New rounding modes
/// (e.g. away from zero) can be added with three or four lines of code.
///
/// Four formats are built-in: IEEE single precision, double precision,
/// quadruple precision, and x87 80-bit extended double (when operating with
/// full extended precision). Adding a new format that obeys IEEE semantics
/// only requires adding two lines of code: a declaration and definition of the
/// format.
///
/// All operations return the status of that operation as an exception bit-mask,
/// so multiple operations can be done consecutively with their results or-ed
/// together. The returned status can be useful for compiler diagnostics; e.g.,
/// inexact, underflow and overflow can be easily diagnosed on constant folding,
/// and compiler optimizers can determine what exceptions would be raised by
/// folding operations and optimize, or perhaps not optimize, accordingly.
///
/// At present, underflow tininess is detected after rounding; it should be
/// straight forward to add support for the before-rounding case too.
///
/// The library reads hexadecimal floating point numbers as per C99, and
/// correctly rounds if necessary according to the specified rounding mode.
/// Syntax is required to have been validated by the caller. It also converts
/// floating point numbers to hexadecimal text as per the C99 %a and %A
/// conversions. The output precision (or alternatively the natural minimal
/// precision) can be specified; if the requested precision is less than the
/// natural precision the output is correctly rounded for the specified rounding
/// mode.
///
/// It also reads decimal floating point numbers and correctly rounds according
/// to the specified rounding mode.
///
/// Conversion to decimal text is not currently implemented.
///
/// Non-zero finite numbers are represented internally as a sign bit, a 16-bit
/// signed exponent, and the significand as an array of integer limbs. After
/// normalization of a number of precision P the exponent is within the range of
/// the format, and if the number is not denormal the P-th bit of the
/// significand is set as an explicit integer bit. For denormals the most
/// significant bit is shifted right so that the exponent is maintained at the
/// format's minimum, so that the smallest denormal has just the least
/// significant bit of the significand set. The sign of zeros and infinities
/// is significant; the exponent and significand of such numbers is not stored,
/// but has a known implicit (deterministic) value: 0 for the significands, 0
/// for zero exponent, all 1 bits for infinity exponent. For NaNs the sign and
/// significand are deterministic, although not really meaningful, and preserved
/// in non-conversion operations. The exponent is implicitly all 1 bits.
///
/// APFloat does not provide any exception handling beyond default exception
/// handling. We represent Signaling NaNs via IEEE-754R 2008 6.2.1 should clause
/// by encoding Signaling NaNs with the first bit of its trailing significand as
/// 0.
///
/// Future work
/// ===========
///
/// Some features that may or may not be worth adding:
///
/// Binary to decimal conversion (hard).
///
/// Optional ability to detect underflow tininess before rounding.
///
/// New formats: x87 in single and double precision mode (IEEE apart from
/// extended exponent range) (hard).
///
/// New operations: sqrt, IEEE remainder, C90 fmod, nexttoward.
///
pub trait Float
    : Copy
    + Default
    + FromStr<Err = ParseError>
    + PartialOrd
    + fmt::Display
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self> {
    /// Factory for Positive Zero.
    fn zero() -> Self;

    /// Factory for Positive Infinity.
    fn inf() -> Self;

    /// Factory for NaN values.
    fn nan() -> Self {
        Self::qnan(None)
    }

    /// Factory for QNaN values.
    fn qnan(payload: Option<u128>) -> Self;

    /// Factory for SNaN values.
    fn snan(payload: Option<u128>) -> Self;

    /// Returns the largest finite number in the given semantics.
    fn largest() -> Self;

    /// Returns the smallest (by magnitude) finite number in the given semantics.
    /// Might be denormalized, which implies a relative loss of precision.
    fn smallest() -> Self;

    /// Returns the smallest (by magnitude) normalized finite number in the given
    /// semantics.
    fn smallest_normalized() -> Self;

    // Arithmetic

    fn add_rounded(&mut self, rhs: Self, round: Round) -> OpStatus;
    fn sub_rounded(&mut self, rhs: Self, round: Round) -> OpStatus {
        self.add_rounded(-rhs, round)
    }
    fn mul_rounded(&mut self, rhs: Self, round: Round) -> OpStatus;
    fn div_rounded(&mut self, rhs: Self, round: Round) -> OpStatus;
    /// IEEE remainder.
    fn remainder(&mut self, rhs: Self) -> OpStatus;
    /// C fmod, or llvm frem.
    fn modulo(&mut self, rhs: Self) -> OpStatus;
    fn fused_mul_add(&mut self, multiplicand: Self, addend: Self, round: Round) -> OpStatus;
    fn round_to_integral(self, round: Round) -> (Self, OpStatus);

    /// IEEE-754R 2008 5.3.1: nextUp.
    fn next_up(&mut self) -> OpStatus;

    /// IEEE-754R 2008 5.3.1: nextDown.
    ///
    /// *NOTE* since nextDown(x) = -nextUp(-x), we only implement nextUp with
    /// appropriate sign switching before/after the computation.
    fn next_down(&mut self) -> OpStatus {
        self.change_sign();
        let result = self.next_up();
        self.change_sign();
        result
    }

    fn change_sign(&mut self);
    fn abs(self) -> Self {
        if self.is_negative() { -self } else { self }
    }
    fn copy_sign(self, rhs: Self) -> Self {
        if self.is_negative() != rhs.is_negative() {
            -self
        } else {
            self
        }
    }

    // Conversions
    fn from_bits(input: u128) -> Self;
    fn from_i128(input: i128, round: Round) -> (Self, OpStatus) {
        if input < 0 {
            let (r, fs) = Self::from_u128(-input as u128, -round);
            (-r, fs)
        } else {
            Self::from_u128(input as u128, round)
        }
    }
    fn from_u128(input: u128, round: Round) -> (Self, OpStatus);
    fn from_str_rounded(s: &str, round: Round) -> Result<(Self, OpStatus), ParseError>;
    fn to_bits(self) -> u128;

    /// Convert a floating point number to an integer according to the
    /// rounding mode. In case of an invalid operation exception,
    /// deterministic values are returned, namely zero for NaNs and the
    /// minimal or maximal value respectively for underflow or overflow.
    /// If the rounded value is in range but the floating point number is
    /// not the exact integer, the C standard doesn't require an inexact
    /// exception to be raised. IEEE-854 does require it so we do that.
    ///
    /// Note that for conversions to integer type the C standard requires
    /// round-to-zero to always be used.
    ///
    /// The *is_exact output tells whether the result is exact, in the sense
    /// that converting it back to the original floating point type produces
    /// the original value. This is almost equivalent to result==OpStatus::OK,
    /// except for negative zeroes.
    fn to_i128(self, width: usize, round: Round, is_exact: &mut bool) -> (i128, OpStatus) {
        if self.is_negative() {
            if self.is_zero() {
                // Negative zero can't be represented as an int.
                *is_exact = false;
            }
            let (r, fs) = (-self).to_u128(width, -round, is_exact);

            // Check for values that don't fit in the signed integer.
            if r > (1 << (width - 1)) {
                // Return the most negative integer for the given width.
                *is_exact = false;
                (-1 << (width - 1), OpStatus::INVALID_OP)
            } else {
                (r.wrapping_neg() as i128, fs)
            }
        } else {
            // Positive case is simpler, can pretend it's a smaller unsigned
            // integer, and `to_u128` will take care of all the edge cases.
            let (r, fs) = self.to_u128(width - 1, round, is_exact);
            (r as i128, fs)
        }
    }
    fn to_u128(self, width: usize, round: Round, is_exact: &mut bool) -> (u128, OpStatus);

    fn cmp_abs_normal(self, rhs: Self) -> Ordering;

    /// Bitwise comparison for equality (QNaNs compare equal, 0!=-0).
    fn bitwise_eq(self, rhs: Self) -> bool;

    // IEEE-754R 5.7.2 General operations.

    /// Implements IEEE minNum semantics. Returns the smaller of the 2 arguments if
    /// both are not NaN. If either argument is a NaN, returns the other argument.
    fn min(self, other: Self) -> Self {
        if self.is_nan() {
            other
        } else if other.is_nan() {
            self
        } else if other.partial_cmp(&self) == Some(Ordering::Less) {
            other
        } else {
            self
        }
    }

    /// Implements IEEE maxNum semantics. Returns the larger of the 2 arguments if
    /// both are not NaN. If either argument is a NaN, returns the other argument.
    fn max(self, other: Self) -> Self {
        if self.is_nan() {
            other
        } else if other.is_nan() {
            self
        } else if self.partial_cmp(&other) == Some(Ordering::Less) {
            other
        } else {
            self
        }
    }

    /// IEEE-754R isSignMinus: Returns true if and only if the current value is
    /// negative.
    ///
    /// This applies to zeros and NaNs as well.
    fn is_negative(self) -> bool;

    /// IEEE-754R isNormal: Returns true if and only if the current value is normal.
    ///
    /// This implies that the current value of the float is not zero, subnormal,
    /// infinite, or NaN following the definition of normality from IEEE-754R.
    fn is_normal(self) -> bool {
        !self.is_denormal() && self.is_finite_non_zero()
    }

    /// Returns true if and only if the current value is zero, subnormal, or
    /// normal.
    ///
    /// This means that the value is not infinite or NaN.
    fn is_finite(self) -> bool {
        !self.is_nan() && !self.is_infinite()
    }

    /// Returns true if and only if the float is plus or minus zero.
    fn is_zero(self) -> bool {
        self.category() == Category::Zero
    }

    /// IEEE-754R isSubnormal(): Returns true if and only if the float is a
    /// denormal.
    fn is_denormal(self) -> bool;

    /// IEEE-754R isInfinite(): Returns true if and only if the float is infinity.
    fn is_infinite(self) -> bool {
        self.category() == Category::Infinity
    }

    /// Returns true if and only if the float is a quiet or signaling NaN.
    fn is_nan(self) -> bool {
        self.category() == Category::NaN
    }

    /// Returns true if and only if the float is a signaling NaN.
    fn is_signaling(self) -> bool;

    // Simple Queries

    fn category(self) -> Category;
    fn is_non_zero(self) -> bool {
        !self.is_zero()
    }
    fn is_finite_non_zero(self) -> bool {
        self.is_finite() && !self.is_zero()
    }
    fn is_pos_zero(self) -> bool {
        self.is_zero() && !self.is_negative()
    }
    fn is_neg_zero(self) -> bool {
        self.is_zero() && self.is_negative()
    }

    /// Returns true if and only if the number has the smallest possible non-zero
    /// magnitude in the current semantics.
    fn is_smallest(self) -> bool;

    /// Returns true if and only if the number has the largest possible finite
    /// magnitude in the current semantics.
    fn is_largest(self) -> bool;

    /// Returns true if and only if the number is an exact integer.
    fn is_integer(self) -> bool;

    /// If this value has an exact multiplicative inverse, return it.
    fn get_exact_inverse(self) -> Option<Self>;

    /// Returns the exponent of the internal representation of the Float.
    ///
    /// Because the radix of Float is 2, this is equivalent to floor(log2(x)).
    /// For special Float values, this returns special error codes:
    ///
    ///   NaN -> \c IEK_NAN
    ///   0   -> \c IEK_ZERO
    ///   Inf -> \c IEK_INF
    ///
    fn ilogb(self) -> i32;

    /// Returns: self * 2^exp for integral exponents.
    fn scalbn(self, exp: i32, round: Round) -> Self;

    /// Equivalent of C standard library function.
    ///
    /// While the C standard says exp is an unspecified value for infinity and nan,
    /// this returns INT_MAX for infinities, and INT_MIN for NaNs.
    fn frexp(self, exp: &mut i32, round: Round) -> Self;
}

macro_rules! proxy_impls {
    ([$($g:tt)*] $ty:ty) => {
        impl<$($g)*> Default for $ty {
            fn default() -> Self {
                Self::zero()
            }
        }

        impl<$($g)*> PartialEq for $ty {
            fn eq(&self, _: &Self) -> bool {
                panic!(concat!(stringify!($ty),
                    "::eq is disallowed, use \
                     a.partial_cmp(&b) == Some(Ordering::Equal) or \
                     a.bitwise_eq(b)"))
            }

            fn ne(&self, _: &Self) -> bool {
                panic!(concat!(stringify!($ty),
                    "::ne is disallowed, use \
                     a.partial_cmp(&b) != Some(Ordering::Equal) or \
                     !a.bitwise_eq(b)"))
            }
        }

        impl<$($g)*> FromStr for $ty {
            type Err = ParseError;
            fn from_str(s: &str) -> Result<Self, ParseError> {
                Self::from_str_rounded(s, Round::NearestTiesToEven)
                    .map(|(x, _)| x)
            }
        }

        impl<$($g)*> Neg for $ty {
            type Output = Self;
            fn neg(mut self) -> Self {
                self.change_sign();
                self
            }
        }

        // Rounding ties to the nearest even, by default.

        impl<$($g)*> AddAssign for $ty {
            fn add_assign(&mut self, rhs: Self) {
                let _: OpStatus = self.add_rounded(rhs, Round::NearestTiesToEven);
            }
        }

        impl<$($g)*> SubAssign for $ty {
            fn sub_assign(&mut self, rhs: Self) {
                let _: OpStatus = self.sub_rounded(rhs, Round::NearestTiesToEven);
            }
        }

        impl<$($g)*> MulAssign for $ty {
            fn mul_assign(&mut self, rhs: Self) {
                let _: OpStatus = self.mul_rounded(rhs, Round::NearestTiesToEven);
            }
        }

        impl<$($g)*> DivAssign for $ty {
            fn div_assign(&mut self, rhs: Self) {
                let _: OpStatus = self.div_rounded(rhs, Round::NearestTiesToEven);
            }
        }

        impl<$($g)*> RemAssign for $ty {
            fn rem_assign(&mut self, rhs: Self) {
                let _: OpStatus = self.modulo(rhs);
            }
        }

        impl<$($g)*> Add for $ty {
            type Output = Self;
            fn add(mut self, rhs: Self) -> Self {
                self += rhs;
                self
            }
        }

        impl<$($g)*> Sub for $ty {
            type Output = Self;
            fn sub(mut self, rhs: Self) -> Self {
                self -= rhs;
                self
            }
        }

        impl<$($g)*> Mul for $ty {
            type Output = Self;
            fn mul(mut self, rhs: Self) -> Self {
                self *= rhs;
                self
            }
        }

        impl<$($g)*> Div for $ty {
            type Output = Self;
            fn div(mut self, rhs: Self) -> Self {
                self /= rhs;
                self
            }
        }

        impl<$($g)*> Rem for $ty {
            type Output = Self;
            fn rem(mut self, rhs: Self) -> Self {
                self %= rhs;
                self
            }
        }
    }
}

/// Fundamental unit of big integer arithmetic, but also
/// large to store the largest significands by itself.
type Limb = u128;
const LIMB_BITS: usize = 128;
fn limbs_for_bits(bits: usize) -> usize {
    (bits + LIMB_BITS - 1) / LIMB_BITS
}

/// A signed type to represent a floating point number's unbiased exponent.
type ExpInt = i16;

/// Enum that represents what fraction of the LSB truncated bits of an fp number
/// represent.
///
/// This essentially combines the roles of guard and sticky bits.
#[must_use]
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum Loss {
    // Example of truncated bits:
    ExactlyZero, // 000000
    LessThanHalf, // 0xxxxx  x's not all zero
    ExactlyHalf, // 100000
    MoreThanHalf, // 1xxxxx  x's not all zero
}

/// Represents floating point arithmetic semantics.
pub trait IeeeSemantics: Sized + fmt::Debug {
    /// Number of bits in the exponent.
    const EXPONENT_BITS: usize;

    /// The largest E such that 2^E is representable; this matches the
    /// definition of IEEE 754.
    const MAX_EXPONENT: ExpInt = (1 << (Self::EXPONENT_BITS - 1)) - 1;

    /// The smallest E such that 2^E is a normalized number; this
    /// matches the definition of IEEE 754.
    const MIN_EXPONENT: ExpInt = -(1 << (Self::EXPONENT_BITS - 1)) + 2;

    /// Number of bits in the significand. This includes the integer bit.
    const PRECISION: usize;

    /// Total number of bits in the in-memory format.
    const BITS: usize = Self::EXPONENT_BITS + (Self::PRECISION - 1) + 1;

    /// The significand bit that marks NaN as quiet.
    const QNAN_BIT: usize = Self::PRECISION - 2;

    /// The significand bitpattern to mark a NaN as quiet.
    /// NOTE: for X87DoubleExtended we need to set two bits instead of 2.
    const QNAN_SIGNIFICAND: Limb = 1 << Self::QNAN_BIT;

    fn from_bits(bits: u128) -> Ieee<Self> {
        let exponent = (bits >> (Self::PRECISION - 1)) & ((1 << Self::EXPONENT_BITS) - 1);
        let mut r = Ieee {
            sig: [bits & ((1 << (Self::PRECISION - 1)) - 1)],
            // Convert the exponent from its bias representation to a signed integer.
            exp: (exponent as ExpInt) - Self::MAX_EXPONENT,
            category: Category::Zero,
            sign: (bits & (1 << (Self::BITS - 1))) != 0,
            marker: PhantomData,
        };

        if r.exp == Self::MIN_EXPONENT - 1 && r.sig == [0] {
            // Exponent, significand meaningless.
            r.category = Category::Zero;
        } else if r.exp == Self::MAX_EXPONENT + 1 && r.sig == [0] {
            // Exponent, significand meaningless.
            r.category = Category::Infinity;
        } else if r.exp == Self::MAX_EXPONENT + 1 && r.sig != [0] {
            // Sign, exponent, significand meaningless.
            r.category = Category::NaN;
        } else {
            r.category = Category::Normal;
            if r.exp == Self::MIN_EXPONENT - 1 {
                // Denormal.
                r.exp = Self::MIN_EXPONENT;
            } else {
                // Set integer bit.
                sig::set_bit(&mut r.sig, Self::PRECISION - 1);
            }
        }

        r
    }

    fn to_bits(x: Ieee<Self>) -> u128 {
        // Split integer bit from significand.
        let integer_bit = sig::get_bit(&x.sig, Self::PRECISION - 1);
        let mut significand = x.sig[0] & ((1 << (Self::PRECISION - 1)) - 1);
        let exponent = match x.category {
            Category::Normal => {
                if x.exp == Self::MIN_EXPONENT && !integer_bit {
                    // Denormal.
                    Self::MIN_EXPONENT - 1
                } else {
                    x.exp
                }
            }
            Category::Zero => {
                // FIXME(eddyb) Maybe we should guarantee an invariant instead?
                significand = 0;
                Self::MIN_EXPONENT - 1
            }
            Category::Infinity => {
                // FIXME(eddyb) Maybe we should guarantee an invariant instead?
                significand = 0;
                Self::MAX_EXPONENT + 1
            }
            Category::NaN => Self::MAX_EXPONENT + 1,
        };

        // Convert the exponent from a signed integer to its bias representation.
        let exponent = (exponent + Self::MAX_EXPONENT) as u128;

        ((x.sign as u128) << (Self::BITS - 1)) | (exponent << (Self::PRECISION - 1)) | significand
    }
}

#[derive(Debug)]
pub struct Ieee<S: IeeeSemantics> {
    /// Absolute significand value (including the integer bit).
    sig: [Limb; 1],

    /// The signed unbiased exponent of the value.
    exp: ExpInt,

    /// What kind of floating point number this is.
    category: Category,

    /// Sign bit of the number.
    sign: bool,

    marker: PhantomData<S>,
}

impl<S: IeeeSemantics> Copy for Ieee<S> {}
impl<S: IeeeSemantics> Clone for Ieee<S> {
    fn clone(&self) -> Self {
        *self
    }
}

macro_rules! ieee_semantics {
    ($($name:ident { $($items:tt)* })*) => {
        mod ieee_semantics { $(#[derive(Debug)] pub struct $name;)* }
        $(impl IeeeSemantics for ieee_semantics::$name { $($items)* })*
        $(pub type $name = Ieee<ieee_semantics::$name>;)*
    }
}

ieee_semantics! {
    IeeeHalf { const EXPONENT_BITS: usize = 5; const PRECISION: usize = 11; }
    IeeeSingle { const EXPONENT_BITS: usize = 8; const PRECISION: usize = 24; }
    IeeeDouble { const EXPONENT_BITS: usize = 11; const PRECISION: usize = 53; }
    IeeeQuad { const EXPONENT_BITS: usize = 15; const PRECISION: usize = 113; }
    X87DoubleExtended {
        const EXPONENT_BITS: usize = 15;
        const PRECISION: usize = 64;
        const BITS: usize = 80;

        /// For x87 extended precision, we want to make a NaN, not a
        /// pseudo-NaN. Maybe we should expose the ability to make
        /// pseudo-NaNs?
        const QNAN_SIGNIFICAND: Limb = 0b11 << Self::QNAN_BIT;

        /// Integer bit is explicit in this format. Intel hardware (387 and later)
        /// does not support these bit patterns:
        ///  exponent = all 1's, integer bit 0, significand 0 ("pseudoinfinity")
        ///  exponent = all 1's, integer bit 0, significand nonzero ("pseudoNaN")
        ///  exponent = 0, integer bit 1 ("pseudodenormal")
        ///  exponent!=0 nor all 1's, integer bit 0 ("unnormal")
        /// At the moment, the first two are treated as NaNs, the second two as Normal.
        fn from_bits(bits: u128) -> Ieee<Self> {
            let exponent = (bits >> Self::PRECISION) & ((1 << Self::EXPONENT_BITS) - 1);
            let mut r = Ieee {
                sig: [bits & ((1 << (Self::PRECISION - 1)) - 1)],
                // Convert the exponent from its bias representation to a signed integer.
                exp: (exponent as ExpInt) - Self::MAX_EXPONENT,
                category: Category::Zero,
                sign: (bits & (1 << (Self::BITS - 1))) != 0,
                marker: PhantomData
            };

            if r.exp == Self::MIN_EXPONENT - 1 && r.sig == [0] {
                // Exponent, significand meaningless.
                r.category = Category::Zero;
            } else if r.exp == Self::MAX_EXPONENT + 1
                && r.sig == [1 << (Self::PRECISION - 1)] {
                // Exponent, significand meaningless.
                r.category = Category::Infinity;
            } else if r.exp == Self::MAX_EXPONENT + 1
                && r.sig != [1 << (Self::PRECISION - 1)] {
                // Sign, exponent, significand meaningless.
                r.category = Category::NaN;
            } else {
                r.category = Category::Normal;
                if r.exp == Self::MIN_EXPONENT - 1 {
                    // Denormal.
                    r.exp = Self::MIN_EXPONENT;
                }
            }

            r
        }

        fn to_bits(x: Ieee<Self>) -> u128 {
            // Get integer bit from significand.
            let integer_bit = sig::get_bit(&x.sig, Self::PRECISION - 1);
            let mut significand = x.sig[0] & ((1 << Self::PRECISION) - 1);
            let exponent = match x.category {
                Category::Normal => {
                    if x.exp == Self::MIN_EXPONENT && !integer_bit {
                        // Denormal.
                        Self::MIN_EXPONENT - 1
                    } else {
                        x.exp
                    }
                }
                Category::Zero => {
                    // FIXME(eddyb) Maybe we should guarantee an invariant instead?
                    significand = 0;
                    Self::MIN_EXPONENT - 1
                }
                Category::Infinity => {
                    // FIXME(eddyb) Maybe we should guarantee an invariant instead?
                    significand = 1 << (Self::PRECISION - 1);
                    Self::MAX_EXPONENT + 1
                }
                Category::NaN => Self::MAX_EXPONENT + 1
            };

            // Convert the exponent from a signed integer to its bias representation.
            let exponent = (exponent + Self::MAX_EXPONENT) as u128;

            ((x.sign as u128) << (Self::BITS - 1)) | (exponent << Self::PRECISION) | significand
        }
    }
}

proxy_impls!([S: IeeeSemantics] Ieee<S>);

impl<S: IeeeSemantics> PartialOrd for Ieee<S> {
    fn partial_cmp(&self, rhs: &Self) -> Option<Ordering> {
        match (self.category, rhs.category) {
            (Category::NaN, _) |
            (_, Category::NaN) => None,

            (Category::Infinity, Category::Infinity) => Some((!self.sign).cmp(&(!rhs.sign))),

            (Category::Zero, Category::Zero) => Some(Ordering::Equal),

            (Category::Infinity, _) |
            (Category::Normal, Category::Zero) => Some((!self.sign).cmp(&self.sign)),

            (_, Category::Infinity) |
            (Category::Zero, Category::Normal) => Some(rhs.sign.cmp(&(!rhs.sign))),

            (Category::Normal, Category::Normal) => {
                // Two normal numbers. Do they have the same sign?
                Some((!self.sign).cmp(&(!rhs.sign)).then_with(|| {
                    // Compare absolute values; invert result if negative.
                    let result = self.cmp_abs_normal(*rhs);

                    if self.sign { result.reverse() } else { result }
                }))
            }
        }
    }
}

/// Prints this value as a decimal string.
///
/// \param precision The maximum number of digits of
///   precision to output. If there are fewer digits available,
///   zero padding will not be used unless the value is
///   integral and small enough to be expressed in
///   precision digits. 0 means to use the natural
///   precision of the number.
/// \param width The maximum number of zeros to
///   consider inserting before falling back to scientific
///   notation. 0 means to always use scientific notation.
///
/// \param alternate Indicate whether to remove the trailing zero in
///   fraction part or not. Also setting this parameter to true forces
///   producing of output more similar to default printf behavior.
///   Specifically the lower e is used as exponent delimiter and exponent
///   always contains no less than two digits.
///
/// Number       precision    width      Result
/// ------       ---------    -----      ------
/// 1.01E+4              5        2       10100
/// 1.01E+4              4        2       1.01E+4
/// 1.01E+4              5        1       1.01E+4
/// 1.01E-2              5        2       0.0101
/// 1.01E-2              4        2       0.0101
/// 1.01E-2              4        1       1.01E-2
impl<S: IeeeSemantics> fmt::Display for Ieee<S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let width = f.width().unwrap_or(3);
        let alternate = f.alternate();

        match self.category {
            Category::Infinity => {
                if self.sign {
                    return f.write_str("-Inf");
                } else {
                    return f.write_str("+Inf");
                }
            }

            Category::NaN => return f.write_str("NaN"),

            Category::Zero => {
                if self.sign {
                    f.write_char('-')?;
                }

                if width == 0 {
                    if alternate {
                        f.write_str("0.0")?;
                        if let Some(n) = f.precision() {
                            for _ in 1..n {
                                f.write_char('0')?;
                            }
                        }
                        f.write_str("e+00")?;
                    } else {
                        f.write_str("0.0E+0")?;
                    }
                } else {
                    f.write_char('0')?;
                }
                return Ok(());
            }

            Category::Normal => {}
        }

        if self.sign {
            f.write_char('-')?;
        }

        // We use enough digits so the number can be round-tripped back to an
        // APFloat. The formula comes from "How to Print Floating-Point Numbers
        // Accurately" by Steele and White.
        // FIXME: Using a formula based purely on the precision is conservative;
        // we can print fewer digits depending on the actual value being printed.

        // precision = 2 + floor(S::PRECISION / lg_2(10))
        let precision = f.precision().unwrap_or(2 + S::PRECISION * 59 / 196);

        // Decompose the number into an APInt and an exponent.
        let mut exp = self.exp - (S::PRECISION as ExpInt - 1);
        let mut sig = vec![self.sig[0]];

        // Ignore trailing binary zeros.
        let trailing_zeros = sig[0].trailing_zeros();
        let _: Loss = sig::shift_right(&mut sig, &mut exp, trailing_zeros as usize);

        // Change the exponent from 2^e to 10^e.
        if exp == 0 {
            // Nothing to do.
        } else if exp > 0 {
            // Just shift left.
            let shift = exp as usize;
            sig.resize(limbs_for_bits(S::PRECISION + shift), 0);
            sig::shift_left(&mut sig, &mut exp, shift);
        } else {
            // exp < 0
            let mut texp = -exp as usize;

            // We transform this using the identity:
            //   (N)(2^-e) == (N)(5^e)(10^-e)

            // Multiply significand by 5^e.
            //   N * 5^0101 == N * 5^(1*1) * 5^(0*2) * 5^(1*4) * 5^(0*8)
            let mut sig_scratch = vec![];
            let mut p5 = vec![];
            let mut p5_scratch = vec![];
            while texp != 0 {
                if p5.is_empty() {
                    p5.push(5);
                } else {
                    p5_scratch.resize(p5.len() * 2, 0);
                    let _: Loss =
                        sig::mul(&mut p5_scratch, &mut 0, &p5, &p5, p5.len() * 2 * LIMB_BITS);
                    while p5_scratch.last() == Some(&0) {
                        p5_scratch.pop();
                    }
                    mem::swap(&mut p5, &mut p5_scratch);
                }
                if texp & 1 != 0 {
                    sig_scratch.resize(sig.len() + p5.len(), 0);
                    let _: Loss = sig::mul(
                        &mut sig_scratch,
                        &mut 0,
                        &sig,
                        &p5,
                        (sig.len() + p5.len()) * LIMB_BITS,
                    );
                    while sig_scratch.last() == Some(&0) {
                        sig_scratch.pop();
                    }
                    mem::swap(&mut sig, &mut sig_scratch);
                }
                texp >>= 1;
            }
        }

        // Fill the buffer.
        let mut buffer = vec![];

        // Ignore digits from the significand until it is no more
        // precise than is required for the desired precision.
        // 196/59 is a very slight overestimate of lg_2(10).
        let required = (precision * 196 + 58) / 59;
        let mut discard_digits = sig::omsb(&sig).saturating_sub(required) * 59 / 196;
        let mut in_trail = true;
        while !sig.is_empty() {
            // Perform short division by 10 to extract the rightmost digit.
            // rem <- sig % 10
            // sig <- sig / 10
            let mut rem = 0;
            for limb in sig.iter_mut().rev() {
                // We don't have an integer doubly wide than Limb,
                // so we have to split the divrem on two halves.
                const HALF_BITS: usize = LIMB_BITS / 2;
                let mut halves = [*limb & ((1 << HALF_BITS) - 1), *limb >> HALF_BITS];
                for half in halves.iter_mut().rev() {
                    *half |= rem << HALF_BITS;
                    rem = *half % 10;
                    *half /= 10;
                }
                *limb = halves[0] | (halves[1] << HALF_BITS);
            }
            // Reduce the sigificand to avoid wasting time dividing 0's.
            while sig.last() == Some(&0) {
                sig.pop();
            }

            let digit = rem;

            // Ignore digits we don't need.
            if discard_digits > 0 {
                discard_digits -= 1;
                exp += 1;
                continue;
            }

            // Drop trailing zeros.
            if in_trail && digit == 0 {
                exp += 1;
            } else {
                in_trail = false;
                buffer.push(b'0' + digit as u8);
            }
        }

        assert!(!buffer.is_empty(), "no characters in buffer!");

        // Drop down to precision.
        // FIXME: don't do more precise calculations above than are required.
        if buffer.len() > precision {
            // The most significant figures are the last ones in the buffer.
            let mut first_sig = buffer.len() - precision;

            // Round.
            // FIXME: this probably shouldn't use 'round half up'.

            // Rounding down is just a truncation, except we also want to drop
            // trailing zeros from the new result.
            if buffer[first_sig - 1] < b'5' {
                while first_sig < buffer.len() && buffer[first_sig] == b'0' {
                    first_sig += 1;
                }
            } else {
                // Rounding up requires a decimal add-with-carry. If we continue
                // the carry, the newly-introduced zeros will just be truncated.
                for x in &mut buffer[first_sig..] {
                    if *x == b'9' {
                        first_sig += 1;
                    } else {
                        *x += 1;
                        break;
                    }
                }
            }

            exp += first_sig as ExpInt;
            buffer.drain(..first_sig);

            // If we carried through, we have exactly one digit of precision.
            if buffer.is_empty() {
                buffer.push(b'1');
            }
        }

        let digits = buffer.len();

        // Check whether we should use scientific notation.
        let scientific = if width == 0 {
            true
        } else {
            if exp >= 0 {
                // 765e3 --> 765000
                //              ^^^
                // But we shouldn't make the number look more precise than it is.
                exp as usize > width || digits + exp as usize > precision
            } else {
                // Power of the most significant digit.
                let msd = exp + (digits - 1) as ExpInt;
                if msd >= 0 {
                    // 765e-2 == 7.65
                    false
                } else {
                    // 765e-5 == 0.00765
                    //           ^ ^^
                    -msd as usize > width
                }
            }
        };

        // Scientific formatting is pretty straightforward.
        if scientific {
            exp += digits as ExpInt - 1;

            f.write_char(buffer[digits - 1] as char)?;
            f.write_char('.')?;
            let truncate_zero = !alternate;
            if digits == 1 && truncate_zero {
                f.write_char('0')?;
            } else {
                for &d in buffer[..digits - 1].iter().rev() {
                    f.write_char(d as char)?;
                }
            }
            // Fill with zeros up to precision.
            if !truncate_zero && precision > digits - 1 {
                for _ in 0..precision - digits + 1 {
                    f.write_char('0')?;
                }
            }
            // For alternate we use lower 'e'.
            f.write_char(if alternate { 'e' } else { 'E' })?;

            // Exponent always at least two digits if we do not truncate zeros.
            if truncate_zero {
                write!(f, "{:+}", exp)?;
            } else {
                write!(f, "{:+03}", exp)?;
            }

            return Ok(());
        }

        // Non-scientific, positive exponents.
        if exp >= 0 {
            for &d in buffer.iter().rev() {
                f.write_char(d as char)?;
            }
            for _ in 0..exp {
                f.write_char('0')?;
            }
            return Ok(());
        }

        // Non-scientific, negative exponents.
        let unit_place = -exp as usize;
        if unit_place < digits {
            for &d in buffer[unit_place..].iter().rev() {
                f.write_char(d as char)?;
            }
            f.write_char('.')?;
            for &d in buffer[..unit_place].iter().rev() {
                f.write_char(d as char)?;
            }
        } else {
            f.write_str("0.")?;
            for _ in digits..unit_place {
                f.write_char('0')?;
            }
            for &d in buffer.iter().rev() {
                f.write_char(d as char)?;
            }
        }

        Ok(())
    }
}

impl<S: IeeeSemantics> Float for Ieee<S> {
    fn zero() -> Self {
        Ieee {
            sig: [0],
            exp: S::MIN_EXPONENT - 1,
            category: Category::Zero,
            sign: false,
            marker: PhantomData,
        }
    }

    fn inf() -> Self {
        Ieee {
            sig: [0],
            exp: S::MAX_EXPONENT + 1,
            category: Category::Infinity,
            sign: false,
            marker: PhantomData,
        }
    }

    fn qnan(payload: Option<u128>) -> Self {
        Ieee {
            sig: [
                S::QNAN_SIGNIFICAND |
                    payload.map_or(0, |payload| {
                        // Zero out the excess bits of the significand.
                        payload & ((1 << S::QNAN_BIT) - 1)
                    }),
            ],
            exp: S::MAX_EXPONENT + 1,
            category: Category::NaN,
            sign: false,
            marker: PhantomData,
        }
    }

    fn snan(payload: Option<u128>) -> Self {
        let mut snan = Self::qnan(payload);

        // We always have to clear the QNaN bit to make it an SNaN.
        sig::clear_bit(&mut snan.sig, S::QNAN_BIT);

        // If there are no bits set in the payload, we have to set
        // *something* to make it a NaN instead of an infinity;
        // conventionally, this is the next bit down from the QNaN bit.
        if snan.sig[0] & !S::QNAN_SIGNIFICAND == 0 {
            sig::set_bit(&mut snan.sig, S::QNAN_BIT - 1);
        }

        snan
    }

    fn largest() -> Self {
        // We want (in interchange format):
        //   exponent = 1..10
        //   significand = 1..1
        Ieee {
            sig: [!0 & ((1 << S::PRECISION) - 1)],
            exp: S::MAX_EXPONENT,
            category: Category::Normal,
            sign: false,
            marker: PhantomData,
        }
    }

    fn smallest() -> Self {
        // We want (in interchange format):
        //   exponent = 0..0
        //   significand = 0..01
        Ieee {
            sig: [1],
            exp: S::MIN_EXPONENT,
            category: Category::Normal,
            sign: false,
            marker: PhantomData,
        }
    }

    fn smallest_normalized() -> Self {
        // We want (in interchange format):
        //   exponent = 0..0
        //   significand = 10..0
        Ieee {
            sig: [1 << (S::PRECISION - 1)],
            exp: S::MIN_EXPONENT,
            category: Category::Normal,
            sign: false,
            marker: PhantomData,
        }
    }

    fn add_rounded(&mut self, rhs: Self, round: Round) -> OpStatus {
        let fs = match (self.category, rhs.category) {
            (Category::Infinity, Category::Infinity) => {
                // Differently signed infinities can only be validly
                // subtracted.
                if self.sign != rhs.sign {
                    *self = Self::nan();
                    OpStatus::INVALID_OP
                } else {
                    OpStatus::OK
                }
            }

            // Sign may depend on rounding mode; handled below.
            (_, Category::Zero) |
            (Category::NaN, _) |
            (Category::Infinity, Category::Normal) => OpStatus::OK,

            (Category::Zero, _) |
            (_, Category::NaN) |
            (_, Category::Infinity) => {
                *self = rhs;
                OpStatus::OK
            }

            // This return code means it was not a simple case.
            (Category::Normal, Category::Normal) => {
                let loss = sig::add_or_sub(
                    &mut self.sig,
                    &mut self.exp,
                    &mut self.sign,
                    &mut [rhs.sig[0]],
                    rhs.exp,
                    rhs.sign,
                );
                let fs = self.normalize(round, loss);

                // Can only be zero if we lost no fraction.
                assert!(self.category != Category::Zero || loss == Loss::ExactlyZero);

                fs
            }
        };

        // If two numbers add (exactly) to zero, IEEE 754 decrees it is a
        // positive zero unless rounding to minus infinity, except that
        // adding two like-signed zeroes gives that zero.
        if self.category == Category::Zero &&
            (rhs.category != Category::Zero || self.sign != rhs.sign)
        {
            self.sign = round == Round::TowardNegative;
        }

        fs
    }

    fn mul_rounded(&mut self, rhs: Self, round: Round) -> OpStatus {
        self.sign ^= rhs.sign;

        match (self.category, rhs.category) {
            (Category::NaN, _) => {
                self.sign = false;
                OpStatus::OK
            }

            (_, Category::NaN) => {
                self.sign = false;
                self.category = Category::NaN;
                self.sig = rhs.sig;
                OpStatus::OK
            }

            (Category::Zero, Category::Infinity) |
            (Category::Infinity, Category::Zero) => {
                *self = Self::nan();
                OpStatus::INVALID_OP
            }

            (_, Category::Infinity) |
            (Category::Infinity, _) => {
                self.category = Category::Infinity;
                OpStatus::OK
            }

            (Category::Zero, _) |
            (_, Category::Zero) => {
                self.category = Category::Zero;
                OpStatus::OK
            }

            (Category::Normal, Category::Normal) => {
                self.exp += rhs.exp;
                let mut wide_sig = [0; 2];
                let loss = sig::mul(
                    &mut wide_sig,
                    &mut self.exp,
                    &self.sig,
                    &rhs.sig,
                    S::PRECISION,
                );
                self.sig = [wide_sig[0]];
                let mut fs = self.normalize(round, loss);
                if loss != Loss::ExactlyZero {
                    fs |= OpStatus::INEXACT;
                }
                fs
            }
        }
    }

    fn div_rounded(&mut self, rhs: Self, round: Round) -> OpStatus {
        self.sign ^= rhs.sign;

        match (self.category, rhs.category) {
            (Category::NaN, _) => {
                self.sign = false;
                OpStatus::OK
            }

            (_, Category::NaN) => {
                self.category = Category::NaN;
                self.sig = rhs.sig;
                self.sign = false;
                OpStatus::OK
            }

            (Category::Infinity, Category::Infinity) |
            (Category::Zero, Category::Zero) => {
                *self = Self::nan();
                OpStatus::INVALID_OP
            }

            (Category::Infinity, _) |
            (Category::Zero, _) => OpStatus::OK,

            (Category::Normal, Category::Infinity) => {
                self.category = Category::Zero;
                OpStatus::OK
            }

            (Category::Normal, Category::Zero) => {
                self.category = Category::Infinity;
                OpStatus::DIV_BY_ZERO
            }

            (Category::Normal, Category::Normal) => {
                self.exp -= rhs.exp;
                let dividend = self.sig[0];
                let loss = sig::div(
                    &mut self.sig,
                    &mut self.exp,
                    &mut [dividend],
                    &mut [rhs.sig[0]],
                    S::PRECISION,
                );
                let mut fs = self.normalize(round, loss);
                if loss != Loss::ExactlyZero {
                    fs |= OpStatus::INEXACT;
                }
                fs
            }
        }
    }

    // This is not currently correct in all cases.
    fn remainder(&mut self, rhs: Self) -> OpStatus {
        let mut v = *self;
        let orig_sign = self.sign;

        let fs = v.div_rounded(rhs, Round::NearestTiesToEven);
        if fs == OpStatus::DIV_BY_ZERO {
            return fs;
        }

        assert!(S::PRECISION < LIMB_BITS);
        assert!(LIMB_BITS <= 128);

        let (x, fs) = v.to_i128(LIMB_BITS, Round::NearestTiesToEven, &mut false);
        if fs == OpStatus::INVALID_OP {
            return fs;
        }

        let (mut v, fs) = Self::from_i128(x, Round::NearestTiesToEven);
        assert_eq!(fs, OpStatus::OK); // should always work

        let fs = v.mul_rounded(rhs, Round::NearestTiesToEven);
        assert!(fs == OpStatus::OK || fs == OpStatus::INEXACT); // should not overflow or underflow

        let fs = self.sub_rounded(v, Round::NearestTiesToEven);
        assert!(fs == OpStatus::OK || fs == OpStatus::INEXACT); // likewise

        if self.is_zero() {
            self.sign = orig_sign; // IEEE754 requires this
        }
        fs
    }

    fn modulo(&mut self, rhs: Self) -> OpStatus {
        match (self.category, rhs.category) {
            (Category::NaN, _) |
            (Category::Zero, Category::Infinity) |
            (Category::Zero, Category::Normal) |
            (Category::Normal, Category::Infinity) => OpStatus::OK,

            (_, Category::NaN) => {
                self.sign = false;
                self.category = Category::NaN;
                self.sig = rhs.sig;
                OpStatus::OK
            }

            (Category::Infinity, _) |
            (_, Category::Zero) => {
                *self = Self::nan();
                OpStatus::INVALID_OP
            }

            (Category::Normal, Category::Normal) => {
                while self.is_finite_non_zero() && rhs.is_finite_non_zero() &&
                    self.cmp_abs_normal(rhs) != Ordering::Less
                {
                    let mut v = rhs.scalbn(self.ilogb() - rhs.ilogb(), Round::NearestTiesToEven);
                    if self.cmp_abs_normal(v) == Ordering::Less {
                        v = v.scalbn(-1, Round::NearestTiesToEven);
                    }
                    v.sign = self.sign;

                    assert_eq!(self.sub_rounded(v, Round::NearestTiesToEven), OpStatus::OK);
                }
                OpStatus::OK
            }
        }
    }

    fn fused_mul_add(&mut self, multiplicand: Self, addend: Self, round: Round) -> OpStatus {
        // If and only if all arguments are normal do we need to do an
        // extended-precision calculation.
        if !self.is_finite_non_zero() || !multiplicand.is_finite_non_zero() || !addend.is_finite() {
            let mut fs = self.mul_rounded(multiplicand, round);

            // FS can only be OpStatus::OK or OpStatus::INVALID_OP. There is no more work
            // to do in the latter case. The IEEE-754R standard says it is
            // implementation-defined in this case whether, if ADDEND is a
            // quiet NaN, we raise invalid op; this implementation does so.
            //
            // If we need to do the addition we can do so with normal
            // precision.
            if fs == OpStatus::OK {
                fs = self.add_rounded(addend, round);
            }
            return fs;
        }

        // Post-multiplication sign, before addition.
        self.sign ^= multiplicand.sign;

        // Allocate space for twice as many bits as the original significand, plus one
        // extra bit for the addition to overflow into.
        assert!(limbs_for_bits(S::PRECISION * 2 + 1) <= 2);
        let mut wide_sig = sig::widening_mul(self.sig[0], multiplicand.sig[0]);

        let mut loss = Loss::ExactlyZero;
        let mut omsb = sig::omsb(&wide_sig);
        self.exp += multiplicand.exp;

        // Assume the operands involved in the multiplication are single-precision
        // FP, and the two multiplicants are:
        //     lhs = a23 . a22 ... a0 * 2^e1
        //     rhs = b23 . b22 ... b0 * 2^e2
        // the result of multiplication is:
        //     lhs = c48 c47 c46 . c45 ... c0 * 2^(e1+e2)
        // Note that there are three significant bits at the left-hand side of the
        // radix point: two for the multiplication, and an overflow bit for the
        // addition (that will always be zero at this point). Move the radix point
        // toward left by two bits, and adjust exponent accordingly.
        self.exp += 2;

        if addend.is_non_zero() {
            // Normalize our MSB to one below the top bit to allow for overflow.
            let ext_precision = 2 * S::PRECISION + 1;
            if omsb != ext_precision - 1 {
                assert!(ext_precision > omsb);
                sig::shift_left(&mut wide_sig, &mut self.exp, (ext_precision - 1) - omsb);
            }

            // The intermediate result of the multiplication has "2 * S::PRECISION"
            // signicant bit; adjust the addend to be consistent with mul result.
            let mut ext_addend_sig = [addend.sig[0], 0];

            // Extend the addend significand to ext_precision - 1. This guarantees
            // that the high bit of the significand is zero (same as wide_sig),
            // so the addition will overflow (if it does overflow at all) into the top bit.
            sig::shift_left(
                &mut ext_addend_sig,
                &mut 0,
                ext_precision - 1 - S::PRECISION,
            );
            loss = sig::add_or_sub(
                &mut wide_sig,
                &mut self.exp,
                &mut self.sign,
                &mut ext_addend_sig,
                addend.exp + 1,
                addend.sign,
            );

            omsb = sig::omsb(&wide_sig);
        }

        // Convert the result having "2 * S::PRECISION" significant-bits back to the one
        // having "S::PRECISION" significant-bits. First, move the radix point from
        // poision "2*S::PRECISION - 1" to "S::PRECISION - 1". The exponent need to be
        // adjusted by "2*S::PRECISION - 1" - "S::PRECISION - 1" = "S::PRECISION".
        self.exp -= S::PRECISION as ExpInt + 1;

        // In case MSB resides at the left-hand side of radix point, shift the
        // mantissa right by some amount to make sure the MSB reside right before
        // the radix point (i.e. "MSB . rest-significant-bits").
        if omsb > S::PRECISION {
            let bits = omsb - S::PRECISION;
            loss = sig::shift_right(&mut wide_sig, &mut self.exp, bits).combine(loss);
        }

        self.sig[0] = wide_sig[0];

        let mut fs = self.normalize(round, loss);
        if loss != Loss::ExactlyZero {
            fs |= OpStatus::INEXACT;
        }

        // If two numbers add (exactly) to zero, IEEE 754 decrees it is a
        // positive zero unless rounding to minus infinity, except that
        // adding two like-signed zeroes gives that zero.
        if self.category == Category::Zero && !fs.intersects(OpStatus::UNDERFLOW) &&
            self.sign != addend.sign
        {
            self.sign = round == Round::TowardNegative;
        }

        fs
    }

    fn round_to_integral(self, round: Round) -> (Self, OpStatus) {
        // If the exponent is large enough, we know that this value is already
        // integral, and the arithmetic below would potentially cause it to saturate
        // to +/-Inf. Bail out early instead.
        if self.is_finite_non_zero() && self.exp + 1 >= S::PRECISION as ExpInt {
            return (self, OpStatus::OK);
        }

        // The algorithm here is quite simple: we add 2^(p-1), where p is the
        // precision of our format, and then subtract it back off again. The choice
        // of rounding modes for the addition/subtraction determines the rounding mode
        // for our integral rounding as well.
        // NOTE: When the input value is negative, we do subtraction followed by
        // addition instead.
        assert!(S::PRECISION <= 128);
        let (mut magic_const, fs) =
            Self::from_u128(1 << (S::PRECISION - 1), Round::NearestTiesToEven);
        magic_const.sign = self.sign;

        if fs != OpStatus::OK {
            return (self, fs);
        }

        let mut r = self;
        let fs = r.add_rounded(magic_const, round);
        if fs != OpStatus::OK && fs != OpStatus::INEXACT {
            return (self, fs);
        }

        let fs = r.sub_rounded(magic_const, round);

        // Restore the input sign to handle 0.0/-0.0 cases correctly.
        r.sign = self.sign;

        (r, fs)
    }

    fn next_up(&mut self) -> OpStatus {
        // Compute nextUp(x), handling each float category separately.
        match self.category {
            Category::Infinity => {
                if self.sign {
                    // nextUp(-inf) = -largest
                    *self = -Self::largest();
                    OpStatus::OK
                } else {
                    // nextUp(+inf) = +inf
                    OpStatus::OK
                }
            }
            Category::NaN => {
                // IEEE-754R 2008 6.2 Par 2: nextUp(sNaN) = qNaN. Set Invalid flag.
                // IEEE-754R 2008 6.2: nextUp(qNaN) = qNaN. Must be identity so we do not
                //                     change the payload.
                if self.is_signaling() {
                    // For consistency, propagate the sign of the sNaN to the qNaN.
                    *self = Self::nan().copy_sign(*self);
                    OpStatus::INVALID_OP
                } else {
                    OpStatus::OK
                }
            }
            Category::Zero => {
                // nextUp(pm 0) = +smallest
                *self = Self::smallest();
                OpStatus::OK
            }
            Category::Normal => {
                // nextUp(-smallest) = -0
                if self.is_smallest() && self.sign {
                    *self = -Self::zero();
                    return OpStatus::OK;
                }

                // nextUp(largest) == INFINITY
                if self.is_largest() && !self.sign {
                    *self = Self::inf();
                    return OpStatus::OK;
                }

                // Excluding the integral bit. This allows us to test for binade boundaries.
                let sig_mask = (1 << (S::PRECISION - 1)) - 1;

                // nextUp(normal) == normal + inc.
                if self.sign {
                    // If we are negative, we need to decrement the significand.

                    // We only cross a binade boundary that requires adjusting the exponent
                    // if:
                    //   1. exponent != S::MIN_EXPONENT. This implies we are not in the
                    //   smallest binade or are dealing with denormals.
                    //   2. Our significand excluding the integral bit is all zeros.
                    let crossing_binade_boundary = self.exp != S::MIN_EXPONENT &&
                        self.sig[0] & sig_mask == 0;

                    // Decrement the significand.
                    //
                    // We always do this since:
                    //   1. If we are dealing with a non-binade decrement, by definition we
                    //   just decrement the significand.
                    //   2. If we are dealing with a normal -> normal binade decrement, since
                    //   we have an explicit integral bit the fact that all bits but the
                    //   integral bit are zero implies that subtracting one will yield a
                    //   significand with 0 integral bit and 1 in all other spots. Thus we
                    //   must just adjust the exponent and set the integral bit to 1.
                    //   3. If we are dealing with a normal -> denormal binade decrement,
                    //   since we set the integral bit to 0 when we represent denormals, we
                    //   just decrement the significand.
                    sig::decrement(&mut self.sig);

                    if crossing_binade_boundary {
                        // Our result is a normal number. Do the following:
                        // 1. Set the integral bit to 1.
                        // 2. Decrement the exponent.
                        sig::set_bit(&mut self.sig, S::PRECISION - 1);
                        self.exp -= 1;
                    }
                } else {
                    // If we are positive, we need to increment the significand.

                    // We only cross a binade boundary that requires adjusting the exponent if
                    // the input is not a denormal and all of said input's significand bits
                    // are set. If all of said conditions are true: clear the significand, set
                    // the integral bit to 1, and increment the exponent. If we have a
                    // denormal always increment since moving denormals and the numbers in the
                    // smallest normal binade have the same exponent in our representation.
                    let crossing_binade_boundary = !self.is_denormal() &&
                        self.sig[0] & sig_mask == sig_mask;

                    if crossing_binade_boundary {
                        self.sig = [0];
                        sig::set_bit(&mut self.sig, S::PRECISION - 1);
                        assert_ne!(
                            self.exp,
                            S::MAX_EXPONENT,
                            "We can not increment an exponent beyond the MAX_EXPONENT \
                             allowed by the given floating point semantics."
                        );
                        self.exp += 1;
                    } else {
                        sig::increment(&mut self.sig);
                    }
                }
                OpStatus::OK
            }
        }
    }

    fn change_sign(&mut self) {
        self.sign = !self.sign;
    }

    fn from_bits(input: u128) -> Self {
        // Dispatch to semantics.
        S::from_bits(input)
    }

    fn from_u128(input: u128, round: Round) -> (Self, OpStatus) {
        let mut r = Ieee {
            sig: [input],
            exp: S::PRECISION as ExpInt - 1,
            category: Category::Normal,
            sign: false,
            marker: PhantomData,
        };
        let fs = r.normalize(round, Loss::ExactlyZero);
        (r, fs)
    }

    fn from_str_rounded(mut s: &str, mut round: Round) -> Result<(Self, OpStatus), ParseError> {
        if s.is_empty() {
            return Err(ParseError("Invalid string length"));
        }

        // Handle special cases.
        match s {
            "inf" | "INFINITY" => return Ok((Self::inf(), OpStatus::OK)),
            "-inf" | "-INFINITY" => return Ok((-Self::inf(), OpStatus::OK)),
            "nan" | "NaN" => return Ok((Self::nan(), OpStatus::OK)),
            "-nan" | "-NaN" => return Ok((-Self::nan(), OpStatus::OK)),
            _ => {}
        }

        // Handle a leading minus sign.
        let minus = s.starts_with("-");
        if minus || s.starts_with("+") {
            s = &s[1..];
            if s.is_empty() {
                return Err(ParseError("String has no digits"));
            }
        }

        // Adjust the rounding mode for the absolute value below.
        if minus {
            round = -round;
        }

        let (mut r, fs) = if s.starts_with("0x") || s.starts_with("0X") {
            s = &s[2..];
            if s.is_empty() {
                return Err(ParseError("Invalid string"));
            }
            Self::from_hexadecimal_string(s, round)?
        } else {
            Self::from_decimal_string(s, round)?
        };

        if minus {
            r.change_sign();
        }

        Ok((r, fs))
    }

    fn to_bits(self) -> u128 {
        // Dispatch to semantics.
        S::to_bits(self)
    }

    fn to_u128(self, width: usize, round: Round, is_exact: &mut bool) -> (u128, OpStatus) {
        // The result of trying to convert a number too large.
        let overflow = if self.sign {
            // Negative numbers cannot be represented as unsigned.
            0
        } else {
            // Largest unsigned integer of the given width.
            !0 >> (128 - width)
        };

        *is_exact = false;

        match self.category {
            Category::NaN => (0, OpStatus::INVALID_OP),

            Category::Infinity => (overflow, OpStatus::INVALID_OP),

            Category::Zero => {
                // Negative zero can't be represented as an int.
                *is_exact = !self.sign;
                (0, OpStatus::OK)
            }

            Category::Normal => {
                let mut r = 0;

                // Step 1: place our absolute value, with any fraction truncated, in
                // the destination.
                let truncated_bits = if self.exp < 0 {
                    // Our absolute value is less than one; truncate everything.
                    // For exponent -1 the integer bit represents .5, look at that.
                    // For smaller exponents leftmost truncated bit is 0.
                    S::PRECISION - 1 + (-self.exp) as usize
                } else {
                    // We want the most significant (exponent + 1) bits; the rest are
                    // truncated.
                    let bits = self.exp as usize + 1;

                    // Hopelessly large in magnitude?
                    if bits > width {
                        return (overflow, OpStatus::INVALID_OP);
                    }

                    if bits < S::PRECISION {
                        // We truncate (S::PRECISION - bits) bits.
                        r = self.sig[0] >> (S::PRECISION - bits);
                        S::PRECISION - bits
                    } else {
                        // We want at least as many bits as are available.
                        r = self.sig[0] << (bits - S::PRECISION);
                        0
                    }
                };

                // Step 2: work out any lost fraction, and increment the absolute
                // value if we would round away from zero.
                let mut loss = Loss::ExactlyZero;
                if truncated_bits > 0 {
                    loss = Loss::through_truncation(&self.sig, truncated_bits);
                    if loss != Loss::ExactlyZero &&
                        self.round_away_from_zero(round, loss, truncated_bits)
                    {
                        r = r.wrapping_add(1);
                        if r == 0 {
                            return (overflow, OpStatus::INVALID_OP); // Overflow.
                        }
                    }
                }

                // Step 3: check if we fit in the destination.
                if r > overflow {
                    return (overflow, OpStatus::INVALID_OP);
                }

                if loss == Loss::ExactlyZero {
                    *is_exact = true;
                    (r, OpStatus::OK)
                } else {
                    (r, OpStatus::INEXACT)
                }
            }
        }
    }

    fn cmp_abs_normal(self, rhs: Self) -> Ordering {
        assert!(self.is_finite_non_zero());
        assert!(rhs.is_finite_non_zero());

        // If exponents are equal, do an unsigned comparison of the significands.
        self.exp.cmp(&rhs.exp).then_with(
            || sig::cmp(&self.sig, &rhs.sig),
        )
    }

    fn bitwise_eq(self, rhs: Self) -> bool {
        if self.category != rhs.category || self.sign != rhs.sign {
            return false;
        }

        if self.category == Category::Zero || self.category == Category::Infinity {
            return true;
        }

        if self.is_finite_non_zero() && self.exp != rhs.exp {
            return false;
        }

        self.sig == rhs.sig
    }

    fn is_negative(self) -> bool {
        self.sign
    }

    fn is_denormal(self) -> bool {
        self.is_finite_non_zero() && self.exp == S::MIN_EXPONENT &&
            !sig::get_bit(&self.sig, S::PRECISION - 1)
    }

    fn is_signaling(self) -> bool {
        // Ieee-754R 2008 6.2.1: A signaling NaN bit string should be encoded with the
        // first bit of the trailing significand being 0.
        self.is_nan() && !sig::get_bit(&self.sig, S::QNAN_BIT)
    }

    fn category(self) -> Category {
        self.category
    }

    fn is_smallest(self) -> bool {
        // The smallest number by magnitude in our format will be the smallest
        // denormal, i.e. the floating point number with exponent being minimum
        // exponent and significand bitwise equal to 1.
        self.is_denormal() && self.sig == [1]
    }

    fn is_largest(self) -> bool {
        // The largest number by magnitude in our format will be the floating point
        // number with maximum exponent and with significand that is all ones.
        self.is_finite_non_zero() && self.exp == S::MAX_EXPONENT &&
            self.sig == [!0 & ((1 << S::PRECISION) - 1)]
    }

    fn is_integer(self) -> bool {
        // This could be made more efficient; I'm going for obviously correct.
        if !self.is_finite() {
            return false;
        }
        let (truncated, _) = self.round_to_integral(Round::TowardZero);
        self.partial_cmp(&truncated) == Some(Ordering::Equal)
    }

    #[allow(unused)]
    fn get_exact_inverse(self) -> Option<Self> {
        // Special floats and denormals have no exact inverse.
        if !self.is_finite_non_zero() {
            return None;
        }

        // Check that the number is a power of two by making sure that only the
        // integer bit is set in the significand.
        if self.sig != [1 << (S::PRECISION - 1)] {
            return None;
        }

        // Get the inverse.
        let (mut reciprocal, _) = Self::from_u128(1, Round::NearestTiesToEven);
        if reciprocal.div_rounded(self, Round::NearestTiesToEven) != OpStatus::OK {
            return None;
        }

        // Avoid multiplication with a denormal, it is not safe on all platforms and
        // may be slower than a normal division.
        if reciprocal.is_denormal() {
            return None;
        }

        assert!(reciprocal.is_finite_non_zero() && reciprocal.sig == [1 << (S::PRECISION - 1)]);

        Some(reciprocal)
    }

    fn ilogb(mut self) -> i32 {
        if self.is_nan() {
            return IEK_NAN;
        }
        if self.is_zero() {
            return IEK_ZERO;
        }
        if self.is_infinite() {
            return IEK_INF;
        }
        if !self.is_denormal() {
            return self.exp as i32;
        }

        let sig_bits = (S::PRECISION - 1) as ExpInt;
        self.exp += sig_bits;
        let _: OpStatus = self.normalize(Round::NearestTiesToEven, Loss::ExactlyZero);
        (self.exp - sig_bits) as i32
    }

    fn scalbn(mut self, exp: i32, round: Round) -> Self {
        // If exp is wildly out-of-scale, simply adding it to self.exp will
        // overflow; clamp it to a safe range before adding, but ensure that the range
        // is large enough that the clamp does not change the result. The range we
        // need to support is the difference between the largest possible exponent and
        // the normalized exponent of half the smallest denormal.

        let sig_bits = (S::PRECISION - 1) as ExpInt;
        let max_change = (S::MAX_EXPONENT - (S::MIN_EXPONENT - sig_bits) + 1) as i32;

        // Clamp to one past the range ends to let normalize handle overlflow.
        self.exp += cmp::min(cmp::max(exp, (-max_change - 1)), max_change) as ExpInt;
        let _: OpStatus = self.normalize(round, Loss::ExactlyZero);
        if self.is_nan() {
            sig::set_bit(&mut self.sig, S::QNAN_BIT);
        }
        self
    }

    fn frexp(mut self, exp: &mut i32, round: Round) -> Self {
        *exp = self.ilogb();

        // Quiet signalling nans.
        if *exp == IEK_NAN {
            sig::set_bit(&mut self.sig, S::QNAN_BIT);
            return self;
        }

        if *exp == IEK_INF {
            return self;
        }

        // 1 is added because frexp is defined to return a normalized fraction in
        // +/-[0.5, 1.0), rather than the usual +/-[1.0, 2.0).
        if *exp == IEK_ZERO {
            *exp = 0;
        } else {
            *exp += 1;
        }
        self.scalbn(-*exp, round)
    }
}

impl<S: IeeeSemantics> Ieee<S> {
    /// Handle positive overflow. We either return infinity or
    /// the largest finite number. For negative overflow,
    /// negate the `round` argument before calling.
    fn overflow_result(round: Round) -> (Self, OpStatus) {
        match round {
            // Infinity?
            Round::NearestTiesToEven | Round::NearestTiesToAway | Round::TowardPositive => {
                (Self::inf(), OpStatus::OVERFLOW | OpStatus::INEXACT)
            }
            // Otherwise we become the largest finite number.
            Round::TowardNegative | Round::TowardZero => (Self::largest(), OpStatus::INEXACT),
        }
    }

    /// Returns TRUE if, when truncating the current number, with BIT the
    /// new LSB, with the given lost fraction and rounding mode, the result
    /// would need to be rounded away from zero (i.e., by increasing the
    /// signficand). This routine must work for Category::Zero of both signs, and
    /// Category::Normal numbers.
    fn round_away_from_zero(&self, round: Round, loss: Loss, bit: usize) -> bool {
        // NaNs and infinities should not have lost fractions.
        assert!(self.is_finite_non_zero() || self.category == Category::Zero);

        // Current callers never pass this so we don't handle it.
        assert_ne!(loss, Loss::ExactlyZero);

        match round {
            Round::NearestTiesToAway => loss == Loss::ExactlyHalf || loss == Loss::MoreThanHalf,
            Round::NearestTiesToEven => {
                if loss == Loss::MoreThanHalf {
                    return true;
                }

                // Our zeros don't have a significand to test.
                if loss == Loss::ExactlyHalf && self.category != Category::Zero {
                    return sig::get_bit(&self.sig, bit);
                }

                false
            }
            Round::TowardZero => false,
            Round::TowardPositive => !self.sign,
            Round::TowardNegative => self.sign,
        }
    }

    fn normalize(&mut self, round: Round, mut loss: Loss) -> OpStatus {
        if !self.is_finite_non_zero() {
            return OpStatus::OK;
        }

        // Before rounding normalize the exponent of Category::Normal numbers.
        let mut omsb = sig::omsb(&self.sig);

        if omsb > 0 {
            // OMSB is numbered from 1. We want to place it in the integer
            // bit numbered PRECISION if possible, with a compensating change in
            // the exponent.
            let mut final_exp = self.exp.saturating_add(
                omsb as ExpInt - S::PRECISION as ExpInt,
            );

            // If the resulting exponent is too high, overflow according to
            // the rounding mode.
            if final_exp > S::MAX_EXPONENT {
                let round = if self.sign { -round } else { round };
                let (r, fs) = Self::overflow_result(round);
                *self = r.copy_sign(*self);
                return fs;
            }

            // Subnormal numbers have exponent MIN_EXPONENT, and their MSB
            // is forced based on that.
            if final_exp < S::MIN_EXPONENT {
                final_exp = S::MIN_EXPONENT;
            }

            // Shifting left is easy as we don't lose precision.
            if final_exp < self.exp {
                assert_eq!(loss, Loss::ExactlyZero);

                let exp_change = (self.exp - final_exp) as usize;
                sig::shift_left(&mut self.sig, &mut self.exp, exp_change);

                return OpStatus::OK;
            }

            // Shift right and capture any new lost fraction.
            if final_exp > self.exp {
                let exp_change = (final_exp - self.exp) as usize;
                loss = sig::shift_right(&mut self.sig, &mut self.exp, exp_change).combine(loss);

                // Keep OMSB up-to-date.
                omsb = omsb.saturating_sub(exp_change);
            }
        }

        // Now round the number according to round given the lost
        // fraction.

        // As specified in IEEE 754, since we do not trap we do not report
        // underflow for exact results.
        if loss == Loss::ExactlyZero {
            // Canonicalize zeros.
            if omsb == 0 {
                self.category = Category::Zero;
            }

            return OpStatus::OK;
        }

        // Increment the significand if we're rounding away from zero.
        if self.round_away_from_zero(round, loss, 0) {
            if omsb == 0 {
                self.exp = S::MIN_EXPONENT;
            }

            // We should never overflow.
            assert_eq!(sig::increment(&mut self.sig), 0);
            omsb = sig::omsb(&self.sig);

            // Did the significand increment overflow?
            if omsb == S::PRECISION + 1 {
                // Renormalize by incrementing the exponent and shifting our
                // significand right one. However if we already have the
                // maximum exponent we overflow to infinity.
                if self.exp == S::MAX_EXPONENT {
                    self.category = Category::Infinity;

                    return OpStatus::OVERFLOW | OpStatus::INEXACT;
                }

                let _: Loss = sig::shift_right(&mut self.sig, &mut self.exp, 1);

                return OpStatus::INEXACT;
            }
        }

        // The normal case - we were and are not denormal, and any
        // significand increment above didn't overflow.
        if omsb == S::PRECISION {
            return OpStatus::INEXACT;
        }

        // We have a non-zero denormal.
        assert!(omsb < S::PRECISION);

        // Canonicalize zeros.
        if omsb == 0 {
            self.category = Category::Zero;
        }

        // The Category::Zero case is a denormal that underflowed to zero.
        OpStatus::UNDERFLOW | OpStatus::INEXACT
    }

    fn from_hexadecimal_string(s: &str, round: Round) -> Result<(Self, OpStatus), ParseError> {
        let mut r = Ieee {
            sig: [0],
            exp: 0,
            category: Category::Normal,
            sign: false,
            marker: PhantomData,
        };

        let mut any_digits = false;
        let mut has_exp = false;
        let mut bit_pos = LIMB_BITS as isize;
        let mut loss = None;

        // Without leading or trailing zeros, irrespective of the dot.
        let mut first_sig_digit = None;
        let mut dot = s.len();

        for (p, c) in s.char_indices() {
            // Skip leading zeros and any (hexa)decimal point.
            if c == '.' {
                if dot != s.len() {
                    return Err(ParseError("String contains multiple dots"));
                }
                dot = p;
            } else if let Some(hex_value) = c.to_digit(16) {
                any_digits = true;

                if first_sig_digit.is_none() {
                    if hex_value == 0 {
                        continue;
                    }
                    first_sig_digit = Some(p);
                }

                // Store the number while we have space.
                bit_pos -= 4;
                if bit_pos >= 0 {
                    r.sig[0] |= (hex_value as Limb) << bit_pos;
                } else {
                    // If zero or one-half (the hexadecimal digit 8) are followed
                    // by non-zero, they're a little more than zero or one-half.
                    if let Some(ref mut loss) = loss {
                        if hex_value != 0 {
                            if *loss == Loss::ExactlyZero {
                                *loss = Loss::LessThanHalf;
                            }
                            if *loss == Loss::ExactlyHalf {
                                *loss = Loss::MoreThanHalf;
                            }
                        }
                    } else {
                        loss = Some(match hex_value {
                            0 => Loss::ExactlyZero,
                            1...7 => Loss::LessThanHalf,
                            8 => Loss::ExactlyHalf,
                            9...15 => Loss::MoreThanHalf,
                            _ => unreachable!(),
                        });
                    }
                }
            } else if c == 'p' || c == 'P' {
                if !any_digits {
                    return Err(ParseError("Significand has no digits"));
                }

                if dot == s.len() {
                    dot = p;
                }

                let mut chars = s[p + 1..].chars().peekable();

                // Adjust for the given exponent.
                let exp_minus = chars.peek() == Some(&'-');
                if exp_minus || chars.peek() == Some(&'+') {
                    chars.next();
                }

                for c in chars {
                    if let Some(value) = c.to_digit(10) {
                        has_exp = true;
                        r.exp = r.exp.saturating_mul(10).saturating_add(value as ExpInt);
                    } else {
                        return Err(ParseError("Invalid character in exponent"));
                    }
                }
                if !has_exp {
                    return Err(ParseError("Exponent has no digits"));
                }

                if exp_minus {
                    r.exp = -r.exp;
                }

                break;
            } else {
                return Err(ParseError("Invalid character in significand"));
            }
        }
        if !any_digits {
            return Err(ParseError("Significand has no digits"));
        }

        // Hex floats require an exponent but not a hexadecimal point.
        if !has_exp {
            return Err(ParseError("Hex strings require an exponent"));
        }

        // Ignore the exponent if we are zero.
        let first_sig_digit = match first_sig_digit {
            Some(p) => p,
            None => return Ok((Self::zero(), OpStatus::OK)),
        };

        // Calculate the exponent adjustment implicit in the number of
        // significant digits and adjust for writing the significand starting
        // at the most significant nibble.
        let exp_adjustment = if dot > first_sig_digit {
            ExpInt::try_from(dot - first_sig_digit).unwrap()
        } else {
            -ExpInt::try_from(first_sig_digit - dot - 1).unwrap()
        };
        let exp_adjustment = exp_adjustment
            .saturating_mul(4)
            .saturating_sub(1)
            .saturating_add(S::PRECISION as ExpInt)
            .saturating_sub(LIMB_BITS as ExpInt);
        r.exp = r.exp.saturating_add(exp_adjustment);

        let fs = r.normalize(round, loss.unwrap_or(Loss::ExactlyZero));
        Ok((r, fs))
    }

    fn from_decimal_string(s: &str, round: Round) -> Result<(Self, OpStatus), ParseError> {
        // Given a normal decimal floating point number of the form
        //
        //   dddd.dddd[eE][+-]ddd
        //
        // where the decimal point and exponent are optional, fill out the
        // variables below. Exponent is appropriate if the significand is
        // treated as an integer, and normalized_exp if the significand
        // is taken to have the decimal point after a single leading
        // non-zero digit.
        //
        // If the value is zero, first_sig_digit is None.

        let mut any_digits = false;
        let mut dec_exp = 0i32;

        // Without leading or trailing zeros, irrespective of the dot.
        let mut first_sig_digit = None;
        let mut last_sig_digit = 0;
        let mut dot = s.len();

        for (p, c) in s.char_indices() {
            if c == '.' {
                if dot != s.len() {
                    return Err(ParseError("String contains multiple dots"));
                }
                dot = p;
            } else if let Some(dec_value) = c.to_digit(10) {
                any_digits = true;

                if dec_value != 0 {
                    if first_sig_digit.is_none() {
                        first_sig_digit = Some(p);
                    }
                    last_sig_digit = p;
                }
            } else if c == 'e' || c == 'E' {
                if !any_digits {
                    return Err(ParseError("Significand has no digits"));
                }

                if dot == s.len() {
                    dot = p;
                }

                let mut chars = s[p + 1..].chars().peekable();

                // Adjust for the given exponent.
                let exp_minus = chars.peek() == Some(&'-');
                if exp_minus || chars.peek() == Some(&'+') {
                    chars.next();
                }

                any_digits = false;
                for c in chars {
                    if let Some(value) = c.to_digit(10) {
                        any_digits = true;
                        dec_exp = dec_exp.saturating_mul(10).saturating_add(value as i32);
                    } else {
                        return Err(ParseError("Invalid character in exponent"));
                    }
                }
                if !any_digits {
                    return Err(ParseError("Exponent has no digits"));
                }

                if exp_minus {
                    dec_exp = -dec_exp;
                }

                break;
            } else {
                return Err(ParseError("Invalid character in significand"));
            }
        }
        if !any_digits {
            return Err(ParseError("Significand has no digits"));
        }

        // Test if we have a zero number allowing for non-zero exponents.
        let first_sig_digit = match first_sig_digit {
            Some(p) => p,
            None => return Ok((Self::zero(), OpStatus::OK)),
        };

        // Adjust the exponents for any decimal point.
        if dot > last_sig_digit {
            dec_exp = dec_exp.saturating_add((dot - last_sig_digit - 1) as i32);
        } else {
            dec_exp = dec_exp.saturating_sub((last_sig_digit - dot) as i32);
        }
        let significand_digits = last_sig_digit - first_sig_digit + 1 -
            (dot > first_sig_digit && dot < last_sig_digit) as usize;
        let normalized_exp = dec_exp.saturating_add(significand_digits as i32 - 1);

        // Handle the cases where exponents are obviously too large or too
        // small. Writing L for log 10 / log 2, a number d.ddddd*10^dec_exp
        // definitely overflows if
        //
        //       (dec_exp - 1) * L >= MAX_EXPONENT
        //
        // and definitely underflows to zero where
        //
        //       (dec_exp + 1) * L <= MIN_EXPONENT - PRECISION
        //
        // With integer arithmetic the tightest bounds for L are
        //
        //       93/28 < L < 196/59            [ numerator <= 256 ]
        //       42039/12655 < L < 28738/8651  [ numerator <= 65536 ]

        // Check for MAX_EXPONENT.
        if normalized_exp.saturating_sub(1).saturating_mul(42039) >=
            12655 * S::MAX_EXPONENT as i32
        {
            // Overflow and round.
            return Ok(Self::overflow_result(round));
        }

        // Check for MIN_EXPONENT.
        if normalized_exp.saturating_add(1).saturating_mul(28738) <=
            8651 * (S::MIN_EXPONENT as i32 - S::PRECISION as i32)
        {
            // Underflow to zero and round.
            let r = if round == Round::TowardPositive {
                Ieee::smallest()
            } else {
                Ieee::zero()
            };
            return Ok((r, OpStatus::UNDERFLOW | OpStatus::INEXACT));
        }

        // A tight upper bound on number of bits required to hold an
        // N-digit decimal integer is N * 196 / 59. Allocate enough space
        // to hold the full significand, and an extra limb required by
        // tcMultiplyPart.
        let max_limbs = limbs_for_bits(1 + 196 * significand_digits / 59);
        let mut dec_sig = Vec::with_capacity(max_limbs);

        // Convert to binary efficiently - we do almost all multiplication
        // in a Limb. When this would overflow do we do a single
        // bignum multiplication, and then revert again to multiplication
        // in a Limb.
        let mut chars = s[first_sig_digit..last_sig_digit + 1].chars();
        loop {
            let mut val = 0;
            let mut multiplier = 1;

            loop {
                let dec_value = match chars.next() {
                    Some('.') => continue,
                    Some(c) => c.to_digit(10).unwrap(),
                    None => break,
                };

                multiplier *= 10;
                val = val * 10 + dec_value as Limb;

                // The maximum number that can be multiplied by ten with any
                // digit added without overflowing a Limb.
                if multiplier > (!0 - 9) / 10 {
                    break;
                }
            }

            // If we've consumed no digits, we're done.
            if multiplier == 1 {
                break;
            }

            // Multiply out the current limb.
            let mut carry = val;
            for x in &mut dec_sig {
                let [low, mut high] = sig::widening_mul(*x, multiplier);

                // Now add carry.
                let (low, overflow) = low.overflowing_add(carry);
                high += overflow as Limb;

                *x = low;
                carry = high;
            }

            // If we had carry, we need another limb (likely but not guaranteed).
            if carry > 0 {
                dec_sig.push(carry);
            }
        }

        // Calculate pow(5, abs(dec_exp)) into `pow5_full`.
        // The *_calc Vec's are reused scratch space, as an optimization.
        let (pow5_full, mut pow5_calc, mut sig_calc, mut sig_scratch_calc) = {
            let mut power = dec_exp.abs() as usize;

            const FIRST_EIGHT_POWERS: [Limb; 8] = [1, 5, 25, 125, 625, 3125, 15625, 78125];

            let mut p5_scratch = vec![];
            let mut p5 = vec![FIRST_EIGHT_POWERS[4]];

            let mut r_scratch = vec![];
            let mut r = vec![FIRST_EIGHT_POWERS[power & 7]];
            power >>= 3;

            while power > 0 {
                // Calculate pow(5,pow(2,n+3)).
                p5_scratch.resize(p5.len() * 2, 0);
                let _: Loss = sig::mul(&mut p5_scratch, &mut 0, &p5, &p5, p5.len() * 2 * LIMB_BITS);
                while p5_scratch.last() == Some(&0) {
                    p5_scratch.pop();
                }
                mem::swap(&mut p5, &mut p5_scratch);

                if power & 1 != 0 {
                    r_scratch.resize(r.len() + p5.len(), 0);
                    let _: Loss = sig::mul(
                        &mut r_scratch,
                        &mut 0,
                        &r,
                        &p5,
                        (r.len() + p5.len()) * LIMB_BITS,
                    );
                    while r_scratch.last() == Some(&0) {
                        r_scratch.pop();
                    }
                    mem::swap(&mut r, &mut r_scratch);
                }

                power >>= 1;
            }

            (r, r_scratch, p5, p5_scratch)
        };

        // Attempt dec_sig * 10^dec_exp with increasing precision.
        let mut attempt = 1;
        loop {
            let calc_precision = (LIMB_BITS << attempt) - 1;
            attempt += 1;

            let calc_normal_from_limbs = |sig: &mut Vec<Limb>,
                                          limbs: &[Limb]|
             -> (ExpInt, OpStatus) {
                sig.resize(limbs_for_bits(calc_precision), 0);
                let (mut loss, mut exp) = sig::from_limbs(sig, limbs, calc_precision);

                // Before rounding normalize the exponent of Category::Normal numbers.
                let mut omsb = sig::omsb(sig);

                assert_ne!(omsb, 0);

                // OMSB is numbered from 1. We want to place it in the integer
                // bit numbered PRECISION if possible, with a compensating change in
                // the exponent.
                let final_exp = exp.saturating_add(omsb as ExpInt - calc_precision as ExpInt);

                // Shifting left is easy as we don't lose precision.
                if final_exp < exp {
                    assert_eq!(loss, Loss::ExactlyZero);

                    let exp_change = (exp - final_exp) as usize;
                    sig::shift_left(sig, &mut exp, exp_change);

                    return (exp, OpStatus::OK);
                }

                // Shift right and capture any new lost fraction.
                if final_exp > exp {
                    let exp_change = (final_exp - exp) as usize;
                    loss = sig::shift_right(sig, &mut exp, exp_change).combine(loss);

                    // Keep OMSB up-to-date.
                    omsb = omsb.saturating_sub(exp_change);
                }

                assert_eq!(omsb, calc_precision);

                // Now round the number according to round given the lost
                // fraction.

                // As specified in IEEE 754, since we do not trap we do not report
                // underflow for exact results.
                if loss == Loss::ExactlyZero {
                    return (exp, OpStatus::OK);
                }

                // Increment the significand if we're rounding away from zero.
                if loss == Loss::MoreThanHalf || loss == Loss::ExactlyHalf && sig::get_bit(sig, 0) {
                    // We should never overflow.
                    assert_eq!(sig::increment(sig), 0);
                    omsb = sig::omsb(sig);

                    // Did the significand increment overflow?
                    if omsb == calc_precision + 1 {
                        let _: Loss = sig::shift_right(sig, &mut exp, 1);

                        return (exp, OpStatus::INEXACT);
                    }
                }

                // The normal case - we were and are not denormal, and any
                // significand increment above didn't overflow.
                (exp, OpStatus::INEXACT)
            };

            let (mut exp, fs) = calc_normal_from_limbs(&mut sig_calc, &dec_sig);
            let (pow5_exp, pow5_fs) = calc_normal_from_limbs(&mut pow5_calc, &pow5_full);

            // Add dec_exp, as 10^n = 5^n * 2^n.
            exp += dec_exp as ExpInt;

            let mut used_bits = S::PRECISION;
            let mut truncated_bits = calc_precision - used_bits;

            let half_ulp_err1 = (fs != OpStatus::OK) as Limb;
            let (calc_loss, half_ulp_err2);
            if dec_exp >= 0 {
                exp += pow5_exp;

                sig_scratch_calc.resize(sig_calc.len() + pow5_calc.len(), 0);
                calc_loss = sig::mul(
                    &mut sig_scratch_calc,
                    &mut exp,
                    &sig_calc,
                    &pow5_calc,
                    calc_precision,
                );
                mem::swap(&mut sig_calc, &mut sig_scratch_calc);

                half_ulp_err2 = (pow5_fs != OpStatus::OK) as Limb;
            } else {
                exp -= pow5_exp;

                sig_scratch_calc.resize(sig_calc.len(), 0);
                calc_loss = sig::div(
                    &mut sig_scratch_calc,
                    &mut exp,
                    &mut sig_calc,
                    &mut pow5_calc,
                    calc_precision,
                );
                mem::swap(&mut sig_calc, &mut sig_scratch_calc);

                // Denormal numbers have less precision.
                if exp < S::MIN_EXPONENT {
                    truncated_bits += (S::MIN_EXPONENT - exp) as usize;
                    used_bits = calc_precision.saturating_sub(truncated_bits);
                }
                // Extra half-ulp lost in reciprocal of exponent.
                half_ulp_err2 = 2 *
                    (pow5_fs != OpStatus::OK || calc_loss != Loss::ExactlyZero) as Limb;
            }

            // Both sig::mul and sig::div return the
            // result with the integer bit set.
            assert!(sig::get_bit(&sig_calc, calc_precision - 1));

            // The error from the true value, in half-ulps, on multiplying two
            // floating point numbers, which differ from the value they
            // approximate by at most half_ulp_err1 and half_ulp_err2 half-ulps, is strictly less
            // than the returned value.
            //
            // See "How to Read Floating Point Numbers Accurately" by William D Clinger.
            assert!(half_ulp_err1 < 2 || half_ulp_err2 < 2 || (half_ulp_err1 + half_ulp_err2 < 8));

            let inexact = (calc_loss != Loss::ExactlyZero) as Limb;
            let half_ulp_err = if half_ulp_err1 + half_ulp_err2 == 0 {
                inexact * 2 // <= inexact half-ulps.
            } else {
                inexact + 2 * (half_ulp_err1 + half_ulp_err2)
            };

            let ulps_from_boundary = {
                let bits = calc_precision - used_bits - 1;

                let i = bits / LIMB_BITS;
                let limb = sig_calc[i] & (!0 >> (LIMB_BITS - 1 - bits % LIMB_BITS));
                let boundary = match round {
                    Round::NearestTiesToEven | Round::NearestTiesToAway => 1 << (bits % LIMB_BITS),
                    _ => 0,
                };
                if i == 0 {
                    let delta = limb.wrapping_sub(boundary);
                    cmp::min(delta, delta.wrapping_neg())
                } else if limb == boundary {
                    if !sig::is_zero(&sig_calc[1..i]) {
                        !0 // A lot.
                    } else {
                        sig_calc[0]
                    }
                } else if limb == boundary.wrapping_sub(1) {
                    if sig_calc[1..i].iter().any(|&x| x.wrapping_neg() != 1) {
                        !0 // A lot.
                    } else {
                        sig_calc[0].wrapping_neg()
                    }
                } else {
                    !0 // A lot.
                }
            };

            // Are we guaranteed to round correctly if we truncate?
            if ulps_from_boundary.saturating_mul(2) >= half_ulp_err {
                let mut r = Ieee {
                    sig: [0],
                    exp,
                    category: Category::Normal,
                    sign: false,
                    marker: PhantomData,
                };
                sig::extract(&mut r.sig, &sig_calc, used_bits, calc_precision - used_bits);
                // If we extracted less bits above we must adjust our exponent
                // to compensate for the implicit right shift.
                r.exp += (S::PRECISION - used_bits) as ExpInt;
                let loss = Loss::through_truncation(&sig_calc, truncated_bits);
                let fs = r.normalize(round, loss);
                return Ok((r, fs));
            }
        }
    }

    /// IEEE::convert - convert a value of one floating point type to another.
    /// The return value corresponds to the IEEE754 exceptions. *loses_info
    /// records whether the transformation lost information, i.e. whether
    /// converting the result back to the original type will produce the
    /// original value (this is almost the same as return value==OpStatus::OK,
    /// but there are edge cases where this is not so).
    pub fn convert<T: IeeeSemantics>(
        self,
        round: Round,
        loses_info: &mut bool,
    ) -> (Ieee<T>, OpStatus) {
        let mut r = Ieee {
            sig: self.sig,
            exp: self.exp,
            category: self.category,
            sign: self.sign,
            marker: PhantomData,
        };

        // x86 has some unusual NaNs which cannot be represented in any other
        // format; note them here.
        fn is_x87_double_extended<S: IeeeSemantics>() -> bool {
            S::QNAN_SIGNIFICAND == ieee_semantics::X87DoubleExtended::QNAN_SIGNIFICAND
        }
        let x87_special_nan = is_x87_double_extended::<S>() && !is_x87_double_extended::<T>() &&
            r.category == Category::NaN &&
            (r.sig[0] & S::QNAN_SIGNIFICAND) != S::QNAN_SIGNIFICAND;

        // If this is a truncation of a denormal number, and the target semantics
        // has larger exponent range than the source semantics (this can happen
        // when truncating from PowerPC double-double to double format), the
        // right shift could lose result mantissa bits. Adjust exponent instead
        // of performing excessive shift.
        let mut shift = T::PRECISION as ExpInt - S::PRECISION as ExpInt;
        if shift < 0 && r.is_finite_non_zero() {
            let mut exp_change = sig::omsb(&r.sig) as ExpInt - S::PRECISION as ExpInt;
            if r.exp + exp_change < T::MIN_EXPONENT {
                exp_change = T::MIN_EXPONENT - r.exp;
            }
            if exp_change < shift {
                exp_change = shift;
            }
            if exp_change < 0 {
                shift -= exp_change;
                r.exp += exp_change;
            }
        }

        // If this is a truncation, perform the shift.
        let mut loss = Loss::ExactlyZero;
        if shift < 0 && (r.is_finite_non_zero() || r.category == Category::NaN) {
            loss = sig::shift_right(&mut r.sig, &mut 0, -shift as usize);
        }

        // If this is an extension, perform the shift.
        if shift > 0 && (r.is_finite_non_zero() || r.category == Category::NaN) {
            sig::shift_left(&mut r.sig, &mut 0, shift as usize);
        }

        let fs;
        if r.is_finite_non_zero() {
            fs = r.normalize(round, loss);
            *loses_info = fs != OpStatus::OK;
        } else if r.category == Category::NaN {
            *loses_info = loss != Loss::ExactlyZero || x87_special_nan;

            // For x87 extended precision, we want to make a NaN, not a special NaN if
            // the input wasn't special either.
            if !x87_special_nan && is_x87_double_extended::<T>() {
                sig::set_bit(&mut r.sig, T::PRECISION - 1);
            }

            // gcc forces the Quiet bit on, which means (float)(double)(float_sNan)
            // does not give you back the same bits. This is dubious, and we
            // don't currently do it. You're really supposed to get
            // an invalid operation signal at runtime, but nobody does that.
            fs = OpStatus::OK;
        } else {
            *loses_info = false;
            fs = OpStatus::OK;
        }

        (r, fs)
    }
}

impl Loss {
    /// Combine the effect of two lost fractions.
    fn combine(self, less_significant: Loss) -> Loss {
        let mut more_significant = self;
        if less_significant != Loss::ExactlyZero {
            if more_significant == Loss::ExactlyZero {
                more_significant = Loss::LessThanHalf;
            } else if more_significant == Loss::ExactlyHalf {
                more_significant = Loss::MoreThanHalf;
            }
        }

        more_significant
    }

    /// Return the fraction lost were a bignum truncated losing the least
    /// significant BITS bits.
    fn through_truncation(limbs: &[Limb], bits: usize) -> Loss {
        if bits == 0 {
            return Loss::ExactlyZero;
        }

        let half_bit = bits - 1;
        let half_limb = half_bit / LIMB_BITS;
        let (half_limb, rest) = if half_limb < limbs.len() {
            (limbs[half_limb], &limbs[..half_limb])
        } else {
            (0, limbs)
        };
        let half = 1 << (half_bit % LIMB_BITS);
        let has_half = half_limb & half != 0;
        let has_rest = half_limb & (half - 1) != 0 || !sig::is_zero(rest);

        match (has_half, has_rest) {
            (false, false) => Loss::ExactlyZero,
            (false, true) => Loss::LessThanHalf,
            (true, false) => Loss::ExactlyHalf,
            (true, true) => Loss::MoreThanHalf,
        }
    }
}

/// Implementation details of Ieee significands, such as big integer arithmetic.
/// As a rule of thumb, no functions in this module should dynamically allocate.
mod sig {
    use std::cmp::Ordering;
    use std::mem;
    use super::{ExpInt, Limb, LIMB_BITS, limbs_for_bits, Loss};

    pub(super) fn is_zero(limbs: &[Limb]) -> bool {
        limbs.iter().all(|&l| l == 0)
    }

    /// One, not zero, based MSB. That is, returns 0 for a zeroed significand.
    pub(super) fn omsb(limbs: &[Limb]) -> usize {
        for i in (0..limbs.len()).rev() {
            if limbs[i] != 0 {
                return (i + 1) * LIMB_BITS - limbs[i].leading_zeros() as usize;
            }
        }

        0
    }

    /// Comparison (unsigned) of two significands.
    pub(super) fn cmp(a: &[Limb], b: &[Limb]) -> Ordering {
        assert_eq!(a.len(), b.len());
        for (a, b) in a.iter().zip(b).rev() {
            match a.cmp(b) {
                Ordering::Equal => {}
                o => return o,
            }
        }

        Ordering::Equal
    }

    /// Extract the given bit.
    pub(super) fn get_bit(limbs: &[Limb], bit: usize) -> bool {
        limbs[bit / LIMB_BITS] & (1 << (bit % LIMB_BITS)) != 0
    }

    /// Set the given bit.
    pub(super) fn set_bit(limbs: &mut [Limb], bit: usize) {
        limbs[bit / LIMB_BITS] |= 1 << (bit % LIMB_BITS);
    }

    /// Clear the given bit.
    pub(super) fn clear_bit(limbs: &mut [Limb], bit: usize) {
        limbs[bit / LIMB_BITS] &= !(1 << (bit % LIMB_BITS));
    }

    /// Shift DST left BITS bits, subtract BITS from its exponent.
    pub(super) fn shift_left(dst: &mut [Limb], exp: &mut ExpInt, bits: usize) {
        if bits > 0 {
            // Our exponent should not underflow.
            *exp = exp.checked_sub(bits as ExpInt).unwrap();

            // Jump is the inter-limb jump; shift is is intra-limb shift.
            let jump = bits / LIMB_BITS;
            let shift = bits % LIMB_BITS;

            for i in (0..dst.len()).rev() {
                let mut limb;

                if i < jump {
                    limb = 0;
                } else {
                    // dst[i] comes from the two limbs src[i - jump] and, if we have
                    // an intra-limb shift, src[i - jump - 1].
                    limb = dst[i - jump];
                    if shift > 0 {
                        limb <<= shift;
                        if i >= jump + 1 {
                            limb |= dst[i - jump - 1] >> (LIMB_BITS - shift);
                        }
                    }
                }

                dst[i] = limb;
            }
        }
    }

    /// Shift DST right BITS bits noting lost fraction.
    pub(super) fn shift_right(dst: &mut [Limb], exp: &mut ExpInt, bits: usize) -> Loss {
        let loss = Loss::through_truncation(dst, bits);

        if bits > 0 {
            // Our exponent should not overflow.
            *exp = exp.checked_add(bits as ExpInt).unwrap();

            // Jump is the inter-limb jump; shift is is intra-limb shift.
            let jump = bits / LIMB_BITS;
            let shift = bits % LIMB_BITS;

            // Perform the shift. This leaves the most significant BITS bits
            // of the result at zero.
            for i in 0..dst.len() {
                let mut limb;

                if i + jump >= dst.len() {
                    limb = 0;
                } else {
                    limb = dst[i + jump];
                    if shift > 0 {
                        limb >>= shift;
                        if i + jump + 1 < dst.len() {
                            limb |= dst[i + jump + 1] << (LIMB_BITS - shift);
                        }
                    }
                }

                dst[i] = limb;
            }
        }

        loss
    }

    /// Copy the bit vector of width SRC_BITS from SRC, starting at bit SRC_LSB,
    /// to DST, such that the bit SRC_LSB becomes the least significant bit of DST.
    /// All high bits above SRC_BITS in DST are zero-filled.
    pub(super) fn extract(dst: &mut [Limb], src: &[Limb], src_bits: usize, src_lsb: usize) {
        if src_bits == 0 {
            return;
        }

        let dst_limbs = limbs_for_bits(src_bits);
        assert!(dst_limbs <= dst.len());

        let src = &src[src_lsb / LIMB_BITS..];
        dst[..dst_limbs].copy_from_slice(&src[..dst_limbs]);

        let shift = src_lsb % LIMB_BITS;
        let _: Loss = shift_right(&mut dst[..dst_limbs], &mut 0, shift);

        // We now have (dst_limbs * LIMB_BITS - shift) bits from SRC
        // in DST.  If this is less that src_bits, append the rest, else
        // clear the high bits.
        let n = dst_limbs * LIMB_BITS - shift;
        if n < src_bits {
            let mask = (1 << (src_bits - n)) - 1;
            dst[dst_limbs - 1] |= (src[dst_limbs] & mask) << n % LIMB_BITS;
        } else if n > src_bits && src_bits % LIMB_BITS > 0 {
            dst[dst_limbs - 1] &= (1 << (src_bits % LIMB_BITS)) - 1;
        }

        // Clear high limbs.
        for x in &mut dst[dst_limbs..] {
            *x = 0;
        }
    }

    /// We want the most significant PRECISION bits of SRC. There may not
    /// be that many; extract what we can.
    pub(super) fn from_limbs(dst: &mut [Limb], src: &[Limb], precision: usize) -> (Loss, ExpInt) {
        let omsb = omsb(src);

        if precision <= omsb {
            extract(dst, src, precision, omsb - precision);
            (
                Loss::through_truncation(src, omsb - precision),
                omsb as ExpInt - 1,
            )
        } else {
            extract(dst, src, omsb, 0);
            (Loss::ExactlyZero, precision as ExpInt - 1)
        }
    }

    /// Increment in-place, return the carry flag.
    pub(super) fn increment(dst: &mut [Limb]) -> Limb {
        for x in dst {
            *x = x.wrapping_add(1);
            if *x != 0 {
                return 0;
            }
        }

        1
    }

    /// Decrement in-place, return the borrow flag.
    pub(super) fn decrement(dst: &mut [Limb]) -> Limb {
        for x in dst {
            *x = x.wrapping_sub(1);
            if *x != !0 {
                return 0;
            }
        }

        1
    }

    /// A += B + C where C is zero or one. Returns the carry flag.
    pub(super) fn add(a: &mut [Limb], b: &[Limb], mut c: Limb) -> Limb {
        assert!(c <= 1);

        for (a, &b) in a.iter_mut().zip(b) {
            let (r, overflow) = a.overflowing_add(b);
            let (r, overflow2) = r.overflowing_add(c);
            *a = r;
            c = (overflow | overflow2) as Limb;
        }

        c
    }

    /// A -= B + C where C is zero or one. Returns the borrow flag.
    pub(super) fn sub(a: &mut [Limb], b: &[Limb], mut c: Limb) -> Limb {
        assert!(c <= 1);

        for (a, &b) in a.iter_mut().zip(b) {
            let (r, overflow) = a.overflowing_sub(b);
            let (r, overflow2) = r.overflowing_sub(c);
            *a = r;
            c = (overflow | overflow2) as Limb;
        }

        c
    }

    /// A += B or A -= B. Does not preserve B.
    pub(super) fn add_or_sub(
        a_sig: &mut [Limb],
        a_exp: &mut ExpInt,
        a_sign: &mut bool,
        b_sig: &mut [Limb],
        b_exp: ExpInt,
        b_sign: bool,
    ) -> Loss {
        // Are we bigger exponent-wise than the RHS?
        let bits = *a_exp - b_exp;

        // Determine if the operation on the absolute values is effectively
        // an addition or subtraction.
        // Subtraction is more subtle than one might naively expect.
        if *a_sign ^ b_sign {
            let (reverse, loss);

            if bits == 0 {
                reverse = cmp(a_sig, b_sig) == Ordering::Less;
                loss = Loss::ExactlyZero;
            } else if bits > 0 {
                loss = shift_right(b_sig, &mut 0, (bits - 1) as usize);
                shift_left(a_sig, a_exp, 1);
                reverse = false;
            } else {
                loss = shift_right(a_sig, a_exp, (-bits - 1) as usize);
                shift_left(b_sig, &mut 0, 1);
                reverse = true;
            }

            let borrow = (loss != Loss::ExactlyZero) as Limb;
            if reverse {
                // The code above is intended to ensure that no borrow is necessary.
                assert_eq!(sub(b_sig, a_sig, borrow), 0);
                a_sig.copy_from_slice(b_sig);
                *a_sign = !*a_sign;
            } else {
                // The code above is intended to ensure that no borrow is necessary.
                assert_eq!(sub(a_sig, b_sig, borrow), 0);
            }

            // Invert the lost fraction - it was on the RHS and subtracted.
            match loss {
                Loss::LessThanHalf => Loss::MoreThanHalf,
                Loss::MoreThanHalf => Loss::LessThanHalf,
                _ => loss,
            }
        } else {
            let loss = if bits > 0 {
                shift_right(b_sig, &mut 0, bits as usize)
            } else {
                shift_right(a_sig, a_exp, -bits as usize)
            };
            // We have a guard bit; generating a carry cannot happen.
            assert_eq!(add(a_sig, b_sig, 0), 0);
            loss
        }
    }

    /// [LOW, HIGH] = A * B.
    ///
    /// This cannot overflow, because
    ///
    /// (n - 1) * (n - 1) + 2 (n - 1) = (n - 1) * (n + 1)
    ///
    /// which is less than n^2.
    pub(super) fn widening_mul(a: Limb, b: Limb) -> [Limb; 2] {
        let mut wide = [0, 0];

        if a == 0 || b == 0 {
            return wide;
        }

        const HALF_BITS: usize = LIMB_BITS / 2;

        let select = |limb, i| (limb >> (i * HALF_BITS)) & ((1 << HALF_BITS) - 1);
        for i in 0..2 {
            for j in 0..2 {
                let mut x = [select(a, i) * select(b, j), 0];
                shift_left(&mut x, &mut 0, (i + j) * HALF_BITS);
                assert_eq!(add(&mut wide, &x, 0), 0);
            }
        }

        wide
    }

    /// Multiply (normal) A by B into DST. Returns the lost fraction.
    pub(super) fn mul<'a>(
        dst: &mut [Limb],
        exp: &mut ExpInt,
        mut a: &'a [Limb],
        mut b: &'a [Limb],
        precision: usize,
    ) -> Loss {
        // Put the narrower number on the A for less loops below.
        if a.len() > b.len() {
            mem::swap(&mut a, &mut b);
        }

        for x in &mut dst[..b.len()] {
            *x = 0;
        }

        for i in 0..a.len() {
            let mut carry = 0;
            for j in 0..b.len() {
                let [low, mut high] = widening_mul(a[i], b[j]);

                // Now add carry.
                let (low, overflow) = low.overflowing_add(carry);
                high += overflow as Limb;

                // And now DST[i + j], and store the new low part there.
                let (low, overflow) = low.overflowing_add(dst[i + j]);
                high += overflow as Limb;

                dst[i + j] = low;
                carry = high;
            }
            dst[i + b.len()] = carry;
        }

        // Assume the operands involved in the multiplication are single-precision
        // FP, and the two multiplicants are:
        //     a = a23 . a22 ... a0 * 2^e1
        //     b = b23 . b22 ... b0 * 2^e2
        // the result of multiplication is:
        //     dst = c48 c47 c46 . c45 ... c0 * 2^(e1+e2)
        // Note that there are three significant bits at the left-hand side of the
        // radix point: two for the multiplication, and an overflow bit for the
        // addition (that will always be zero at this point). Move the radix point
        // toward left by two bits, and adjust exponent accordingly.
        *exp += 2;

        // Convert the result having "2 * precision" significant-bits back to the one
        // having "precision" significant-bits. First, move the radix point from
        // poision "2*precision - 1" to "precision - 1". The exponent need to be
        // adjusted by "2*precision - 1" - "precision - 1" = "precision".
        *exp -= precision as ExpInt + 1;

        // In case MSB resides at the left-hand side of radix point, shift the
        // mantissa right by some amount to make sure the MSB reside right before
        // the radix point (i.e. "MSB . rest-significant-bits").
        //
        // Note that the result is not normalized when "omsb < precision". So, the
        // caller needs to call Ieee::normalize() if normalized value is
        // expected.
        let omsb = omsb(dst);
        if omsb <= precision {
            Loss::ExactlyZero
        } else {
            shift_right(dst, exp, omsb - precision)
        }
    }

    /// Divide DIVIDEND by DIVISOR into QUOTIENT. Returns the lost fraction.
    /// Does not preserve DIVIDEND or DIVISOR.
    pub(super) fn div(
        quotient: &mut [Limb],
        exp: &mut ExpInt,
        dividend: &mut [Limb],
        divisor: &mut [Limb],
        precision: usize,
    ) -> Loss {
        // Zero the quotient before setting bits in it.
        for x in &mut quotient[..limbs_for_bits(precision)] {
            *x = 0;
        }

        // Normalize the divisor.
        let bits = precision - omsb(divisor);
        shift_left(divisor, &mut 0, bits);
        *exp += bits as ExpInt;

        // Normalize the dividend.
        let bits = precision - omsb(dividend);
        shift_left(dividend, exp, bits);

        // Ensure the dividend >= divisor initially for the loop below.
        // Incidentally, this means that the division loop below is
        // guaranteed to set the integer bit to one.
        if cmp(dividend, divisor) == Ordering::Less {
            shift_left(dividend, exp, 1);
            assert_ne!(cmp(dividend, divisor), Ordering::Less)
        }

        // Long division.
        for bit in (0..precision).rev() {
            if cmp(dividend, divisor) != Ordering::Less {
                sub(dividend, divisor, 0);
                set_bit(quotient, bit);
            }
            shift_left(dividend, &mut 0, 1);
        }

        // Figure out the lost fraction.
        match cmp(dividend, divisor) {
            Ordering::Greater => Loss::MoreThanHalf,
            Ordering::Equal => Loss::ExactlyHalf,
            Ordering::Less => {
                if is_zero(dividend) {
                    Loss::ExactlyZero
                } else {
                    Loss::LessThanHalf
                }
            }
        }
    }
}

#[derive(Debug)]
pub struct IeeePair<S: IeeeSemantics>(Ieee<S>, Ieee<S>);

impl<S: IeeeSemantics> Copy for IeeePair<S> {}
impl<S: IeeeSemantics> Clone for IeeePair<S> {
    fn clone(&self) -> Self {
        *self
    }
}

#[derive(Debug)]
pub struct IeeePairLegacySemantics<S: IeeeSemantics>(S);
pub type IeeePairLegacy<S> = Ieee<IeeePairLegacySemantics<S>>;

pub type PpcDoubleDouble = IeeePair<ieee_semantics::IeeeDouble>;
pub type PpcDoubleDoubleLegacy = IeeePairLegacy<ieee_semantics::IeeeDouble>;

// These are legacy semantics for the fallback, inaccrurate implementation of
// IBM double-double, if the accurate PpcDoubleDouble doesn't handle the
// operation. It's equivalent to having an IEEE number with consecutive 106
// bits of mantissa and 11 bits of exponent.
//
// It's not equivalent to IBM double-double. For example, a legit IBM
// double-double, 1 + epsilon:
//
//   1 + epsilon = 1 + (1 >> 1076)
//
// is not representable by a consecutive 106 bits of mantissa.
//
// Currently, these semantics are used in the following way:
//
//   PpcDoubleDouble -> (IeeeDouble, IeeeDouble) ->
//   (64 bits, 64 bits) -> (128 bits) ->
//   PpcDoubleDoubleLegacy -> IEEE operations
//
// We use to_bits() to get the bit representation of the
// underlying IeeeDouble, then construct the legacy IEEE float.
//
// FIXME: Implement all operations in PpcDoubleDouble, and delete these
// semantics.
impl<S: IeeeSemantics> IeeeSemantics for IeeePairLegacySemantics<S> {
    const EXPONENT_BITS: usize = S::EXPONENT_BITS;
    const MIN_EXPONENT: ExpInt = S::MIN_EXPONENT + (S::PRECISION as ExpInt);
    const PRECISION: usize = S::PRECISION * 2;
    const BITS: usize = S::BITS * 2;

    fn from_bits(bits: u128) -> Ieee<Self> {
        let mut loses_info = false;

        // Get the first double and convert to our format.
        let a = Ieee::<S>::from_bits(bits & ((1 << S::BITS) - 1));
        let (a, fs) = a.convert(Round::NearestTiesToEven, &mut loses_info);
        assert!(fs == OpStatus::OK && !loses_info);

        // Unless we have a special case, add in second double.
        if a.is_finite_non_zero() {
            let b = Ieee::<S>::from_bits(bits >> S::BITS);
            let (b, fs) = b.convert(Round::NearestTiesToEven, &mut loses_info);
            assert!(fs == OpStatus::OK && !loses_info);

            a + b
        } else {
            a
        }
    }

    fn to_bits(x: Ieee<Self>) -> u128 {
        #[derive(Debug)]
        struct Extended<S: IeeeSemantics>(S);

        // Convert number to double. To avoid spurious underflows, we re-
        // normalize against the "double" MIN_EXPONENT first, and only *then*
        // truncate the mantissa. The result of that second conversion
        // may be inexact, but should never underflow.
        impl<S: IeeeSemantics> IeeeSemantics for Extended<S> {
            const EXPONENT_BITS: usize = S::EXPONENT_BITS;
            const PRECISION: usize = S::PRECISION * 2;
        }

        let mut loses_info = false;
        let (extended, fs) = x.convert::<Extended<S>>(Round::NearestTiesToEven, &mut loses_info);
        assert!(fs == OpStatus::OK && !loses_info);

        let (u, fs) = extended.convert(Round::NearestTiesToEven, &mut loses_info);
        assert!(fs == OpStatus::OK || fs == OpStatus::INEXACT);
        let a = Ieee::<S>::to_bits(u);

        // If conversion was exact or resulted in a special case, we're done;
        // just set the second double to zero. Otherwise, re-convert back to
        // the extended format and compute the difference. This now should
        // convert exactly to double.
        let b = if u.is_finite_non_zero() && loses_info {
            let (u, fs) = u.convert::<Extended<S>>(Round::NearestTiesToEven, &mut loses_info);
            assert!(fs == OpStatus::OK && !loses_info);

            let v = extended - u;
            let (v, fs) = v.convert(Round::NearestTiesToEven, &mut loses_info);
            assert!(fs == OpStatus::OK && !loses_info);
            Ieee::<S>::to_bits(v)
        } else {
            0
        };

        a | (b << S::BITS)
    }
}

proxy_impls!([S: IeeeSemantics] IeeePair<S>);

impl<S: IeeeSemantics> PartialOrd for IeeePair<S> {
    fn partial_cmp(&self, rhs: &Self) -> Option<Ordering> {
        let result = self.0.partial_cmp(&rhs.0);
        if result == Some(Ordering::Equal) {
            return self.1.partial_cmp(&rhs.1);
        }
        result
    }
}

impl<S: IeeeSemantics> fmt::Display for IeeePair<S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&IeeePairLegacy::<S>::from_bits(self.to_bits()), f)
    }
}

impl<S: IeeeSemantics> Float for IeeePair<S> {
    fn zero() -> Self {
        IeeePair(Ieee::zero(), Ieee::zero())
    }

    fn inf() -> Self {
        IeeePair(Ieee::inf(), Ieee::zero())
    }

    fn qnan(payload: Option<u128>) -> Self {
        IeeePair(Ieee::qnan(payload), Ieee::zero())
    }

    fn snan(payload: Option<u128>) -> Self {
        IeeePair(Ieee::snan(payload), Ieee::zero())
    }

    fn largest() -> Self {
        let mut r = IeeePair(Ieee::largest(), Ieee::largest());
        r.1.exp -= S::PRECISION as ExpInt + 1;
        r.1.sig[0] -= 1;
        r
    }

    fn smallest() -> Self {
        IeeePair(Ieee::smallest(), Ieee::zero())
    }

    fn smallest_normalized() -> Self {
        let mut r = IeeePair(Ieee::smallest_normalized(), Ieee::zero());
        r.0.exp += S::PRECISION as ExpInt;
        r
    }

    // Implement addition, subtraction, multiplication and division based on:
    // "Software for Doubled-Precision Floating-Point Computations",
    // by Seppo Linnainmaa, ACM TOMS vol 7 no 3, September 1981, pages 272-283.

    fn add_rounded(&mut self, rhs: Self, round: Round) -> OpStatus {
        match (self.category(), rhs.category()) {
            (Category::Infinity, Category::Infinity) => {
                if self.is_negative() != rhs.is_negative() {
                    *self = Self::nan().copy_sign(*self);
                    OpStatus::INVALID_OP
                } else {
                    OpStatus::OK
                }
            }

            (_, Category::Zero) |
            (Category::NaN, _) |
            (Category::Infinity, Category::Normal) => OpStatus::OK,

            (Category::Zero, _) |
            (_, Category::NaN) |
            (_, Category::Infinity) => {
                *self = rhs;
                OpStatus::OK
            }

            (Category::Normal, Category::Normal) => {
                let mut fs = OpStatus::OK;
                let (a, aa, c, cc) = (self.0, self.1, rhs.0, rhs.1);
                let mut z = a;
                fs |= z.add_rounded(c, round);
                if !z.is_finite() {
                    if !z.is_infinite() {
                        self.0 = z;
                        self.1 = Ieee::zero();
                        return fs;
                    }
                    fs = OpStatus::OK;
                    let a_cmp_c = a.cmp_abs_normal(c);
                    z = cc;
                    fs |= z.add_rounded(aa, round);
                    if a_cmp_c == Ordering::Greater {
                        // z = cc + aa + c + a;
                        fs |= z.add_rounded(c, round);
                        fs |= z.add_rounded(a, round);
                    } else {
                        // z = cc + aa + a + c;
                        fs |= z.add_rounded(a, round);
                        fs |= z.add_rounded(c, round);
                    }
                    if !z.is_finite() {
                        self.0 = z;
                        self.1 = Ieee::zero();
                        return fs;
                    }
                    self.0 = z;
                    let mut zz = aa;
                    fs |= zz.add_rounded(cc, round);
                    if a_cmp_c == Ordering::Greater {
                        // self.1 = a - z + c + zz;
                        self.1 = a;
                        fs |= self.1.sub_rounded(z, round);
                        fs |= self.1.add_rounded(c, round);
                        fs |= self.1.add_rounded(zz, round);
                    } else {
                        // self.1 = c - z + a + zz;
                        self.1 = c;
                        fs |= self.1.sub_rounded(z, round);
                        fs |= self.1.add_rounded(a, round);
                        fs |= self.1.add_rounded(zz, round);
                    }
                } else {
                    // q = a - z;
                    let mut q = a;
                    fs |= q.sub_rounded(z, round);

                    // zz = q + c + (a - (q + z)) + aa + cc;
                    // Compute a - (q + z) as -((q + z) - a) to avoid temporary copies.
                    let mut zz = q;
                    fs |= zz.add_rounded(c, round);
                    fs |= q.add_rounded(z, round);
                    fs |= q.sub_rounded(a, round);
                    q.change_sign();
                    fs |= zz.add_rounded(q, round);
                    fs |= zz.add_rounded(aa, round);
                    fs |= zz.add_rounded(cc, round);
                    if zz.is_zero() && !zz.is_negative() {
                        self.0 = z;
                        self.1 = Ieee::zero();
                        return OpStatus::OK;
                    }
                    self.0 = z;
                    fs |= self.0.add_rounded(zz, round);
                    if !self.0.is_finite() {
                        self.1 = Ieee::zero();
                        return fs;
                    }
                    self.1 = z;
                    fs |= self.1.sub_rounded(self.0, round);
                    fs |= self.1.add_rounded(zz, round);
                }
                fs
            }
        }
    }

    fn mul_rounded(&mut self, rhs: Self, round: Round) -> OpStatus {
        // Interesting observation: For special categories, finding the lowest
        // common ancestor of the following layered graph gives the correct
        // return category:
        //
        //    NaN
        //   /   \
        // Zero  Inf
        //   \   /
        //   Normal
        //
        // e.g. NaN * NaN = NaN
        //      Zero * Inf = NaN
        //      Normal * Zero = Zero
        //      Normal * Inf = Inf
        match (self.category(), rhs.category()) {
            (Category::NaN, _) => OpStatus::OK,

            (_, Category::NaN) => {
                *self = rhs;
                OpStatus::OK
            }

            (Category::Zero, Category::Infinity) |
            (Category::Infinity, Category::Zero) => {
                *self = Self::nan();
                OpStatus::OK
            }

            (Category::Zero, _) |
            (Category::Infinity, _) => OpStatus::OK,

            (_, Category::Zero) |
            (_, Category::Infinity) => {
                *self = rhs;
                OpStatus::OK
            }

            (Category::Normal, Category::Normal) => {
                let mut fs = OpStatus::OK;
                let (a, b, c, d) = (self.0, self.1, rhs.0, rhs.1);
                // t = a * c
                let mut t = a;
                fs |= t.mul_rounded(c, round);
                if !t.is_finite_non_zero() {
                    self.0 = t;
                    self.1 = Ieee::zero();
                    return fs;
                }

                // tau = fmsub(a, c, t), that is -fmadd(-a, c, t).
                let mut tau = a;
                t.change_sign();
                fs |= tau.fused_mul_add(c, t, round);
                t.change_sign();
                // v = a * d
                let mut v = a;
                fs |= v.mul_rounded(d, round);
                // w = b * c
                let mut w = b;
                fs |= w.mul_rounded(c, round);
                fs |= v.add_rounded(w, round);
                // tau += v + w
                fs |= tau.add_rounded(v, round);
                // u = t + tau
                let mut u = t;
                fs |= u.add_rounded(tau, round);

                self.0 = u;
                if !u.is_finite() {
                    self.1 = Ieee::zero();
                } else {
                    // self.1 = (t - u) + tau
                    fs |= t.sub_rounded(u, round);
                    fs |= t.add_rounded(tau, round);
                    self.1 = t;
                }
                fs
            }
        }
    }

    fn div_rounded(&mut self, rhs: Self, round: Round) -> OpStatus {
        let mut lhs = IeeePairLegacy::<S>::from_bits(self.to_bits());
        let rhs = IeeePairLegacy::<S>::from_bits(rhs.to_bits());
        let fs = lhs.div_rounded(rhs, round);
        *self = Self::from_bits(lhs.to_bits());
        fs
    }

    fn remainder(&mut self, rhs: Self) -> OpStatus {
        let mut lhs = IeeePairLegacy::<S>::from_bits(self.to_bits());
        let rhs = IeeePairLegacy::<S>::from_bits(rhs.to_bits());
        let fs = lhs.remainder(rhs);
        *self = Self::from_bits(lhs.to_bits());
        fs
    }

    fn modulo(&mut self, rhs: Self) -> OpStatus {
        let mut lhs = IeeePairLegacy::<S>::from_bits(self.to_bits());
        let rhs = IeeePairLegacy::<S>::from_bits(rhs.to_bits());
        let fs = lhs.modulo(rhs);
        *self = Self::from_bits(lhs.to_bits());
        fs
    }

    fn fused_mul_add(&mut self, multiplicand: Self, addend: Self, round: Round) -> OpStatus {
        let mut lhs = IeeePairLegacy::<S>::from_bits(self.to_bits());
        let multiplicand = IeeePairLegacy::<S>::from_bits(multiplicand.to_bits());
        let addend = IeeePairLegacy::<S>::from_bits(addend.to_bits());
        let fs = lhs.fused_mul_add(multiplicand, addend, round);
        *self = Self::from_bits(lhs.to_bits());
        fs
    }

    fn round_to_integral(self, round: Round) -> (Self, OpStatus) {
        let (r, fs) = IeeePairLegacy::<S>::from_bits(self.to_bits()).round_to_integral(round);
        (Self::from_bits(r.to_bits()), fs)
    }

    fn next_up(&mut self) -> OpStatus {
        let mut lhs = IeeePairLegacy::<S>::from_bits(self.to_bits());
        let fs = lhs.next_up();
        *self = Self::from_bits(lhs.to_bits());
        fs
    }

    fn change_sign(&mut self) {
        self.0.change_sign();
        if self.1.is_finite_non_zero() {
            self.1.change_sign();
        }
    }

    fn from_bits(input: u128) -> Self {
        let (a, b) = (input, input >> S::BITS);
        IeeePair(
            Ieee::from_bits(a & ((1 << S::BITS) - 1)),
            Ieee::from_bits(b & ((1 << S::BITS) - 1)),
        )
    }

    fn from_u128(input: u128, round: Round) -> (Self, OpStatus) {
        let (r, fs) = IeeePairLegacy::<S>::from_u128(input, round);
        (Self::from_bits(r.to_bits()), fs)
    }

    fn from_str_rounded(s: &str, round: Round) -> Result<(Self, OpStatus), ParseError> {
        IeeePairLegacy::<S>::from_str_rounded(s, round).map(
            |(r, fs)| {
                (Self::from_bits(r.to_bits()), fs)
            },
        )
    }

    fn to_bits(self) -> u128 {
        self.0.to_bits() | (self.1.to_bits() << S::BITS)
    }

    fn to_u128(self, width: usize, round: Round, is_exact: &mut bool) -> (u128, OpStatus) {
        IeeePairLegacy::<S>::from_bits(self.to_bits()).to_u128(width, round, is_exact)
    }

    fn cmp_abs_normal(self, rhs: Self) -> Ordering {
        self.0.cmp_abs_normal(rhs.0).then_with(|| {
            let result = self.1.cmp_abs_normal(rhs.1);
            if result != Ordering::Equal {
                let against = self.0.sign ^ self.1.sign;
                let rhs_against = rhs.0.sign ^ rhs.1.sign;
                (!against).cmp(&!rhs_against).then_with(|| if against {
                    result.reverse()
                } else {
                    result
                })
            } else {
                result
            }
        })
    }

    fn bitwise_eq(self, rhs: Self) -> bool {
        self.0.bitwise_eq(rhs.0) && self.1.bitwise_eq(rhs.1)
    }

    fn is_negative(self) -> bool {
        self.0.is_negative()
    }

    fn is_denormal(self) -> bool {
        self.category() == Category::Normal &&
            (self.0.is_denormal() || self.0.is_denormal() ||
          // (double)(Hi + Lo) == Hi defines a normal number.
          (self.0 + self.1).partial_cmp(&self.0) != Some(Ordering::Equal))
    }

    fn is_signaling(self) -> bool {
        self.0.is_signaling()
    }

    fn category(self) -> Category {
        self.0.category()
    }

    fn is_smallest(self) -> bool {
        Self::smallest().copy_sign(self).partial_cmp(&self) == Some(Ordering::Equal)
    }

    fn is_largest(self) -> bool {
        Self::largest().copy_sign(self).partial_cmp(&self) == Some(Ordering::Equal)
    }

    fn is_integer(self) -> bool {
        self.0.is_integer() && self.1.is_integer()
    }

    fn get_exact_inverse(self) -> Option<Self> {
        let r = IeeePairLegacy::<S>::from_bits(self.to_bits()).get_exact_inverse();
        r.map(|r| Self::from_bits(r.to_bits()))
    }

    fn ilogb(self) -> i32 {
        self.0.ilogb()
    }

    fn scalbn(self, exp: i32, round: Round) -> Self {
        IeeePair(self.0.scalbn(exp, round), self.1.scalbn(exp, round))
    }

    fn frexp(self, exp: &mut i32, round: Round) -> Self {
        let a = self.0.frexp(exp, round);
        let mut b = self.1;
        if self.category() == Category::Normal {
            b = b.scalbn(-*exp, round);
        }
        IeeePair(a, b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    impl IeeeSingle {
        fn from_f32(input: f32) -> Self {
            Self::from_bits(input.to_bits() as u128)
        }

        fn to_f32(self) -> f32 {
            f32::from_bits(self.to_bits() as u32)
        }
    }

    impl IeeeDouble {
        fn from_f64(input: f64) -> Self {
            Self::from_bits(input.to_bits() as u128)
        }

        fn to_f64(self) -> f64 {
            f64::from_bits(self.to_bits() as u64)
        }
    }

    #[test]
    fn is_signaling() {
        // We test qNaN, -qNaN, +sNaN, -sNaN with and without payloads.
        let payload = 4;
        assert!(!IeeeSingle::qnan(None).is_signaling());
        assert!(!(-IeeeSingle::qnan(None)).is_signaling());
        assert!(!IeeeSingle::qnan(Some(payload)).is_signaling());
        assert!(!(-IeeeSingle::qnan(Some(payload))).is_signaling());
        assert!(IeeeSingle::snan(None).is_signaling());
        assert!((-IeeeSingle::snan(None)).is_signaling());
        assert!(IeeeSingle::snan(Some(payload)).is_signaling());
        assert!((-IeeeSingle::snan(Some(payload))).is_signaling());
    }

    #[test]
    fn next() {
        // 1. Test Special Cases Values.
        //
        // Test all special values for nextUp and nextDown perscribed by IEEE-754R
        // 2008. These are:
        //   1. +inf
        //   2. -inf
        //   3. largest
        //   4. -largest
        //   5. smallest
        //   6. -smallest
        //   7. qNaN
        //   8. sNaN
        //   9. +0
        //   10. -0

        // nextUp(+inf) = +inf.
        let mut test = IeeeQuad::inf();
        let expected = IeeeQuad::inf();
        assert_eq!(test.next_up(), OpStatus::OK);
        assert!(test.is_infinite());
        assert!(!test.is_negative());
        assert!(test.bitwise_eq(expected));

        // nextDown(+inf) = -nextUp(-inf) = -(-largest) = largest
        let mut test = IeeeQuad::inf();
        let expected = IeeeQuad::largest();
        assert_eq!(test.next_down(), OpStatus::OK);
        assert!(!test.is_negative());
        assert!(test.bitwise_eq(expected));

        // nextUp(-inf) = -largest
        let mut test = -IeeeQuad::inf();
        let expected = -IeeeQuad::largest();
        assert_eq!(test.next_up(), OpStatus::OK);
        assert!(test.is_negative());
        assert!(test.bitwise_eq(expected));

        // nextDown(-inf) = -nextUp(+inf) = -(+inf) = -inf.
        let mut test = -IeeeQuad::inf();
        let expected = -IeeeQuad::inf();
        assert_eq!(test.next_down(), OpStatus::OK);
        assert!(test.is_infinite() && test.is_negative());
        assert!(test.bitwise_eq(expected));

        // nextUp(largest) = +inf
        let mut test = IeeeQuad::largest();
        let expected = IeeeQuad::inf();
        assert_eq!(test.next_up(), OpStatus::OK);
        assert!(test.is_infinite() && !test.is_negative());
        assert!(test.bitwise_eq(expected));

        // nextDown(largest) = -nextUp(-largest)
        //                        = -(-largest + inc)
        //                        = largest - inc.
        let mut test = IeeeQuad::largest();
        let expected = "0x1.fffffffffffffffffffffffffffep+16383"
            .parse::<IeeeQuad>()
            .unwrap();
        assert_eq!(test.next_down(), OpStatus::OK);
        assert!(!test.is_infinite() && !test.is_negative());
        assert!(test.bitwise_eq(expected));

        // nextUp(-largest) = -largest + inc.
        let mut test = -IeeeQuad::largest();
        let expected = "-0x1.fffffffffffffffffffffffffffep+16383"
            .parse::<IeeeQuad>()
            .unwrap();
        assert_eq!(test.next_up(), OpStatus::OK);
        assert!(test.bitwise_eq(expected));

        // nextDown(-largest) = -nextUp(largest) = -(inf) = -inf.
        let mut test = -IeeeQuad::largest();
        let expected = -IeeeQuad::inf();
        assert_eq!(test.next_down(), OpStatus::OK);
        assert!(test.is_infinite() && test.is_negative());
        assert!(test.bitwise_eq(expected));

        // nextUp(smallest) = smallest + inc.
        let mut test = "0x0.0000000000000000000000000001p-16382"
            .parse::<IeeeQuad>()
            .unwrap();
        let expected = "0x0.0000000000000000000000000002p-16382"
            .parse::<IeeeQuad>()
            .unwrap();
        assert_eq!(test.next_up(), OpStatus::OK);
        assert!(test.bitwise_eq(expected));

        // nextDown(smallest) = -nextUp(-smallest) = -(-0) = +0.
        let mut test = "0x0.0000000000000000000000000001p-16382"
            .parse::<IeeeQuad>()
            .unwrap();
        let expected = IeeeQuad::zero();
        assert_eq!(test.next_down(), OpStatus::OK);
        assert!(test.is_pos_zero());
        assert!(test.bitwise_eq(expected));

        // nextUp(-smallest) = -0.
        let mut test = "-0x0.0000000000000000000000000001p-16382"
            .parse::<IeeeQuad>()
            .unwrap();
        let expected = -IeeeQuad::zero();
        assert_eq!(test.next_up(), OpStatus::OK);
        assert!(test.is_neg_zero());
        assert!(test.bitwise_eq(expected));

        // nextDown(-smallest) = -nextUp(smallest) = -smallest - inc.
        let mut test = "-0x0.0000000000000000000000000001p-16382"
            .parse::<IeeeQuad>()
            .unwrap();
        let expected = "-0x0.0000000000000000000000000002p-16382"
            .parse::<IeeeQuad>()
            .unwrap();
        assert_eq!(test.next_down(), OpStatus::OK);
        assert!(test.bitwise_eq(expected));

        // nextUp(qNaN) = qNaN
        let mut test = IeeeQuad::qnan(None);
        let expected = IeeeQuad::qnan(None);
        assert_eq!(test.next_up(), OpStatus::OK);
        assert!(test.bitwise_eq(expected));

        // nextDown(qNaN) = qNaN
        let mut test = IeeeQuad::qnan(None);
        let expected = IeeeQuad::qnan(None);
        assert_eq!(test.next_down(), OpStatus::OK);
        assert!(test.bitwise_eq(expected));

        // nextUp(sNaN) = qNaN
        let mut test = IeeeQuad::snan(None);
        let expected = IeeeQuad::qnan(None);
        assert_eq!(test.next_up(), OpStatus::INVALID_OP);
        assert!(test.bitwise_eq(expected));

        // nextDown(sNaN) = qNaN
        let mut test = IeeeQuad::snan(None);
        let expected = IeeeQuad::qnan(None);
        assert_eq!(test.next_down(), OpStatus::INVALID_OP);
        assert!(test.bitwise_eq(expected));

        // nextUp(+0) = +smallest
        let mut test = IeeeQuad::zero();
        let expected = IeeeQuad::smallest();
        assert_eq!(test.next_up(), OpStatus::OK);
        assert!(test.bitwise_eq(expected));

        // nextDown(+0) = -nextUp(-0) = -smallest
        let mut test = IeeeQuad::zero();
        let expected = -IeeeQuad::smallest();
        assert_eq!(test.next_down(), OpStatus::OK);
        assert!(test.bitwise_eq(expected));

        // nextUp(-0) = +smallest
        let mut test = -IeeeQuad::zero();
        let expected = IeeeQuad::smallest();
        assert_eq!(test.next_up(), OpStatus::OK);
        assert!(test.bitwise_eq(expected));

        // nextDown(-0) = -nextUp(0) = -smallest
        let mut test = -IeeeQuad::zero();
        let expected = -IeeeQuad::smallest();
        assert_eq!(test.next_down(), OpStatus::OK);
        assert!(test.bitwise_eq(expected));

        // 2. Binade Boundary Tests.

        // 2a. Test denormal <-> normal binade boundaries.
        //     * nextUp(+Largest Denormal) -> +Smallest Normal.
        //     * nextDown(-Largest Denormal) -> -Smallest Normal.
        //     * nextUp(-Smallest Normal) -> -Largest Denormal.
        //     * nextDown(+Smallest Normal) -> +Largest Denormal.

        // nextUp(+Largest Denormal) -> +Smallest Normal.
        let mut test = "0x0.ffffffffffffffffffffffffffffp-16382"
            .parse::<IeeeQuad>()
            .unwrap();
        let expected = "0x1.0000000000000000000000000000p-16382"
            .parse::<IeeeQuad>()
            .unwrap();
        assert_eq!(test.next_up(), OpStatus::OK);
        assert!(!test.is_denormal());
        assert!(test.bitwise_eq(expected));

        // nextDown(-Largest Denormal) -> -Smallest Normal.
        let mut test = "-0x0.ffffffffffffffffffffffffffffp-16382"
            .parse::<IeeeQuad>()
            .unwrap();
        let expected = "-0x1.0000000000000000000000000000p-16382"
            .parse::<IeeeQuad>()
            .unwrap();
        assert_eq!(test.next_down(), OpStatus::OK);
        assert!(!test.is_denormal());
        assert!(test.bitwise_eq(expected));

        // nextUp(-Smallest Normal) -> -Largest Denormal.
        let mut test = "-0x1.0000000000000000000000000000p-16382"
            .parse::<IeeeQuad>()
            .unwrap();
        let expected = "-0x0.ffffffffffffffffffffffffffffp-16382"
            .parse::<IeeeQuad>()
            .unwrap();
        assert_eq!(test.next_up(), OpStatus::OK);
        assert!(test.is_denormal());
        assert!(test.bitwise_eq(expected));

        // nextDown(+Smallest Normal) -> +Largest Denormal.
        let mut test = "+0x1.0000000000000000000000000000p-16382"
            .parse::<IeeeQuad>()
            .unwrap();
        let expected = "+0x0.ffffffffffffffffffffffffffffp-16382"
            .parse::<IeeeQuad>()
            .unwrap();
        assert_eq!(test.next_down(), OpStatus::OK);
        assert!(test.is_denormal());
        assert!(test.bitwise_eq(expected));

        // 2b. Test normal <-> normal binade boundaries.
        //     * nextUp(-Normal Binade Boundary) -> -Normal Binade Boundary + 1.
        //     * nextDown(+Normal Binade Boundary) -> +Normal Binade Boundary - 1.
        //     * nextUp(+Normal Binade Boundary - 1) -> +Normal Binade Boundary.
        //     * nextDown(-Normal Binade Boundary + 1) -> -Normal Binade Boundary.

        // nextUp(-Normal Binade Boundary) -> -Normal Binade Boundary + 1.
        let mut test = "-0x1p+1".parse::<IeeeQuad>().unwrap();
        let expected = "-0x1.ffffffffffffffffffffffffffffp+0"
            .parse::<IeeeQuad>()
            .unwrap();
        assert_eq!(test.next_up(), OpStatus::OK);
        assert!(test.bitwise_eq(expected));

        // nextDown(+Normal Binade Boundary) -> +Normal Binade Boundary - 1.
        let mut test = "0x1p+1".parse::<IeeeQuad>().unwrap();
        let expected = "0x1.ffffffffffffffffffffffffffffp+0"
            .parse::<IeeeQuad>()
            .unwrap();
        assert_eq!(test.next_down(), OpStatus::OK);
        assert!(test.bitwise_eq(expected));

        // nextUp(+Normal Binade Boundary - 1) -> +Normal Binade Boundary.
        let mut test = "0x1.ffffffffffffffffffffffffffffp+0"
            .parse::<IeeeQuad>()
            .unwrap();
        let expected = "0x1p+1".parse::<IeeeQuad>().unwrap();
        assert_eq!(test.next_up(), OpStatus::OK);
        assert!(test.bitwise_eq(expected));

        // nextDown(-Normal Binade Boundary + 1) -> -Normal Binade Boundary.
        let mut test = "-0x1.ffffffffffffffffffffffffffffp+0"
            .parse::<IeeeQuad>()
            .unwrap();
        let expected = "-0x1p+1".parse::<IeeeQuad>().unwrap();
        assert_eq!(test.next_down(), OpStatus::OK);
        assert!(test.bitwise_eq(expected));

        // 2c. Test using next at binade boundaries with a direction away from the
        // binade boundary. Away from denormal <-> normal boundaries.
        //
        // This is to make sure that even though we are at a binade boundary, since
        // we are rounding away, we do not trigger the binade boundary code. Thus we
        // test:
        //   * nextUp(-Largest Denormal) -> -Largest Denormal + inc.
        //   * nextDown(+Largest Denormal) -> +Largest Denormal - inc.
        //   * nextUp(+Smallest Normal) -> +Smallest Normal + inc.
        //   * nextDown(-Smallest Normal) -> -Smallest Normal - inc.

        // nextUp(-Largest Denormal) -> -Largest Denormal + inc.
        let mut test = "-0x0.ffffffffffffffffffffffffffffp-16382"
            .parse::<IeeeQuad>()
            .unwrap();
        let expected = "-0x0.fffffffffffffffffffffffffffep-16382"
            .parse::<IeeeQuad>()
            .unwrap();
        assert_eq!(test.next_up(), OpStatus::OK);
        assert!(test.is_denormal());
        assert!(test.is_negative());
        assert!(test.bitwise_eq(expected));

        // nextDown(+Largest Denormal) -> +Largest Denormal - inc.
        let mut test = "0x0.ffffffffffffffffffffffffffffp-16382"
            .parse::<IeeeQuad>()
            .unwrap();
        let expected = "0x0.fffffffffffffffffffffffffffep-16382"
            .parse::<IeeeQuad>()
            .unwrap();
        assert_eq!(test.next_down(), OpStatus::OK);
        assert!(test.is_denormal());
        assert!(!test.is_negative());
        assert!(test.bitwise_eq(expected));

        // nextUp(+Smallest Normal) -> +Smallest Normal + inc.
        let mut test = "0x1.0000000000000000000000000000p-16382"
            .parse::<IeeeQuad>()
            .unwrap();
        let expected = "0x1.0000000000000000000000000001p-16382"
            .parse::<IeeeQuad>()
            .unwrap();
        assert_eq!(test.next_up(), OpStatus::OK);
        assert!(!test.is_denormal());
        assert!(!test.is_negative());
        assert!(test.bitwise_eq(expected));

        // nextDown(-Smallest Normal) -> -Smallest Normal - inc.
        let mut test = "-0x1.0000000000000000000000000000p-16382"
            .parse::<IeeeQuad>()
            .unwrap();
        let expected = "-0x1.0000000000000000000000000001p-16382"
            .parse::<IeeeQuad>()
            .unwrap();
        assert_eq!(test.next_down(), OpStatus::OK);
        assert!(!test.is_denormal());
        assert!(test.is_negative());
        assert!(test.bitwise_eq(expected));

        // 2d. Test values which cause our exponent to go to min exponent. This
        // is to ensure that guards in the code to check for min exponent
        // trigger properly.
        //     * nextUp(-0x1p-16381) -> -0x1.ffffffffffffffffffffffffffffp-16382
        //     * nextDown(-0x1.ffffffffffffffffffffffffffffp-16382) ->
        //         -0x1p-16381
        //     * nextUp(0x1.ffffffffffffffffffffffffffffp-16382) -> 0x1p-16382
        //     * nextDown(0x1p-16382) -> 0x1.ffffffffffffffffffffffffffffp-16382

        // nextUp(-0x1p-16381) -> -0x1.ffffffffffffffffffffffffffffp-16382
        let mut test = "-0x1p-16381".parse::<IeeeQuad>().unwrap();
        let expected = "-0x1.ffffffffffffffffffffffffffffp-16382"
            .parse::<IeeeQuad>()
            .unwrap();
        assert_eq!(test.next_up(), OpStatus::OK);
        assert!(test.bitwise_eq(expected));

        // nextDown(-0x1.ffffffffffffffffffffffffffffp-16382) ->
        //         -0x1p-16381
        let mut test = "-0x1.ffffffffffffffffffffffffffffp-16382"
            .parse::<IeeeQuad>()
            .unwrap();
        let expected = "-0x1p-16381".parse::<IeeeQuad>().unwrap();
        assert_eq!(test.next_down(), OpStatus::OK);
        assert!(test.bitwise_eq(expected));

        // nextUp(0x1.ffffffffffffffffffffffffffffp-16382) -> 0x1p-16381
        let mut test = "0x1.ffffffffffffffffffffffffffffp-16382"
            .parse::<IeeeQuad>()
            .unwrap();
        let expected = "0x1p-16381".parse::<IeeeQuad>().unwrap();
        assert_eq!(test.next_up(), OpStatus::OK);
        assert!(test.bitwise_eq(expected));

        // nextDown(0x1p-16381) -> 0x1.ffffffffffffffffffffffffffffp-16382
        let mut test = "0x1p-16381".parse::<IeeeQuad>().unwrap();
        let expected = "0x1.ffffffffffffffffffffffffffffp-16382"
            .parse::<IeeeQuad>()
            .unwrap();
        assert_eq!(test.next_down(), OpStatus::OK);
        assert!(test.bitwise_eq(expected));

        // 3. Now we test both denormal/normal computation which will not cause us
        // to go across binade boundaries. Specifically we test:
        //   * nextUp(+Denormal) -> +Denormal.
        //   * nextDown(+Denormal) -> +Denormal.
        //   * nextUp(-Denormal) -> -Denormal.
        //   * nextDown(-Denormal) -> -Denormal.
        //   * nextUp(+Normal) -> +Normal.
        //   * nextDown(+Normal) -> +Normal.
        //   * nextUp(-Normal) -> -Normal.
        //   * nextDown(-Normal) -> -Normal.

        // nextUp(+Denormal) -> +Denormal.
        let mut test = "0x0.ffffffffffffffffffffffff000cp-16382"
            .parse::<IeeeQuad>()
            .unwrap();
        let expected = "0x0.ffffffffffffffffffffffff000dp-16382"
            .parse::<IeeeQuad>()
            .unwrap();
        assert_eq!(test.next_up(), OpStatus::OK);
        assert!(test.is_denormal());
        assert!(!test.is_negative());
        assert!(test.bitwise_eq(expected));

        // nextDown(+Denormal) -> +Denormal.
        let mut test = "0x0.ffffffffffffffffffffffff000cp-16382"
            .parse::<IeeeQuad>()
            .unwrap();
        let expected = "0x0.ffffffffffffffffffffffff000bp-16382"
            .parse::<IeeeQuad>()
            .unwrap();
        assert_eq!(test.next_down(), OpStatus::OK);
        assert!(test.is_denormal());
        assert!(!test.is_negative());
        assert!(test.bitwise_eq(expected));

        // nextUp(-Denormal) -> -Denormal.
        let mut test = "-0x0.ffffffffffffffffffffffff000cp-16382"
            .parse::<IeeeQuad>()
            .unwrap();
        let expected = "-0x0.ffffffffffffffffffffffff000bp-16382"
            .parse::<IeeeQuad>()
            .unwrap();
        assert_eq!(test.next_up(), OpStatus::OK);
        assert!(test.is_denormal());
        assert!(test.is_negative());
        assert!(test.bitwise_eq(expected));

        // nextDown(-Denormal) -> -Denormal
        let mut test = "-0x0.ffffffffffffffffffffffff000cp-16382"
            .parse::<IeeeQuad>()
            .unwrap();
        let expected = "-0x0.ffffffffffffffffffffffff000dp-16382"
            .parse::<IeeeQuad>()
            .unwrap();
        assert_eq!(test.next_down(), OpStatus::OK);
        assert!(test.is_denormal());
        assert!(test.is_negative());
        assert!(test.bitwise_eq(expected));

        // nextUp(+Normal) -> +Normal.
        let mut test = "0x1.ffffffffffffffffffffffff000cp-16000"
            .parse::<IeeeQuad>()
            .unwrap();
        let expected = "0x1.ffffffffffffffffffffffff000dp-16000"
            .parse::<IeeeQuad>()
            .unwrap();
        assert_eq!(test.next_up(), OpStatus::OK);
        assert!(!test.is_denormal());
        assert!(!test.is_negative());
        assert!(test.bitwise_eq(expected));

        // nextDown(+Normal) -> +Normal.
        let mut test = "0x1.ffffffffffffffffffffffff000cp-16000"
            .parse::<IeeeQuad>()
            .unwrap();
        let expected = "0x1.ffffffffffffffffffffffff000bp-16000"
            .parse::<IeeeQuad>()
            .unwrap();
        assert_eq!(test.next_down(), OpStatus::OK);
        assert!(!test.is_denormal());
        assert!(!test.is_negative());
        assert!(test.bitwise_eq(expected));

        // nextUp(-Normal) -> -Normal.
        let mut test = "-0x1.ffffffffffffffffffffffff000cp-16000"
            .parse::<IeeeQuad>()
            .unwrap();
        let expected = "-0x1.ffffffffffffffffffffffff000bp-16000"
            .parse::<IeeeQuad>()
            .unwrap();
        assert_eq!(test.next_up(), OpStatus::OK);
        assert!(!test.is_denormal());
        assert!(test.is_negative());
        assert!(test.bitwise_eq(expected));

        // nextDown(-Normal) -> -Normal.
        let mut test = "-0x1.ffffffffffffffffffffffff000cp-16000"
            .parse::<IeeeQuad>()
            .unwrap();
        let expected = "-0x1.ffffffffffffffffffffffff000dp-16000"
            .parse::<IeeeQuad>()
            .unwrap();
        assert_eq!(test.next_down(), OpStatus::OK);
        assert!(!test.is_denormal());
        assert!(test.is_negative());
        assert!(test.bitwise_eq(expected));
    }

    #[test]
    fn fma() {
        let rdmd = Round::NearestTiesToEven;

        {
            let mut f1 = IeeeSingle::from_f32(14.5);
            let f2 = IeeeSingle::from_f32(-14.5);
            let f3 = IeeeSingle::from_f32(225.0);
            let _: OpStatus = f1.fused_mul_add(f2, f3, Round::NearestTiesToEven);
            assert_eq!(14.75, f1.to_f32());
        }

        {
            let val2 = IeeeSingle::from_f32(2.0);
            let mut f1 = IeeeSingle::from_f32(1.17549435e-38);
            let mut f2 = IeeeSingle::from_f32(1.17549435e-38);
            let _: OpStatus = f1.div_rounded(val2, rdmd);
            let _: OpStatus = f2.div_rounded(val2, rdmd);
            let f3 = IeeeSingle::from_f32(12.0);
            let _: OpStatus = f1.fused_mul_add(f2, f3, Round::NearestTiesToEven);
            assert_eq!(12.0, f1.to_f32());
        }

        // Test for correct zero sign when answer is exactly zero.
        // fma(1.0, -1.0, 1.0) -> +ve 0.
        {
            let mut f1 = IeeeDouble::from_f64(1.0);
            let f2 = IeeeDouble::from_f64(-1.0);
            let f3 = IeeeDouble::from_f64(1.0);
            let _: OpStatus = f1.fused_mul_add(f2, f3, Round::NearestTiesToEven);
            assert!(!f1.is_negative() && f1.is_zero());
        }

        // Test for correct zero sign when answer is exactly zero and rounding towards
        // negative.
        // fma(1.0, -1.0, 1.0) -> +ve 0.
        {
            let mut f1 = IeeeDouble::from_f64(1.0);
            let f2 = IeeeDouble::from_f64(-1.0);
            let f3 = IeeeDouble::from_f64(1.0);
            let _: OpStatus = f1.fused_mul_add(f2, f3, Round::TowardNegative);
            assert!(f1.is_negative() && f1.is_zero());
        }

        // Test for correct (in this case -ve) sign when adding like signed zeros.
        // Test fma(0.0, -0.0, -0.0) -> -ve 0.
        {
            let mut f1 = IeeeDouble::from_f64(0.0);
            let f2 = IeeeDouble::from_f64(-0.0);
            let f3 = IeeeDouble::from_f64(-0.0);
            let _: OpStatus = f1.fused_mul_add(f2, f3, Round::NearestTiesToEven);
            assert!(f1.is_negative() && f1.is_zero());
        }

        // Test -ve sign preservation when small negative results underflow.
        {
            let mut f1 = "-0x1p-1074".parse::<IeeeDouble>().unwrap();
            let f2 = "+0x1p-1074".parse::<IeeeDouble>().unwrap();
            let f3 = IeeeDouble::from_f64(0.0);
            let _: OpStatus = f1.fused_mul_add(f2, f3, Round::NearestTiesToEven);
            assert!(f1.is_negative() && f1.is_zero());
        }

        // Test x87 extended precision case from http://llvm.org/PR20728.
        {
            let mut m1 = X87DoubleExtended::from_u128(1, Round::NearestTiesToEven).0;
            let m2 = X87DoubleExtended::from_u128(1, Round::NearestTiesToEven).0;
            let a = X87DoubleExtended::from_u128(3, Round::NearestTiesToEven).0;

            let mut loses_info = false;
            let _: OpStatus = m1.fused_mul_add(m2, a, Round::NearestTiesToEven);
            let r: IeeeSingle = m1.convert(Round::NearestTiesToEven, &mut loses_info).0;
            assert!(!loses_info);
            assert_eq!(4.0, r.to_f32());
        }
    }

    #[test]
    fn min_num() {
        let f1 = IeeeDouble::from_f64(1.0);
        let f2 = IeeeDouble::from_f64(2.0);
        let nan = IeeeDouble::nan();

        assert_eq!(1.0, f1.min(f2).to_f64());
        assert_eq!(1.0, f2.min(f1).to_f64());
        assert_eq!(1.0, f1.min(nan).to_f64());
        assert_eq!(1.0, nan.min(f1).to_f64());
    }

    #[test]
    fn max_num() {
        let f1 = IeeeDouble::from_f64(1.0);
        let f2 = IeeeDouble::from_f64(2.0);
        let nan = IeeeDouble::nan();

        assert_eq!(2.0, f1.max(f2).to_f64());
        assert_eq!(2.0, f2.max(f1).to_f64());
        assert_eq!(1.0, f1.max(nan).to_f64());
        assert_eq!(1.0, nan.max(f1).to_f64());
    }

    #[test]
    fn denormal() {
        let round = Round::NearestTiesToEven;

        // Test single precision
        {
            assert!(!IeeeSingle::from_f32(0.0).is_denormal());

            let mut t = "1.17549435082228750797e-38".parse::<IeeeSingle>().unwrap();
            assert!(!t.is_denormal());

            let val2 = IeeeSingle::from_f32(2.0e0);
            let _: OpStatus = t.div_rounded(val2, round);
            assert!(t.is_denormal());
        }

        // Test double precision
        {
            assert!(!IeeeDouble::from_f64(0.0).is_denormal());

            let mut t = "2.22507385850720138309e-308".parse::<IeeeDouble>().unwrap();
            assert!(!t.is_denormal());

            let val2 = IeeeDouble::from_f64(2.0e0);
            let _: OpStatus = t.div_rounded(val2, round);
            assert!(t.is_denormal());
        }

        // Test Intel double-ext
        {
            assert!(!X87DoubleExtended::from_u128(0, Round::NearestTiesToEven)
                .0
                .is_denormal());

            let mut t = "3.36210314311209350626e-4932"
                .parse::<X87DoubleExtended>()
                .unwrap();
            assert!(!t.is_denormal());

            let val2 = X87DoubleExtended::from_u128(2, Round::NearestTiesToEven).0;
            let _: OpStatus = t.div_rounded(val2, round);
            assert!(t.is_denormal());
        }

        // Test quadruple precision
        {
            assert!(!IeeeQuad::from_u128(0, Round::NearestTiesToEven)
                .0
                .is_denormal());

            let mut t = "3.36210314311209350626267781732175260e-4932"
                .parse::<IeeeQuad>()
                .unwrap();
            assert!(!t.is_denormal());

            let val2 = IeeeQuad::from_u128(2, Round::NearestTiesToEven).0;
            let _: OpStatus = t.div_rounded(val2, round);
            assert!(t.is_denormal());
        }
    }

    #[test]
    fn decimal_strings_without_null_terminators() {
        // Make sure that we can parse strings without null terminators.
        // rdar://14323230.
        let val = "0.00"[..3].parse::<IeeeDouble>().unwrap();
        assert_eq!(val.to_f64(), 0.0);
        let val = "0.01"[..3].parse::<IeeeDouble>().unwrap();
        assert_eq!(val.to_f64(), 0.0);
        let val = "0.09"[..3].parse::<IeeeDouble>().unwrap();
        assert_eq!(val.to_f64(), 0.0);
        let val = "0.095"[..4].parse::<IeeeDouble>().unwrap();
        assert_eq!(val.to_f64(), 0.09);
        let val = "0.00e+3"[..7].parse::<IeeeDouble>().unwrap();
        assert_eq!(val.to_f64(), 0.00);
        let val = "0e+3"[..4].parse::<IeeeDouble>().unwrap();
        assert_eq!(val.to_f64(), 0.00);

    }

    #[test]
    fn from_zero_decimal_string() {
        assert_eq!(0.0, "0".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(0.0, "+0".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(-0.0, "-0".parse::<IeeeDouble>().unwrap().to_f64());

        assert_eq!(0.0, "0.".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(0.0, "+0.".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(-0.0, "-0.".parse::<IeeeDouble>().unwrap().to_f64());

        assert_eq!(0.0, ".0".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(0.0, "+.0".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(-0.0, "-.0".parse::<IeeeDouble>().unwrap().to_f64());

        assert_eq!(0.0, "0.0".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(0.0, "+0.0".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(-0.0, "-0.0".parse::<IeeeDouble>().unwrap().to_f64());

        assert_eq!(0.0, "00000.".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(0.0, "+00000.".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(-0.0, "-00000.".parse::<IeeeDouble>().unwrap().to_f64());

        assert_eq!(0.0, ".00000".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(0.0, "+.00000".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(-0.0, "-.00000".parse::<IeeeDouble>().unwrap().to_f64());

        assert_eq!(0.0, "0000.00000".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(0.0, "+0000.00000".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(-0.0, "-0000.00000".parse::<IeeeDouble>().unwrap().to_f64());
    }

    #[test]
    fn from_zero_decimal_single_exponent_string() {
        assert_eq!(0.0, "0e1".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(0.0, "+0e1".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(-0.0, "-0e1".parse::<IeeeDouble>().unwrap().to_f64());

        assert_eq!(0.0, "0e+1".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(0.0, "+0e+1".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(-0.0, "-0e+1".parse::<IeeeDouble>().unwrap().to_f64());

        assert_eq!(0.0, "0e-1".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(0.0, "+0e-1".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(-0.0, "-0e-1".parse::<IeeeDouble>().unwrap().to_f64());


        assert_eq!(0.0, "0.e1".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(0.0, "+0.e1".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(-0.0, "-0.e1".parse::<IeeeDouble>().unwrap().to_f64());

        assert_eq!(0.0, "0.e+1".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(0.0, "+0.e+1".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(-0.0, "-0.e+1".parse::<IeeeDouble>().unwrap().to_f64());

        assert_eq!(0.0, "0.e-1".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(0.0, "+0.e-1".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(-0.0, "-0.e-1".parse::<IeeeDouble>().unwrap().to_f64());

        assert_eq!(0.0, ".0e1".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(0.0, "+.0e1".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(-0.0, "-.0e1".parse::<IeeeDouble>().unwrap().to_f64());

        assert_eq!(0.0, ".0e+1".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(0.0, "+.0e+1".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(-0.0, "-.0e+1".parse::<IeeeDouble>().unwrap().to_f64());

        assert_eq!(0.0, ".0e-1".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(0.0, "+.0e-1".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(-0.0, "-.0e-1".parse::<IeeeDouble>().unwrap().to_f64());


        assert_eq!(0.0, "0.0e1".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(0.0, "+0.0e1".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(-0.0, "-0.0e1".parse::<IeeeDouble>().unwrap().to_f64());

        assert_eq!(0.0, "0.0e+1".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(0.0, "+0.0e+1".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(-0.0, "-0.0e+1".parse::<IeeeDouble>().unwrap().to_f64());

        assert_eq!(0.0, "0.0e-1".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(0.0, "+0.0e-1".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(-0.0, "-0.0e-1".parse::<IeeeDouble>().unwrap().to_f64());


        assert_eq!(0.0, "000.0000e1".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(0.0, "+000.0000e+1".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(-0.0, "-000.0000e+1".parse::<IeeeDouble>().unwrap().to_f64());
    }

    #[test]
    fn from_zero_decimal_large_exponent_string() {
        assert_eq!(0.0, "0e1234".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(0.0, "+0e1234".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(-0.0, "-0e1234".parse::<IeeeDouble>().unwrap().to_f64());

        assert_eq!(0.0, "0e+1234".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(0.0, "+0e+1234".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(-0.0, "-0e+1234".parse::<IeeeDouble>().unwrap().to_f64());

        assert_eq!(0.0, "0e-1234".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(0.0, "+0e-1234".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(-0.0, "-0e-1234".parse::<IeeeDouble>().unwrap().to_f64());

        assert_eq!(0.0, "000.0000e1234".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(
            0.0,
            "000.0000e-1234".parse::<IeeeDouble>().unwrap().to_f64()
        );
    }

    #[test]
    fn from_zero_hexadecimal_string() {
        assert_eq!(0.0, "0x0p1".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(0.0, "+0x0p1".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(-0.0, "-0x0p1".parse::<IeeeDouble>().unwrap().to_f64());

        assert_eq!(0.0, "0x0p+1".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(0.0, "+0x0p+1".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(-0.0, "-0x0p+1".parse::<IeeeDouble>().unwrap().to_f64());

        assert_eq!(0.0, "0x0p-1".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(0.0, "+0x0p-1".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(-0.0, "-0x0p-1".parse::<IeeeDouble>().unwrap().to_f64());


        assert_eq!(0.0, "0x0.p1".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(0.0, "+0x0.p1".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(-0.0, "-0x0.p1".parse::<IeeeDouble>().unwrap().to_f64());

        assert_eq!(0.0, "0x0.p+1".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(0.0, "+0x0.p+1".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(-0.0, "-0x0.p+1".parse::<IeeeDouble>().unwrap().to_f64());

        assert_eq!(0.0, "0x0.p-1".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(0.0, "+0x0.p-1".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(-0.0, "-0x0.p-1".parse::<IeeeDouble>().unwrap().to_f64());


        assert_eq!(0.0, "0x.0p1".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(0.0, "+0x.0p1".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(-0.0, "-0x.0p1".parse::<IeeeDouble>().unwrap().to_f64());

        assert_eq!(0.0, "0x.0p+1".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(0.0, "+0x.0p+1".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(-0.0, "-0x.0p+1".parse::<IeeeDouble>().unwrap().to_f64());

        assert_eq!(0.0, "0x.0p-1".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(0.0, "+0x.0p-1".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(-0.0, "-0x.0p-1".parse::<IeeeDouble>().unwrap().to_f64());


        assert_eq!(0.0, "0x0.0p1".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(0.0, "+0x0.0p1".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(-0.0, "-0x0.0p1".parse::<IeeeDouble>().unwrap().to_f64());

        assert_eq!(0.0, "0x0.0p+1".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(0.0, "+0x0.0p+1".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(-0.0, "-0x0.0p+1".parse::<IeeeDouble>().unwrap().to_f64());

        assert_eq!(0.0, "0x0.0p-1".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(0.0, "+0x0.0p-1".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(-0.0, "-0x0.0p-1".parse::<IeeeDouble>().unwrap().to_f64());


        assert_eq!(0.0, "0x00000.p1".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(
            0.0,
            "0x0000.00000p1".parse::<IeeeDouble>().unwrap().to_f64()
        );
        assert_eq!(0.0, "0x.00000p1".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(0.0, "0x0.p1".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(0.0, "0x0p1234".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(-0.0, "-0x0p1234".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(0.0, "0x00000.p1234".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(
            0.0,
            "0x0000.00000p1234".parse::<IeeeDouble>().unwrap().to_f64()
        );
        assert_eq!(0.0, "0x.00000p1234".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(0.0, "0x0.p1234".parse::<IeeeDouble>().unwrap().to_f64());
    }

    #[test]
    fn from_decimal_string() {
        assert_eq!(1.0, "1".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(2.0, "2.".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(0.5, ".5".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(1.0, "1.0".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(-2.0, "-2".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(-4.0, "-4.".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(-0.5, "-.5".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(-1.5, "-1.5".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(1.25e12, "1.25e12".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(1.25e+12, "1.25e+12".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(1.25e-12, "1.25e-12".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(1024.0, "1024.".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(
            1024.05,
            "1024.05000".parse::<IeeeDouble>().unwrap().to_f64()
        );
        assert_eq!(0.05, ".05000".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(2.0, "2.".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(2.0e2, "2.e2".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(2.0e+2, "2.e+2".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(2.0e-2, "2.e-2".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(
            2.05e2,
            "002.05000e2".parse::<IeeeDouble>().unwrap().to_f64()
        );
        assert_eq!(
            2.05e+2,
            "002.05000e+2".parse::<IeeeDouble>().unwrap().to_f64()
        );
        assert_eq!(
            2.05e-2,
            "002.05000e-2".parse::<IeeeDouble>().unwrap().to_f64()
        );
        assert_eq!(
            2.05e12,
            "002.05000e12".parse::<IeeeDouble>().unwrap().to_f64()
        );
        assert_eq!(
            2.05e+12,
            "002.05000e+12".parse::<IeeeDouble>().unwrap().to_f64()
        );
        assert_eq!(
            2.05e-12,
            "002.05000e-12".parse::<IeeeDouble>().unwrap().to_f64()
        );

        // These are "carefully selected" to overflow the fast log-base
        // calculations in the implementation.
        assert!("99e99999".parse::<IeeeDouble>().unwrap().is_infinite());
        assert!("-99e99999".parse::<IeeeDouble>().unwrap().is_infinite());
        assert!("1e-99999".parse::<IeeeDouble>().unwrap().is_pos_zero());
        assert!("-1e-99999".parse::<IeeeDouble>().unwrap().is_neg_zero());

        assert_eq!(2.71828, "2.71828".parse::<IeeeDouble>().unwrap().to_f64());
    }

    #[test]
    fn from_hexadecimal_string() {
        assert_eq!(1.0, "0x1p0".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(1.0, "+0x1p0".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(-1.0, "-0x1p0".parse::<IeeeDouble>().unwrap().to_f64());

        assert_eq!(1.0, "0x1p+0".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(1.0, "+0x1p+0".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(-1.0, "-0x1p+0".parse::<IeeeDouble>().unwrap().to_f64());

        assert_eq!(1.0, "0x1p-0".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(1.0, "+0x1p-0".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(-1.0, "-0x1p-0".parse::<IeeeDouble>().unwrap().to_f64());


        assert_eq!(2.0, "0x1p1".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(2.0, "+0x1p1".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(-2.0, "-0x1p1".parse::<IeeeDouble>().unwrap().to_f64());

        assert_eq!(2.0, "0x1p+1".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(2.0, "+0x1p+1".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(-2.0, "-0x1p+1".parse::<IeeeDouble>().unwrap().to_f64());

        assert_eq!(0.5, "0x1p-1".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(0.5, "+0x1p-1".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(-0.5, "-0x1p-1".parse::<IeeeDouble>().unwrap().to_f64());


        assert_eq!(3.0, "0x1.8p1".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(3.0, "+0x1.8p1".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(-3.0, "-0x1.8p1".parse::<IeeeDouble>().unwrap().to_f64());

        assert_eq!(3.0, "0x1.8p+1".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(3.0, "+0x1.8p+1".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(-3.0, "-0x1.8p+1".parse::<IeeeDouble>().unwrap().to_f64());

        assert_eq!(0.75, "0x1.8p-1".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(0.75, "+0x1.8p-1".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(-0.75, "-0x1.8p-1".parse::<IeeeDouble>().unwrap().to_f64());


        assert_eq!(
            8192.0,
            "0x1000.000p1".parse::<IeeeDouble>().unwrap().to_f64()
        );
        assert_eq!(
            8192.0,
            "+0x1000.000p1".parse::<IeeeDouble>().unwrap().to_f64()
        );
        assert_eq!(
            -8192.0,
            "-0x1000.000p1".parse::<IeeeDouble>().unwrap().to_f64()
        );

        assert_eq!(
            8192.0,
            "0x1000.000p+1".parse::<IeeeDouble>().unwrap().to_f64()
        );
        assert_eq!(
            8192.0,
            "+0x1000.000p+1".parse::<IeeeDouble>().unwrap().to_f64()
        );
        assert_eq!(
            -8192.0,
            "-0x1000.000p+1".parse::<IeeeDouble>().unwrap().to_f64()
        );

        assert_eq!(
            2048.0,
            "0x1000.000p-1".parse::<IeeeDouble>().unwrap().to_f64()
        );
        assert_eq!(
            2048.0,
            "+0x1000.000p-1".parse::<IeeeDouble>().unwrap().to_f64()
        );
        assert_eq!(
            -2048.0,
            "-0x1000.000p-1".parse::<IeeeDouble>().unwrap().to_f64()
        );


        assert_eq!(8192.0, "0x1000p1".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(8192.0, "+0x1000p1".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(-8192.0, "-0x1000p1".parse::<IeeeDouble>().unwrap().to_f64());

        assert_eq!(8192.0, "0x1000p+1".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(8192.0, "+0x1000p+1".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(
            -8192.0,
            "-0x1000p+1".parse::<IeeeDouble>().unwrap().to_f64()
        );

        assert_eq!(2048.0, "0x1000p-1".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(2048.0, "+0x1000p-1".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(
            -2048.0,
            "-0x1000p-1".parse::<IeeeDouble>().unwrap().to_f64()
        );


        assert_eq!(16384.0, "0x10p10".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(16384.0, "+0x10p10".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(-16384.0, "-0x10p10".parse::<IeeeDouble>().unwrap().to_f64());

        assert_eq!(16384.0, "0x10p+10".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(16384.0, "+0x10p+10".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(
            -16384.0,
            "-0x10p+10".parse::<IeeeDouble>().unwrap().to_f64()
        );

        assert_eq!(0.015625, "0x10p-10".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(
            0.015625,
            "+0x10p-10".parse::<IeeeDouble>().unwrap().to_f64()
        );
        assert_eq!(
            -0.015625,
            "-0x10p-10".parse::<IeeeDouble>().unwrap().to_f64()
        );

        assert_eq!(1.0625, "0x1.1p0".parse::<IeeeDouble>().unwrap().to_f64());
        assert_eq!(1.0, "0x1p0".parse::<IeeeDouble>().unwrap().to_f64());

        assert_eq!(
            "0x1p-150".parse::<IeeeDouble>().unwrap().to_f64(),
            "+0x800000000000000001.p-221"
                .parse::<IeeeDouble>()
                .unwrap()
                .to_f64()
        );
        assert_eq!(
            2251799813685248.5,
            "0x80000000000004000000.010p-28"
                .parse::<IeeeDouble>()
                .unwrap()
                .to_f64()
        );
    }

    #[test]
    fn to_string() {
        let to_string = |d: f64, precision: usize, width: usize| {
            let x = IeeeDouble::from_f64(d);
            if precision == 0 {
                format!("{:1$}", x, width)
            } else {
                format!("{:2$.1$}", x, precision, width)
            }
        };
        assert_eq!("10", to_string(10.0, 6, 3));
        assert_eq!("1.0E+1", to_string(10.0, 6, 0));
        assert_eq!("10100", to_string(1.01E+4, 5, 2));
        assert_eq!("1.01E+4", to_string(1.01E+4, 4, 2));
        assert_eq!("1.01E+4", to_string(1.01E+4, 5, 1));
        assert_eq!("0.0101", to_string(1.01E-2, 5, 2));
        assert_eq!("0.0101", to_string(1.01E-2, 4, 2));
        assert_eq!("1.01E-2", to_string(1.01E-2, 5, 1));
        assert_eq!(
            "0.78539816339744828",
            to_string(0.78539816339744830961, 0, 3)
        );
        assert_eq!(
            "4.9406564584124654E-324",
            to_string(4.9406564584124654e-324, 0, 3)
        );
        assert_eq!("873.18340000000001", to_string(873.1834, 0, 1));
        assert_eq!("8.7318340000000001E+2", to_string(873.1834, 0, 0));
        assert_eq!(
            "1.7976931348623157E+308",
            to_string(1.7976931348623157E+308, 0, 0)
        );

        let to_string = |d: f64, precision: usize, width: usize| {
            let x = IeeeDouble::from_f64(d);
            if precision == 0 {
                format!("{:#1$}", x, width)
            } else {
                format!("{:#2$.1$}", x, precision, width)
            }
        };
        assert_eq!("10", to_string(10.0, 6, 3));
        assert_eq!("1.000000e+01", to_string(10.0, 6, 0));
        assert_eq!("10100", to_string(1.01E+4, 5, 2));
        assert_eq!("1.0100e+04", to_string(1.01E+4, 4, 2));
        assert_eq!("1.01000e+04", to_string(1.01E+4, 5, 1));
        assert_eq!("0.0101", to_string(1.01E-2, 5, 2));
        assert_eq!("0.0101", to_string(1.01E-2, 4, 2));
        assert_eq!("1.01000e-02", to_string(1.01E-2, 5, 1));
        assert_eq!(
            "0.78539816339744828",
            to_string(0.78539816339744830961, 0, 3)
        );
        assert_eq!(
            "4.94065645841246540e-324",
            to_string(4.9406564584124654e-324, 0, 3)
        );
        assert_eq!("873.18340000000001", to_string(873.1834, 0, 1));
        assert_eq!("8.73183400000000010e+02", to_string(873.1834, 0, 0));
        assert_eq!(
            "1.79769313486231570e+308",
            to_string(1.7976931348623157E+308, 0, 0)
        );
    }

    #[test]
    fn to_integer() {
        let mut is_exact = false;

        assert_eq!(
            (10, OpStatus::OK),
            "10".parse::<IeeeDouble>().unwrap().to_u128(
                5,
                Round::TowardZero,
                &mut is_exact,
            )
        );
        assert!(is_exact);

        assert_eq!(
            (0, OpStatus::INVALID_OP),
            "-10".parse::<IeeeDouble>().unwrap().to_u128(
                5,
                Round::TowardZero,
                &mut is_exact,
            )
        );
        assert!(!is_exact);

        assert_eq!(
            (31, OpStatus::INVALID_OP),
            "32".parse::<IeeeDouble>().unwrap().to_u128(
                5,
                Round::TowardZero,
                &mut is_exact,
            )
        );
        assert!(!is_exact);

        assert_eq!(
            (7, OpStatus::INEXACT),
            "7.9".parse::<IeeeDouble>().unwrap().to_u128(
                5,
                Round::TowardZero,
                &mut is_exact,
            )
        );
        assert!(!is_exact);

        assert_eq!(
            (-10, OpStatus::OK),
            "-10".parse::<IeeeDouble>().unwrap().to_i128(
                5,
                Round::TowardZero,
                &mut is_exact,
            )
        );
        assert!(is_exact);

        assert_eq!(
            (-16, OpStatus::INVALID_OP),
            "-17".parse::<IeeeDouble>().unwrap().to_i128(
                5,
                Round::TowardZero,
                &mut is_exact,
            )
        );
        assert!(!is_exact);

        assert_eq!(
            (15, OpStatus::INVALID_OP),
            "16".parse::<IeeeDouble>().unwrap().to_i128(
                5,
                Round::TowardZero,
                &mut is_exact,
            )
        );
        assert!(!is_exact);
    }

    #[test]
    fn nan() {
        fn nanbits<T: Float>(signaling: bool, negative: bool, fill: u128) -> u128 {
            let mut x = if signaling {
                T::snan(Some(fill))
            } else {
                T::qnan(Some(fill))
            };
            if negative {
                x.change_sign();
            }
            x.to_bits()
        }

        assert_eq!(0x7fc00000, nanbits::<IeeeSingle>(false, false, 0));
        assert_eq!(0xffc00000, nanbits::<IeeeSingle>(false, true, 0));
        assert_eq!(0x7fc0ae72, nanbits::<IeeeSingle>(false, false, 0xae72));
        assert_eq!(0x7fffae72, nanbits::<IeeeSingle>(false, false, 0xffffae72));
        assert_eq!(0x7fa00000, nanbits::<IeeeSingle>(true, false, 0));
        assert_eq!(0xffa00000, nanbits::<IeeeSingle>(true, true, 0));
        assert_eq!(0x7f80ae72, nanbits::<IeeeSingle>(true, false, 0xae72));
        assert_eq!(0x7fbfae72, nanbits::<IeeeSingle>(true, false, 0xffffae72));

        assert_eq!(0x7ff8000000000000, nanbits::<IeeeDouble>(false, false, 0));
        assert_eq!(0xfff8000000000000, nanbits::<IeeeDouble>(false, true, 0));
        assert_eq!(
            0x7ff800000000ae72,
            nanbits::<IeeeDouble>(false, false, 0xae72)
        );
        assert_eq!(
            0x7fffffffffffae72,
            nanbits::<IeeeDouble>(false, false, 0xffffffffffffae72)
        );
        assert_eq!(0x7ff4000000000000, nanbits::<IeeeDouble>(true, false, 0));
        assert_eq!(0xfff4000000000000, nanbits::<IeeeDouble>(true, true, 0));
        assert_eq!(
            0x7ff000000000ae72,
            nanbits::<IeeeDouble>(true, false, 0xae72)
        );
        assert_eq!(
            0x7ff7ffffffffae72,
            nanbits::<IeeeDouble>(true, false, 0xffffffffffffae72)
        );
    }

    #[test]
    fn string_decimal_death() {
        assert_eq!(
            "".parse::<IeeeDouble>(),
            Err(ParseError("Invalid string length"))
        );
        assert_eq!(
            "+".parse::<IeeeDouble>(),
            Err(ParseError("String has no digits"))
        );
        assert_eq!(
            "-".parse::<IeeeDouble>(),
            Err(ParseError("String has no digits"))
        );

        assert_eq!(
            "\0".parse::<IeeeDouble>(),
            Err(ParseError("Invalid character in significand"))
        );
        assert_eq!(
            "1\0".parse::<IeeeDouble>(),
            Err(ParseError("Invalid character in significand"))
        );
        assert_eq!(
            "1\02".parse::<IeeeDouble>(),
            Err(ParseError("Invalid character in significand"))
        );
        assert_eq!(
            "1\02e1".parse::<IeeeDouble>(),
            Err(ParseError("Invalid character in significand"))
        );
        assert_eq!(
            "1e\0".parse::<IeeeDouble>(),
            Err(ParseError("Invalid character in exponent"))
        );
        assert_eq!(
            "1e1\0".parse::<IeeeDouble>(),
            Err(ParseError("Invalid character in exponent"))
        );
        assert_eq!(
            "1e1\02".parse::<IeeeDouble>(),
            Err(ParseError("Invalid character in exponent"))
        );

        assert_eq!(
            "1.0f".parse::<IeeeDouble>(),
            Err(ParseError("Invalid character in significand"))
        );

        assert_eq!(
            "..".parse::<IeeeDouble>(),
            Err(ParseError("String contains multiple dots"))
        );
        assert_eq!(
            "..0".parse::<IeeeDouble>(),
            Err(ParseError("String contains multiple dots"))
        );
        assert_eq!(
            "1.0.0".parse::<IeeeDouble>(),
            Err(ParseError("String contains multiple dots"))
        );
    }

    #[test]
    fn string_decimal_significand_death() {
        assert_eq!(
            ".".parse::<IeeeDouble>(),
            Err(ParseError("Significand has no digits"))
        );
        assert_eq!(
            "+.".parse::<IeeeDouble>(),
            Err(ParseError("Significand has no digits"))
        );
        assert_eq!(
            "-.".parse::<IeeeDouble>(),
            Err(ParseError("Significand has no digits"))
        );


        assert_eq!(
            "e".parse::<IeeeDouble>(),
            Err(ParseError("Significand has no digits"))
        );
        assert_eq!(
            "+e".parse::<IeeeDouble>(),
            Err(ParseError("Significand has no digits"))
        );
        assert_eq!(
            "-e".parse::<IeeeDouble>(),
            Err(ParseError("Significand has no digits"))
        );

        assert_eq!(
            "e1".parse::<IeeeDouble>(),
            Err(ParseError("Significand has no digits"))
        );
        assert_eq!(
            "+e1".parse::<IeeeDouble>(),
            Err(ParseError("Significand has no digits"))
        );
        assert_eq!(
            "-e1".parse::<IeeeDouble>(),
            Err(ParseError("Significand has no digits"))
        );

        assert_eq!(
            ".e1".parse::<IeeeDouble>(),
            Err(ParseError("Significand has no digits"))
        );
        assert_eq!(
            "+.e1".parse::<IeeeDouble>(),
            Err(ParseError("Significand has no digits"))
        );
        assert_eq!(
            "-.e1".parse::<IeeeDouble>(),
            Err(ParseError("Significand has no digits"))
        );


        assert_eq!(
            ".e".parse::<IeeeDouble>(),
            Err(ParseError("Significand has no digits"))
        );
        assert_eq!(
            "+.e".parse::<IeeeDouble>(),
            Err(ParseError("Significand has no digits"))
        );
        assert_eq!(
            "-.e".parse::<IeeeDouble>(),
            Err(ParseError("Significand has no digits"))
        );
    }

    #[test]
    fn string_decimal_exponent_death() {
        assert_eq!(
            "1e".parse::<IeeeDouble>(),
            Err(ParseError("Exponent has no digits"))
        );
        assert_eq!(
            "+1e".parse::<IeeeDouble>(),
            Err(ParseError("Exponent has no digits"))
        );
        assert_eq!(
            "-1e".parse::<IeeeDouble>(),
            Err(ParseError("Exponent has no digits"))
        );

        assert_eq!(
            "1.e".parse::<IeeeDouble>(),
            Err(ParseError("Exponent has no digits"))
        );
        assert_eq!(
            "+1.e".parse::<IeeeDouble>(),
            Err(ParseError("Exponent has no digits"))
        );
        assert_eq!(
            "-1.e".parse::<IeeeDouble>(),
            Err(ParseError("Exponent has no digits"))
        );

        assert_eq!(
            ".1e".parse::<IeeeDouble>(),
            Err(ParseError("Exponent has no digits"))
        );
        assert_eq!(
            "+.1e".parse::<IeeeDouble>(),
            Err(ParseError("Exponent has no digits"))
        );
        assert_eq!(
            "-.1e".parse::<IeeeDouble>(),
            Err(ParseError("Exponent has no digits"))
        );

        assert_eq!(
            "1.1e".parse::<IeeeDouble>(),
            Err(ParseError("Exponent has no digits"))
        );
        assert_eq!(
            "+1.1e".parse::<IeeeDouble>(),
            Err(ParseError("Exponent has no digits"))
        );
        assert_eq!(
            "-1.1e".parse::<IeeeDouble>(),
            Err(ParseError("Exponent has no digits"))
        );


        assert_eq!(
            "1e+".parse::<IeeeDouble>(),
            Err(ParseError("Exponent has no digits"))
        );
        assert_eq!(
            "1e-".parse::<IeeeDouble>(),
            Err(ParseError("Exponent has no digits"))
        );

        assert_eq!(
            ".1e".parse::<IeeeDouble>(),
            Err(ParseError("Exponent has no digits"))
        );
        assert_eq!(
            ".1e+".parse::<IeeeDouble>(),
            Err(ParseError("Exponent has no digits"))
        );
        assert_eq!(
            ".1e-".parse::<IeeeDouble>(),
            Err(ParseError("Exponent has no digits"))
        );

        assert_eq!(
            "1.0e".parse::<IeeeDouble>(),
            Err(ParseError("Exponent has no digits"))
        );
        assert_eq!(
            "1.0e+".parse::<IeeeDouble>(),
            Err(ParseError("Exponent has no digits"))
        );
        assert_eq!(
            "1.0e-".parse::<IeeeDouble>(),
            Err(ParseError("Exponent has no digits"))
        );
    }

    #[test]
    fn string_hexadecimal_death() {
        assert_eq!(
            "0x".parse::<IeeeDouble>(),
            Err(ParseError("Invalid string"))
        );
        assert_eq!(
            "+0x".parse::<IeeeDouble>(),
            Err(ParseError("Invalid string"))
        );
        assert_eq!(
            "-0x".parse::<IeeeDouble>(),
            Err(ParseError("Invalid string"))
        );

        assert_eq!(
            "0x0".parse::<IeeeDouble>(),
            Err(ParseError("Hex strings require an exponent"))
        );
        assert_eq!(
            "+0x0".parse::<IeeeDouble>(),
            Err(ParseError("Hex strings require an exponent"))
        );
        assert_eq!(
            "-0x0".parse::<IeeeDouble>(),
            Err(ParseError("Hex strings require an exponent"))
        );

        assert_eq!(
            "0x0.".parse::<IeeeDouble>(),
            Err(ParseError("Hex strings require an exponent"))
        );
        assert_eq!(
            "+0x0.".parse::<IeeeDouble>(),
            Err(ParseError("Hex strings require an exponent"))
        );
        assert_eq!(
            "-0x0.".parse::<IeeeDouble>(),
            Err(ParseError("Hex strings require an exponent"))
        );

        assert_eq!(
            "0x.0".parse::<IeeeDouble>(),
            Err(ParseError("Hex strings require an exponent"))
        );
        assert_eq!(
            "+0x.0".parse::<IeeeDouble>(),
            Err(ParseError("Hex strings require an exponent"))
        );
        assert_eq!(
            "-0x.0".parse::<IeeeDouble>(),
            Err(ParseError("Hex strings require an exponent"))
        );

        assert_eq!(
            "0x0.0".parse::<IeeeDouble>(),
            Err(ParseError("Hex strings require an exponent"))
        );
        assert_eq!(
            "+0x0.0".parse::<IeeeDouble>(),
            Err(ParseError("Hex strings require an exponent"))
        );
        assert_eq!(
            "-0x0.0".parse::<IeeeDouble>(),
            Err(ParseError("Hex strings require an exponent"))
        );

        assert_eq!(
            "0x\0".parse::<IeeeDouble>(),
            Err(ParseError("Invalid character in significand"))
        );
        assert_eq!(
            "0x1\0".parse::<IeeeDouble>(),
            Err(ParseError("Invalid character in significand"))
        );
        assert_eq!(
            "0x1\02".parse::<IeeeDouble>(),
            Err(ParseError("Invalid character in significand"))
        );
        assert_eq!(
            "0x1\02p1".parse::<IeeeDouble>(),
            Err(ParseError("Invalid character in significand"))
        );
        assert_eq!(
            "0x1p\0".parse::<IeeeDouble>(),
            Err(ParseError("Invalid character in exponent"))
        );
        assert_eq!(
            "0x1p1\0".parse::<IeeeDouble>(),
            Err(ParseError("Invalid character in exponent"))
        );
        assert_eq!(
            "0x1p1\02".parse::<IeeeDouble>(),
            Err(ParseError("Invalid character in exponent"))
        );

        assert_eq!(
            "0x1p0f".parse::<IeeeDouble>(),
            Err(ParseError("Invalid character in exponent"))
        );

        assert_eq!(
            "0x..p1".parse::<IeeeDouble>(),
            Err(ParseError("String contains multiple dots"))
        );
        assert_eq!(
            "0x..0p1".parse::<IeeeDouble>(),
            Err(ParseError("String contains multiple dots"))
        );
        assert_eq!(
            "0x1.0.0p1".parse::<IeeeDouble>(),
            Err(ParseError("String contains multiple dots"))
        );
    }

    #[test]
    fn string_hexadecimal_significand_death() {
        assert_eq!(
            "0x.".parse::<IeeeDouble>(),
            Err(ParseError("Significand has no digits"))
        );
        assert_eq!(
            "+0x.".parse::<IeeeDouble>(),
            Err(ParseError("Significand has no digits"))
        );
        assert_eq!(
            "-0x.".parse::<IeeeDouble>(),
            Err(ParseError("Significand has no digits"))
        );

        assert_eq!(
            "0xp".parse::<IeeeDouble>(),
            Err(ParseError("Significand has no digits"))
        );
        assert_eq!(
            "+0xp".parse::<IeeeDouble>(),
            Err(ParseError("Significand has no digits"))
        );
        assert_eq!(
            "-0xp".parse::<IeeeDouble>(),
            Err(ParseError("Significand has no digits"))
        );

        assert_eq!(
            "0xp+".parse::<IeeeDouble>(),
            Err(ParseError("Significand has no digits"))
        );
        assert_eq!(
            "+0xp+".parse::<IeeeDouble>(),
            Err(ParseError("Significand has no digits"))
        );
        assert_eq!(
            "-0xp+".parse::<IeeeDouble>(),
            Err(ParseError("Significand has no digits"))
        );

        assert_eq!(
            "0xp-".parse::<IeeeDouble>(),
            Err(ParseError("Significand has no digits"))
        );
        assert_eq!(
            "+0xp-".parse::<IeeeDouble>(),
            Err(ParseError("Significand has no digits"))
        );
        assert_eq!(
            "-0xp-".parse::<IeeeDouble>(),
            Err(ParseError("Significand has no digits"))
        );


        assert_eq!(
            "0x.p".parse::<IeeeDouble>(),
            Err(ParseError("Significand has no digits"))
        );
        assert_eq!(
            "+0x.p".parse::<IeeeDouble>(),
            Err(ParseError("Significand has no digits"))
        );
        assert_eq!(
            "-0x.p".parse::<IeeeDouble>(),
            Err(ParseError("Significand has no digits"))
        );

        assert_eq!(
            "0x.p+".parse::<IeeeDouble>(),
            Err(ParseError("Significand has no digits"))
        );
        assert_eq!(
            "+0x.p+".parse::<IeeeDouble>(),
            Err(ParseError("Significand has no digits"))
        );
        assert_eq!(
            "-0x.p+".parse::<IeeeDouble>(),
            Err(ParseError("Significand has no digits"))
        );

        assert_eq!(
            "0x.p-".parse::<IeeeDouble>(),
            Err(ParseError("Significand has no digits"))
        );
        assert_eq!(
            "+0x.p-".parse::<IeeeDouble>(),
            Err(ParseError("Significand has no digits"))
        );
        assert_eq!(
            "-0x.p-".parse::<IeeeDouble>(),
            Err(ParseError("Significand has no digits"))
        );
    }

    #[test]
    fn string_hexadecimal_exponent_death() {
        assert_eq!(
            "0x1p".parse::<IeeeDouble>(),
            Err(ParseError("Exponent has no digits"))
        );
        assert_eq!(
            "+0x1p".parse::<IeeeDouble>(),
            Err(ParseError("Exponent has no digits"))
        );
        assert_eq!(
            "-0x1p".parse::<IeeeDouble>(),
            Err(ParseError("Exponent has no digits"))
        );

        assert_eq!(
            "0x1p+".parse::<IeeeDouble>(),
            Err(ParseError("Exponent has no digits"))
        );
        assert_eq!(
            "+0x1p+".parse::<IeeeDouble>(),
            Err(ParseError("Exponent has no digits"))
        );
        assert_eq!(
            "-0x1p+".parse::<IeeeDouble>(),
            Err(ParseError("Exponent has no digits"))
        );

        assert_eq!(
            "0x1p-".parse::<IeeeDouble>(),
            Err(ParseError("Exponent has no digits"))
        );
        assert_eq!(
            "+0x1p-".parse::<IeeeDouble>(),
            Err(ParseError("Exponent has no digits"))
        );
        assert_eq!(
            "-0x1p-".parse::<IeeeDouble>(),
            Err(ParseError("Exponent has no digits"))
        );


        assert_eq!(
            "0x1.p".parse::<IeeeDouble>(),
            Err(ParseError("Exponent has no digits"))
        );
        assert_eq!(
            "+0x1.p".parse::<IeeeDouble>(),
            Err(ParseError("Exponent has no digits"))
        );
        assert_eq!(
            "-0x1.p".parse::<IeeeDouble>(),
            Err(ParseError("Exponent has no digits"))
        );

        assert_eq!(
            "0x1.p+".parse::<IeeeDouble>(),
            Err(ParseError("Exponent has no digits"))
        );
        assert_eq!(
            "+0x1.p+".parse::<IeeeDouble>(),
            Err(ParseError("Exponent has no digits"))
        );
        assert_eq!(
            "-0x1.p+".parse::<IeeeDouble>(),
            Err(ParseError("Exponent has no digits"))
        );

        assert_eq!(
            "0x1.p-".parse::<IeeeDouble>(),
            Err(ParseError("Exponent has no digits"))
        );
        assert_eq!(
            "+0x1.p-".parse::<IeeeDouble>(),
            Err(ParseError("Exponent has no digits"))
        );
        assert_eq!(
            "-0x1.p-".parse::<IeeeDouble>(),
            Err(ParseError("Exponent has no digits"))
        );


        assert_eq!(
            "0x.1p".parse::<IeeeDouble>(),
            Err(ParseError("Exponent has no digits"))
        );
        assert_eq!(
            "+0x.1p".parse::<IeeeDouble>(),
            Err(ParseError("Exponent has no digits"))
        );
        assert_eq!(
            "-0x.1p".parse::<IeeeDouble>(),
            Err(ParseError("Exponent has no digits"))
        );

        assert_eq!(
            "0x.1p+".parse::<IeeeDouble>(),
            Err(ParseError("Exponent has no digits"))
        );
        assert_eq!(
            "+0x.1p+".parse::<IeeeDouble>(),
            Err(ParseError("Exponent has no digits"))
        );
        assert_eq!(
            "-0x.1p+".parse::<IeeeDouble>(),
            Err(ParseError("Exponent has no digits"))
        );

        assert_eq!(
            "0x.1p-".parse::<IeeeDouble>(),
            Err(ParseError("Exponent has no digits"))
        );
        assert_eq!(
            "+0x.1p-".parse::<IeeeDouble>(),
            Err(ParseError("Exponent has no digits"))
        );
        assert_eq!(
            "-0x.1p-".parse::<IeeeDouble>(),
            Err(ParseError("Exponent has no digits"))
        );


        assert_eq!(
            "0x1.1p".parse::<IeeeDouble>(),
            Err(ParseError("Exponent has no digits"))
        );
        assert_eq!(
            "+0x1.1p".parse::<IeeeDouble>(),
            Err(ParseError("Exponent has no digits"))
        );
        assert_eq!(
            "-0x1.1p".parse::<IeeeDouble>(),
            Err(ParseError("Exponent has no digits"))
        );

        assert_eq!(
            "0x1.1p+".parse::<IeeeDouble>(),
            Err(ParseError("Exponent has no digits"))
        );
        assert_eq!(
            "+0x1.1p+".parse::<IeeeDouble>(),
            Err(ParseError("Exponent has no digits"))
        );
        assert_eq!(
            "-0x1.1p+".parse::<IeeeDouble>(),
            Err(ParseError("Exponent has no digits"))
        );

        assert_eq!(
            "0x1.1p-".parse::<IeeeDouble>(),
            Err(ParseError("Exponent has no digits"))
        );
        assert_eq!(
            "+0x1.1p-".parse::<IeeeDouble>(),
            Err(ParseError("Exponent has no digits"))
        );
        assert_eq!(
            "-0x1.1p-".parse::<IeeeDouble>(),
            Err(ParseError("Exponent has no digits"))
        );
    }

    #[test]
    fn exact_inverse() {
        // Trivial operation.
        assert!(
            IeeeDouble::from_f64(2.0)
                .get_exact_inverse()
                .unwrap()
                .bitwise_eq(IeeeDouble::from_f64(0.5))
        );
        assert!(
            IeeeSingle::from_f32(2.0)
                .get_exact_inverse()
                .unwrap()
                .bitwise_eq(IeeeSingle::from_f32(0.5))
        );
        assert!(
            "2.0"
                .parse::<IeeeQuad>()
                .unwrap()
                .get_exact_inverse()
                .unwrap()
                .bitwise_eq("0.5".parse::<IeeeQuad>().unwrap())
        );
        assert!(
            "2.0"
                .parse::<PpcDoubleDouble>()
                .unwrap()
                .get_exact_inverse()
                .unwrap()
                .bitwise_eq("0.5".parse::<PpcDoubleDouble>().unwrap())
        );
        assert!(
            "2.0"
                .parse::<X87DoubleExtended>()
                .unwrap()
                .get_exact_inverse()
                .unwrap()
                .bitwise_eq("0.5".parse::<X87DoubleExtended>().unwrap())
        );

        // FLT_MIN
        assert!(
            IeeeSingle::from_f32(1.17549435e-38)
                .get_exact_inverse()
                .unwrap()
                .bitwise_eq(IeeeSingle::from_f32(8.5070592e+37))
        );

        // Large float, inverse is a denormal.
        assert!(
            IeeeSingle::from_f32(1.7014118e38)
                .get_exact_inverse()
                .is_none()
        );
        // Zero
        assert!(IeeeDouble::from_f64(0.0).get_exact_inverse().is_none());
        // Denormalized float
        assert!(
            IeeeSingle::from_f32(1.40129846e-45)
                .get_exact_inverse()
                .is_none()
        );
    }

    #[test]
    fn round_to_integral() {
        let t = IeeeDouble::from_f64(-0.5);
        assert_eq!(-0.0, t.round_to_integral(Round::TowardZero).0.to_f64());
        assert_eq!(-1.0, t.round_to_integral(Round::TowardNegative).0.to_f64());
        assert_eq!(-0.0, t.round_to_integral(Round::TowardPositive).0.to_f64());
        assert_eq!(
            -0.0,
            t.round_to_integral(Round::NearestTiesToEven).0.to_f64()
        );

        let s = IeeeDouble::from_f64(3.14);
        assert_eq!(3.0, s.round_to_integral(Round::TowardZero).0.to_f64());
        assert_eq!(3.0, s.round_to_integral(Round::TowardNegative).0.to_f64());
        assert_eq!(4.0, s.round_to_integral(Round::TowardPositive).0.to_f64());
        assert_eq!(
            3.0,
            s.round_to_integral(Round::NearestTiesToEven).0.to_f64()
        );

        let r = IeeeDouble::largest();
        assert_eq!(
            r.to_f64(),
            r.round_to_integral(Round::TowardZero).0.to_f64()
        );
        assert_eq!(
            r.to_f64(),
            r.round_to_integral(Round::TowardNegative).0.to_f64()
        );
        assert_eq!(
            r.to_f64(),
            r.round_to_integral(Round::TowardPositive).0.to_f64()
        );
        assert_eq!(
            r.to_f64(),
            r.round_to_integral(Round::NearestTiesToEven).0.to_f64()
        );

        let p = IeeeDouble::zero().round_to_integral(Round::TowardZero).0;
        assert_eq!(0.0, p.to_f64());
        let p = (-IeeeDouble::zero()).round_to_integral(Round::TowardZero).0;
        assert_eq!(-0.0, p.to_f64());
        let p = IeeeDouble::nan().round_to_integral(Round::TowardZero).0;
        assert!(p.to_f64().is_nan());
        let p = IeeeDouble::inf().round_to_integral(Round::TowardZero).0;
        assert!(p.to_f64().is_infinite() && p.to_f64() > 0.0);
        let p = (-IeeeDouble::inf()).round_to_integral(Round::TowardZero).0;
        assert!(p.to_f64().is_infinite() && p.to_f64() < 0.0);
    }

    #[test]
    fn is_integer() {
        let t = IeeeDouble::from_f64(-0.0);
        assert!(t.is_integer());
        let t = IeeeDouble::from_f64(3.14159);
        assert!(!t.is_integer());
        let t = IeeeDouble::nan();
        assert!(!t.is_integer());
        let t = IeeeDouble::inf();
        assert!(!t.is_integer());
        let t = -IeeeDouble::inf();
        assert!(!t.is_integer());
        let t = IeeeDouble::largest();
        assert!(t.is_integer());
    }

    #[test]
    fn largest() {
        assert_eq!(3.402823466e+38, IeeeSingle::largest().to_f32());
        assert_eq!(1.7976931348623158e+308, IeeeDouble::largest().to_f64());
    }

    #[test]
    fn smallest() {
        let test = IeeeSingle::smallest();
        let expected = "0x0.000002p-126".parse::<IeeeSingle>().unwrap();
        assert!(!test.is_negative());
        assert!(test.is_finite_non_zero());
        assert!(test.is_denormal());
        assert!(test.bitwise_eq(expected));

        let test = -IeeeSingle::smallest();
        let expected = "-0x0.000002p-126".parse::<IeeeSingle>().unwrap();
        assert!(test.is_negative());
        assert!(test.is_finite_non_zero());
        assert!(test.is_denormal());
        assert!(test.bitwise_eq(expected));

        let test = IeeeQuad::smallest();
        let expected = "0x0.0000000000000000000000000001p-16382"
            .parse::<IeeeQuad>()
            .unwrap();
        assert!(!test.is_negative());
        assert!(test.is_finite_non_zero());
        assert!(test.is_denormal());
        assert!(test.bitwise_eq(expected));

        let test = -IeeeQuad::smallest();
        let expected = "-0x0.0000000000000000000000000001p-16382"
            .parse::<IeeeQuad>()
            .unwrap();
        assert!(test.is_negative());
        assert!(test.is_finite_non_zero());
        assert!(test.is_denormal());
        assert!(test.bitwise_eq(expected));
    }

    #[test]
    fn smallest_normalized() {
        let test = IeeeSingle::smallest_normalized();
        let expected = "0x1p-126".parse::<IeeeSingle>().unwrap();
        assert!(!test.is_negative());
        assert!(test.is_finite_non_zero());
        assert!(!test.is_denormal());
        assert!(test.bitwise_eq(expected));

        let test = -IeeeSingle::smallest_normalized();
        let expected = "-0x1p-126".parse::<IeeeSingle>().unwrap();
        assert!(test.is_negative());
        assert!(test.is_finite_non_zero());
        assert!(!test.is_denormal());
        assert!(test.bitwise_eq(expected));

        let test = IeeeQuad::smallest_normalized();
        let expected = "0x1p-16382".parse::<IeeeQuad>().unwrap();
        assert!(!test.is_negative());
        assert!(test.is_finite_non_zero());
        assert!(!test.is_denormal());
        assert!(test.bitwise_eq(expected));

        let test = -IeeeQuad::smallest_normalized();
        let expected = "-0x1p-16382".parse::<IeeeQuad>().unwrap();
        assert!(test.is_negative());
        assert!(test.is_finite_non_zero());
        assert!(!test.is_denormal());
        assert!(test.bitwise_eq(expected));
    }

    #[test]
    fn zero() {
        assert_eq!(0.0, IeeeSingle::from_f32(0.0).to_f32());
        assert_eq!(-0.0, IeeeSingle::from_f32(-0.0).to_f32());
        assert!(IeeeSingle::from_f32(-0.0).is_negative());

        assert_eq!(0.0, IeeeDouble::from_f64(0.0).to_f64());
        assert_eq!(-0.0, IeeeDouble::from_f64(-0.0).to_f64());
        assert!(IeeeDouble::from_f64(-0.0).is_negative());

        fn test<T: Float>(sign: bool, bits: u128) {
            let test = if sign { -T::zero() } else { T::zero() };
            let pattern = if sign { "-0x0p+0" } else { "0x0p+0" };
            let expected = pattern.parse::<T>().unwrap();
            assert!(test.is_zero());
            assert_eq!(sign, test.is_negative());
            assert!(test.bitwise_eq(expected));
            assert_eq!(bits, test.to_bits());
        }
        test::<IeeeHalf>(false, 0);
        test::<IeeeHalf>(true, 0x8000);
        test::<IeeeSingle>(false, 0);
        test::<IeeeSingle>(true, 0x80000000);
        test::<IeeeDouble>(false, 0);
        test::<IeeeDouble>(true, 0x8000000000000000);
        test::<IeeeQuad>(false, 0);
        test::<IeeeQuad>(true, 0x8000000000000000_0000000000000000);
        test::<PpcDoubleDouble>(false, 0);
        test::<PpcDoubleDouble>(true, 0x8000000000000000);
        test::<X87DoubleExtended>(false, 0);
        test::<X87DoubleExtended>(true, 0x8000_0000000000000000);
    }

    #[test]
    fn copy_sign() {
        assert!(IeeeDouble::from_f64(-42.0).bitwise_eq(
            IeeeDouble::from_f64(42.0).copy_sign(IeeeDouble::from_f64(-1.0)),
        ));
        assert!(IeeeDouble::from_f64(42.0).bitwise_eq(
            IeeeDouble::from_f64(-42.0).copy_sign(IeeeDouble::from_f64(1.0)),
        ));
        assert!(IeeeDouble::from_f64(-42.0).bitwise_eq(
            IeeeDouble::from_f64(-42.0).copy_sign(IeeeDouble::from_f64(-1.0)),
        ));
        assert!(IeeeDouble::from_f64(42.0).bitwise_eq(
            IeeeDouble::from_f64(42.0).copy_sign(IeeeDouble::from_f64(1.0)),
        ));
    }

    #[test]
    fn convert() {
        let mut loses_info = false;
        let test = "1.0".parse::<IeeeDouble>().unwrap();
        let test: IeeeSingle = test.convert(Round::NearestTiesToEven, &mut loses_info).0;
        assert_eq!(1.0, test.to_f32());
        assert!(!loses_info);

        let mut test = "0x1p-53".parse::<X87DoubleExtended>().unwrap();
        let _: OpStatus = test.add_rounded(
            "1.0".parse::<X87DoubleExtended>().unwrap(),
            Round::NearestTiesToEven,
        );
        let test: IeeeDouble = test.convert(Round::NearestTiesToEven, &mut loses_info).0;
        assert_eq!(1.0, test.to_f64());
        assert!(loses_info);

        let mut test = "0x1p-53".parse::<IeeeQuad>().unwrap();
        let _: OpStatus =
            test.add_rounded("1.0".parse::<IeeeQuad>().unwrap(), Round::NearestTiesToEven);
        let test: IeeeDouble = test.convert(Round::NearestTiesToEven, &mut loses_info).0;
        assert_eq!(1.0, test.to_f64());
        assert!(loses_info);

        let test = "0xf.fffffffp+28".parse::<X87DoubleExtended>().unwrap();
        let test: IeeeDouble = test.convert(Round::NearestTiesToEven, &mut loses_info).0;
        assert_eq!(4294967295.0, test.to_f64());
        assert!(!loses_info);

        let test = IeeeSingle::snan(None);
        let x87_snan = X87DoubleExtended::snan(None);
        let test: X87DoubleExtended = test.convert(Round::NearestTiesToEven, &mut loses_info).0;
        assert!(test.bitwise_eq(x87_snan));
        assert!(!loses_info);

        let test = IeeeSingle::qnan(None);
        let x87_qnan = X87DoubleExtended::qnan(None);
        let test: X87DoubleExtended = test.convert(Round::NearestTiesToEven, &mut loses_info).0;
        assert!(test.bitwise_eq(x87_qnan));
        assert!(!loses_info);

        let test = X87DoubleExtended::snan(None);
        let test: X87DoubleExtended = test.convert(Round::NearestTiesToEven, &mut loses_info).0;
        assert!(test.bitwise_eq(x87_snan));
        assert!(!loses_info);

        let test = X87DoubleExtended::qnan(None);
        let test: X87DoubleExtended = test.convert(Round::NearestTiesToEven, &mut loses_info).0;
        assert!(test.bitwise_eq(x87_qnan));
        assert!(!loses_info);
    }

    #[test]
    fn ppc_double_double() {
        let test = "1.0".parse::<PpcDoubleDouble>().unwrap();
        assert_eq!(0x3ff0000000000000, test.to_bits());

        // LDBL_MAX
        let test = "1.79769313486231580793728971405301e+308"
            .parse::<PpcDoubleDouble>()
            .unwrap();
        assert_eq!(0x7c8ffffffffffffe_7fefffffffffffff, test.to_bits());

        // LDBL_MIN
        let test = "2.00416836000897277799610805135016e-292"
            .parse::<PpcDoubleDouble>()
            .unwrap();
        assert_eq!(0x0000000000000000_0360000000000000, test.to_bits());

        // PR30869
        {
            let result = "1.0".parse::<PpcDoubleDouble>().unwrap() +
                "1.0".parse::<PpcDoubleDouble>().unwrap();
            let _: PpcDoubleDouble = result;

            let result = "1.0".parse::<PpcDoubleDouble>().unwrap() -
                "1.0".parse::<PpcDoubleDouble>().unwrap();
            let _: PpcDoubleDouble = result;

            let result = "1.0".parse::<PpcDoubleDouble>().unwrap() *
                "1.0".parse::<PpcDoubleDouble>().unwrap();
            let _: PpcDoubleDouble = result;

            let result = "1.0".parse::<PpcDoubleDouble>().unwrap() /
                "1.0".parse::<PpcDoubleDouble>().unwrap();
            let _: PpcDoubleDouble = result;

            let mut exp = 0;
            let result = "1.0".parse::<PpcDoubleDouble>().unwrap().frexp(
                &mut exp,
                Round::NearestTiesToEven,
            );
            let _: PpcDoubleDouble = result;

            let result = "1.0".parse::<PpcDoubleDouble>().unwrap().scalbn(
                1,
                Round::NearestTiesToEven,
            );
            let _: PpcDoubleDouble = result;
        }
    }

    #[test]
    fn is_negative() {
        let t = "0x1p+0".parse::<IeeeSingle>().unwrap();
        assert!(!t.is_negative());
        let t = "-0x1p+0".parse::<IeeeSingle>().unwrap();
        assert!(t.is_negative());

        assert!(!IeeeSingle::inf().is_negative());
        assert!((-IeeeSingle::inf()).is_negative());

        assert!(!IeeeSingle::zero().is_negative());
        assert!((-IeeeSingle::zero()).is_negative());

        assert!(!IeeeSingle::nan().is_negative());
        assert!((-IeeeSingle::nan()).is_negative());

        assert!(!IeeeSingle::snan(None).is_negative());
        assert!((-IeeeSingle::snan(None)).is_negative());
    }

    #[test]
    fn is_normal() {
        let t = "0x1p+0".parse::<IeeeSingle>().unwrap();
        assert!(t.is_normal());

        assert!(!IeeeSingle::inf().is_normal());
        assert!(!IeeeSingle::zero().is_normal());
        assert!(!IeeeSingle::nan().is_normal());
        assert!(!IeeeSingle::snan(None).is_normal());
        assert!(!"0x1p-149".parse::<IeeeSingle>().unwrap().is_normal());
    }

    #[test]
    fn is_finite() {
        let t = "0x1p+0".parse::<IeeeSingle>().unwrap();
        assert!(t.is_finite());
        assert!(!IeeeSingle::inf().is_finite());
        assert!(IeeeSingle::zero().is_finite());
        assert!(!IeeeSingle::nan().is_finite());
        assert!(!IeeeSingle::snan(None).is_finite());
        assert!("0x1p-149".parse::<IeeeSingle>().unwrap().is_finite());
    }

    #[test]
    fn is_infinite() {
        let t = "0x1p+0".parse::<IeeeSingle>().unwrap();
        assert!(!t.is_infinite());
        assert!(IeeeSingle::inf().is_infinite());
        assert!(!IeeeSingle::zero().is_infinite());
        assert!(!IeeeSingle::nan().is_infinite());
        assert!(!IeeeSingle::snan(None).is_infinite());
        assert!(!"0x1p-149".parse::<IeeeSingle>().unwrap().is_infinite());
    }

    #[test]
    fn is_nan() {
        let t = "0x1p+0".parse::<IeeeSingle>().unwrap();
        assert!(!t.is_nan());
        assert!(!IeeeSingle::inf().is_nan());
        assert!(!IeeeSingle::zero().is_nan());
        assert!(IeeeSingle::nan().is_nan());
        assert!(IeeeSingle::snan(None).is_nan());
        assert!(!"0x1p-149".parse::<IeeeSingle>().unwrap().is_nan());
    }

    #[test]
    fn is_finite_non_zero() {
        // Test positive/negative normal value.
        assert!("0x1p+0".parse::<IeeeSingle>().unwrap().is_finite_non_zero());
        assert!(
            "-0x1p+0"
                .parse::<IeeeSingle>()
                .unwrap()
                .is_finite_non_zero()
        );

        // Test positive/negative denormal value.
        assert!(
            "0x1p-149"
                .parse::<IeeeSingle>()
                .unwrap()
                .is_finite_non_zero()
        );
        assert!(
            "-0x1p-149"
                .parse::<IeeeSingle>()
                .unwrap()
                .is_finite_non_zero()
        );

        // Test +/- Infinity.
        assert!(!IeeeSingle::inf().is_finite_non_zero());
        assert!(!(-IeeeSingle::inf()).is_finite_non_zero());

        // Test +/- Zero.
        assert!(!IeeeSingle::zero().is_finite_non_zero());
        assert!(!(-IeeeSingle::zero()).is_finite_non_zero());

        // Test +/- qNaN. +/- dont mean anything with qNaN but paranoia can't hurt in
        // this instance.
        assert!(!IeeeSingle::nan().is_finite_non_zero());
        assert!(!(-IeeeSingle::nan()).is_finite_non_zero());

        // Test +/- sNaN. +/- dont mean anything with sNaN but paranoia can't hurt in
        // this instance.
        assert!(!IeeeSingle::snan(None).is_finite_non_zero());
        assert!(!(-IeeeSingle::snan(None)).is_finite_non_zero());
    }

    #[test]
    fn add() {
        // Test Special Cases against each other and normal values.

        // FIXMES/NOTES:
        // 1. Since we perform only default exception handling all operations with
        // signaling NaNs should have a result that is a quiet NaN. Currently they
        // return sNaN.

        let p_inf = IeeeSingle::inf();
        let m_inf = -IeeeSingle::inf();
        let p_zero = IeeeSingle::zero();
        let m_zero = -IeeeSingle::zero();
        let qnan = IeeeSingle::nan();
        let p_normal_value = "0x1p+0".parse::<IeeeSingle>().unwrap();
        let m_normal_value = "-0x1p+0".parse::<IeeeSingle>().unwrap();
        let p_largest_value = IeeeSingle::largest();
        let m_largest_value = -IeeeSingle::largest();
        let p_smallest_value = IeeeSingle::smallest();
        let m_smallest_value = -IeeeSingle::smallest();
        let p_smallest_normalized = IeeeSingle::smallest_normalized();
        let m_smallest_normalized = -IeeeSingle::smallest_normalized();

        let overflow_status = OpStatus::OVERFLOW | OpStatus::INEXACT;

        let special_cases = [
            (p_inf, p_inf, "inf", OpStatus::OK, Category::Infinity),
            (p_inf, m_inf, "nan", OpStatus::INVALID_OP, Category::NaN),
            (p_inf, p_zero, "inf", OpStatus::OK, Category::Infinity),
            (p_inf, m_zero, "inf", OpStatus::OK, Category::Infinity),
            (p_inf, qnan, "nan", OpStatus::OK, Category::NaN),
            /*
    // See Note 1.
    (p_inf, snan, "nan", OpStatus::INVALID_OP, Category::NaN),
            */
            (
                p_inf,
                p_normal_value,
                "inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                p_inf,
                m_normal_value,
                "inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                p_inf,
                p_largest_value,
                "inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                p_inf,
                m_largest_value,
                "inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                p_inf,
                p_smallest_value,
                "inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                p_inf,
                m_smallest_value,
                "inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                p_inf,
                p_smallest_normalized,
                "inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                p_inf,
                m_smallest_normalized,
                "inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (m_inf, p_inf, "nan", OpStatus::INVALID_OP, Category::NaN),
            (m_inf, m_inf, "-inf", OpStatus::OK, Category::Infinity),
            (m_inf, p_zero, "-inf", OpStatus::OK, Category::Infinity),
            (m_inf, m_zero, "-inf", OpStatus::OK, Category::Infinity),
            (m_inf, qnan, "nan", OpStatus::OK, Category::NaN),
            /*
    // See Note 1.
    (m_inf, snan, "nan", OpStatus::INVALID_OP, Category::NaN),
            */
            (
                m_inf,
                p_normal_value,
                "-inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                m_inf,
                m_normal_value,
                "-inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                m_inf,
                p_largest_value,
                "-inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                m_inf,
                m_largest_value,
                "-inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                m_inf,
                p_smallest_value,
                "-inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                m_inf,
                m_smallest_value,
                "-inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                m_inf,
                p_smallest_normalized,
                "-inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                m_inf,
                m_smallest_normalized,
                "-inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (p_zero, p_inf, "inf", OpStatus::OK, Category::Infinity),
            (p_zero, m_inf, "-inf", OpStatus::OK, Category::Infinity),
            (p_zero, p_zero, "0x0p+0", OpStatus::OK, Category::Zero),
            (p_zero, m_zero, "0x0p+0", OpStatus::OK, Category::Zero),
            (p_zero, qnan, "nan", OpStatus::OK, Category::NaN),
            /*
    // See Note 1.
    (p_zero, snan, "nan", OpStatus::INVALID_OP, Category::NaN),
            */
            (
                p_zero,
                p_normal_value,
                "0x1p+0",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                p_zero,
                m_normal_value,
                "-0x1p+0",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                p_zero,
                p_largest_value,
                "0x1.fffffep+127",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                p_zero,
                m_largest_value,
                "-0x1.fffffep+127",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                p_zero,
                p_smallest_value,
                "0x1p-149",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                p_zero,
                m_smallest_value,
                "-0x1p-149",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                p_zero,
                p_smallest_normalized,
                "0x1p-126",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                p_zero,
                m_smallest_normalized,
                "-0x1p-126",
                OpStatus::OK,
                Category::Normal,
            ),
            (m_zero, p_inf, "inf", OpStatus::OK, Category::Infinity),
            (m_zero, m_inf, "-inf", OpStatus::OK, Category::Infinity),
            (m_zero, p_zero, "0x0p+0", OpStatus::OK, Category::Zero),
            (m_zero, m_zero, "-0x0p+0", OpStatus::OK, Category::Zero),
            (m_zero, qnan, "nan", OpStatus::OK, Category::NaN),
            /*
    // See Note 1.
    (m_zero, snan, "nan", OpStatus::INVALID_OP, Category::NaN),
            */
            (
                m_zero,
                p_normal_value,
                "0x1p+0",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                m_zero,
                m_normal_value,
                "-0x1p+0",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                m_zero,
                p_largest_value,
                "0x1.fffffep+127",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                m_zero,
                m_largest_value,
                "-0x1.fffffep+127",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                m_zero,
                p_smallest_value,
                "0x1p-149",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                m_zero,
                m_smallest_value,
                "-0x1p-149",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                m_zero,
                p_smallest_normalized,
                "0x1p-126",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                m_zero,
                m_smallest_normalized,
                "-0x1p-126",
                OpStatus::OK,
                Category::Normal,
            ),
            (qnan, p_inf, "nan", OpStatus::OK, Category::NaN),
            (qnan, m_inf, "nan", OpStatus::OK, Category::NaN),
            (qnan, p_zero, "nan", OpStatus::OK, Category::NaN),
            (qnan, m_zero, "nan", OpStatus::OK, Category::NaN),
            (qnan, qnan, "nan", OpStatus::OK, Category::NaN),
            /*
    // See Note 1.
    (qnan, snan, "nan", OpStatus::INVALID_OP, Category::NaN),
            */
            (qnan, p_normal_value, "nan", OpStatus::OK, Category::NaN),
            (qnan, m_normal_value, "nan", OpStatus::OK, Category::NaN),
            (qnan, p_largest_value, "nan", OpStatus::OK, Category::NaN),
            (qnan, m_largest_value, "nan", OpStatus::OK, Category::NaN),
            (qnan, p_smallest_value, "nan", OpStatus::OK, Category::NaN),
            (qnan, m_smallest_value, "nan", OpStatus::OK, Category::NaN),
            (
                qnan,
                p_smallest_normalized,
                "nan",
                OpStatus::OK,
                Category::NaN,
            ),
            (
                qnan,
                m_smallest_normalized,
                "nan",
                OpStatus::OK,
                Category::NaN,
            ),
            /*
    // See Note 1.
    (snan, p_inf, "nan", OpStatus::INVALID_OP, Category::NaN),
    (snan, m_inf, "nan", OpStatus::INVALID_OP, Category::NaN),
    (snan, p_zero, "nan", OpStatus::INVALID_OP, Category::NaN),
    (snan, m_zero, "nan", OpStatus::INVALID_OP, Category::NaN),
    (snan, qnan, "nan", OpStatus::INVALID_OP, Category::NaN),
    (snan, snan, "nan", OpStatus::INVALID_OP, Category::NaN),
    (snan, p_normal_value, "nan", OpStatus::INVALID_OP, Category::NaN),
    (snan, m_normal_value, "nan", OpStatus::INVALID_OP, Category::NaN),
    (snan, p_largest_value, "nan", OpStatus::INVALID_OP, Category::NaN),
    (snan, m_largest_value, "nan", OpStatus::INVALID_OP, Category::NaN),
    (snan, p_smallest_value, "nan", OpStatus::INVALID_OP, Category::NaN),
    (snan, m_smallest_value, "nan", OpStatus::INVALID_OP, Category::NaN),
    (snan, p_smallest_normalized, "nan", OpStatus::INVALID_OP, Category::NaN),
    (snan, m_smallest_normalized, "nan", OpStatus::INVALID_OP, Category::NaN),
            */
            (
                p_normal_value,
                p_inf,
                "inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                p_normal_value,
                m_inf,
                "-inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                p_normal_value,
                p_zero,
                "0x1p+0",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                p_normal_value,
                m_zero,
                "0x1p+0",
                OpStatus::OK,
                Category::Normal,
            ),
            (p_normal_value, qnan, "nan", OpStatus::OK, Category::NaN),
            /*
    // See Note 1.
    (p_normal_value, snan, "nan", OpStatus::INVALID_OP, Category::NaN),
            */
            (
                p_normal_value,
                p_normal_value,
                "0x1p+1",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                p_normal_value,
                m_normal_value,
                "0x0p+0",
                OpStatus::OK,
                Category::Zero,
            ),
            (
                p_normal_value,
                p_largest_value,
                "0x1.fffffep+127",
                OpStatus::INEXACT,
                Category::Normal,
            ),
            (
                p_normal_value,
                m_largest_value,
                "-0x1.fffffep+127",
                OpStatus::INEXACT,
                Category::Normal,
            ),
            (
                p_normal_value,
                p_smallest_value,
                "0x1p+0",
                OpStatus::INEXACT,
                Category::Normal,
            ),
            (
                p_normal_value,
                m_smallest_value,
                "0x1p+0",
                OpStatus::INEXACT,
                Category::Normal,
            ),
            (
                p_normal_value,
                p_smallest_normalized,
                "0x1p+0",
                OpStatus::INEXACT,
                Category::Normal,
            ),
            (
                p_normal_value,
                m_smallest_normalized,
                "0x1p+0",
                OpStatus::INEXACT,
                Category::Normal,
            ),
            (
                m_normal_value,
                p_inf,
                "inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                m_normal_value,
                m_inf,
                "-inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                m_normal_value,
                p_zero,
                "-0x1p+0",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                m_normal_value,
                m_zero,
                "-0x1p+0",
                OpStatus::OK,
                Category::Normal,
            ),
            (m_normal_value, qnan, "nan", OpStatus::OK, Category::NaN),
            /*
    // See Note 1.
    (m_normal_value, snan, "nan", OpStatus::INVALID_OP, Category::NaN),
            */
            (
                m_normal_value,
                p_normal_value,
                "0x0p+0",
                OpStatus::OK,
                Category::Zero,
            ),
            (
                m_normal_value,
                m_normal_value,
                "-0x1p+1",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                m_normal_value,
                p_largest_value,
                "0x1.fffffep+127",
                OpStatus::INEXACT,
                Category::Normal,
            ),
            (
                m_normal_value,
                m_largest_value,
                "-0x1.fffffep+127",
                OpStatus::INEXACT,
                Category::Normal,
            ),
            (
                m_normal_value,
                p_smallest_value,
                "-0x1p+0",
                OpStatus::INEXACT,
                Category::Normal,
            ),
            (
                m_normal_value,
                m_smallest_value,
                "-0x1p+0",
                OpStatus::INEXACT,
                Category::Normal,
            ),
            (
                m_normal_value,
                p_smallest_normalized,
                "-0x1p+0",
                OpStatus::INEXACT,
                Category::Normal,
            ),
            (
                m_normal_value,
                m_smallest_normalized,
                "-0x1p+0",
                OpStatus::INEXACT,
                Category::Normal,
            ),
            (
                p_largest_value,
                p_inf,
                "inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                p_largest_value,
                m_inf,
                "-inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                p_largest_value,
                p_zero,
                "0x1.fffffep+127",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                p_largest_value,
                m_zero,
                "0x1.fffffep+127",
                OpStatus::OK,
                Category::Normal,
            ),
            (p_largest_value, qnan, "nan", OpStatus::OK, Category::NaN),
            /*
    // See Note 1.
    (p_largest_value, snan, "nan", OpStatus::INVALID_OP, Category::NaN),
            */
            (
                p_largest_value,
                p_normal_value,
                "0x1.fffffep+127",
                OpStatus::INEXACT,
                Category::Normal,
            ),
            (
                p_largest_value,
                m_normal_value,
                "0x1.fffffep+127",
                OpStatus::INEXACT,
                Category::Normal,
            ),
            (
                p_largest_value,
                p_largest_value,
                "inf",
                overflow_status,
                Category::Infinity,
            ),
            (
                p_largest_value,
                m_largest_value,
                "0x0p+0",
                OpStatus::OK,
                Category::Zero,
            ),
            (
                p_largest_value,
                p_smallest_value,
                "0x1.fffffep+127",
                OpStatus::INEXACT,
                Category::Normal,
            ),
            (
                p_largest_value,
                m_smallest_value,
                "0x1.fffffep+127",
                OpStatus::INEXACT,
                Category::Normal,
            ),
            (
                p_largest_value,
                p_smallest_normalized,
                "0x1.fffffep+127",
                OpStatus::INEXACT,
                Category::Normal,
            ),
            (
                p_largest_value,
                m_smallest_normalized,
                "0x1.fffffep+127",
                OpStatus::INEXACT,
                Category::Normal,
            ),
            (
                m_largest_value,
                p_inf,
                "inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                m_largest_value,
                m_inf,
                "-inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                m_largest_value,
                p_zero,
                "-0x1.fffffep+127",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                m_largest_value,
                m_zero,
                "-0x1.fffffep+127",
                OpStatus::OK,
                Category::Normal,
            ),
            (m_largest_value, qnan, "nan", OpStatus::OK, Category::NaN),
            /*
    // See Note 1.
    (m_largest_value, snan, "nan", OpStatus::INVALID_OP, Category::NaN),
            */
            (
                m_largest_value,
                p_normal_value,
                "-0x1.fffffep+127",
                OpStatus::INEXACT,
                Category::Normal,
            ),
            (
                m_largest_value,
                m_normal_value,
                "-0x1.fffffep+127",
                OpStatus::INEXACT,
                Category::Normal,
            ),
            (
                m_largest_value,
                p_largest_value,
                "0x0p+0",
                OpStatus::OK,
                Category::Zero,
            ),
            (
                m_largest_value,
                m_largest_value,
                "-inf",
                overflow_status,
                Category::Infinity,
            ),
            (
                m_largest_value,
                p_smallest_value,
                "-0x1.fffffep+127",
                OpStatus::INEXACT,
                Category::Normal,
            ),
            (
                m_largest_value,
                m_smallest_value,
                "-0x1.fffffep+127",
                OpStatus::INEXACT,
                Category::Normal,
            ),
            (
                m_largest_value,
                p_smallest_normalized,
                "-0x1.fffffep+127",
                OpStatus::INEXACT,
                Category::Normal,
            ),
            (
                m_largest_value,
                m_smallest_normalized,
                "-0x1.fffffep+127",
                OpStatus::INEXACT,
                Category::Normal,
            ),
            (
                p_smallest_value,
                p_inf,
                "inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                p_smallest_value,
                m_inf,
                "-inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                p_smallest_value,
                p_zero,
                "0x1p-149",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                p_smallest_value,
                m_zero,
                "0x1p-149",
                OpStatus::OK,
                Category::Normal,
            ),
            (p_smallest_value, qnan, "nan", OpStatus::OK, Category::NaN),
            /*
    // See Note 1.
    (p_smallest_value, snan, "nan", OpStatus::INVALID_OP, Category::NaN),
            */
            (
                p_smallest_value,
                p_normal_value,
                "0x1p+0",
                OpStatus::INEXACT,
                Category::Normal,
            ),
            (
                p_smallest_value,
                m_normal_value,
                "-0x1p+0",
                OpStatus::INEXACT,
                Category::Normal,
            ),
            (
                p_smallest_value,
                p_largest_value,
                "0x1.fffffep+127",
                OpStatus::INEXACT,
                Category::Normal,
            ),
            (
                p_smallest_value,
                m_largest_value,
                "-0x1.fffffep+127",
                OpStatus::INEXACT,
                Category::Normal,
            ),
            (
                p_smallest_value,
                p_smallest_value,
                "0x1p-148",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                p_smallest_value,
                m_smallest_value,
                "0x0p+0",
                OpStatus::OK,
                Category::Zero,
            ),
            (
                p_smallest_value,
                p_smallest_normalized,
                "0x1.000002p-126",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                p_smallest_value,
                m_smallest_normalized,
                "-0x1.fffffcp-127",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                m_smallest_value,
                p_inf,
                "inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                m_smallest_value,
                m_inf,
                "-inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                m_smallest_value,
                p_zero,
                "-0x1p-149",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                m_smallest_value,
                m_zero,
                "-0x1p-149",
                OpStatus::OK,
                Category::Normal,
            ),
            (m_smallest_value, qnan, "nan", OpStatus::OK, Category::NaN),
            /*
    // See Note 1.
    (m_smallest_value, snan, "nan", OpStatus::INVALID_OP, Category::NaN),
            */
            (
                m_smallest_value,
                p_normal_value,
                "0x1p+0",
                OpStatus::INEXACT,
                Category::Normal,
            ),
            (
                m_smallest_value,
                m_normal_value,
                "-0x1p+0",
                OpStatus::INEXACT,
                Category::Normal,
            ),
            (
                m_smallest_value,
                p_largest_value,
                "0x1.fffffep+127",
                OpStatus::INEXACT,
                Category::Normal,
            ),
            (
                m_smallest_value,
                m_largest_value,
                "-0x1.fffffep+127",
                OpStatus::INEXACT,
                Category::Normal,
            ),
            (
                m_smallest_value,
                p_smallest_value,
                "0x0p+0",
                OpStatus::OK,
                Category::Zero,
            ),
            (
                m_smallest_value,
                m_smallest_value,
                "-0x1p-148",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                m_smallest_value,
                p_smallest_normalized,
                "0x1.fffffcp-127",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                m_smallest_value,
                m_smallest_normalized,
                "-0x1.000002p-126",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                p_smallest_normalized,
                p_inf,
                "inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                p_smallest_normalized,
                m_inf,
                "-inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                p_smallest_normalized,
                p_zero,
                "0x1p-126",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                p_smallest_normalized,
                m_zero,
                "0x1p-126",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                p_smallest_normalized,
                qnan,
                "nan",
                OpStatus::OK,
                Category::NaN,
            ),
            /*
    // See Note 1.
    (p_smallest_normalized, snan, "nan", OpStatus::INVALID_OP, Category::NaN),
            */
            (
                p_smallest_normalized,
                p_normal_value,
                "0x1p+0",
                OpStatus::INEXACT,
                Category::Normal,
            ),
            (
                p_smallest_normalized,
                m_normal_value,
                "-0x1p+0",
                OpStatus::INEXACT,
                Category::Normal,
            ),
            (
                p_smallest_normalized,
                p_largest_value,
                "0x1.fffffep+127",
                OpStatus::INEXACT,
                Category::Normal,
            ),
            (
                p_smallest_normalized,
                m_largest_value,
                "-0x1.fffffep+127",
                OpStatus::INEXACT,
                Category::Normal,
            ),
            (
                p_smallest_normalized,
                p_smallest_value,
                "0x1.000002p-126",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                p_smallest_normalized,
                m_smallest_value,
                "0x1.fffffcp-127",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                p_smallest_normalized,
                p_smallest_normalized,
                "0x1p-125",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                p_smallest_normalized,
                m_smallest_normalized,
                "0x0p+0",
                OpStatus::OK,
                Category::Zero,
            ),
            (
                m_smallest_normalized,
                p_inf,
                "inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                m_smallest_normalized,
                m_inf,
                "-inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                m_smallest_normalized,
                p_zero,
                "-0x1p-126",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                m_smallest_normalized,
                m_zero,
                "-0x1p-126",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                m_smallest_normalized,
                qnan,
                "nan",
                OpStatus::OK,
                Category::NaN,
            ),
            /*
    // See Note 1.
    (m_smallest_normalized, snan, "nan", OpStatus::INVALID_OP, Category::NaN),
            */
            (
                m_smallest_normalized,
                p_normal_value,
                "0x1p+0",
                OpStatus::INEXACT,
                Category::Normal,
            ),
            (
                m_smallest_normalized,
                m_normal_value,
                "-0x1p+0",
                OpStatus::INEXACT,
                Category::Normal,
            ),
            (
                m_smallest_normalized,
                p_largest_value,
                "0x1.fffffep+127",
                OpStatus::INEXACT,
                Category::Normal,
            ),
            (
                m_smallest_normalized,
                m_largest_value,
                "-0x1.fffffep+127",
                OpStatus::INEXACT,
                Category::Normal,
            ),
            (
                m_smallest_normalized,
                p_smallest_value,
                "-0x1.fffffcp-127",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                m_smallest_normalized,
                m_smallest_value,
                "-0x1.000002p-126",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                m_smallest_normalized,
                p_smallest_normalized,
                "0x0p+0",
                OpStatus::OK,
                Category::Zero,
            ),
            (
                m_smallest_normalized,
                m_smallest_normalized,
                "-0x1p-125",
                OpStatus::OK,
                Category::Normal,
            ),
        ];

        for &(mut x, y, result, status, category) in &special_cases[..] {
            assert_eq!(x.add_rounded(y, Round::NearestTiesToEven), status);
            assert_eq!(x.category(), category);
            assert!(result.parse::<IeeeSingle>().unwrap().bitwise_eq(x));
        }
    }

    #[test]
    fn subtract() {
        // Test Special Cases against each other and normal values.

        // FIXMES/NOTES:
        // 1. Since we perform only default exception handling all operations with
        // signaling NaNs should have a result that is a quiet NaN. Currently they
        // return sNaN.

        let p_inf = IeeeSingle::inf();
        let m_inf = -IeeeSingle::inf();
        let p_zero = IeeeSingle::zero();
        let m_zero = -IeeeSingle::zero();
        let qnan = IeeeSingle::nan();
        let p_normal_value = "0x1p+0".parse::<IeeeSingle>().unwrap();
        let m_normal_value = "-0x1p+0".parse::<IeeeSingle>().unwrap();
        let p_largest_value = IeeeSingle::largest();
        let m_largest_value = -IeeeSingle::largest();
        let p_smallest_value = IeeeSingle::smallest();
        let m_smallest_value = -IeeeSingle::smallest();
        let p_smallest_normalized = IeeeSingle::smallest_normalized();
        let m_smallest_normalized = -IeeeSingle::smallest_normalized();

        let overflow_status = OpStatus::OVERFLOW | OpStatus::INEXACT;

        let special_cases = [
            (p_inf, p_inf, "nan", OpStatus::INVALID_OP, Category::NaN),
            (p_inf, m_inf, "inf", OpStatus::OK, Category::Infinity),
            (p_inf, p_zero, "inf", OpStatus::OK, Category::Infinity),
            (p_inf, m_zero, "inf", OpStatus::OK, Category::Infinity),
            (p_inf, qnan, "-nan", OpStatus::OK, Category::NaN),
            /*
    // See Note 1.
    (p_inf, snan, "-nan", OpStatus::INVALID_OP, Category::NaN),
            */
            (
                p_inf,
                p_normal_value,
                "inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                p_inf,
                m_normal_value,
                "inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                p_inf,
                p_largest_value,
                "inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                p_inf,
                m_largest_value,
                "inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                p_inf,
                p_smallest_value,
                "inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                p_inf,
                m_smallest_value,
                "inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                p_inf,
                p_smallest_normalized,
                "inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                p_inf,
                m_smallest_normalized,
                "inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (m_inf, p_inf, "-inf", OpStatus::OK, Category::Infinity),
            (m_inf, m_inf, "nan", OpStatus::INVALID_OP, Category::NaN),
            (m_inf, p_zero, "-inf", OpStatus::OK, Category::Infinity),
            (m_inf, m_zero, "-inf", OpStatus::OK, Category::Infinity),
            (m_inf, qnan, "-nan", OpStatus::OK, Category::NaN),
            /*
    // See Note 1.
    (m_inf, snan, "-nan", OpStatus::INVALID_OP, Category::NaN),
            */
            (
                m_inf,
                p_normal_value,
                "-inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                m_inf,
                m_normal_value,
                "-inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                m_inf,
                p_largest_value,
                "-inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                m_inf,
                m_largest_value,
                "-inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                m_inf,
                p_smallest_value,
                "-inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                m_inf,
                m_smallest_value,
                "-inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                m_inf,
                p_smallest_normalized,
                "-inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                m_inf,
                m_smallest_normalized,
                "-inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (p_zero, p_inf, "-inf", OpStatus::OK, Category::Infinity),
            (p_zero, m_inf, "inf", OpStatus::OK, Category::Infinity),
            (p_zero, p_zero, "0x0p+0", OpStatus::OK, Category::Zero),
            (p_zero, m_zero, "0x0p+0", OpStatus::OK, Category::Zero),
            (p_zero, qnan, "-nan", OpStatus::OK, Category::NaN),
            /*
    // See Note 1.
    (p_zero, snan, "-nan", OpStatus::INVALID_OP, Category::NaN),
            */
            (
                p_zero,
                p_normal_value,
                "-0x1p+0",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                p_zero,
                m_normal_value,
                "0x1p+0",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                p_zero,
                p_largest_value,
                "-0x1.fffffep+127",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                p_zero,
                m_largest_value,
                "0x1.fffffep+127",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                p_zero,
                p_smallest_value,
                "-0x1p-149",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                p_zero,
                m_smallest_value,
                "0x1p-149",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                p_zero,
                p_smallest_normalized,
                "-0x1p-126",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                p_zero,
                m_smallest_normalized,
                "0x1p-126",
                OpStatus::OK,
                Category::Normal,
            ),
            (m_zero, p_inf, "-inf", OpStatus::OK, Category::Infinity),
            (m_zero, m_inf, "inf", OpStatus::OK, Category::Infinity),
            (m_zero, p_zero, "-0x0p+0", OpStatus::OK, Category::Zero),
            (m_zero, m_zero, "0x0p+0", OpStatus::OK, Category::Zero),
            (m_zero, qnan, "-nan", OpStatus::OK, Category::NaN),
            /*
    // See Note 1.
    (m_zero, snan, "-nan", OpStatus::INVALID_OP, Category::NaN),
            */
            (
                m_zero,
                p_normal_value,
                "-0x1p+0",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                m_zero,
                m_normal_value,
                "0x1p+0",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                m_zero,
                p_largest_value,
                "-0x1.fffffep+127",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                m_zero,
                m_largest_value,
                "0x1.fffffep+127",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                m_zero,
                p_smallest_value,
                "-0x1p-149",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                m_zero,
                m_smallest_value,
                "0x1p-149",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                m_zero,
                p_smallest_normalized,
                "-0x1p-126",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                m_zero,
                m_smallest_normalized,
                "0x1p-126",
                OpStatus::OK,
                Category::Normal,
            ),
            (qnan, p_inf, "nan", OpStatus::OK, Category::NaN),
            (qnan, m_inf, "nan", OpStatus::OK, Category::NaN),
            (qnan, p_zero, "nan", OpStatus::OK, Category::NaN),
            (qnan, m_zero, "nan", OpStatus::OK, Category::NaN),
            (qnan, qnan, "nan", OpStatus::OK, Category::NaN),
            /*
    // See Note 1.
    (qnan, snan, "nan", OpStatus::INVALID_OP, Category::NaN),
            */
            (qnan, p_normal_value, "nan", OpStatus::OK, Category::NaN),
            (qnan, m_normal_value, "nan", OpStatus::OK, Category::NaN),
            (qnan, p_largest_value, "nan", OpStatus::OK, Category::NaN),
            (qnan, m_largest_value, "nan", OpStatus::OK, Category::NaN),
            (qnan, p_smallest_value, "nan", OpStatus::OK, Category::NaN),
            (qnan, m_smallest_value, "nan", OpStatus::OK, Category::NaN),
            (
                qnan,
                p_smallest_normalized,
                "nan",
                OpStatus::OK,
                Category::NaN,
            ),
            (
                qnan,
                m_smallest_normalized,
                "nan",
                OpStatus::OK,
                Category::NaN,
            ),
            /*
    // See Note 1.
    (snan, p_inf, "nan", OpStatus::INVALID_OP, Category::NaN),
    (snan, m_inf, "nan", OpStatus::INVALID_OP, Category::NaN),
    (snan, p_zero, "nan", OpStatus::INVALID_OP, Category::NaN),
    (snan, m_zero, "nan", OpStatus::INVALID_OP, Category::NaN),
    (snan, qnan, "nan", OpStatus::INVALID_OP, Category::NaN),
    (snan, snan, "nan", OpStatus::INVALID_OP, Category::NaN),
    (snan, p_normal_value, "nan", OpStatus::INVALID_OP, Category::NaN),
    (snan, m_normal_value, "nan", OpStatus::INVALID_OP, Category::NaN),
    (snan, p_largest_value, "nan", OpStatus::INVALID_OP, Category::NaN),
    (snan, m_largest_value, "nan", OpStatus::INVALID_OP, Category::NaN),
    (snan, p_smallest_value, "nan", OpStatus::INVALID_OP, Category::NaN),
    (snan, m_smallest_value, "nan", OpStatus::INVALID_OP, Category::NaN),
    (snan, p_smallest_normalized, "nan", OpStatus::INVALID_OP, Category::NaN),
    (snan, m_smallest_normalized, "nan", OpStatus::INVALID_OP, Category::NaN),
            */
            (
                p_normal_value,
                p_inf,
                "-inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                p_normal_value,
                m_inf,
                "inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                p_normal_value,
                p_zero,
                "0x1p+0",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                p_normal_value,
                m_zero,
                "0x1p+0",
                OpStatus::OK,
                Category::Normal,
            ),
            (p_normal_value, qnan, "-nan", OpStatus::OK, Category::NaN),
            /*
    // See Note 1.
    (p_normal_value, snan, "-nan", OpStatus::INVALID_OP, Category::NaN),
            */
            (
                p_normal_value,
                p_normal_value,
                "0x0p+0",
                OpStatus::OK,
                Category::Zero,
            ),
            (
                p_normal_value,
                m_normal_value,
                "0x1p+1",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                p_normal_value,
                p_largest_value,
                "-0x1.fffffep+127",
                OpStatus::INEXACT,
                Category::Normal,
            ),
            (
                p_normal_value,
                m_largest_value,
                "0x1.fffffep+127",
                OpStatus::INEXACT,
                Category::Normal,
            ),
            (
                p_normal_value,
                p_smallest_value,
                "0x1p+0",
                OpStatus::INEXACT,
                Category::Normal,
            ),
            (
                p_normal_value,
                m_smallest_value,
                "0x1p+0",
                OpStatus::INEXACT,
                Category::Normal,
            ),
            (
                p_normal_value,
                p_smallest_normalized,
                "0x1p+0",
                OpStatus::INEXACT,
                Category::Normal,
            ),
            (
                p_normal_value,
                m_smallest_normalized,
                "0x1p+0",
                OpStatus::INEXACT,
                Category::Normal,
            ),
            (
                m_normal_value,
                p_inf,
                "-inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                m_normal_value,
                m_inf,
                "inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                m_normal_value,
                p_zero,
                "-0x1p+0",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                m_normal_value,
                m_zero,
                "-0x1p+0",
                OpStatus::OK,
                Category::Normal,
            ),
            (m_normal_value, qnan, "-nan", OpStatus::OK, Category::NaN),
            /*
    // See Note 1.
    (m_normal_value, snan, "-nan", OpStatus::INVALID_OP, Category::NaN),
            */
            (
                m_normal_value,
                p_normal_value,
                "-0x1p+1",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                m_normal_value,
                m_normal_value,
                "0x0p+0",
                OpStatus::OK,
                Category::Zero,
            ),
            (
                m_normal_value,
                p_largest_value,
                "-0x1.fffffep+127",
                OpStatus::INEXACT,
                Category::Normal,
            ),
            (
                m_normal_value,
                m_largest_value,
                "0x1.fffffep+127",
                OpStatus::INEXACT,
                Category::Normal,
            ),
            (
                m_normal_value,
                p_smallest_value,
                "-0x1p+0",
                OpStatus::INEXACT,
                Category::Normal,
            ),
            (
                m_normal_value,
                m_smallest_value,
                "-0x1p+0",
                OpStatus::INEXACT,
                Category::Normal,
            ),
            (
                m_normal_value,
                p_smallest_normalized,
                "-0x1p+0",
                OpStatus::INEXACT,
                Category::Normal,
            ),
            (
                m_normal_value,
                m_smallest_normalized,
                "-0x1p+0",
                OpStatus::INEXACT,
                Category::Normal,
            ),
            (
                p_largest_value,
                p_inf,
                "-inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                p_largest_value,
                m_inf,
                "inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                p_largest_value,
                p_zero,
                "0x1.fffffep+127",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                p_largest_value,
                m_zero,
                "0x1.fffffep+127",
                OpStatus::OK,
                Category::Normal,
            ),
            (p_largest_value, qnan, "-nan", OpStatus::OK, Category::NaN),
            /*
    // See Note 1.
    (p_largest_value, snan, "-nan", OpStatus::INVALID_OP, Category::NaN),
            */
            (
                p_largest_value,
                p_normal_value,
                "0x1.fffffep+127",
                OpStatus::INEXACT,
                Category::Normal,
            ),
            (
                p_largest_value,
                m_normal_value,
                "0x1.fffffep+127",
                OpStatus::INEXACT,
                Category::Normal,
            ),
            (
                p_largest_value,
                p_largest_value,
                "0x0p+0",
                OpStatus::OK,
                Category::Zero,
            ),
            (
                p_largest_value,
                m_largest_value,
                "inf",
                overflow_status,
                Category::Infinity,
            ),
            (
                p_largest_value,
                p_smallest_value,
                "0x1.fffffep+127",
                OpStatus::INEXACT,
                Category::Normal,
            ),
            (
                p_largest_value,
                m_smallest_value,
                "0x1.fffffep+127",
                OpStatus::INEXACT,
                Category::Normal,
            ),
            (
                p_largest_value,
                p_smallest_normalized,
                "0x1.fffffep+127",
                OpStatus::INEXACT,
                Category::Normal,
            ),
            (
                p_largest_value,
                m_smallest_normalized,
                "0x1.fffffep+127",
                OpStatus::INEXACT,
                Category::Normal,
            ),
            (
                m_largest_value,
                p_inf,
                "-inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                m_largest_value,
                m_inf,
                "inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                m_largest_value,
                p_zero,
                "-0x1.fffffep+127",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                m_largest_value,
                m_zero,
                "-0x1.fffffep+127",
                OpStatus::OK,
                Category::Normal,
            ),
            (m_largest_value, qnan, "-nan", OpStatus::OK, Category::NaN),
            /*
    // See Note 1.
    (m_largest_value, snan, "-nan", OpStatus::INVALID_OP, Category::NaN),
            */
            (
                m_largest_value,
                p_normal_value,
                "-0x1.fffffep+127",
                OpStatus::INEXACT,
                Category::Normal,
            ),
            (
                m_largest_value,
                m_normal_value,
                "-0x1.fffffep+127",
                OpStatus::INEXACT,
                Category::Normal,
            ),
            (
                m_largest_value,
                p_largest_value,
                "-inf",
                overflow_status,
                Category::Infinity,
            ),
            (
                m_largest_value,
                m_largest_value,
                "0x0p+0",
                OpStatus::OK,
                Category::Zero,
            ),
            (
                m_largest_value,
                p_smallest_value,
                "-0x1.fffffep+127",
                OpStatus::INEXACT,
                Category::Normal,
            ),
            (
                m_largest_value,
                m_smallest_value,
                "-0x1.fffffep+127",
                OpStatus::INEXACT,
                Category::Normal,
            ),
            (
                m_largest_value,
                p_smallest_normalized,
                "-0x1.fffffep+127",
                OpStatus::INEXACT,
                Category::Normal,
            ),
            (
                m_largest_value,
                m_smallest_normalized,
                "-0x1.fffffep+127",
                OpStatus::INEXACT,
                Category::Normal,
            ),
            (
                p_smallest_value,
                p_inf,
                "-inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                p_smallest_value,
                m_inf,
                "inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                p_smallest_value,
                p_zero,
                "0x1p-149",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                p_smallest_value,
                m_zero,
                "0x1p-149",
                OpStatus::OK,
                Category::Normal,
            ),
            (p_smallest_value, qnan, "-nan", OpStatus::OK, Category::NaN),
            /*
    // See Note 1.
    (p_smallest_value, snan, "-nan", OpStatus::INVALID_OP, Category::NaN),
            */
            (
                p_smallest_value,
                p_normal_value,
                "-0x1p+0",
                OpStatus::INEXACT,
                Category::Normal,
            ),
            (
                p_smallest_value,
                m_normal_value,
                "0x1p+0",
                OpStatus::INEXACT,
                Category::Normal,
            ),
            (
                p_smallest_value,
                p_largest_value,
                "-0x1.fffffep+127",
                OpStatus::INEXACT,
                Category::Normal,
            ),
            (
                p_smallest_value,
                m_largest_value,
                "0x1.fffffep+127",
                OpStatus::INEXACT,
                Category::Normal,
            ),
            (
                p_smallest_value,
                p_smallest_value,
                "0x0p+0",
                OpStatus::OK,
                Category::Zero,
            ),
            (
                p_smallest_value,
                m_smallest_value,
                "0x1p-148",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                p_smallest_value,
                p_smallest_normalized,
                "-0x1.fffffcp-127",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                p_smallest_value,
                m_smallest_normalized,
                "0x1.000002p-126",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                m_smallest_value,
                p_inf,
                "-inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                m_smallest_value,
                m_inf,
                "inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                m_smallest_value,
                p_zero,
                "-0x1p-149",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                m_smallest_value,
                m_zero,
                "-0x1p-149",
                OpStatus::OK,
                Category::Normal,
            ),
            (m_smallest_value, qnan, "-nan", OpStatus::OK, Category::NaN),
            /*
    // See Note 1.
    (m_smallest_value, snan, "-nan", OpStatus::INVALID_OP, Category::NaN),
            */
            (
                m_smallest_value,
                p_normal_value,
                "-0x1p+0",
                OpStatus::INEXACT,
                Category::Normal,
            ),
            (
                m_smallest_value,
                m_normal_value,
                "0x1p+0",
                OpStatus::INEXACT,
                Category::Normal,
            ),
            (
                m_smallest_value,
                p_largest_value,
                "-0x1.fffffep+127",
                OpStatus::INEXACT,
                Category::Normal,
            ),
            (
                m_smallest_value,
                m_largest_value,
                "0x1.fffffep+127",
                OpStatus::INEXACT,
                Category::Normal,
            ),
            (
                m_smallest_value,
                p_smallest_value,
                "-0x1p-148",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                m_smallest_value,
                m_smallest_value,
                "0x0p+0",
                OpStatus::OK,
                Category::Zero,
            ),
            (
                m_smallest_value,
                p_smallest_normalized,
                "-0x1.000002p-126",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                m_smallest_value,
                m_smallest_normalized,
                "0x1.fffffcp-127",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                p_smallest_normalized,
                p_inf,
                "-inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                p_smallest_normalized,
                m_inf,
                "inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                p_smallest_normalized,
                p_zero,
                "0x1p-126",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                p_smallest_normalized,
                m_zero,
                "0x1p-126",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                p_smallest_normalized,
                qnan,
                "-nan",
                OpStatus::OK,
                Category::NaN,
            ),
            /*
    // See Note 1.
    (p_smallest_normalized, snan, "-nan", OpStatus::INVALID_OP, Category::NaN),
            */
            (
                p_smallest_normalized,
                p_normal_value,
                "-0x1p+0",
                OpStatus::INEXACT,
                Category::Normal,
            ),
            (
                p_smallest_normalized,
                m_normal_value,
                "0x1p+0",
                OpStatus::INEXACT,
                Category::Normal,
            ),
            (
                p_smallest_normalized,
                p_largest_value,
                "-0x1.fffffep+127",
                OpStatus::INEXACT,
                Category::Normal,
            ),
            (
                p_smallest_normalized,
                m_largest_value,
                "0x1.fffffep+127",
                OpStatus::INEXACT,
                Category::Normal,
            ),
            (
                p_smallest_normalized,
                p_smallest_value,
                "0x1.fffffcp-127",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                p_smallest_normalized,
                m_smallest_value,
                "0x1.000002p-126",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                p_smallest_normalized,
                p_smallest_normalized,
                "0x0p+0",
                OpStatus::OK,
                Category::Zero,
            ),
            (
                p_smallest_normalized,
                m_smallest_normalized,
                "0x1p-125",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                m_smallest_normalized,
                p_inf,
                "-inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                m_smallest_normalized,
                m_inf,
                "inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                m_smallest_normalized,
                p_zero,
                "-0x1p-126",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                m_smallest_normalized,
                m_zero,
                "-0x1p-126",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                m_smallest_normalized,
                qnan,
                "-nan",
                OpStatus::OK,
                Category::NaN,
            ),
            /*
    // See Note 1.
    (m_smallest_normalized, snan, "-nan", OpStatus::INVALID_OP, Category::NaN),
            */
            (
                m_smallest_normalized,
                p_normal_value,
                "-0x1p+0",
                OpStatus::INEXACT,
                Category::Normal,
            ),
            (
                m_smallest_normalized,
                m_normal_value,
                "0x1p+0",
                OpStatus::INEXACT,
                Category::Normal,
            ),
            (
                m_smallest_normalized,
                p_largest_value,
                "-0x1.fffffep+127",
                OpStatus::INEXACT,
                Category::Normal,
            ),
            (
                m_smallest_normalized,
                m_largest_value,
                "0x1.fffffep+127",
                OpStatus::INEXACT,
                Category::Normal,
            ),
            (
                m_smallest_normalized,
                p_smallest_value,
                "-0x1.000002p-126",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                m_smallest_normalized,
                m_smallest_value,
                "-0x1.fffffcp-127",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                m_smallest_normalized,
                p_smallest_normalized,
                "-0x1p-125",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                m_smallest_normalized,
                m_smallest_normalized,
                "0x0p+0",
                OpStatus::OK,
                Category::Zero,
            ),
        ];

        for &(mut x, y, result, status, category) in &special_cases[..] {
            assert_eq!(x.sub_rounded(y, Round::NearestTiesToEven), status);
            assert_eq!(x.category(), category);
            assert!(result.parse::<IeeeSingle>().unwrap().bitwise_eq(x));
        }
    }

    #[test]
    fn multiply() {
        // Test Special Cases against each other and normal values.

        // FIXMES/NOTES:
        // 1. Since we perform only default exception handling all operations with
        // signaling NaNs should have a result that is a quiet NaN. Currently they
        // return sNaN.

        let p_inf = IeeeSingle::inf();
        let m_inf = -IeeeSingle::inf();
        let p_zero = IeeeSingle::zero();
        let m_zero = -IeeeSingle::zero();
        let qnan = IeeeSingle::nan();
        let p_normal_value = "0x1p+0".parse::<IeeeSingle>().unwrap();
        let m_normal_value = "-0x1p+0".parse::<IeeeSingle>().unwrap();
        let p_largest_value = IeeeSingle::largest();
        let m_largest_value = -IeeeSingle::largest();
        let p_smallest_value = IeeeSingle::smallest();
        let m_smallest_value = -IeeeSingle::smallest();
        let p_smallest_normalized = IeeeSingle::smallest_normalized();
        let m_smallest_normalized = -IeeeSingle::smallest_normalized();

        let overflow_status = OpStatus::OVERFLOW | OpStatus::INEXACT;
        let underflow_status = OpStatus::UNDERFLOW | OpStatus::INEXACT;

        let special_cases = [
            (p_inf, p_inf, "inf", OpStatus::OK, Category::Infinity),
            (p_inf, m_inf, "-inf", OpStatus::OK, Category::Infinity),
            (p_inf, p_zero, "nan", OpStatus::INVALID_OP, Category::NaN),
            (p_inf, m_zero, "nan", OpStatus::INVALID_OP, Category::NaN),
            (p_inf, qnan, "nan", OpStatus::OK, Category::NaN),
            /*
    // See Note 1.
    (p_inf, snan, "nan", OpStatus::INVALID_OP, Category::NaN),
            */
            (
                p_inf,
                p_normal_value,
                "inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                p_inf,
                m_normal_value,
                "-inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                p_inf,
                p_largest_value,
                "inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                p_inf,
                m_largest_value,
                "-inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                p_inf,
                p_smallest_value,
                "inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                p_inf,
                m_smallest_value,
                "-inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                p_inf,
                p_smallest_normalized,
                "inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                p_inf,
                m_smallest_normalized,
                "-inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (m_inf, p_inf, "-inf", OpStatus::OK, Category::Infinity),
            (m_inf, m_inf, "inf", OpStatus::OK, Category::Infinity),
            (m_inf, p_zero, "nan", OpStatus::INVALID_OP, Category::NaN),
            (m_inf, m_zero, "nan", OpStatus::INVALID_OP, Category::NaN),
            (m_inf, qnan, "nan", OpStatus::OK, Category::NaN),
            /*
    // See Note 1.
    (m_inf, snan, "nan", OpStatus::INVALID_OP, Category::NaN),
            */
            (
                m_inf,
                p_normal_value,
                "-inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                m_inf,
                m_normal_value,
                "inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                m_inf,
                p_largest_value,
                "-inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                m_inf,
                m_largest_value,
                "inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                m_inf,
                p_smallest_value,
                "-inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                m_inf,
                m_smallest_value,
                "inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                m_inf,
                p_smallest_normalized,
                "-inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                m_inf,
                m_smallest_normalized,
                "inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (p_zero, p_inf, "nan", OpStatus::INVALID_OP, Category::NaN),
            (p_zero, m_inf, "nan", OpStatus::INVALID_OP, Category::NaN),
            (p_zero, p_zero, "0x0p+0", OpStatus::OK, Category::Zero),
            (p_zero, m_zero, "-0x0p+0", OpStatus::OK, Category::Zero),
            (p_zero, qnan, "nan", OpStatus::OK, Category::NaN),
            /*
    // See Note 1.
    (p_zero, snan, "nan", OpStatus::INVALID_OP, Category::NaN),
            */
            (
                p_zero,
                p_normal_value,
                "0x0p+0",
                OpStatus::OK,
                Category::Zero,
            ),
            (
                p_zero,
                m_normal_value,
                "-0x0p+0",
                OpStatus::OK,
                Category::Zero,
            ),
            (
                p_zero,
                p_largest_value,
                "0x0p+0",
                OpStatus::OK,
                Category::Zero,
            ),
            (
                p_zero,
                m_largest_value,
                "-0x0p+0",
                OpStatus::OK,
                Category::Zero,
            ),
            (
                p_zero,
                p_smallest_value,
                "0x0p+0",
                OpStatus::OK,
                Category::Zero,
            ),
            (
                p_zero,
                m_smallest_value,
                "-0x0p+0",
                OpStatus::OK,
                Category::Zero,
            ),
            (
                p_zero,
                p_smallest_normalized,
                "0x0p+0",
                OpStatus::OK,
                Category::Zero,
            ),
            (
                p_zero,
                m_smallest_normalized,
                "-0x0p+0",
                OpStatus::OK,
                Category::Zero,
            ),
            (m_zero, p_inf, "nan", OpStatus::INVALID_OP, Category::NaN),
            (m_zero, m_inf, "nan", OpStatus::INVALID_OP, Category::NaN),
            (m_zero, p_zero, "-0x0p+0", OpStatus::OK, Category::Zero),
            (m_zero, m_zero, "0x0p+0", OpStatus::OK, Category::Zero),
            (m_zero, qnan, "nan", OpStatus::OK, Category::NaN),
            /*
    // See Note 1.
    (m_zero, snan, "nan", OpStatus::INVALID_OP, Category::NaN),
            */
            (
                m_zero,
                p_normal_value,
                "-0x0p+0",
                OpStatus::OK,
                Category::Zero,
            ),
            (
                m_zero,
                m_normal_value,
                "0x0p+0",
                OpStatus::OK,
                Category::Zero,
            ),
            (
                m_zero,
                p_largest_value,
                "-0x0p+0",
                OpStatus::OK,
                Category::Zero,
            ),
            (
                m_zero,
                m_largest_value,
                "0x0p+0",
                OpStatus::OK,
                Category::Zero,
            ),
            (
                m_zero,
                p_smallest_value,
                "-0x0p+0",
                OpStatus::OK,
                Category::Zero,
            ),
            (
                m_zero,
                m_smallest_value,
                "0x0p+0",
                OpStatus::OK,
                Category::Zero,
            ),
            (
                m_zero,
                p_smallest_normalized,
                "-0x0p+0",
                OpStatus::OK,
                Category::Zero,
            ),
            (
                m_zero,
                m_smallest_normalized,
                "0x0p+0",
                OpStatus::OK,
                Category::Zero,
            ),
            (qnan, p_inf, "nan", OpStatus::OK, Category::NaN),
            (qnan, m_inf, "nan", OpStatus::OK, Category::NaN),
            (qnan, p_zero, "nan", OpStatus::OK, Category::NaN),
            (qnan, m_zero, "nan", OpStatus::OK, Category::NaN),
            (qnan, qnan, "nan", OpStatus::OK, Category::NaN),
            /*
    // See Note 1.
    (qnan, snan, "nan", OpStatus::INVALID_OP, Category::NaN),
            */
            (qnan, p_normal_value, "nan", OpStatus::OK, Category::NaN),
            (qnan, m_normal_value, "nan", OpStatus::OK, Category::NaN),
            (qnan, p_largest_value, "nan", OpStatus::OK, Category::NaN),
            (qnan, m_largest_value, "nan", OpStatus::OK, Category::NaN),
            (qnan, p_smallest_value, "nan", OpStatus::OK, Category::NaN),
            (qnan, m_smallest_value, "nan", OpStatus::OK, Category::NaN),
            (
                qnan,
                p_smallest_normalized,
                "nan",
                OpStatus::OK,
                Category::NaN,
            ),
            (
                qnan,
                m_smallest_normalized,
                "nan",
                OpStatus::OK,
                Category::NaN,
            ),
            /*
    // See Note 1.
    (snan, p_inf, "nan", OpStatus::INVALID_OP, Category::NaN),
    (snan, m_inf, "nan", OpStatus::INVALID_OP, Category::NaN),
    (snan, p_zero, "nan", OpStatus::INVALID_OP, Category::NaN),
    (snan, m_zero, "nan", OpStatus::INVALID_OP, Category::NaN),
    (snan, qnan, "nan", OpStatus::INVALID_OP, Category::NaN),
    (snan, snan, "nan", OpStatus::INVALID_OP, Category::NaN),
    (snan, p_normal_value, "nan", OpStatus::INVALID_OP, Category::NaN),
    (snan, m_normal_value, "nan", OpStatus::INVALID_OP, Category::NaN),
    (snan, p_largest_value, "nan", OpStatus::INVALID_OP, Category::NaN),
    (snan, m_largest_value, "nan", OpStatus::INVALID_OP, Category::NaN),
    (snan, p_smallest_value, "nan", OpStatus::INVALID_OP, Category::NaN),
    (snan, m_smallest_value, "nan", OpStatus::INVALID_OP, Category::NaN),
    (snan, p_smallest_normalized, "nan", OpStatus::INVALID_OP, Category::NaN),
    (snan, m_smallest_normalized, "nan", OpStatus::INVALID_OP, Category::NaN),
            */
            (
                p_normal_value,
                p_inf,
                "inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                p_normal_value,
                m_inf,
                "-inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                p_normal_value,
                p_zero,
                "0x0p+0",
                OpStatus::OK,
                Category::Zero,
            ),
            (
                p_normal_value,
                m_zero,
                "-0x0p+0",
                OpStatus::OK,
                Category::Zero,
            ),
            (p_normal_value, qnan, "nan", OpStatus::OK, Category::NaN),
            /*
    // See Note 1.
    (p_normal_value, snan, "nan", OpStatus::INVALID_OP, Category::NaN),
            */
            (
                p_normal_value,
                p_normal_value,
                "0x1p+0",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                p_normal_value,
                m_normal_value,
                "-0x1p+0",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                p_normal_value,
                p_largest_value,
                "0x1.fffffep+127",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                p_normal_value,
                m_largest_value,
                "-0x1.fffffep+127",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                p_normal_value,
                p_smallest_value,
                "0x1p-149",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                p_normal_value,
                m_smallest_value,
                "-0x1p-149",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                p_normal_value,
                p_smallest_normalized,
                "0x1p-126",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                p_normal_value,
                m_smallest_normalized,
                "-0x1p-126",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                m_normal_value,
                p_inf,
                "-inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                m_normal_value,
                m_inf,
                "inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                m_normal_value,
                p_zero,
                "-0x0p+0",
                OpStatus::OK,
                Category::Zero,
            ),
            (
                m_normal_value,
                m_zero,
                "0x0p+0",
                OpStatus::OK,
                Category::Zero,
            ),
            (m_normal_value, qnan, "nan", OpStatus::OK, Category::NaN),
            /*
    // See Note 1.
    (m_normal_value, snan, "nan", OpStatus::INVALID_OP, Category::NaN),
            */
            (
                m_normal_value,
                p_normal_value,
                "-0x1p+0",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                m_normal_value,
                m_normal_value,
                "0x1p+0",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                m_normal_value,
                p_largest_value,
                "-0x1.fffffep+127",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                m_normal_value,
                m_largest_value,
                "0x1.fffffep+127",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                m_normal_value,
                p_smallest_value,
                "-0x1p-149",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                m_normal_value,
                m_smallest_value,
                "0x1p-149",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                m_normal_value,
                p_smallest_normalized,
                "-0x1p-126",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                m_normal_value,
                m_smallest_normalized,
                "0x1p-126",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                p_largest_value,
                p_inf,
                "inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                p_largest_value,
                m_inf,
                "-inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                p_largest_value,
                p_zero,
                "0x0p+0",
                OpStatus::OK,
                Category::Zero,
            ),
            (
                p_largest_value,
                m_zero,
                "-0x0p+0",
                OpStatus::OK,
                Category::Zero,
            ),
            (p_largest_value, qnan, "nan", OpStatus::OK, Category::NaN),
            /*
    // See Note 1.
    (p_largest_value, snan, "nan", OpStatus::INVALID_OP, Category::NaN),
            */
            (
                p_largest_value,
                p_normal_value,
                "0x1.fffffep+127",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                p_largest_value,
                m_normal_value,
                "-0x1.fffffep+127",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                p_largest_value,
                p_largest_value,
                "inf",
                overflow_status,
                Category::Infinity,
            ),
            (
                p_largest_value,
                m_largest_value,
                "-inf",
                overflow_status,
                Category::Infinity,
            ),
            (
                p_largest_value,
                p_smallest_value,
                "0x1.fffffep-22",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                p_largest_value,
                m_smallest_value,
                "-0x1.fffffep-22",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                p_largest_value,
                p_smallest_normalized,
                "0x1.fffffep+1",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                p_largest_value,
                m_smallest_normalized,
                "-0x1.fffffep+1",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                m_largest_value,
                p_inf,
                "-inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                m_largest_value,
                m_inf,
                "inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                m_largest_value,
                p_zero,
                "-0x0p+0",
                OpStatus::OK,
                Category::Zero,
            ),
            (
                m_largest_value,
                m_zero,
                "0x0p+0",
                OpStatus::OK,
                Category::Zero,
            ),
            (m_largest_value, qnan, "nan", OpStatus::OK, Category::NaN),
            /*
    // See Note 1.
    (m_largest_value, snan, "nan", OpStatus::INVALID_OP, Category::NaN),
            */
            (
                m_largest_value,
                p_normal_value,
                "-0x1.fffffep+127",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                m_largest_value,
                m_normal_value,
                "0x1.fffffep+127",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                m_largest_value,
                p_largest_value,
                "-inf",
                overflow_status,
                Category::Infinity,
            ),
            (
                m_largest_value,
                m_largest_value,
                "inf",
                overflow_status,
                Category::Infinity,
            ),
            (
                m_largest_value,
                p_smallest_value,
                "-0x1.fffffep-22",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                m_largest_value,
                m_smallest_value,
                "0x1.fffffep-22",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                m_largest_value,
                p_smallest_normalized,
                "-0x1.fffffep+1",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                m_largest_value,
                m_smallest_normalized,
                "0x1.fffffep+1",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                p_smallest_value,
                p_inf,
                "inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                p_smallest_value,
                m_inf,
                "-inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                p_smallest_value,
                p_zero,
                "0x0p+0",
                OpStatus::OK,
                Category::Zero,
            ),
            (
                p_smallest_value,
                m_zero,
                "-0x0p+0",
                OpStatus::OK,
                Category::Zero,
            ),
            (p_smallest_value, qnan, "nan", OpStatus::OK, Category::NaN),
            /*
    // See Note 1.
    (p_smallest_value, snan, "nan", OpStatus::INVALID_OP, Category::NaN),
            */
            (
                p_smallest_value,
                p_normal_value,
                "0x1p-149",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                p_smallest_value,
                m_normal_value,
                "-0x1p-149",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                p_smallest_value,
                p_largest_value,
                "0x1.fffffep-22",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                p_smallest_value,
                m_largest_value,
                "-0x1.fffffep-22",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                p_smallest_value,
                p_smallest_value,
                "0x0p+0",
                underflow_status,
                Category::Zero,
            ),
            (
                p_smallest_value,
                m_smallest_value,
                "-0x0p+0",
                underflow_status,
                Category::Zero,
            ),
            (
                p_smallest_value,
                p_smallest_normalized,
                "0x0p+0",
                underflow_status,
                Category::Zero,
            ),
            (
                p_smallest_value,
                m_smallest_normalized,
                "-0x0p+0",
                underflow_status,
                Category::Zero,
            ),
            (
                m_smallest_value,
                p_inf,
                "-inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                m_smallest_value,
                m_inf,
                "inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                m_smallest_value,
                p_zero,
                "-0x0p+0",
                OpStatus::OK,
                Category::Zero,
            ),
            (
                m_smallest_value,
                m_zero,
                "0x0p+0",
                OpStatus::OK,
                Category::Zero,
            ),
            (m_smallest_value, qnan, "nan", OpStatus::OK, Category::NaN),
            /*
    // See Note 1.
    (m_smallest_value, snan, "nan", OpStatus::INVALID_OP, Category::NaN),
            */
            (
                m_smallest_value,
                p_normal_value,
                "-0x1p-149",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                m_smallest_value,
                m_normal_value,
                "0x1p-149",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                m_smallest_value,
                p_largest_value,
                "-0x1.fffffep-22",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                m_smallest_value,
                m_largest_value,
                "0x1.fffffep-22",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                m_smallest_value,
                p_smallest_value,
                "-0x0p+0",
                underflow_status,
                Category::Zero,
            ),
            (
                m_smallest_value,
                m_smallest_value,
                "0x0p+0",
                underflow_status,
                Category::Zero,
            ),
            (
                m_smallest_value,
                p_smallest_normalized,
                "-0x0p+0",
                underflow_status,
                Category::Zero,
            ),
            (
                m_smallest_value,
                m_smallest_normalized,
                "0x0p+0",
                underflow_status,
                Category::Zero,
            ),
            (
                p_smallest_normalized,
                p_inf,
                "inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                p_smallest_normalized,
                m_inf,
                "-inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                p_smallest_normalized,
                p_zero,
                "0x0p+0",
                OpStatus::OK,
                Category::Zero,
            ),
            (
                p_smallest_normalized,
                m_zero,
                "-0x0p+0",
                OpStatus::OK,
                Category::Zero,
            ),
            (
                p_smallest_normalized,
                qnan,
                "nan",
                OpStatus::OK,
                Category::NaN,
            ),
            /*
    // See Note 1.
    (p_smallest_normalized, snan, "nan", OpStatus::INVALID_OP, Category::NaN),
            */
            (
                p_smallest_normalized,
                p_normal_value,
                "0x1p-126",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                p_smallest_normalized,
                m_normal_value,
                "-0x1p-126",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                p_smallest_normalized,
                p_largest_value,
                "0x1.fffffep+1",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                p_smallest_normalized,
                m_largest_value,
                "-0x1.fffffep+1",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                p_smallest_normalized,
                p_smallest_value,
                "0x0p+0",
                underflow_status,
                Category::Zero,
            ),
            (
                p_smallest_normalized,
                m_smallest_value,
                "-0x0p+0",
                underflow_status,
                Category::Zero,
            ),
            (
                p_smallest_normalized,
                p_smallest_normalized,
                "0x0p+0",
                underflow_status,
                Category::Zero,
            ),
            (
                p_smallest_normalized,
                m_smallest_normalized,
                "-0x0p+0",
                underflow_status,
                Category::Zero,
            ),
            (
                m_smallest_normalized,
                p_inf,
                "-inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                m_smallest_normalized,
                m_inf,
                "inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                m_smallest_normalized,
                p_zero,
                "-0x0p+0",
                OpStatus::OK,
                Category::Zero,
            ),
            (
                m_smallest_normalized,
                m_zero,
                "0x0p+0",
                OpStatus::OK,
                Category::Zero,
            ),
            (
                m_smallest_normalized,
                qnan,
                "nan",
                OpStatus::OK,
                Category::NaN,
            ),
            /*
    // See Note 1.
    (m_smallest_normalized, snan, "nan", OpStatus::INVALID_OP, Category::NaN),
            */
            (
                m_smallest_normalized,
                p_normal_value,
                "-0x1p-126",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                m_smallest_normalized,
                m_normal_value,
                "0x1p-126",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                m_smallest_normalized,
                p_largest_value,
                "-0x1.fffffep+1",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                m_smallest_normalized,
                m_largest_value,
                "0x1.fffffep+1",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                m_smallest_normalized,
                p_smallest_value,
                "-0x0p+0",
                underflow_status,
                Category::Zero,
            ),
            (
                m_smallest_normalized,
                m_smallest_value,
                "0x0p+0",
                underflow_status,
                Category::Zero,
            ),
            (
                m_smallest_normalized,
                p_smallest_normalized,
                "-0x0p+0",
                underflow_status,
                Category::Zero,
            ),
            (
                m_smallest_normalized,
                m_smallest_normalized,
                "0x0p+0",
                underflow_status,
                Category::Zero,
            ),
        ];

        for &(mut x, y, result, status, category) in &special_cases[..] {
            assert_eq!(x.mul_rounded(y, Round::NearestTiesToEven), status);
            assert_eq!(x.category(), category);
            assert!(result.parse::<IeeeSingle>().unwrap().bitwise_eq(x));
        }
    }

    #[test]
    fn divide() {
        // Test Special Cases against each other and normal values.

        // FIXMES/NOTES:
        // 1. Since we perform only default exception handling all operations with
        // signaling NaNs should have a result that is a quiet NaN. Currently they
        // return sNaN.

        let p_inf = IeeeSingle::inf();
        let m_inf = -IeeeSingle::inf();
        let p_zero = IeeeSingle::zero();
        let m_zero = -IeeeSingle::zero();
        let qnan = IeeeSingle::nan();
        let p_normal_value = "0x1p+0".parse::<IeeeSingle>().unwrap();
        let m_normal_value = "-0x1p+0".parse::<IeeeSingle>().unwrap();
        let p_largest_value = IeeeSingle::largest();
        let m_largest_value = -IeeeSingle::largest();
        let p_smallest_value = IeeeSingle::smallest();
        let m_smallest_value = -IeeeSingle::smallest();
        let p_smallest_normalized = IeeeSingle::smallest_normalized();
        let m_smallest_normalized = -IeeeSingle::smallest_normalized();

        let overflow_status = OpStatus::OVERFLOW | OpStatus::INEXACT;
        let underflow_status = OpStatus::UNDERFLOW | OpStatus::INEXACT;

        let special_cases = [
            (p_inf, p_inf, "nan", OpStatus::INVALID_OP, Category::NaN),
            (p_inf, m_inf, "nan", OpStatus::INVALID_OP, Category::NaN),
            (p_inf, p_zero, "inf", OpStatus::OK, Category::Infinity),
            (p_inf, m_zero, "-inf", OpStatus::OK, Category::Infinity),
            (p_inf, qnan, "nan", OpStatus::OK, Category::NaN),
            /*
    // See Note 1.
    (p_inf, snan, "nan", OpStatus::INVALID_OP, Category::NaN),
            */
            (
                p_inf,
                p_normal_value,
                "inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                p_inf,
                m_normal_value,
                "-inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                p_inf,
                p_largest_value,
                "inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                p_inf,
                m_largest_value,
                "-inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                p_inf,
                p_smallest_value,
                "inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                p_inf,
                m_smallest_value,
                "-inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                p_inf,
                p_smallest_normalized,
                "inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                p_inf,
                m_smallest_normalized,
                "-inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (m_inf, p_inf, "nan", OpStatus::INVALID_OP, Category::NaN),
            (m_inf, m_inf, "nan", OpStatus::INVALID_OP, Category::NaN),
            (m_inf, p_zero, "-inf", OpStatus::OK, Category::Infinity),
            (m_inf, m_zero, "inf", OpStatus::OK, Category::Infinity),
            (m_inf, qnan, "nan", OpStatus::OK, Category::NaN),
            /*
    // See Note 1.
    (m_inf, snan, "nan", OpStatus::INVALID_OP, Category::NaN),
            */
            (
                m_inf,
                p_normal_value,
                "-inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                m_inf,
                m_normal_value,
                "inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                m_inf,
                p_largest_value,
                "-inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                m_inf,
                m_largest_value,
                "inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                m_inf,
                p_smallest_value,
                "-inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                m_inf,
                m_smallest_value,
                "inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                m_inf,
                p_smallest_normalized,
                "-inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (
                m_inf,
                m_smallest_normalized,
                "inf",
                OpStatus::OK,
                Category::Infinity,
            ),
            (p_zero, p_inf, "0x0p+0", OpStatus::OK, Category::Zero),
            (p_zero, m_inf, "-0x0p+0", OpStatus::OK, Category::Zero),
            (p_zero, p_zero, "nan", OpStatus::INVALID_OP, Category::NaN),
            (p_zero, m_zero, "nan", OpStatus::INVALID_OP, Category::NaN),
            (p_zero, qnan, "nan", OpStatus::OK, Category::NaN),
            /*
    // See Note 1.
    (p_zero, snan, "nan", OpStatus::INVALID_OP, Category::NaN),
            */
            (
                p_zero,
                p_normal_value,
                "0x0p+0",
                OpStatus::OK,
                Category::Zero,
            ),
            (
                p_zero,
                m_normal_value,
                "-0x0p+0",
                OpStatus::OK,
                Category::Zero,
            ),
            (
                p_zero,
                p_largest_value,
                "0x0p+0",
                OpStatus::OK,
                Category::Zero,
            ),
            (
                p_zero,
                m_largest_value,
                "-0x0p+0",
                OpStatus::OK,
                Category::Zero,
            ),
            (
                p_zero,
                p_smallest_value,
                "0x0p+0",
                OpStatus::OK,
                Category::Zero,
            ),
            (
                p_zero,
                m_smallest_value,
                "-0x0p+0",
                OpStatus::OK,
                Category::Zero,
            ),
            (
                p_zero,
                p_smallest_normalized,
                "0x0p+0",
                OpStatus::OK,
                Category::Zero,
            ),
            (
                p_zero,
                m_smallest_normalized,
                "-0x0p+0",
                OpStatus::OK,
                Category::Zero,
            ),
            (m_zero, p_inf, "-0x0p+0", OpStatus::OK, Category::Zero),
            (m_zero, m_inf, "0x0p+0", OpStatus::OK, Category::Zero),
            (m_zero, p_zero, "nan", OpStatus::INVALID_OP, Category::NaN),
            (m_zero, m_zero, "nan", OpStatus::INVALID_OP, Category::NaN),
            (m_zero, qnan, "nan", OpStatus::OK, Category::NaN),
            /*
    // See Note 1.
    (m_zero, snan, "nan", OpStatus::INVALID_OP, Category::NaN),
            */
            (
                m_zero,
                p_normal_value,
                "-0x0p+0",
                OpStatus::OK,
                Category::Zero,
            ),
            (
                m_zero,
                m_normal_value,
                "0x0p+0",
                OpStatus::OK,
                Category::Zero,
            ),
            (
                m_zero,
                p_largest_value,
                "-0x0p+0",
                OpStatus::OK,
                Category::Zero,
            ),
            (
                m_zero,
                m_largest_value,
                "0x0p+0",
                OpStatus::OK,
                Category::Zero,
            ),
            (
                m_zero,
                p_smallest_value,
                "-0x0p+0",
                OpStatus::OK,
                Category::Zero,
            ),
            (
                m_zero,
                m_smallest_value,
                "0x0p+0",
                OpStatus::OK,
                Category::Zero,
            ),
            (
                m_zero,
                p_smallest_normalized,
                "-0x0p+0",
                OpStatus::OK,
                Category::Zero,
            ),
            (
                m_zero,
                m_smallest_normalized,
                "0x0p+0",
                OpStatus::OK,
                Category::Zero,
            ),
            (qnan, p_inf, "nan", OpStatus::OK, Category::NaN),
            (qnan, m_inf, "nan", OpStatus::OK, Category::NaN),
            (qnan, p_zero, "nan", OpStatus::OK, Category::NaN),
            (qnan, m_zero, "nan", OpStatus::OK, Category::NaN),
            (qnan, qnan, "nan", OpStatus::OK, Category::NaN),
            /*
    // See Note 1.
    (qnan, snan, "nan", OpStatus::INVALID_OP, Category::NaN),
            */
            (qnan, p_normal_value, "nan", OpStatus::OK, Category::NaN),
            (qnan, m_normal_value, "nan", OpStatus::OK, Category::NaN),
            (qnan, p_largest_value, "nan", OpStatus::OK, Category::NaN),
            (qnan, m_largest_value, "nan", OpStatus::OK, Category::NaN),
            (qnan, p_smallest_value, "nan", OpStatus::OK, Category::NaN),
            (qnan, m_smallest_value, "nan", OpStatus::OK, Category::NaN),
            (
                qnan,
                p_smallest_normalized,
                "nan",
                OpStatus::OK,
                Category::NaN,
            ),
            (
                qnan,
                m_smallest_normalized,
                "nan",
                OpStatus::OK,
                Category::NaN,
            ),
            /*
    // See Note 1.
    (snan, p_inf, "nan", OpStatus::INVALID_OP, Category::NaN),
    (snan, m_inf, "nan", OpStatus::INVALID_OP, Category::NaN),
    (snan, p_zero, "nan", OpStatus::INVALID_OP, Category::NaN),
    (snan, m_zero, "nan", OpStatus::INVALID_OP, Category::NaN),
    (snan, qnan, "nan", OpStatus::INVALID_OP, Category::NaN),
    (snan, snan, "nan", OpStatus::INVALID_OP, Category::NaN),
    (snan, p_normal_value, "nan", OpStatus::INVALID_OP, Category::NaN),
    (snan, m_normal_value, "nan", OpStatus::INVALID_OP, Category::NaN),
    (snan, p_largest_value, "nan", OpStatus::INVALID_OP, Category::NaN),
    (snan, m_largest_value, "nan", OpStatus::INVALID_OP, Category::NaN),
    (snan, p_smallest_value, "nan", OpStatus::INVALID_OP, Category::NaN),
    (snan, m_smallest_value, "nan", OpStatus::INVALID_OP, Category::NaN),
    (snan, p_smallest_normalized, "nan", OpStatus::INVALID_OP, Category::NaN),
    (snan, m_smallest_normalized, "nan", OpStatus::INVALID_OP, Category::NaN),
            */
            (
                p_normal_value,
                p_inf,
                "0x0p+0",
                OpStatus::OK,
                Category::Zero,
            ),
            (
                p_normal_value,
                m_inf,
                "-0x0p+0",
                OpStatus::OK,
                Category::Zero,
            ),
            (
                p_normal_value,
                p_zero,
                "inf",
                OpStatus::DIV_BY_ZERO,
                Category::Infinity,
            ),
            (
                p_normal_value,
                m_zero,
                "-inf",
                OpStatus::DIV_BY_ZERO,
                Category::Infinity,
            ),
            (p_normal_value, qnan, "nan", OpStatus::OK, Category::NaN),
            /*
    // See Note 1.
    (p_normal_value, snan, "nan", OpStatus::INVALID_OP, Category::NaN),
            */
            (
                p_normal_value,
                p_normal_value,
                "0x1p+0",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                p_normal_value,
                m_normal_value,
                "-0x1p+0",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                p_normal_value,
                p_largest_value,
                "0x1p-128",
                underflow_status,
                Category::Normal,
            ),
            (
                p_normal_value,
                m_largest_value,
                "-0x1p-128",
                underflow_status,
                Category::Normal,
            ),
            (
                p_normal_value,
                p_smallest_value,
                "inf",
                overflow_status,
                Category::Infinity,
            ),
            (
                p_normal_value,
                m_smallest_value,
                "-inf",
                overflow_status,
                Category::Infinity,
            ),
            (
                p_normal_value,
                p_smallest_normalized,
                "0x1p+126",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                p_normal_value,
                m_smallest_normalized,
                "-0x1p+126",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                m_normal_value,
                p_inf,
                "-0x0p+0",
                OpStatus::OK,
                Category::Zero,
            ),
            (
                m_normal_value,
                m_inf,
                "0x0p+0",
                OpStatus::OK,
                Category::Zero,
            ),
            (
                m_normal_value,
                p_zero,
                "-inf",
                OpStatus::DIV_BY_ZERO,
                Category::Infinity,
            ),
            (
                m_normal_value,
                m_zero,
                "inf",
                OpStatus::DIV_BY_ZERO,
                Category::Infinity,
            ),
            (m_normal_value, qnan, "nan", OpStatus::OK, Category::NaN),
            /*
    // See Note 1.
    (m_normal_value, snan, "nan", OpStatus::INVALID_OP, Category::NaN),
            */
            (
                m_normal_value,
                p_normal_value,
                "-0x1p+0",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                m_normal_value,
                m_normal_value,
                "0x1p+0",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                m_normal_value,
                p_largest_value,
                "-0x1p-128",
                underflow_status,
                Category::Normal,
            ),
            (
                m_normal_value,
                m_largest_value,
                "0x1p-128",
                underflow_status,
                Category::Normal,
            ),
            (
                m_normal_value,
                p_smallest_value,
                "-inf",
                overflow_status,
                Category::Infinity,
            ),
            (
                m_normal_value,
                m_smallest_value,
                "inf",
                overflow_status,
                Category::Infinity,
            ),
            (
                m_normal_value,
                p_smallest_normalized,
                "-0x1p+126",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                m_normal_value,
                m_smallest_normalized,
                "0x1p+126",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                p_largest_value,
                p_inf,
                "0x0p+0",
                OpStatus::OK,
                Category::Zero,
            ),
            (
                p_largest_value,
                m_inf,
                "-0x0p+0",
                OpStatus::OK,
                Category::Zero,
            ),
            (
                p_largest_value,
                p_zero,
                "inf",
                OpStatus::DIV_BY_ZERO,
                Category::Infinity,
            ),
            (
                p_largest_value,
                m_zero,
                "-inf",
                OpStatus::DIV_BY_ZERO,
                Category::Infinity,
            ),
            (p_largest_value, qnan, "nan", OpStatus::OK, Category::NaN),
            /*
    // See Note 1.
    (p_largest_value, snan, "nan", OpStatus::INVALID_OP, Category::NaN),
            */
            (
                p_largest_value,
                p_normal_value,
                "0x1.fffffep+127",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                p_largest_value,
                m_normal_value,
                "-0x1.fffffep+127",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                p_largest_value,
                p_largest_value,
                "0x1p+0",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                p_largest_value,
                m_largest_value,
                "-0x1p+0",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                p_largest_value,
                p_smallest_value,
                "inf",
                overflow_status,
                Category::Infinity,
            ),
            (
                p_largest_value,
                m_smallest_value,
                "-inf",
                overflow_status,
                Category::Infinity,
            ),
            (
                p_largest_value,
                p_smallest_normalized,
                "inf",
                overflow_status,
                Category::Infinity,
            ),
            (
                p_largest_value,
                m_smallest_normalized,
                "-inf",
                overflow_status,
                Category::Infinity,
            ),
            (
                m_largest_value,
                p_inf,
                "-0x0p+0",
                OpStatus::OK,
                Category::Zero,
            ),
            (
                m_largest_value,
                m_inf,
                "0x0p+0",
                OpStatus::OK,
                Category::Zero,
            ),
            (
                m_largest_value,
                p_zero,
                "-inf",
                OpStatus::DIV_BY_ZERO,
                Category::Infinity,
            ),
            (
                m_largest_value,
                m_zero,
                "inf",
                OpStatus::DIV_BY_ZERO,
                Category::Infinity,
            ),
            (m_largest_value, qnan, "nan", OpStatus::OK, Category::NaN),
            /*
    // See Note 1.
    (m_largest_value, snan, "nan", OpStatus::INVALID_OP, Category::NaN),
            */
            (
                m_largest_value,
                p_normal_value,
                "-0x1.fffffep+127",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                m_largest_value,
                m_normal_value,
                "0x1.fffffep+127",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                m_largest_value,
                p_largest_value,
                "-0x1p+0",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                m_largest_value,
                m_largest_value,
                "0x1p+0",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                m_largest_value,
                p_smallest_value,
                "-inf",
                overflow_status,
                Category::Infinity,
            ),
            (
                m_largest_value,
                m_smallest_value,
                "inf",
                overflow_status,
                Category::Infinity,
            ),
            (
                m_largest_value,
                p_smallest_normalized,
                "-inf",
                overflow_status,
                Category::Infinity,
            ),
            (
                m_largest_value,
                m_smallest_normalized,
                "inf",
                overflow_status,
                Category::Infinity,
            ),
            (
                p_smallest_value,
                p_inf,
                "0x0p+0",
                OpStatus::OK,
                Category::Zero,
            ),
            (
                p_smallest_value,
                m_inf,
                "-0x0p+0",
                OpStatus::OK,
                Category::Zero,
            ),
            (
                p_smallest_value,
                p_zero,
                "inf",
                OpStatus::DIV_BY_ZERO,
                Category::Infinity,
            ),
            (
                p_smallest_value,
                m_zero,
                "-inf",
                OpStatus::DIV_BY_ZERO,
                Category::Infinity,
            ),
            (p_smallest_value, qnan, "nan", OpStatus::OK, Category::NaN),
            /*
    // See Note 1.
    (p_smallest_value, snan, "nan", OpStatus::INVALID_OP, Category::NaN),
            */
            (
                p_smallest_value,
                p_normal_value,
                "0x1p-149",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                p_smallest_value,
                m_normal_value,
                "-0x1p-149",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                p_smallest_value,
                p_largest_value,
                "0x0p+0",
                underflow_status,
                Category::Zero,
            ),
            (
                p_smallest_value,
                m_largest_value,
                "-0x0p+0",
                underflow_status,
                Category::Zero,
            ),
            (
                p_smallest_value,
                p_smallest_value,
                "0x1p+0",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                p_smallest_value,
                m_smallest_value,
                "-0x1p+0",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                p_smallest_value,
                p_smallest_normalized,
                "0x1p-23",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                p_smallest_value,
                m_smallest_normalized,
                "-0x1p-23",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                m_smallest_value,
                p_inf,
                "-0x0p+0",
                OpStatus::OK,
                Category::Zero,
            ),
            (
                m_smallest_value,
                m_inf,
                "0x0p+0",
                OpStatus::OK,
                Category::Zero,
            ),
            (
                m_smallest_value,
                p_zero,
                "-inf",
                OpStatus::DIV_BY_ZERO,
                Category::Infinity,
            ),
            (
                m_smallest_value,
                m_zero,
                "inf",
                OpStatus::DIV_BY_ZERO,
                Category::Infinity,
            ),
            (m_smallest_value, qnan, "nan", OpStatus::OK, Category::NaN),
            /*
    // See Note 1.
    (m_smallest_value, snan, "nan", OpStatus::INVALID_OP, Category::NaN),
            */
            (
                m_smallest_value,
                p_normal_value,
                "-0x1p-149",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                m_smallest_value,
                m_normal_value,
                "0x1p-149",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                m_smallest_value,
                p_largest_value,
                "-0x0p+0",
                underflow_status,
                Category::Zero,
            ),
            (
                m_smallest_value,
                m_largest_value,
                "0x0p+0",
                underflow_status,
                Category::Zero,
            ),
            (
                m_smallest_value,
                p_smallest_value,
                "-0x1p+0",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                m_smallest_value,
                m_smallest_value,
                "0x1p+0",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                m_smallest_value,
                p_smallest_normalized,
                "-0x1p-23",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                m_smallest_value,
                m_smallest_normalized,
                "0x1p-23",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                p_smallest_normalized,
                p_inf,
                "0x0p+0",
                OpStatus::OK,
                Category::Zero,
            ),
            (
                p_smallest_normalized,
                m_inf,
                "-0x0p+0",
                OpStatus::OK,
                Category::Zero,
            ),
            (
                p_smallest_normalized,
                p_zero,
                "inf",
                OpStatus::DIV_BY_ZERO,
                Category::Infinity,
            ),
            (
                p_smallest_normalized,
                m_zero,
                "-inf",
                OpStatus::DIV_BY_ZERO,
                Category::Infinity,
            ),
            (
                p_smallest_normalized,
                qnan,
                "nan",
                OpStatus::OK,
                Category::NaN,
            ),
            /*
    // See Note 1.
    (p_smallest_normalized, snan, "nan", OpStatus::INVALID_OP, Category::NaN),
            */
            (
                p_smallest_normalized,
                p_normal_value,
                "0x1p-126",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                p_smallest_normalized,
                m_normal_value,
                "-0x1p-126",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                p_smallest_normalized,
                p_largest_value,
                "0x0p+0",
                underflow_status,
                Category::Zero,
            ),
            (
                p_smallest_normalized,
                m_largest_value,
                "-0x0p+0",
                underflow_status,
                Category::Zero,
            ),
            (
                p_smallest_normalized,
                p_smallest_value,
                "0x1p+23",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                p_smallest_normalized,
                m_smallest_value,
                "-0x1p+23",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                p_smallest_normalized,
                p_smallest_normalized,
                "0x1p+0",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                p_smallest_normalized,
                m_smallest_normalized,
                "-0x1p+0",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                m_smallest_normalized,
                p_inf,
                "-0x0p+0",
                OpStatus::OK,
                Category::Zero,
            ),
            (
                m_smallest_normalized,
                m_inf,
                "0x0p+0",
                OpStatus::OK,
                Category::Zero,
            ),
            (
                m_smallest_normalized,
                p_zero,
                "-inf",
                OpStatus::DIV_BY_ZERO,
                Category::Infinity,
            ),
            (
                m_smallest_normalized,
                m_zero,
                "inf",
                OpStatus::DIV_BY_ZERO,
                Category::Infinity,
            ),
            (
                m_smallest_normalized,
                qnan,
                "nan",
                OpStatus::OK,
                Category::NaN,
            ),
            /*
    // See Note 1.
    (m_smallest_normalized, snan, "nan", OpStatus::INVALID_OP, Category::NaN),
            */
            (
                m_smallest_normalized,
                p_normal_value,
                "-0x1p-126",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                m_smallest_normalized,
                m_normal_value,
                "0x1p-126",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                m_smallest_normalized,
                p_largest_value,
                "-0x0p+0",
                underflow_status,
                Category::Zero,
            ),
            (
                m_smallest_normalized,
                m_largest_value,
                "0x0p+0",
                underflow_status,
                Category::Zero,
            ),
            (
                m_smallest_normalized,
                p_smallest_value,
                "-0x1p+23",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                m_smallest_normalized,
                m_smallest_value,
                "0x1p+23",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                m_smallest_normalized,
                p_smallest_normalized,
                "-0x1p+0",
                OpStatus::OK,
                Category::Normal,
            ),
            (
                m_smallest_normalized,
                m_smallest_normalized,
                "0x1p+0",
                OpStatus::OK,
                Category::Normal,
            ),
        ];

        for &(mut x, y, result, status, category) in &special_cases[..] {
            assert_eq!(x.div_rounded(y, Round::NearestTiesToEven), status);
            assert_eq!(x.category(), category);
            assert!(result.parse::<IeeeSingle>().unwrap().bitwise_eq(x));
        }
    }

    #[test]
    fn operator_overloads() {
        // This is mostly testing that these operator overloads compile.
        let one = "0x1p+0".parse::<IeeeSingle>().unwrap();
        let two = "0x2p+0".parse::<IeeeSingle>().unwrap();
        assert!(two.bitwise_eq(one + one));
        assert!(one.bitwise_eq(two - one));
        assert!(two.bitwise_eq(one * two));
        assert!(one.bitwise_eq(two / two));
    }

    #[test]
    fn abs() {
        let p_inf = IeeeSingle::inf();
        let m_inf = -IeeeSingle::inf();
        let p_zero = IeeeSingle::zero();
        let m_zero = -IeeeSingle::zero();
        let p_qnan = IeeeSingle::nan();
        let m_qnan = -IeeeSingle::nan();
        let p_snan = IeeeSingle::snan(None);
        let m_snan = -IeeeSingle::snan(None);
        let p_normal_value = "0x1p+0".parse::<IeeeSingle>().unwrap();
        let m_normal_value = "-0x1p+0".parse::<IeeeSingle>().unwrap();
        let p_largest_value = IeeeSingle::largest();
        let m_largest_value = -IeeeSingle::largest();
        let p_smallest_value = IeeeSingle::smallest();
        let m_smallest_value = -IeeeSingle::smallest();
        let p_smallest_normalized = IeeeSingle::smallest_normalized();
        let m_smallest_normalized = -IeeeSingle::smallest_normalized();

        assert!(p_inf.bitwise_eq(p_inf.abs()));
        assert!(p_inf.bitwise_eq(m_inf.abs()));
        assert!(p_zero.bitwise_eq(p_zero.abs()));
        assert!(p_zero.bitwise_eq(m_zero.abs()));
        assert!(p_qnan.bitwise_eq(p_qnan.abs()));
        assert!(p_qnan.bitwise_eq(m_qnan.abs()));
        assert!(p_snan.bitwise_eq(p_snan.abs()));
        assert!(p_snan.bitwise_eq(m_snan.abs()));
        assert!(p_normal_value.bitwise_eq(p_normal_value.abs()));
        assert!(p_normal_value.bitwise_eq(m_normal_value.abs()));
        assert!(p_largest_value.bitwise_eq(p_largest_value.abs()));
        assert!(p_largest_value.bitwise_eq(m_largest_value.abs()));
        assert!(p_smallest_value.bitwise_eq(p_smallest_value.abs()));
        assert!(p_smallest_value.bitwise_eq(m_smallest_value.abs()));
        assert!(p_smallest_normalized.bitwise_eq(
            p_smallest_normalized.abs(),
        ));
        assert!(p_smallest_normalized.bitwise_eq(
            m_smallest_normalized.abs(),
        ));
    }

    #[test]
    fn neg() {
        let one = "1.0".parse::<IeeeSingle>().unwrap();
        let neg_one = "-1.0".parse::<IeeeSingle>().unwrap();
        let zero = IeeeSingle::zero();
        let neg_zero = -IeeeSingle::zero();
        let inf = IeeeSingle::inf();
        let neg_inf = -IeeeSingle::inf();
        let qnan = IeeeSingle::nan();
        let neg_qnan = -IeeeSingle::nan();

        assert!(neg_one.bitwise_eq(-one));
        assert!(one.bitwise_eq(-neg_one));
        assert!(neg_zero.bitwise_eq(-zero));
        assert!(zero.bitwise_eq(-neg_zero));
        assert!(neg_inf.bitwise_eq(-inf));
        assert!(inf.bitwise_eq(-neg_inf));
        assert!(neg_inf.bitwise_eq(-inf));
        assert!(inf.bitwise_eq(-neg_inf));
        assert!(neg_qnan.bitwise_eq(-qnan));
        assert!(qnan.bitwise_eq(-neg_qnan));
    }

    #[test]
    fn ilogb() {
        assert_eq!(-1074, IeeeDouble::smallest().ilogb());
        assert_eq!(-1074, (-IeeeDouble::smallest()).ilogb());
        assert_eq!(
            -1023,
            "0x1.ffffffffffffep-1024"
                .parse::<IeeeDouble>()
                .unwrap()
                .ilogb()
        );
        assert_eq!(
            -1023,
            "0x1.ffffffffffffep-1023"
                .parse::<IeeeDouble>()
                .unwrap()
                .ilogb()
        );
        assert_eq!(
            -1023,
            "-0x1.ffffffffffffep-1023"
                .parse::<IeeeDouble>()
                .unwrap()
                .ilogb()
        );
        assert_eq!(-51, "0x1p-51".parse::<IeeeDouble>().unwrap().ilogb());
        assert_eq!(
            -1023,
            "0x1.c60f120d9f87cp-1023"
                .parse::<IeeeDouble>()
                .unwrap()
                .ilogb()
        );
        assert_eq!(-2, "0x0.ffffp-1".parse::<IeeeDouble>().unwrap().ilogb());
        assert_eq!(
            -1023,
            "0x1.fffep-1023".parse::<IeeeDouble>().unwrap().ilogb()
        );
        assert_eq!(1023, IeeeDouble::largest().ilogb());
        assert_eq!(1023, (-IeeeDouble::largest()).ilogb());


        assert_eq!(0, "0x1p+0".parse::<IeeeSingle>().unwrap().ilogb());
        assert_eq!(0, "-0x1p+0".parse::<IeeeSingle>().unwrap().ilogb());
        assert_eq!(42, "0x1p+42".parse::<IeeeSingle>().unwrap().ilogb());
        assert_eq!(-42, "0x1p-42".parse::<IeeeSingle>().unwrap().ilogb());

        assert_eq!(IEK_INF, IeeeSingle::inf().ilogb());
        assert_eq!(IEK_INF, (-IeeeSingle::inf()).ilogb());
        assert_eq!(IEK_ZERO, IeeeSingle::zero().ilogb());
        assert_eq!(IEK_ZERO, (-IeeeSingle::zero()).ilogb());
        assert_eq!(IEK_NAN, IeeeSingle::nan().ilogb());
        assert_eq!(IEK_NAN, IeeeSingle::snan(None).ilogb());

        assert_eq!(127, IeeeSingle::largest().ilogb());
        assert_eq!(127, (-IeeeSingle::largest()).ilogb());

        assert_eq!(-149, IeeeSingle::smallest().ilogb());
        assert_eq!(-149, (-IeeeSingle::smallest()).ilogb());
        assert_eq!(-126, IeeeSingle::smallest_normalized().ilogb());
        assert_eq!(-126, (-IeeeSingle::smallest_normalized()).ilogb());
    }

    #[test]
    fn scalbn() {
        let round = Round::NearestTiesToEven;
        assert!("0x1p+0".parse::<IeeeSingle>().unwrap().bitwise_eq(
            "0x1p+0".parse::<IeeeSingle>().unwrap().scalbn(0, round),
        ));
        assert!("0x1p+42".parse::<IeeeSingle>().unwrap().bitwise_eq(
            "0x1p+0".parse::<IeeeSingle>().unwrap().scalbn(42, round),
        ));
        assert!("0x1p-42".parse::<IeeeSingle>().unwrap().bitwise_eq(
            "0x1p+0".parse::<IeeeSingle>().unwrap().scalbn(-42, round),
        ));

        let p_inf = IeeeSingle::inf();
        let m_inf = -IeeeSingle::inf();
        let p_zero = IeeeSingle::zero();
        let m_zero = -IeeeSingle::zero();
        let p_qnan = IeeeSingle::nan();
        let m_qnan = -IeeeSingle::nan();
        let snan = IeeeSingle::snan(None);

        assert!(p_inf.bitwise_eq(p_inf.scalbn(0, round)));
        assert!(m_inf.bitwise_eq(m_inf.scalbn(0, round)));
        assert!(p_zero.bitwise_eq(p_zero.scalbn(0, round)));
        assert!(m_zero.bitwise_eq(m_zero.scalbn(0, round)));
        assert!(p_qnan.bitwise_eq(p_qnan.scalbn(0, round)));
        assert!(m_qnan.bitwise_eq(m_qnan.scalbn(0, round)));
        assert!(!snan.scalbn(0, round).is_signaling());

        let scalbn_snan = snan.scalbn(1, round);
        assert!(scalbn_snan.is_nan() && !scalbn_snan.is_signaling());

        // Make sure highest bit of payload is preserved.
        let payload = (1 << 50) | (1 << 49) | (1234 << 32) | 1;

        let snan_with_payload = IeeeDouble::snan(Some(payload));
        let quiet_payload = snan_with_payload.scalbn(1, round);
        assert!(quiet_payload.is_nan() && !quiet_payload.is_signaling());
        assert_eq!(payload, quiet_payload.to_bits() & ((1 << 51) - 1));

        assert!(p_inf.bitwise_eq(
            "0x1p+0".parse::<IeeeSingle>().unwrap().scalbn(
                128,
                round,
            ),
        ));
        assert!(m_inf.bitwise_eq(
            "-0x1p+0".parse::<IeeeSingle>().unwrap().scalbn(
                128,
                round,
            ),
        ));
        assert!(p_inf.bitwise_eq(
            "0x1p+127".parse::<IeeeSingle>().unwrap().scalbn(
                1,
                round,
            ),
        ));
        assert!(p_zero.bitwise_eq(
            "0x1p-127".parse::<IeeeSingle>().unwrap().scalbn(
                -127,
                round,
            ),
        ));
        assert!(m_zero.bitwise_eq(
            "-0x1p-127".parse::<IeeeSingle>().unwrap().scalbn(
                -127,
                round,
            ),
        ));
        assert!("-0x1p-149".parse::<IeeeSingle>().unwrap().bitwise_eq(
            "-0x1p-127".parse::<IeeeSingle>().unwrap().scalbn(
                -22,
                round,
            ),
        ));
        assert!(p_zero.bitwise_eq(
            "0x1p-126".parse::<IeeeSingle>().unwrap().scalbn(
                -24,
                round,
            ),
        ));


        let smallest_f64 = IeeeDouble::smallest();
        let neg_smallest_f64 = -IeeeDouble::smallest();

        let largest_f64 = IeeeDouble::largest();
        let neg_largest_f64 = -IeeeDouble::largest();

        let largest_denormal_f64 = "0x1.ffffffffffffep-1023".parse::<IeeeDouble>().unwrap();
        let neg_largest_denormal_f64 = "-0x1.ffffffffffffep-1023".parse::<IeeeDouble>().unwrap();


        assert!(smallest_f64.bitwise_eq(
            "0x1p-1074".parse::<IeeeDouble>().unwrap().scalbn(0, round),
        ));
        assert!(neg_smallest_f64.bitwise_eq(
            "-0x1p-1074".parse::<IeeeDouble>().unwrap().scalbn(0, round),
        ));

        assert!("0x1p+1023".parse::<IeeeDouble>().unwrap().bitwise_eq(
            smallest_f64.scalbn(2097, round),
        ));

        assert!(smallest_f64.scalbn(-2097, round).is_pos_zero());
        assert!(smallest_f64.scalbn(-2098, round).is_pos_zero());
        assert!(smallest_f64.scalbn(-2099, round).is_pos_zero());
        assert!("0x1p+1022".parse::<IeeeDouble>().unwrap().bitwise_eq(
            smallest_f64.scalbn(2096, round),
        ));
        assert!("0x1p+1023".parse::<IeeeDouble>().unwrap().bitwise_eq(
            smallest_f64.scalbn(2097, round),
        ));
        assert!(smallest_f64.scalbn(2098, round).is_infinite());
        assert!(smallest_f64.scalbn(2099, round).is_infinite());

        // Test for integer overflows when adding to exponent.
        assert!(smallest_f64.scalbn(-i32::MAX, round).is_pos_zero());
        assert!(largest_f64.scalbn(i32::MAX, round).is_infinite());

        assert!(largest_denormal_f64.bitwise_eq(
            largest_denormal_f64.scalbn(0, round),
        ));
        assert!(neg_largest_denormal_f64.bitwise_eq(
            neg_largest_denormal_f64.scalbn(
                0,
                round,
            ),
        ));

        assert!(
            "0x1.ffffffffffffep-1022"
                .parse::<IeeeDouble>()
                .unwrap()
                .bitwise_eq(largest_denormal_f64.scalbn(1, round))
        );
        assert!(
            "-0x1.ffffffffffffep-1021"
                .parse::<IeeeDouble>()
                .unwrap()
                .bitwise_eq(neg_largest_denormal_f64.scalbn(2, round))
        );

        assert!(
            "0x1.ffffffffffffep+1"
                .parse::<IeeeDouble>()
                .unwrap()
                .bitwise_eq(largest_denormal_f64.scalbn(1024, round))
        );
        assert!(largest_denormal_f64.scalbn(-1023, round).is_pos_zero());
        assert!(largest_denormal_f64.scalbn(-1024, round).is_pos_zero());
        assert!(largest_denormal_f64.scalbn(-2048, round).is_pos_zero());
        assert!(largest_denormal_f64.scalbn(2047, round).is_infinite());
        assert!(largest_denormal_f64.scalbn(2098, round).is_infinite());
        assert!(largest_denormal_f64.scalbn(2099, round).is_infinite());

        assert!(
            "0x1.ffffffffffffep-2"
                .parse::<IeeeDouble>()
                .unwrap()
                .bitwise_eq(largest_denormal_f64.scalbn(1021, round))
        );
        assert!(
            "0x1.ffffffffffffep-1"
                .parse::<IeeeDouble>()
                .unwrap()
                .bitwise_eq(largest_denormal_f64.scalbn(1022, round))
        );
        assert!(
            "0x1.ffffffffffffep+0"
                .parse::<IeeeDouble>()
                .unwrap()
                .bitwise_eq(largest_denormal_f64.scalbn(1023, round))
        );
        assert!(
            "0x1.ffffffffffffep+1023"
                .parse::<IeeeDouble>()
                .unwrap()
                .bitwise_eq(largest_denormal_f64.scalbn(2046, round))
        );
        assert!("0x1p+974".parse::<IeeeDouble>().unwrap().bitwise_eq(
            smallest_f64.scalbn(2048, round),
        ));

        let random_denormal_f64 = "0x1.c60f120d9f87cp+51".parse::<IeeeDouble>().unwrap();
        assert!(
            "0x1.c60f120d9f87cp-972"
                .parse::<IeeeDouble>()
                .unwrap()
                .bitwise_eq(random_denormal_f64.scalbn(-1023, round))
        );
        assert!(
            "0x1.c60f120d9f87cp-1"
                .parse::<IeeeDouble>()
                .unwrap()
                .bitwise_eq(random_denormal_f64.scalbn(-52, round))
        );
        assert!(
            "0x1.c60f120d9f87cp-2"
                .parse::<IeeeDouble>()
                .unwrap()
                .bitwise_eq(random_denormal_f64.scalbn(-53, round))
        );
        assert!(
            "0x1.c60f120d9f87cp+0"
                .parse::<IeeeDouble>()
                .unwrap()
                .bitwise_eq(random_denormal_f64.scalbn(-51, round))
        );

        assert!(random_denormal_f64.scalbn(-2097, round).is_pos_zero());
        assert!(random_denormal_f64.scalbn(-2090, round).is_pos_zero());


        assert!("-0x1p-1073".parse::<IeeeDouble>().unwrap().bitwise_eq(
            neg_largest_f64.scalbn(-2097, round),
        ));

        assert!("-0x1p-1024".parse::<IeeeDouble>().unwrap().bitwise_eq(
            neg_largest_f64.scalbn(-2048, round),
        ));

        assert!("0x1p-1073".parse::<IeeeDouble>().unwrap().bitwise_eq(
            largest_f64.scalbn(-2097, round),
        ));

        assert!("0x1p-1074".parse::<IeeeDouble>().unwrap().bitwise_eq(
            largest_f64.scalbn(-2098, round),
        ));
        assert!("-0x1p-1074".parse::<IeeeDouble>().unwrap().bitwise_eq(
            neg_largest_f64.scalbn(-2098, round),
        ));
        assert!(neg_largest_f64.scalbn(-2099, round).is_neg_zero());
        assert!(largest_f64.scalbn(1, round).is_infinite());


        assert!("0x1p+0".parse::<IeeeDouble>().unwrap().bitwise_eq(
            "0x1p+52".parse::<IeeeDouble>().unwrap().scalbn(-52, round),
        ));

        assert!("0x1p-103".parse::<IeeeDouble>().unwrap().bitwise_eq(
            "0x1p-51".parse::<IeeeDouble>().unwrap().scalbn(-52, round),
        ));
    }

    #[test]
    fn frexp() {
        let round = Round::NearestTiesToEven;

        let p_zero = IeeeDouble::zero();
        let m_zero = -IeeeDouble::zero();
        let one = IeeeDouble::from_f64(1.0);
        let m_one = IeeeDouble::from_f64(-1.0);

        let largest_denormal = "0x1.ffffffffffffep-1023".parse::<IeeeDouble>().unwrap();
        let neg_largest_denormal = "-0x1.ffffffffffffep-1023".parse::<IeeeDouble>().unwrap();

        let smallest = IeeeDouble::smallest();
        let neg_smallest = -IeeeDouble::smallest();

        let largest = IeeeDouble::largest();
        let neg_largest = -IeeeDouble::largest();

        let p_inf = IeeeDouble::inf();
        let m_inf = -IeeeDouble::inf();

        let p_qnan = IeeeDouble::nan();
        let m_qnan = -IeeeDouble::nan();
        let snan = IeeeDouble::snan(None);

        // Make sure highest bit of payload is preserved.
        let payload = (1 << 50) | (1 << 49) | (1234 << 32) | 1;

        let snan_with_payload = IeeeDouble::snan(Some(payload));

        let mut exp = 0;

        let frac = p_zero.frexp(&mut exp, round);
        assert_eq!(0, exp);
        assert!(frac.is_pos_zero());

        let frac = m_zero.frexp(&mut exp, round);
        assert_eq!(0, exp);
        assert!(frac.is_neg_zero());


        let frac = one.frexp(&mut exp, round);
        assert_eq!(1, exp);
        assert!("0x1p-1".parse::<IeeeDouble>().unwrap().bitwise_eq(frac));

        let frac = m_one.frexp(&mut exp, round);
        assert_eq!(1, exp);
        assert!("-0x1p-1".parse::<IeeeDouble>().unwrap().bitwise_eq(frac));

        let frac = largest_denormal.frexp(&mut exp, round);
        assert_eq!(-1022, exp);
        assert!(
            "0x1.ffffffffffffep-1"
                .parse::<IeeeDouble>()
                .unwrap()
                .bitwise_eq(frac)
        );

        let frac = neg_largest_denormal.frexp(&mut exp, round);
        assert_eq!(-1022, exp);
        assert!(
            "-0x1.ffffffffffffep-1"
                .parse::<IeeeDouble>()
                .unwrap()
                .bitwise_eq(frac)
        );


        let frac = smallest.frexp(&mut exp, round);
        assert_eq!(-1073, exp);
        assert!("0x1p-1".parse::<IeeeDouble>().unwrap().bitwise_eq(frac));

        let frac = neg_smallest.frexp(&mut exp, round);
        assert_eq!(-1073, exp);
        assert!("-0x1p-1".parse::<IeeeDouble>().unwrap().bitwise_eq(frac));


        let frac = largest.frexp(&mut exp, round);
        assert_eq!(1024, exp);
        assert!(
            "0x1.fffffffffffffp-1"
                .parse::<IeeeDouble>()
                .unwrap()
                .bitwise_eq(frac)
        );

        let frac = neg_largest.frexp(&mut exp, round);
        assert_eq!(1024, exp);
        assert!(
            "-0x1.fffffffffffffp-1"
                .parse::<IeeeDouble>()
                .unwrap()
                .bitwise_eq(frac)
        );


        let frac = p_inf.frexp(&mut exp, round);
        assert_eq!(i32::MAX, exp);
        assert!(frac.is_infinite() && !frac.is_negative());

        let frac = m_inf.frexp(&mut exp, round);
        assert_eq!(i32::MAX, exp);
        assert!(frac.is_infinite() && frac.is_negative());

        let frac = p_qnan.frexp(&mut exp, round);
        assert_eq!(i32::MIN, exp);
        assert!(frac.is_nan());

        let frac = m_qnan.frexp(&mut exp, round);
        assert_eq!(i32::MIN, exp);
        assert!(frac.is_nan());

        let frac = snan.frexp(&mut exp, round);
        assert_eq!(i32::MIN, exp);
        assert!(frac.is_nan() && !frac.is_signaling());

        let frac = snan_with_payload.frexp(&mut exp, round);
        assert_eq!(i32::MIN, exp);
        assert!(frac.is_nan() && !frac.is_signaling());
        assert_eq!(payload, frac.to_bits() & ((1 << 51) - 1));

        let frac = "0x0.ffffp-1".parse::<IeeeDouble>().unwrap().frexp(
            &mut exp,
            round,
        );
        assert_eq!(-1, exp);
        assert!("0x1.fffep-1".parse::<IeeeDouble>().unwrap().bitwise_eq(
            frac,
        ));

        let frac = "0x1p-51".parse::<IeeeDouble>().unwrap().frexp(
            &mut exp,
            round,
        );
        assert_eq!(-50, exp);
        assert!("0x1p-1".parse::<IeeeDouble>().unwrap().bitwise_eq(frac));

        let frac = "0x1.c60f120d9f87cp+51"
            .parse::<IeeeDouble>()
            .unwrap()
            .frexp(&mut exp, round);
        assert_eq!(52, exp);
        assert!(
            "0x1.c60f120d9f87cp-1"
                .parse::<IeeeDouble>()
                .unwrap()
                .bitwise_eq(frac)
        );
    }

    #[test]
    fn modulo() {
        {
            let mut f1 = "1.5".parse::<IeeeDouble>().unwrap();
            let f2 = "1.0".parse::<IeeeDouble>().unwrap();
            let expected = "0.5".parse::<IeeeDouble>().unwrap();
            assert_eq!(f1.modulo(f2), OpStatus::OK);
            assert!(f1.bitwise_eq(expected));
        }
        {
            let mut f1 = "0.5".parse::<IeeeDouble>().unwrap();
            let f2 = "1.0".parse::<IeeeDouble>().unwrap();
            let expected = "0.5".parse::<IeeeDouble>().unwrap();
            assert_eq!(f1.modulo(f2), OpStatus::OK);
            assert!(f1.bitwise_eq(expected));
        }
        {
            let mut f1 = "0x1.3333333333333p-2".parse::<IeeeDouble>().unwrap(); // 0.3
            let f2 = "0x1.47ae147ae147bp-7".parse::<IeeeDouble>().unwrap(); // 0.01
            // 0.009999999999999983
            let expected = "0x1.47ae147ae1471p-7".parse::<IeeeDouble>().unwrap();
            assert_eq!(f1.modulo(f2), OpStatus::OK);
            assert!(f1.bitwise_eq(expected));
        }
        {
            let mut f1 = "0x1p64".parse::<IeeeDouble>().unwrap(); // 1.8446744073709552e19
            let f2 = "1.5".parse::<IeeeDouble>().unwrap();
            let expected = "1.0".parse::<IeeeDouble>().unwrap();
            assert_eq!(f1.modulo(f2), OpStatus::OK);
            assert!(f1.bitwise_eq(expected));
        }
        {
            let mut f1 = "0x1p1000".parse::<IeeeDouble>().unwrap();
            let f2 = "0x1p-1000".parse::<IeeeDouble>().unwrap();
            let expected = "0.0".parse::<IeeeDouble>().unwrap();
            assert_eq!(f1.modulo(f2), OpStatus::OK);
            assert!(f1.bitwise_eq(expected));
        }
        {
            let mut f1 = "0.0".parse::<IeeeDouble>().unwrap();
            let f2 = "1.0".parse::<IeeeDouble>().unwrap();
            let expected = "0.0".parse::<IeeeDouble>().unwrap();
            assert_eq!(f1.modulo(f2), OpStatus::OK);
            assert!(f1.bitwise_eq(expected));
        }
        {
            let mut f1 = "1.0".parse::<IeeeDouble>().unwrap();
            let f2 = "0.0".parse::<IeeeDouble>().unwrap();
            assert_eq!(f1.modulo(f2), OpStatus::INVALID_OP);
            assert!(f1.is_nan());
        }
        {
            let mut f1 = "0.0".parse::<IeeeDouble>().unwrap();
            let f2 = "0.0".parse::<IeeeDouble>().unwrap();
            assert_eq!(f1.modulo(f2), OpStatus::INVALID_OP);
            assert!(f1.is_nan());
        }
        {
            let mut f1 = IeeeDouble::inf();
            let f2 = "1.0".parse::<IeeeDouble>().unwrap();
            assert_eq!(f1.modulo(f2), OpStatus::INVALID_OP);
            assert!(f1.is_nan());
        }
    }

    #[test]
    fn ppc_double_double_add_special() {
        let data = [
            // (1 + 0) + (-1 + 0) = fcZero
            (
                0x3ff0000000000000,
                0xbff0000000000000,
                Category::Zero,
                Round::NearestTiesToEven,
            ),
            // LDBL_MAX + (1.1 >> (1023 - 106) + 0)) = fcInfinity
            (
                0x7c8ffffffffffffe_7fefffffffffffff,
                0x7948000000000000,
                Category::Infinity,
                Round::NearestTiesToEven,
            ),
            // FIXME: change the 4th 0x75effffffffffffe to 0x75efffffffffffff when
            // semPpcDoubleDoubleLegacy is gone.
            // LDBL_MAX + (1.011111... >> (1023 - 106) + (1.1111111...0 >> (1023 -
            // 160))) = fcNormal
            (
                0x7c8ffffffffffffe_7fefffffffffffff,
                0x75effffffffffffe_7947ffffffffffff,
                Category::Normal,
                Round::NearestTiesToEven,
            ),
            // LDBL_MAX + (1.1 >> (1023 - 106) + 0)) = fcInfinity
            (
                0x7c8ffffffffffffe_7fefffffffffffff,
                0x7c8ffffffffffffe_7fefffffffffffff,
                Category::Infinity,
                Round::NearestTiesToEven,
            ),
            // NaN + (1 + 0) = fcNaN
            (
                0x7ff8000000000000,
                0x3ff0000000000000,
                Category::NaN,
                Round::NearestTiesToEven,
            ),
        ];

        for &(op1, op2, expected, round) in &data {
            {
                let mut a1 = PpcDoubleDouble::from_bits(op1);
                let a2 = PpcDoubleDouble::from_bits(op2);
                let _: OpStatus = a1.add_rounded(a2, round);

                assert_eq!(expected, a1.category(), "{:#x} + {:#x}", op1, op2);
            }
            {
                let a1 = PpcDoubleDouble::from_bits(op1);
                let mut a2 = PpcDoubleDouble::from_bits(op2);
                let _: OpStatus = a2.add_rounded(a1, round);

                assert_eq!(expected, a2.category(), "{:#x} + {:#x}", op2, op1);
            }
        }
    }

    #[test]
    fn ppc_double_double_add() {
        let data = [
            // (1 + 0) + (1e-105 + 0) = (1 + 1e-105)
            (
                0x3ff0000000000000,
                0x3960000000000000,
                0x3960000000000000_3ff0000000000000,
                Round::NearestTiesToEven,
            ),
            // (1 + 0) + (1e-106 + 0) = (1 + 1e-106)
            (
                0x3ff0000000000000,
                0x3950000000000000,
                0x3950000000000000_3ff0000000000000,
                Round::NearestTiesToEven,
            ),
            // (1 + 1e-106) + (1e-106 + 0) = (1 + 1e-105)
            (
                0x3950000000000000_3ff0000000000000,
                0x3950000000000000,
                0x3960000000000000_3ff0000000000000,
                Round::NearestTiesToEven,
            ),
            // (1 + 0) + (epsilon + 0) = (1 + epsilon)
            (
                0x3ff0000000000000,
                0x0000000000000001,
                0x0000000000000001_3ff0000000000000,
                Round::NearestTiesToEven,
            ),
            // FIXME: change 0xf950000000000000 to 0xf940000000000000, when
            // semPpcDoubleDoubleLegacy is gone.
            // (DBL_MAX - 1 << (1023 - 105)) + (1 << (1023 - 53) + 0) = DBL_MAX +
            // 1.11111... << (1023 - 52)
            (
                0xf950000000000000_7fefffffffffffff,
                0x7c90000000000000,
                0x7c8ffffffffffffe_7fefffffffffffff,
                Round::NearestTiesToEven,
            ),
            // FIXME: change 0xf950000000000000 to 0xf940000000000000, when
            // semPpcDoubleDoubleLegacy is gone.
            // (1 << (1023 - 53) + 0) + (DBL_MAX - 1 << (1023 - 105)) = DBL_MAX +
            // 1.11111... << (1023 - 52)
            (
                0x7c90000000000000,
                0xf950000000000000_7fefffffffffffff,
                0x7c8ffffffffffffe_7fefffffffffffff,
                Round::NearestTiesToEven,
            ),
        ];

        for &(op1, op2, expected, round) in &data {
            {
                let mut a1 = PpcDoubleDouble::from_bits(op1);
                let a2 = PpcDoubleDouble::from_bits(op2);
                let _: OpStatus = a1.add_rounded(a2, round);

                assert_eq!(expected, a1.to_bits(), "{:#x} + {:#x}", op1, op2);
            }
            {
                let a1 = PpcDoubleDouble::from_bits(op1);
                let mut a2 = PpcDoubleDouble::from_bits(op2);
                let _: OpStatus = a2.add_rounded(a1, round);

                assert_eq!(expected, a2.to_bits(), "{:#x} + {:#x}", op2, op1);
            }
        }
    }

    #[test]
    fn ppc_double_double_subtract() {
        let data = [
            // (1 + 0) - (-1e-105 + 0) = (1 + 1e-105)
            (
                0x3ff0000000000000,
                0xb960000000000000,
                0x3960000000000000_3ff0000000000000,
                Round::NearestTiesToEven,
            ),
            // (1 + 0) - (-1e-106 + 0) = (1 + 1e-106)
            (
                0x3ff0000000000000,
                0xb950000000000000,
                0x3950000000000000_3ff0000000000000,
                Round::NearestTiesToEven,
            ),
        ];

        for &(op1, op2, expected, round) in &data {
            let mut a1 = PpcDoubleDouble::from_bits(op1);
            let a2 = PpcDoubleDouble::from_bits(op2);
            let _: OpStatus = a1.sub_rounded(a2, round);

            assert_eq!(expected, a1.to_bits(), "{:#x} - {:#x}", op1, op2);
        }
    }

    #[test]
    fn ppc_double_double_multiply_special() {
        let data = [
            // fcNaN * fcNaN = fcNaN
            (
                0x7ff8000000000000,
                0x7ff8000000000000,
                Category::NaN,
                Round::NearestTiesToEven,
            ),
            // fcNaN * fcZero = fcNaN
            (
                0x7ff8000000000000,
                0,
                Category::NaN,
                Round::NearestTiesToEven,
            ),
            // fcNaN * fcInfinity = fcNaN
            (
                0x7ff8000000000000,
                0x7ff0000000000000,
                Category::NaN,
                Round::NearestTiesToEven,
            ),
            // fcNaN * fcNormal = fcNaN
            (
                0x7ff8000000000000,
                0x3ff0000000000000,
                Category::NaN,
                Round::NearestTiesToEven,
            ),
            // fcInfinity * fcInfinity = fcInfinity
            (
                0x7ff0000000000000,
                0x7ff0000000000000,
                Category::Infinity,
                Round::NearestTiesToEven,
            ),
            // fcInfinity * fcZero = fcNaN
            (
                0x7ff0000000000000,
                0,
                Category::NaN,
                Round::NearestTiesToEven,
            ),
            // fcInfinity * fcNormal = fcInfinity
            (
                0x7ff0000000000000,
                0x3ff0000000000000,
                Category::Infinity,
                Round::NearestTiesToEven,
            ),
            // fcZero * fcZero = fcZero
            (0, 0, Category::Zero, Round::NearestTiesToEven),
            // fcZero * fcNormal = fcZero
            (
                0,
                0x3ff0000000000000,
                Category::Zero,
                Round::NearestTiesToEven,
            ),
        ];

        for &(op1, op2, expected, round) in &data {
            {
                let mut a1 = PpcDoubleDouble::from_bits(op1);
                let a2 = PpcDoubleDouble::from_bits(op2);
                let _: OpStatus = a1.mul_rounded(a2, round);

                assert_eq!(expected, a1.category(), "{:#x} * {:#x}", op1, op2);
            }
            {
                let a1 = PpcDoubleDouble::from_bits(op1);
                let mut a2 = PpcDoubleDouble::from_bits(op2);
                let _: OpStatus = a2.mul_rounded(a1, round);

                assert_eq!(expected, a2.category(), "{:#x} * {:#x}", op2, op1);
            }
        }
    }

    #[test]
    fn ppc_double_double_multiply() {
        let data = [
            // 1/3 * 3 = 1.0
            (
                0x3c75555555555556_3fd5555555555555,
                0x4008000000000000,
                0x3ff0000000000000,
                Round::NearestTiesToEven,
            ),
            // (1 + epsilon) * (1 + 0) = fcZero
            (
                0x0000000000000001_3ff0000000000000,
                0x3ff0000000000000,
                0x0000000000000001_3ff0000000000000,
                Round::NearestTiesToEven,
            ),
            // (1 + epsilon) * (1 + epsilon) = 1 + 2 * epsilon
            (
                0x0000000000000001_3ff0000000000000,
                0x0000000000000001_3ff0000000000000,
                0x0000000000000002_3ff0000000000000,
                Round::NearestTiesToEven,
            ),
            // -(1 + epsilon) * (1 + epsilon) = -1
            (
                0x0000000000000001_bff0000000000000,
                0x0000000000000001_3ff0000000000000,
                0xbff0000000000000,
                Round::NearestTiesToEven,
            ),
            // (0.5 + 0) * (1 + 2 * epsilon) = 0.5 + epsilon
            (
                0x3fe0000000000000,
                0x0000000000000002_3ff0000000000000,
                0x0000000000000001_3fe0000000000000,
                Round::NearestTiesToEven,
            ),
            // (0.5 + 0) * (1 + epsilon) = 0.5
            (
                0x3fe0000000000000,
                0x0000000000000001_3ff0000000000000,
                0x3fe0000000000000,
                Round::NearestTiesToEven,
            ),
            // __LDBL_MAX__ * (1 + 1 << 106) = inf
            (
                0x7c8ffffffffffffe_7fefffffffffffff,
                0x3950000000000000_3ff0000000000000,
                0x7ff0000000000000,
                Round::NearestTiesToEven,
            ),
            // __LDBL_MAX__ * (1 + 1 << 107) > __LDBL_MAX__, but not inf, yes =_=|||
            (
                0x7c8ffffffffffffe_7fefffffffffffff,
                0x3940000000000000_3ff0000000000000,
                0x7c8fffffffffffff_7fefffffffffffff,
                Round::NearestTiesToEven,
            ),
            // __LDBL_MAX__ * (1 + 1 << 108) = __LDBL_MAX__
            (
                0x7c8ffffffffffffe_7fefffffffffffff,
                0x3930000000000000_3ff0000000000000,
                0x7c8ffffffffffffe_7fefffffffffffff,
                Round::NearestTiesToEven,
            ),
        ];

        for &(op1, op2, expected, round) in &data {
            {
                let mut a1 = PpcDoubleDouble::from_bits(op1);
                let a2 = PpcDoubleDouble::from_bits(op2);
                let _: OpStatus = a1.mul_rounded(a2, round);

                assert_eq!(expected, a1.to_bits(), "{:#x} * {:#x}", op1, op2);
            }
            {
                let a1 = PpcDoubleDouble::from_bits(op1);
                let mut a2 = PpcDoubleDouble::from_bits(op2);
                let _: OpStatus = a2.mul_rounded(a1, round);

                assert_eq!(expected, a2.to_bits(), "{:#x} * {:#x}", op2, op1);
            }
        }
    }

    #[test]
    fn ppc_double_double_divide() {
        // FIXME: Only a sanity check for now. Add more edge cases when the
        // double-double algorithm is implemented.
        let data = [
            // 1 / 3 = 1/3
            (
                0x3ff0000000000000,
                0x4008000000000000,
                0x3c75555555555556_3fd5555555555555,
                Round::NearestTiesToEven,
            ),
        ];

        for &(op1, op2, expected, round) in &data {
            let mut a1 = PpcDoubleDouble::from_bits(op1);
            let a2 = PpcDoubleDouble::from_bits(op2);
            let _: OpStatus = a1.div_rounded(a2, round);

            assert_eq!(expected, a1.to_bits(), "{:#x} / {:#x}", op1, op2);
        }
    }

    #[test]
    fn ppc_double_double_remainder() {
        let data = [
            // remainder(3.0 + 3.0 << 53, 1.25 + 1.25 << 53) = (0.5 + 0.5 << 53)
            (
                0x3cb8000000000000_4008000000000000,
                0x3ca4000000000000_3ff4000000000000,
                0x3c90000000000000_3fe0000000000000,
            ),
            // remainder(3.0 + 3.0 << 53, 1.75 + 1.75 << 53) = (-0.5 - 0.5 << 53)
            (
                0x3cb8000000000000_4008000000000000,
                0x3cac000000000000_3ffc000000000000,
                0xbc90000000000000_bfe0000000000000,
            ),
        ];

        for &(op1, op2, expected) in &data {
            let mut a1 = PpcDoubleDouble::from_bits(op1);
            let a2 = PpcDoubleDouble::from_bits(op2);
            let _: OpStatus = a1.remainder(a2);

            assert_eq!(expected, a1.to_bits(), "remainder({:#x}, {:#x})", op1, op2);
        }
    }

    #[test]
    fn ppc_double_double_mod() {
        let data = [
            // mod(3.0 + 3.0 << 53, 1.25 + 1.25 << 53) = (0.5 + 0.5 << 53)
            (
                0x3cb8000000000000_4008000000000000,
                0x3ca4000000000000_3ff4000000000000,
                0x3c90000000000000_3fe0000000000000,
            ),
            // mod(3.0 + 3.0 << 53, 1.75 + 1.75 << 53) = (1.25 + 1.25 << 53)
            // 0xbc98000000000000 doesn't seem right, but it's what we currently have.
            // FIXME: investigate
            (
                0x3cb8000000000000_4008000000000000,
                0x3cac000000000000_3ffc000000000000,
                0xbc98000000000000_3ff4000000000001,
            ),
        ];

        for &(op1, op2, expected) in &data {
            let mut a1 = PpcDoubleDouble::from_bits(op1);
            let a2 = PpcDoubleDouble::from_bits(op2);
            let _: OpStatus = a1.modulo(a2);

            assert_eq!(expected, a1.to_bits(), "fmod({:#x}, {:#x})", op1, op2);
        }
    }

    #[test]
    fn ppc_double_double_fma() {
        // Sanity check for now.
        let mut a = "2".parse::<PpcDoubleDouble>().unwrap();
        let _: OpStatus = a.fused_mul_add(
            "3".parse::<PpcDoubleDouble>().unwrap(),
            "4".parse::<PpcDoubleDouble>().unwrap(),
            Round::NearestTiesToEven,
        );
        assert_eq!(
            Some(Ordering::Equal),
            "10".parse::<PpcDoubleDouble>().unwrap().partial_cmp(&a)
        );
    }

    #[test]
    fn ppc_double_double_round_to_integral() {
        {
            let a = "1.5".parse::<PpcDoubleDouble>().unwrap();
            let a = a.round_to_integral(Round::NearestTiesToEven).0;
            assert_eq!(
                Some(Ordering::Equal),
                "2".parse::<PpcDoubleDouble>().unwrap().partial_cmp(&a)
            );
        }
        {
            let a = "2.5".parse::<PpcDoubleDouble>().unwrap();
            let a = a.round_to_integral(Round::NearestTiesToEven).0;
            assert_eq!(
                Some(Ordering::Equal),
                "2".parse::<PpcDoubleDouble>().unwrap().partial_cmp(&a)
            );
        }
    }

    #[test]
    fn ppc_double_double_compare() {
        let data = [
            // (1 + 0) = (1 + 0)
            (
                0x3ff0000000000000,
                0x3ff0000000000000,
                Some(Ordering::Equal),
            ),
            // (1 + 0) < (1.00...1 + 0)
            (0x3ff0000000000000, 0x3ff0000000000001, Some(Ordering::Less)),
            // (1.00...1 + 0) > (1 + 0)
            (
                0x3ff0000000000001,
                0x3ff0000000000000,
                Some(Ordering::Greater),
            ),
            // (1 + 0) < (1 + epsilon)
            (
                0x3ff0000000000000,
                0x0000000000000001_3ff0000000000001,
                Some(Ordering::Less),
            ),
            // NaN != NaN
            (0x7ff8000000000000, 0x7ff8000000000000, None),
            // (1 + 0) != NaN
            (0x3ff0000000000000, 0x7ff8000000000000, None),
            // Inf = Inf
            (
                0x7ff0000000000000,
                0x7ff0000000000000,
                Some(Ordering::Equal),
            ),
        ];

        for &(op1, op2, expected) in &data {
            let a1 = PpcDoubleDouble::from_bits(op1);
            let a2 = PpcDoubleDouble::from_bits(op2);
            assert_eq!(
                expected,
                a1.partial_cmp(&a2),
                "compare({:#x}, {:#x})",
                op1,
                op2
            );
        }
    }

    #[test]
    fn ppc_double_double_bitwise_eq() {
        let data = [
            // (1 + 0) = (1 + 0)
            (0x3ff0000000000000, 0x3ff0000000000000, true),
            // (1 + 0) != (1.00...1 + 0)
            (0x3ff0000000000000, 0x3ff0000000000001, false),
            // NaN = NaN
            (0x7ff8000000000000, 0x7ff8000000000000, true),
            // NaN != NaN with a different bit pattern
            (
                0x7ff8000000000000,
                0x3ff0000000000000_7ff8000000000000,
                false,
            ),
            // Inf = Inf
            (0x7ff0000000000000, 0x7ff0000000000000, true),
        ];

        for &(op1, op2, expected) in &data {
            let a1 = PpcDoubleDouble::from_bits(op1);
            let a2 = PpcDoubleDouble::from_bits(op2);
            assert_eq!(expected, a1.bitwise_eq(a2), "{:#x} = {:#x}", op1, op2);
        }
    }

    #[test]
    fn ppc_double_double_change_sign() {
        let float = PpcDoubleDouble::from_bits(0xbcb0000000000000_400f000000000000);
        {
            let actual = float.copy_sign("1".parse::<PpcDoubleDouble>().unwrap());
            assert_eq!(0xbcb0000000000000_400f000000000000, actual.to_bits());
        }
        {
            let actual = float.copy_sign("-1".parse::<PpcDoubleDouble>().unwrap());
            assert_eq!(0x3cb0000000000000_c00f000000000000, actual.to_bits());
        }
    }

    #[test]
    fn ppc_double_double_factories() {
        assert_eq!(0, PpcDoubleDouble::zero().to_bits());
        assert_eq!(
            0x7c8ffffffffffffe_7fefffffffffffff,
            PpcDoubleDouble::largest().to_bits()
        );
        assert_eq!(0x0000000000000001, PpcDoubleDouble::smallest().to_bits());
        assert_eq!(
            0x0360000000000000,
            PpcDoubleDouble::smallest_normalized().to_bits()
        );
        assert_eq!(
            0x0000000000000000_8000000000000000,
            (-PpcDoubleDouble::zero()).to_bits()
        );
        assert_eq!(
            0xfc8ffffffffffffe_ffefffffffffffff,
            (-PpcDoubleDouble::largest()).to_bits()
        );
        assert_eq!(
            0x0000000000000000_8000000000000001,
            (-PpcDoubleDouble::smallest()).to_bits()
        );
        assert_eq!(
            0x0000000000000000_8360000000000000,
            (-PpcDoubleDouble::smallest_normalized()).to_bits()
        );
        assert!(PpcDoubleDouble::smallest().is_smallest());
        assert!(PpcDoubleDouble::largest().is_largest());
    }

    #[test]
    fn ppc_double_double_is_denormal() {
        assert!(PpcDoubleDouble::smallest().is_denormal());
        assert!(!PpcDoubleDouble::largest().is_denormal());
        assert!(!PpcDoubleDouble::smallest_normalized().is_denormal());
        {
            // (4 + 3) is not normalized
            let data = 0x4008000000000000_4010000000000000;
            assert!(PpcDoubleDouble::from_bits(data).is_denormal());
        }
    }

    #[test]
    fn ppc_double_double_scalbn() {
        // 3.0 + 3.0 << 53
        let input = 0x3cb8000000000000_4008000000000000;
        let result = PpcDoubleDouble::from_bits(input).scalbn(1, Round::NearestTiesToEven);
        // 6.0 + 6.0 << 53
        assert_eq!(0x3cc8000000000000_4018000000000000, result.to_bits());
    }

    #[test]
    fn ppc_double_double_frexp() {
        // 3.0 + 3.0 << 53
        let input = 0x3cb8000000000000_4008000000000000;
        let mut exp = 0;
        // 0.75 + 0.75 << 53
        let result = PpcDoubleDouble::from_bits(input).frexp(&mut exp, Round::NearestTiesToEven);
        assert_eq!(2, exp);
        assert_eq!(0x3c98000000000000_3fe8000000000000, result.to_bits());
    }
}
