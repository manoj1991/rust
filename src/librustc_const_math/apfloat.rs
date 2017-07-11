// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::cmp::Ordering;
use std::fmt;
use std::i32;
use std::marker::PhantomData;
use std::ops::{Neg, Add, Sub, Mul, Div, Rem};
use std::ops::{AddAssign, SubAssign, MulAssign, DivAssign, RemAssign};
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

/// A signed type to represent a floating point number's unbiased exponent.
type ExpInt = i16;

/// Represents floating point arithmetic semantics.
pub trait IeeeSemantics: Copy + Clone + Ord + fmt::Debug {
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
}

#[derive(Copy, Clone, PartialOrd, Debug)]
pub struct Ieee<S: IeeeSemantics> {
    marker: PhantomData<S>,
}

macro_rules! ieee_semantics {
    ($($name:ident { $($items:tt)* })*) => {
        mod ieee_semantics {
            use super::{ExpInt, IeeeSemantics};

            $(
                // FIXME(eddyb) Remove most of these by manual impls
                // on the structs parameterized by S: IeeeSemantics.
                #[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
                pub struct $name;
                impl IeeeSemantics for $name { $($items)* }
            )*
        }

        $(pub type $name = Ieee<ieee_semantics::$name>;)*
    }
}

ieee_semantics! {
    IeeeHalf { const EXPONENT_BITS: usize = 5; const PRECISION: usize = 11; }
    IeeeSingle { const EXPONENT_BITS: usize = 8; const PRECISION: usize = 24; }
    IeeeDouble { const EXPONENT_BITS: usize = 11; const PRECISION: usize = 53; }
    IeeeQuad { const EXPONENT_BITS: usize = 15; const PRECISION: usize = 113; }
    X87DoubleExtended { const EXPONENT_BITS: usize = 15; const PRECISION: usize = 64; }

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
    PpcDoubleDoubleLegacy {
        const EXPONENT_BITS: usize = 11;
        const MIN_EXPONENT: ExpInt = -1022 + 53;
        const PRECISION: usize = 53 + 53;
    }
}

proxy_impls!([S: IeeeSemantics] Ieee<S>);

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
#[allow(unused)]
impl<S: IeeeSemantics> fmt::Display for Ieee<S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let frac_digits = f.precision().unwrap_or(0);
        let width = f.width().unwrap_or(3);
        let alternate = f.alternate();
        panic!("NYI Display::fmt");
    }
}

#[allow(unused)]
impl<S: IeeeSemantics> Float for Ieee<S> {
    fn zero() -> Self {
        panic!("NYI zero")
    }

    fn inf() -> Self {
        panic!("NYI inf")
    }

    fn qnan(payload: Option<u128>) -> Self {
        panic!("NYI qnan")
    }

    fn snan(payload: Option<u128>) -> Self {
        panic!("NYI snan")
    }

    fn largest() -> Self {
        panic!("NYI largest")
    }

    fn smallest() -> Self {
        panic!("NYI smallest")
    }

    fn smallest_normalized() -> Self {
        panic!("NYI smallest_normalized")
    }

    fn add_rounded(&mut self, rhs: Self, round: Round) -> OpStatus {
        panic!("NYI add_rounded")
    }

    fn mul_rounded(&mut self, rhs: Self, round: Round) -> OpStatus {
        panic!("NYI mul_rounded")
    }

    fn div_rounded(&mut self, rhs: Self, round: Round) -> OpStatus {
        panic!("NYI div_rounded")
    }

    fn remainder(&mut self, rhs: Self) -> OpStatus {
        panic!("NYI remainder")
    }

    fn modulo(&mut self, rhs: Self) -> OpStatus {
        panic!("NYI modulo")
    }

    fn fused_mul_add(&mut self, multiplicand: Self, addend: Self, round: Round) -> OpStatus {
        panic!("NYI fused_mul_add")
    }

    fn round_to_integral(self, round: Round) -> (Self, OpStatus) {
        panic!("NYI round_to_integral")
    }

    fn next_up(&mut self) -> OpStatus {
        panic!("NYI next_up")
    }

    fn change_sign(&mut self) {
        panic!("NYI change_sign")
    }

    fn from_bits(input: u128) -> Self {
        panic!("NYI from_bits")
    }

    fn from_u128(input: u128, round: Round) -> (Self, OpStatus) {
        panic!("NYI from_u128")
    }

    fn from_str_rounded(s: &str, round: Round) -> Result<(Self, OpStatus), ParseError> {
        panic!("NYI from_str_rounded")
    }

    fn to_bits(self) -> u128 {
        panic!("NYI to_bits")
    }

    fn to_u128(self, width: usize, round: Round, is_exact: &mut bool) -> (u128, OpStatus) {
        panic!("NYI to_u128");
    }

    fn cmp_abs_normal(self, rhs: Self) -> Ordering {
        panic!("NYI cmp_abs_normal")
    }

    fn bitwise_eq(self, rhs: Self) -> bool {
        panic!("NYI bitwise_eq")
    }

    fn is_negative(self) -> bool {
        panic!("NYI is_negative")
    }

    fn is_denormal(self) -> bool {
        panic!("NYI is_denormal")
    }

    fn is_signaling(self) -> bool {
        panic!("NYI is_signaling")
    }

    fn category(self) -> Category {
        panic!("NYI category")
    }

    fn is_smallest(self) -> bool {
        panic!("NYI is_smallest")
    }

    fn is_largest(self) -> bool {
        panic!("NYI is_largest")
    }

    fn is_integer(self) -> bool {
        panic!("NYI is_integer")
    }

    fn get_exact_inverse(self) -> Option<Self> {
        panic!("NYI get_exact_inverse")
    }

    fn ilogb(self) -> i32 {
        panic!("NYI ilogb")
    }

    fn scalbn(self, exp: i32, round: Round) -> Self {
        panic!("NYI scalbn")
    }

    fn frexp(self, exp: &mut i32, round: Round) -> Self {
        panic!("NYI frexp")
    }
}

#[allow(unused)]
impl<S: IeeeSemantics> Ieee<S> {
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
        panic!("NYI convert");
    }
}

#[derive(Copy, Clone, PartialOrd, Debug)]
pub struct IeeePair<S: IeeeSemantics>(Ieee<S>, Ieee<S>);

pub type PpcDoubleDouble = IeeePair<ieee_semantics::IeeeDouble>;

proxy_impls!([S: IeeeSemantics] IeeePair<S>);

#[allow(unused)]
impl<S: IeeeSemantics> fmt::Display for IeeePair<S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        panic!("NYI Display::fmt");
    }
}

#[allow(unused)]
impl<S: IeeeSemantics> Float for IeeePair<S> {
    fn zero() -> Self {
        panic!("NYI zero")
    }

    fn inf() -> Self {
        panic!("NYI inf")
    }

    fn qnan(payload: Option<u128>) -> Self {
        panic!("NYI qnan")
    }

    fn snan(payload: Option<u128>) -> Self {
        panic!("NYI snan")
    }

    fn largest() -> Self {
        panic!("NYI largest")
    }

    fn smallest() -> Self {
        panic!("NYI smallest")
    }

    fn smallest_normalized() -> Self {
        panic!("NYI smallest_normalized")
    }

    fn add_rounded(&mut self, rhs: Self, round: Round) -> OpStatus {
        panic!("NYI add_rounded")
    }

    fn mul_rounded(&mut self, rhs: Self, round: Round) -> OpStatus {
        panic!("NYI mul_rounded")
    }

    fn div_rounded(&mut self, rhs: Self, round: Round) -> OpStatus {
        panic!("NYI div_rounded")
    }

    fn remainder(&mut self, rhs: Self) -> OpStatus {
        panic!("NYI remainder")
    }

    fn modulo(&mut self, rhs: Self) -> OpStatus {
        panic!("NYI modulo")
    }

    fn fused_mul_add(&mut self, multiplicand: Self, addend: Self, round: Round) -> OpStatus {
        panic!("NYI fused_mul_add")
    }

    fn round_to_integral(self, round: Round) -> (Self, OpStatus) {
        panic!("NYI round_to_integral")
    }

    fn next_up(&mut self) -> OpStatus {
        panic!("NYI next_up")
    }

    fn change_sign(&mut self) {
        panic!("NYI change_sign")
    }

    fn from_bits(input: u128) -> Self {
        panic!("NYI from_bits")
    }

    fn from_u128(input: u128, round: Round) -> (Self, OpStatus) {
        panic!("NYI from_u128")
    }

    fn from_str_rounded(s: &str, round: Round) -> Result<(Self, OpStatus), ParseError> {
        panic!("NYI from_str_rounded")
    }

    fn to_bits(self) -> u128 {
        panic!("NYI to_bits")
    }

    fn to_u128(self, width: usize, round: Round, is_exact: &mut bool) -> (u128, OpStatus) {
        panic!("NYI to_u128");
    }

    fn cmp_abs_normal(self, rhs: Self) -> Ordering {
        panic!("NYI cmp_abs_normal")
    }

    fn bitwise_eq(self, rhs: Self) -> bool {
        panic!("NYI bitwise_eq")
    }

    fn is_negative(self) -> bool {
        panic!("NYI is_negative")
    }

    fn is_denormal(self) -> bool {
        panic!("NYI is_denormal")
    }

    fn is_signaling(self) -> bool {
        panic!("NYI is_signaling")
    }

    fn category(self) -> Category {
        panic!("NYI category")
    }

    fn is_smallest(self) -> bool {
        panic!("NYI is_smallest")
    }

    fn is_largest(self) -> bool {
        panic!("NYI is_largest")
    }

    fn is_integer(self) -> bool {
        panic!("NYI is_integer")
    }

    fn get_exact_inverse(self) -> Option<Self> {
        panic!("NYI get_exact_inverse")
    }

    fn ilogb(self) -> i32 {
        panic!("NYI ilogb")
    }

    fn scalbn(self, exp: i32, round: Round) -> Self {
        panic!("NYI scalbn")
    }

    fn frexp(self, exp: &mut i32, round: Round) -> Self {
        panic!("NYI frexp")
    }
}
