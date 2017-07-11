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
            format!("{:2$.1$}", IeeeDouble::from_f64(d), precision, width)
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
            format!("{:#2$.1$}", IeeeDouble::from_f64(d), precision, width)
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
