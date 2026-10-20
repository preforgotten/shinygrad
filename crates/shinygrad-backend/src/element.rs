use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Sub};

use crate::dtype::DType;

pub trait FloatElement:
    Copy
    + Clone
    + Send
    + Sync
    + Debug
    + 'static
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + std::ops::Neg<Output = Self>
    + PartialOrd
    + Into<f64>
{
    fn from_f64(value: f64) -> Self;
    fn to_f64(self) -> f64;
    fn dtype() -> DType;
    fn zero() -> Self;
    fn one() -> Self;
    fn max_value() -> Self;
    fn min_value() -> Self;
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn powf(self, n: Self) -> Self;
}

impl FloatElement for f32 {
    fn from_f64(value: f64) -> Self {
        value as f32
    }
    fn to_f64(self) -> f64 {
        self as f64
    }
    fn dtype() -> DType {
        DType::F32
    }
    fn zero() -> Self {
        0.0
    }
    fn one() -> Self {
        1.0
    }
    fn max_value() -> Self {
        f32::MAX
    }
    fn min_value() -> Self {
        f32::MIN
    }
    fn exp(self) -> Self {
        f32::exp(self)
    }
    fn ln(self) -> Self {
        f32::ln(self)
    }
    fn powf(self, n: Self) -> Self {
        f32::powf(self, n)
    }
}

impl FloatElement for f64 {
    fn from_f64(value: f64) -> Self {
        value
    }
    fn to_f64(self) -> f64 {
        self
    }
    fn dtype() -> DType {
        DType::F64
    }
    fn zero() -> Self {
        0.0
    }
    fn one() -> Self {
        1.0
    }
    fn max_value() -> Self {
        f64::MAX
    }
    fn min_value() -> Self {
        f64::MIN
    }
    fn exp(self) -> Self {
        f64::exp(self)
    }
    fn ln(self) -> Self {
        f64::ln(self)
    }
    fn powf(self, n: Self) -> Self {
        f64::powf(self, n)
    }
}
