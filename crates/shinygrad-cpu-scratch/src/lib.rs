//! # shinygrad-cpu-scratch
//!
//! CPU backend implementation using [ndarray](https://crates.io/crates/ndarray).
//!
//! `CpuBackend<E>` implements the full `Backend` + `FloatOps` traits for
//! the CPU device. All tensor operations are executed eagerly on CPU via ndarray's
//! element-wise, matrix multiplication, and reduction routines.
//!
//! ## Supported element types
//!
//! - `CpuBackend<f32>` — single precision
//! - `CpuBackend<f64>` — double precision
//!
//! ## Internal representation
//!
//! `CpuTensorPrimitive<E>` wraps `Arc<ArrayD<E>>` — a reference-counted
//! dynamic-dimensional ndarray. Clones share the underlying data.
//!
//! ## Softmax
//!
//! Numerically stable via the standard max-subtract trick:
//! `softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))`

#![recursion_limit = "256"]

pub mod backend;

pub use backend::CpuBackend;
