//! # shinygrad-nn
//!
//! Neural network building blocks — layers, activations, initializers, and loss functions.
//!
//! All modules are generic over `B: Backend` and work with any backend,
//! including autodiff-wrapped backends.
//!
//! ## Layers
//!
//! | Module | Description |
//! |---|---|
//! | [`Linear`] | Fully-connected layer `y = xW^T + b` |
//! | [`Sequential`] | Chain multiple modules in sequence |
//!
//! ## Activations (stateless modules)
//!
//! | Module | Forward |
//! |---|---|
//! | [`ReLU`] | `max(0, x)` |
//! | [`Softmax`] | `exp(x) / sum(exp(x))` along a dimension |
//!
//! ## Initializers
//!
//! - [`kaiming_uniform`] — He initialization (scale `√(2/fan_in)`)
//! - [`xavier_uniform`] — Glorot initialization (scale `√(1/fan_in)`)
//!
//! ## Loss functions
//!
//! | Loss | Formula |
//! |---|---|
//! | [`CrossEntropyLoss`] | `-mean(log_softmax * one_hot_targets)` |
//!
//! ## Usage
//!
//! ```ignore
//! use shinygrad_nn::{Linear, LinearConfig, ReLU, Sequential, CrossEntropyLoss};
//!
//! let device = Device::Cpu;
//! let model: Sequential<MyBackend> = Sequential::new(vec![
//!     Box::new(LinearConfig::new(784, 64).init(&device)),
//!     Box::new(ReLU::new()),
//!     Box::new(LinearConfig::new(64, 10).init(&device)),
//! ]);
//! ```

pub mod linear;
pub mod sequential;
pub mod activations;
pub mod init;
pub mod loss;

pub use linear::{Linear, LinearConfig};
pub use sequential::Sequential;
pub use activations::ReLU;
pub use activations::Softmax;
pub use init::{kaiming_uniform, xavier_uniform};
pub use loss::CrossEntropyLoss;
