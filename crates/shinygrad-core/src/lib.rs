//! # shinygrad-core
//!
//! High-level neural network abstractions.
//!
//! Defines the `Module` trait and `Param` wrapper that form the foundation
//! for all layer implementations in `shinygrad-nn`.
//!
//! ## Module trait
//!
//! Every layer implements `Module<B, D>`:
//!
//! ```ignore
//! impl<B: Backend> Module<B, 2> for Linear<B> {
//!     fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2>;
//!     fn parameters(&self) -> Vec<Param<B, 2>>;
//!     fn load_parameters(&mut self, params: &[Param<B, 2>]);
//! }
//! ```
//!
//! ## Param
//!
//! `Param<B, D>` wraps a `Tensor<B, D>` with an identifier. Parameters are
//! the leaves of the computation graph — their gradients are accumulated
//! during the backward pass and used by optimizers to update the model.

pub mod module;

pub use module::Module;
pub use module::Param;
