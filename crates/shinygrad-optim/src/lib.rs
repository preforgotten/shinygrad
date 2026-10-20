//! # shinygrad-optim
//!
//! Optimization algorithms for updating model parameters via gradient descent.
//!
//! ## Optimizer trait
//!
//! ```ignore
//! pub trait Optimizer<B: Backend, const D: usize> {
//!     type State;
//!
//!     fn step(
//!         &mut self,
//!         param: Tensor<B, D>,
//!         grad: Tensor<B, D>,
//!         state: Option<&Self::State>,
//!     ) -> (Tensor<B, D>, Self::State);
//! }
//! ```
//!
//! Each optimizer is stateless — it receives per-parameter state and returns
//! the updated parameter together with new state. This design keeps optimizers
//! backend-agnostic and agnostic to how parameters are stored.
//!
//! ## Implemented optimizers
//!
//! | Optimizer | State | Key parameter |
//! |---|---|---|
//! | [`SGD`] | `SgdState` | `learning_rate` |
//! | [`Momentum`] | `MomentumState { velocity }` | `learning_rate`, `momentum` |
//! | [`Adam`] | `AdamState { m, v, t }` | `learning_rate`, `beta1`, `beta2`, `epsilon` |
//!
//! ## Per-parameter state
//!
//! State is tracked per-parameter. Each call to `step` passes the previous state
//! and receives the new state. Store states in a `Vec<Option<State>>` alongside
//! your parameters:
//!
//! ```ignore
//! let mut states: Vec<Option<AdamState<B, 2>>> = vec![None; num_params];
//!
//! for (i, param) in params.iter().enumerate() {
//!     let (new_val, new_state) = optim.step(val, grad, states[i].as_ref());
//!     states[i] = Some(new_state);
//!     // update param with new_val
//! }
//! ```

pub mod sgd;
pub mod adam;
pub mod momentum;

pub use sgd::SGD;
pub use sgd::SgdState;
pub use adam::Adam;
pub use adam::AdamState;
pub use momentum::Momentum;
pub use momentum::MomentumState;

use shinygrad_backend::Backend;
use shinygrad_tensor::Tensor;

pub trait Optimizer<B: Backend, const D: usize> {
    type State;

    fn step(
        &mut self,
        param: Tensor<B, D>,
        grad: Tensor<B, D>,
        state: Option<&Self::State>,
    ) -> (Tensor<B, D>, Self::State);
}
