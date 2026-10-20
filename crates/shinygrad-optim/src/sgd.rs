use shinygrad_backend::{Backend, FloatElement};
use shinygrad_tensor::Tensor;
use crate::Optimizer;

#[derive(Debug, Clone)]
pub struct SgdState;

pub struct SGD {
    pub learning_rate: f64,
}

impl SGD {
    pub fn new(learning_rate: f64) -> Self {
        SGD { learning_rate }
    }
}

impl<B: Backend, const D: usize> Optimizer<B, D> for SGD {
    type State = SgdState;

    fn step(
        &mut self,
        param: Tensor<B, D>,
        grad: Tensor<B, D>,
        _state: Option<&Self::State>,
    ) -> (Tensor<B, D>, Self::State) {
        let lr = <B::FloatElem as FloatElement>::from_f64(self.learning_rate);
        let update = grad.mul_scalar(lr);
        (param.sub(update), SgdState)
    }
}
