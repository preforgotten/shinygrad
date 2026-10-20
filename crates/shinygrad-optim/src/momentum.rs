use shinygrad_backend::{Backend, BackendTypes, FloatElement};
use shinygrad_tensor::Tensor;
use crate::Optimizer;

#[derive(Debug, Clone)]
pub struct MomentumState<B: Backend, const D: usize> {
    pub velocity: Tensor<B, D>,
}

pub struct Momentum<B: Backend, const D: usize> {
    pub learning_rate: f64,
    pub momentum: f64,
    _phantom: std::marker::PhantomData<(B, [usize; D])>,
}

impl<B: Backend, const D: usize> Momentum<B, D> {
    pub fn new(learning_rate: f64, momentum: f64) -> Self {
        Momentum {
            learning_rate,
            momentum,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<B: Backend, const D: usize> Optimizer<B, D> for Momentum<B, D> {
    type State = MomentumState<B, D>;

    fn step(
        &mut self,
        param: Tensor<B, D>,
        grad: Tensor<B, D>,
        state: Option<&Self::State>,
    ) -> (Tensor<B, D>, Self::State) {
        let device: <B as BackendTypes>::Device = Default::default();
        let shape = param.shape();

        let velocity = match state {
            Some(s) => s.velocity.clone(),
            None => Tensor::zeros(&device, shape),
        };

        let mu = <B::FloatElem as FloatElement>::from_f64(self.momentum);
        let velocity = velocity.mul_scalar(mu).add(grad);
        let lr = <B::FloatElem as FloatElement>::from_f64(self.learning_rate);
        let update = velocity.clone().mul_scalar(lr);
        let new_param = param.sub(update);

        (new_param, MomentumState { velocity })
    }
}
