use shinygrad_backend::{Backend, BackendTypes, FloatElement};
use shinygrad_tensor::Tensor;
use crate::Optimizer;

#[derive(Debug, Clone)]
pub struct AdamState<B: Backend, const D: usize> {
    pub m: Tensor<B, D>,
    pub v: Tensor<B, D>,
    pub t: usize,
}

pub struct Adam<B: Backend, const D: usize> {
    pub learning_rate: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    _phantom: std::marker::PhantomData<(B, [usize; D])>,
}

impl<B: Backend, const D: usize> Adam<B, D> {
    pub fn new(learning_rate: f64) -> Self {
        Adam {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<B: Backend, const D: usize> Optimizer<B, D> for Adam<B, D> {
    type State = AdamState<B, D>;

    fn step(
        &mut self,
        param: Tensor<B, D>,
        grad: Tensor<B, D>,
        state: Option<&Self::State>,
    ) -> (Tensor<B, D>, Self::State) {
        let device: <B as BackendTypes>::Device = Default::default();
        let shape = param.shape();

        let (mut m, mut v, mut t) = match state {
            Some(s) => (s.m.clone(), s.v.clone(), s.t),
            None => {
                let m = Tensor::zeros(&device, shape.clone());
                let v = Tensor::zeros(&device, shape.clone());
                (m, v, 0)
            }
        };

        t += 1;
        use FloatElement;
        let beta1 = <B::FloatElem>::from_f64(self.beta1);
        let beta2 = <B::FloatElem>::from_f64(self.beta2);
        m = m.mul_scalar(beta1).add(grad.clone().mul_scalar(<B::FloatElem>::from_f64(1.0 - self.beta1)));
        v = v.mul_scalar(beta2).add(grad.clone().mul(grad).mul_scalar(<B::FloatElem>::from_f64(1.0 - self.beta2)));

        let m_hat = m.clone().mul_scalar(<B::FloatElem>::from_f64(1.0 / (1.0 - self.beta1.powi(t as i32))));
        let v_hat = v.clone().mul_scalar(<B::FloatElem>::from_f64(1.0 / (1.0 - self.beta2.powi(t as i32))));

        let eps = <B::FloatElem>::from_f64(self.epsilon);
        let denom = v_hat.pow_scalar(0.5).add_scalar(eps);
        let lr = <B::FloatElem>::from_f64(self.learning_rate);
        let update = m_hat.div(denom).mul_scalar(lr);

        let new_param = param.sub(update);

        (new_param, AdamState { m, v, t })
    }
}
