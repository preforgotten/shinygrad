use shinygrad_backend::Backend;
use shinygrad_tensor::Tensor;
use shinygrad_core::module::{Module, Param};

#[derive(Debug, Clone)]
pub struct ReLU<B: Backend> {
    _b: std::marker::PhantomData<B>,
}

impl<B: Backend> ReLU<B> {
    pub fn new() -> Self {
        ReLU { _b: std::marker::PhantomData }
    }
}

impl<B: Backend> Module<B, 2> for ReLU<B> {
    fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        input.relu()
    }

    fn parameters(&self) -> Vec<Param<B, 2>> {
        vec![]
    }

    fn load_parameters(&mut self, _params: &[Param<B, 2>]) {}
}

#[derive(Debug, Clone)]
pub struct Softmax<B: Backend> {
    pub dim: usize,
    _b: std::marker::PhantomData<B>,
}

impl<B: Backend> Softmax<B> {
    pub fn new(dim: usize) -> Self {
        Softmax { dim, _b: std::marker::PhantomData }
    }
}

impl<B: Backend> Module<B, 2> for Softmax<B> {
    fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        input.softmax(self.dim)
    }

    fn parameters(&self) -> Vec<Param<B, 2>> {
        vec![]
    }

    fn load_parameters(&mut self, _params: &[Param<B, 2>]) {}
}
