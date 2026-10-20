use shinygrad_tensor::Tensor;
use shinygrad_backend::Backend;

#[derive(Debug, Clone)]
pub struct ParamId(pub String);

#[derive(Debug, Clone)]
pub struct Param<B: Backend, const D: usize> {
    pub id: ParamId,
    pub value: Tensor<B, D>,
}

impl<B: Backend, const D: usize> Param<B, D> {
    pub fn new(id: &str, value: Tensor<B, D>) -> Self {
        Param {
            id: ParamId(id.to_string()),
            value,
        }
    }
}

pub trait Module<B: Backend, const D: usize> {
    fn forward(&self, input: Tensor<B, D>) -> Tensor<B, D>;
    fn parameters(&self) -> Vec<Param<B, D>>;
    fn load_parameters(&mut self, params: &[Param<B, D>]);
}
