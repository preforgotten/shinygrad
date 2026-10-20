use shinygrad_backend::Backend;
use shinygrad_tensor::Tensor;
use shinygrad_core::module::{Module, Param};

pub struct Sequential<B: Backend> {
    layers: Vec<Box<dyn Module<B, 2>>>,
}

impl<B: Backend> Sequential<B> {
    pub fn new(layers: Vec<Box<dyn Module<B, 2>>>) -> Self {
        Sequential { layers }
    }
}

impl<B: Backend> core::fmt::Debug for Sequential<B> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Sequential")
            .field("num_layers", &self.layers.len())
            .finish()
    }
}

impl<B: Backend> Module<B, 2> for Sequential<B> {
    fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut x = input;
        for layer in &self.layers {
            x = layer.forward(x);
        }
        x
    }

    fn parameters(&self) -> Vec<Param<B, 2>> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }

    fn load_parameters(&mut self, params: &[Param<B, 2>]) {
        let mut offset = 0;
        for layer in &mut self.layers {
            let n = layer.parameters().len();
            layer.load_parameters(&params[offset..offset + n]);
            offset += n;
        }
    }
}
