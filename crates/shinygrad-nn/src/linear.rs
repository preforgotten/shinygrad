use shinygrad_backend::Backend;
use shinygrad_tensor::Tensor;
use shinygrad_backend::Shape;

use shinygrad_core::module::{Module, Param};

use crate::init::kaiming_uniform;

pub struct LinearConfig {
    pub in_features: usize,
    pub out_features: usize,
    pub bias: bool,
}

impl LinearConfig {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        LinearConfig {
            in_features,
            out_features,
            bias: true,
        }
    }

    pub fn with_bias(mut self, bias: bool) -> Self {
        self.bias = bias;
        self
    }

    pub fn init<B: Backend>(&self, device: &<B as shinygrad_backend::BackendTypes>::Device) -> Linear<B> {
        let mut weight = Tensor::zeros(device, Shape::new(vec![self.out_features, self.in_features]));
        kaiming_uniform(&mut weight, self.in_features);
        let bias = if self.bias {
            let bias_tensor = Tensor::zeros(device, Shape::new(vec![1, self.out_features]));
            Some(Param::new("bias", bias_tensor))
        } else {
            None
        };
        Linear {
            weight: Param::new("weight", weight),
            bias,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Linear<B: Backend> {
    pub weight: Param<B, 2>,
    pub bias: Option<Param<B, 2>>,
}

impl<B: Backend> Module<B, 2> for Linear<B> {
    fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let w = self.weight.value.clone();
        let out = input.matmul(w.transpose());
        match &self.bias {
            Some(bias) => {
                let out_shape = out.shape();
                let bias_broadcast = bias.value.clone().broadcast_to(out_shape);
                out.add(bias_broadcast)
            }
            None => out,
        }
    }

    fn parameters(&self) -> Vec<Param<B, 2>> {
        let mut params = vec![self.weight.clone()];
        if let Some(ref bias) = self.bias {
            params.push(bias.clone());
        }
        params
    }

    fn load_parameters(&mut self, params: &[Param<B, 2>]) {
        self.weight = params[0].clone();
        if self.bias.is_some() && params.len() > 1 {
            self.bias = Some(params[1].clone());
        }
    }
}
