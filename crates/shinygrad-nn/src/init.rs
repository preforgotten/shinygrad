use rand::Rng;
use shinygrad_backend::{Backend, FloatElement, TensorData};
use shinygrad_tensor::Tensor;

pub fn kaiming_uniform<B: Backend, const D: usize>(tensor: &mut Tensor<B, D>, fan_in: usize) {
    let scale: f64 = (2.0 / fan_in as f64).sqrt();
    let shape = tensor.shape();
    let mut rng = rand::rng();
    let data: Vec<B::FloatElem> = (0..shape.num_elements())
        .map(|_| {
            let val: f64 = rng.random_range(-scale..scale);
            <B::FloatElem as FloatElement>::from_f64(val)
        })
        .collect();
    let device = Default::default();
    *tensor = Tensor::from_data(&device, TensorData::new(data, shape));
}

pub fn xavier_uniform<B: Backend, const D: usize>(tensor: &mut Tensor<B, D>, fan_in: usize) {
    let scale: f64 = (1.0 / fan_in as f64).sqrt();
    let shape = tensor.shape();
    let mut rng = rand::rng();
    let data: Vec<B::FloatElem> = (0..shape.num_elements())
        .map(|_| {
            let val: f64 = rng.random_range(-scale..scale);
            <B::FloatElem as FloatElement>::from_f64(val)
        })
        .collect();
    let device = Default::default();
    *tensor = Tensor::from_data(&device, TensorData::new(data, shape));
}
