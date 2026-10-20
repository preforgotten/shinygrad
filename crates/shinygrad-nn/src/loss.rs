use shinygrad_backend::{Backend, FloatElement, Shape, TensorData};
use shinygrad_tensor::Tensor;

pub struct CrossEntropyLoss;

impl CrossEntropyLoss {
    pub fn new() -> Self {
        CrossEntropyLoss
    }

    pub fn forward<B: Backend>(
        &self,
        predictions: Tensor<B, 2>,
        targets: &[usize],
    ) -> Tensor<B, 1> {
        let batch_size = predictions.dims()[0];
        let num_classes = predictions.dims()[1];
        let device: <B as shinygrad_backend::BackendTypes>::Device = Default::default();

        let log_softmax = predictions.softmax(1).log();

        let mut flat = vec![<B::FloatElem as FloatElement>::zero(); batch_size * num_classes];
        for (i, &t) in targets.iter().enumerate() {
            flat[i * num_classes + t] = <B::FloatElem as FloatElement>::one();
        }
        let one_hot = Tensor::<B, 2>::from_data(
            &device,
            TensorData::new(flat, Shape::new(vec![batch_size, num_classes])),
        );

        let ce = log_softmax.mul(one_hot).neg().sum();
        let inv_n = <B::FloatElem as FloatElement>::from_f64(1.0 / batch_size as f64);
        ce.mul_scalar(inv_n)
    }

    pub fn forward_mse<B: Backend>(
        predictions: Tensor<B, 2>,
        targets: &[usize],
    ) -> Tensor<B, 1> {
        let batch_size = predictions.dims()[0];
        let num_classes = predictions.dims()[1];
        let device: <B as shinygrad_backend::BackendTypes>::Device = Default::default();

        let mut flat = vec![<B::FloatElem as FloatElement>::zero(); batch_size * num_classes];
        for (i, &t) in targets.iter().enumerate() {
            flat[i * num_classes + t] = <B::FloatElem as FloatElement>::one();
        }
        let one_hot = Tensor::<B, 2>::from_data(
            &device,
            TensorData::new(flat, Shape::new(vec![batch_size, num_classes])),
        );

        let diff = predictions.sub(one_hot);
        let sq = diff.clone().mul(diff);
        let inv_n = <B::FloatElem as FloatElement>::from_f64(1.0 / (batch_size * num_classes) as f64);
        sq.sum().mul_scalar(inv_n)
    }

    pub fn accuracy<B: Backend>(
        predictions: &Tensor<B, 2>,
        targets: &[usize],
    ) -> f64 {
        let data = predictions.clone().into_data();
        let num_classes = predictions.dims()[1];
        let mut correct = 0;
        for (i, &t) in targets.iter().enumerate() {
            let start = i * num_classes;
            let row = &data.value[start..start + num_classes];
            let pred_idx = row
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap();
            if pred_idx == t {
                correct += 1;
            }
        }
        correct as f64 / targets.len() as f64
    }
}
