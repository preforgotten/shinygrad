use shinygrad::prelude::*;
use std::fs;

type Inner = CpuBackend<f64>;
type Backend = Autodiff<Inner>;

fn load_csv(path: &str) -> (Vec<Vec<f64>>, Vec<usize>) {
    let content = fs::read_to_string(path).expect("Failed to read CSV");
    let mut features = Vec::new();
    let mut labels = Vec::new();
    for line in content.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let values: Vec<f64> = line
            .split(',')
            .map(|s| s.trim().parse::<f64>().unwrap())
            .collect();
        if values.is_empty() {
            continue;
        }
        labels.push(values[0] as usize);
        features.push(values[1..].to_vec());
    }
    (features, labels)
}

fn main() {
    let device = Device::Cpu;

    let train_path = "data/mnist_train_100.csv";
    let test_path = "data/mnist_test_10.csv";

    let (train_x, train_y) = load_csv(train_path);
    let (test_x, test_y) = load_csv(test_path);

    let num_train = train_x.len();
    let num_test = test_x.len();
    let num_pixels = 784;
    let num_classes = 10;

    println!("Train: {num_train} samples, Test: {num_test} samples");

    // Create training tensors
    let train_x_flat: Vec<f64> = train_x.iter().flatten().cloned().collect();
    let x_train = Tensor::<Backend, 2>::from_data(
        &device,
        TensorData::new(train_x_flat, Shape::new(vec![num_train, num_pixels])),
    );

    let test_x_flat: Vec<f64> = test_x.iter().flatten().cloned().collect();
    let x_test = Tensor::<Backend, 2>::from_data(
        &device,
        TensorData::new(test_x_flat, Shape::new(vec![num_test, num_pixels])),
    );

    // Model: Linear(784, 64) -> ReLU -> Linear(64, 10)
    let mut model: Sequential<Backend> = Sequential::new(vec![
        Box::new(LinearConfig::new(784, 64).init(&device)),
        Box::new(ReLU::new()),
        Box::new(LinearConfig::new(64, 10).init(&device)),
    ]);

    let mut optim = Adam::new(0.001);
    let num_params = model.parameters().len();
    let mut optim_states: Vec<Option<AdamState<Inner, 2>>> = vec![None; num_params];

    for epoch in 0..50 {
        let pred = model.forward(x_train.clone());

        let mut oh = vec![0.0f64; num_train * num_classes];
        for (i, &t) in train_y.iter().enumerate() {
            oh[i * num_classes + t] = 1.0;
        }
        let one_hot = Tensor::<Backend, 2>::from_data(
            &device,
            TensorData::new(oh, Shape::new(vec![num_train, num_classes])),
        );

        let diff = pred.sub(one_hot);
        let n = (num_train * num_classes) as f64;
        let loss = diff.clone().mul(diff).sum().mul_scalar(1.0 / n);
        let loss_val = loss.clone().into_data().value[0];

        let grads = Autodiff::<Inner>::backward(loss.into_primitive());

        let params = model.parameters();
        let mut new_params: Vec<Param<Backend, 2>> = Vec::with_capacity(num_params);

        for (i, param) in params.iter().enumerate() {
            let grad_prim = Autodiff::<Inner>::grad(&param.value.primitive, &grads);
            if let Some(grad) = grad_prim {
                let grad_t: Tensor<Inner, 2> = Tensor::from_primitive(grad);
                let inner_prim = Autodiff::<Inner>::inner(param.value.clone().into_primitive());
                let val_t: Tensor<Inner, 2> = Tensor::from_primitive(inner_prim);
                let (new_val_t, state) = optim.step(val_t, grad_t, optim_states[i].as_ref());
                optim_states[i] = Some(state);
                let ad_prim = Autodiff::<Inner>::from_inner(new_val_t.into_primitive());
                let ad_val = Tensor::<Backend, 2>::from_primitive(ad_prim);
                new_params.push(Param::new(&param.id.0, ad_val));
            } else {
                new_params.push(param.clone());
            }
        }

        model.load_parameters(&new_params);

        // Training accuracy
        let pred_data = model.forward(x_train.clone()).into_data();
        let mut correct = 0;
        for i in 0..num_train {
            let start = i * num_classes;
            let row = &pred_data.value[start..start + num_classes];
            let pred_idx = row
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap();
            if pred_idx == train_y[i] {
                correct += 1;
            }
        }
        let train_acc = correct as f64 / num_train as f64 * 100.0;

        // Test accuracy
        let test_pred_data = model.forward(x_test.clone()).into_data();
        let mut test_correct = 0;
        for i in 0..num_test {
            let start = i * num_classes;
            let row = &test_pred_data.value[start..start + num_classes];
            let pred_idx = row
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap();
            if pred_idx == test_y[i] {
                test_correct += 1;
            }
        }
        let test_acc = test_correct as f64 / num_test as f64 * 100.0;

        println!(
            "epoch {epoch:2}: loss={:.6} train_acc={train_acc:.1}% test_acc={test_acc:.1}%",
            loss_val
        );
    }
}
