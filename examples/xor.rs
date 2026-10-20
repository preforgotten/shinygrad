use shinygrad::prelude::*;

type Inner = CpuBackend<f64>;
type MyBackend = Autodiff<Inner>;

fn main() {
    let device = Device::Cpu;

    let x = Tensor::<MyBackend, 2>::from_data(
        &device,
        TensorData::new(
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
            Shape::new(vec![4, 2]),
        ),
    );
    let targets: Vec<usize> = vec![0, 1, 1, 0];

    let mut model: Sequential<MyBackend> = Sequential::new(vec![
        Box::new(LinearConfig::new(2, 4).init(&device)),
        Box::new(ReLU::new()),
        Box::new(LinearConfig::new(4, 4).init(&device)),
        Box::new(ReLU::new()),
        Box::new(LinearConfig::new(4, 2).init(&device)),
    ]);

    let mut optim = Adam::new(0.01);
    let num_params = model.parameters().len();
    let mut states: Vec<Option<AdamState<Inner, 2>>> = vec![None; num_params];

    for epoch in 0..400 {
        let pred = model.forward(x.clone());

        let mut oh = vec![0.0f64; 8];
        for (i, &t) in targets.iter().enumerate() {
            oh[i * 2 + t] = 1.0;
        }
        let one_hot = Tensor::<MyBackend, 2>::from_data(
            &device,
            TensorData::new(oh, Shape::new(vec![4, 2])),
        );

        let diff = pred.sub(one_hot);
        let loss = diff.clone().mul(diff).sum();
        let loss_clone = loss.clone();
        let grads = Autodiff::<Inner>::backward(loss.into_primitive());

        let params = model.parameters();
        let mut new_params: Vec<Param<MyBackend, 2>> = Vec::with_capacity(num_params);
        for (i, param) in params.iter().enumerate() {
            if let Some(grad) = Autodiff::<Inner>::grad(&param.value.primitive, &grads) {
                let g: Tensor<Inner, 2> = Tensor::from_primitive(grad);
                let val_p = Autodiff::<Inner>::inner(param.value.clone().into_primitive());
                let v: Tensor<Inner, 2> = Tensor::from_primitive(val_p);
                let (new_v, s) = optim.step(v, g, states[i].as_ref());
                states[i] = Some(s);
                let ad_p = Autodiff::<Inner>::from_inner(new_v.into_primitive());
                new_params.push(Param::new(&param.id.0, Tensor::from_primitive(ad_p)));
            } else {
                new_params.push(param.clone());
            }
        }
        model.load_parameters(&new_params);

        if epoch % 50 == 0 {
            let lv = loss_clone.into_data().value[0];
            println!("epoch {epoch:3}: loss = {lv:.6}");
        }
    }

    let final_pred = model.forward(x);
    let pd = final_pred.into_data();
    let x_data = Tensor::<MyBackend, 2>::from_data(
        &device,
        TensorData::new(
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
            Shape::new(vec![4, 2]),
        ),
    );
    let xd = x_data.into_data();
    println!("\nPredictions:");
    for i in 0..4 {
        let a = pd.value[i * 2];
        let b = pd.value[i * 2 + 1];
        let pred = if a > b { 0 } else { 1 };
        println!(
            "  [{:.0},{:.0}] -> {pred} (expected {})",
            xd.value[i * 2],
            xd.value[i * 2 + 1],
            targets[i],
        );
    }
}
