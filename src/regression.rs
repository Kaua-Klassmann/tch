use tch::{
    Device, Kind, Reduction, Tensor,
    nn::{self, OptimizerConfig, VarStore},
};

fn main() {
    let device = Device::cuda_if_available();

    let features = Tensor::from_slice(&[1., 0.5, 2., 1., 3., 1.5, 4., 2., 5., 2.5])
        .reshape([5, 2])
        .to_kind(Kind::Float)
        .to_device(device);
    let targets = Tensor::from_slice(&[2., 4., 6., 8., 10.])
        .reshape([5, 1])
        .to_kind(Kind::Float)
        .to_device(device);

    let vs = VarStore::new(device);
    let root = &vs.root();

    let model = nn::seq()
        .add(nn::linear(root, 2, 16, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(root, 16, 1, Default::default()));

    let mut opt = nn::Adam::default().build(&vs, 1e-2).unwrap();

    for _ in 1..1_000 {
        let y_pred = features.apply(&model);

        let loss = y_pred.mse_loss(&targets, Reduction::Mean);

        opt.backward_step(&loss);
    }

    let test = Tensor::from_slice(&[6., 3.])
        .to_kind(Kind::Float)
        .apply(&model);

    test.print();
}
