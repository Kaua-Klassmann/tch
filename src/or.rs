use tch::{
    Device, Kind, Reduction, Tensor,
    nn::{self, OptimizerConfig},
};

fn main() {
    let device = Device::cuda_if_available();

    let features = Tensor::from_slice(&[0, 0, 0, 1, 1, 0, 1, 1])
        .reshape(&[4, 2])
        .to_kind(Kind::Float)
        .to_device(device);

    let targets = Tensor::from_slice(&[0, 1, 1, 1])
        .reshape(&[4, 1])
        .to_kind(Kind::Float)
        .to_device(device);

    let x_train = features.narrow(0, 0, 3);
    let y_train = targets.narrow(0, 0, 3);

    println!("X_train: {}", x_train);
    println!("y_train: {}", y_train);

    let x_test = features.narrow(0, 3, 1);
    let y_test = targets.narrow(0, 3, 1);

    println!("X_test: {}", x_test);
    println!("y_test: {}", y_test);

    // Lugar que guarda os pesos
    let vs = nn::VarStore::new(device);
    // Cara que consegue editar os pesos
    let root = &vs.root();

    // Rede neural com 2 entradas e 1 saida e aplica sigmoid no final
    // Sigmoid tende a ir para 0 ou 1
    let model = nn::seq()
        .add(nn::linear(root, 2, 1, Default::default()))
        .add_fn(|xs| xs.sigmoid());

    // Cria o otimizador, que é quem decide quanto mudar os pesos com base na taxa de aprendizado
    let mut opt = nn::Adam::default().build(&vs, 1e-1).unwrap();

    for _ in 1..1000 {
        let y_pred = x_train.apply(&model);

        let loss = y_pred.binary_cross_entropy::<Tensor>(&y_train, None, Reduction::Mean);

        opt.backward_step(&loss);
    }

    let y_pred = x_test.apply(&model);
    println!("\nX_test: {}", x_test);
    println!("y_test: {}", y_test);
    println!("Predição no teste: {}", y_pred.round().int64_value(&[]));
    println!("Igual: {}", y_pred == y_test);
}
