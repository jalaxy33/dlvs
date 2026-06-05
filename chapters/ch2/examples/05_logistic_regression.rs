use ndarray::{Array1, Array2, arr1, arr2};

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn logistic_regression_train(
    inputs: &Array2<f64>,
    targets: &Array1<f64>,
    learning_rate: f64,
    num_iterations: usize,
) -> Array1<f64> {
    let mut weights: Array1<f64> = Array1::zeros(inputs.ncols());

    for _ in 0..num_iterations {
        let predictions: Array1<f64> = inputs.dot(&weights).mapv(sigmoid);
        let errors = &predictions - targets;
        let gradient = inputs.t().dot(&errors) / targets.len() as f64;
        weights = weights - learning_rate * gradient;
    }

    weights
}

fn main() {
    let inputs: Array2<f64> = arr2(&[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
    let targets: Array1<f64> = arr1(&[0.0, 1.0, 1.0]);

    let learning_rate = 0.01;
    let num_iterations = 1000;

    let weights = logistic_regression_train(&inputs, &targets, learning_rate, num_iterations);
    println!("Trained weights: {:?}", weights);
}
