use na::DMatrix;
use nalgebra as na;
use ndarray::{self, Array2, arr2, parallel::prelude::*};

fn parallel_matrix_multiplication(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let mut result = Array2::zeros((a.nrows(), b.ncols()));
    result
        .axis_iter_mut(ndarray::Axis(0))
        // perform parallel iterations using rayon
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut row)| {
            row.assign(&a.row(i).dot(b));
        });
    result
}

fn main() {
    // Example of matrix multiplication using ndarray
    {
        let a: Array2<f64> = arr2(&[[1., 2.], [3., 4.]]);
        let b: Array2<f64> = arr2(&[[5., 6.], [7., 8.]]);
        let c = a.dot(&b); // Matrix multiplication
        println!("Matrix product:\n{}", c);
    }

    // Example of compute SVD using nalgebra
    {
        let a = DMatrix::<f64>::from_vec(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let svd = a.svd(true, true); // Perform SVD
        println!("Singular values: {:?}", svd.singular_values);
    }

    // Example of parallelize matrix multiplication using rayon
    {
        let a: Array2<f64> = arr2(&[[1., 2.], [3., 4.]]);
        let b: Array2<f64> = arr2(&[[5., 6.], [7., 8.]]);
        let result = parallel_matrix_multiplication(&a, &b);
        println!("Parallel Matrix product:\n{}", result);
    }
}
