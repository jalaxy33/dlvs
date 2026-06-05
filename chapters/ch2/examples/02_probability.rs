use ndarray::{self, Array1, arr1};
use rand;
use rand_distr::{Distribution, Normal};
use statrs::statistics::{Data, Distribution as StatrsDistribution};

fn main() {
    // Compute mean and variance
    {
        let data: Array1<f64> = arr1(&[1.0, 2.0, 3.0, 4.0, 5.0]);

        let mean = data.mean().unwrap();
        println!("Mean: {}", mean);

        let variance = data.mapv(|x| (x - mean).powi(2)).mean().unwrap();
        println!("Variance: {}", variance);
    }

    // Compute covariance
    {
        let data1: Array1<f64> = arr1(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let data2: Array1<f64> = arr1(&[2.0, 4.0, 6.0, 8.0, 10.0]);

        let mean1 = data1.mean().unwrap();
        let mean2 = data2.mean().unwrap();

        let covariance = (data1.clone() - mean1) * (data2.clone() - mean2);
        let covariance_mean = covariance.mean().unwrap();
        println!("Covariance: {}", covariance_mean);
    }

    // Simulate data from Gaussian distribution using rand and rand_distr
    {
        let normal = Normal::new(0.0, 1.0).unwrap(); // mean = 0, standard deviation = 1
        let mut rng = rand::rng();

        let sample: f64 = normal.sample(&mut rng);
        println!("Random sample from Gaussian distribution: {}", sample);
    }

    // Statistical analysis with statrs
    {
        let normal = Normal::new(0.0, 1.0).unwrap();
        let mut rng = rand::rng();

        // Generate a sample of random data
        let samples: Vec<f64> = (0..1000).map(|_| normal.sample(&mut rng)).collect();

        // Use statrs crate for statistical analysis
        let data = Data::new(samples);

        println!("Mean: {}", data.mean().unwrap_or_default());
        println!("Variance: {}", data.variance().unwrap_or_default());
        println!("Skewness: {}", data.skewness().unwrap_or_default());
    }
}
