use std::f64;

use ndarray::{Array1, Array2};
use rand::{RngExt, distr::Uniform};

// L2 regularization
fn l2_regularization(weights: &Array2<f64>, lambda: f64) -> f64 {
    lambda * weights.mapv(|w| w.powi(2)).sum()
}

// Dropout implementation
fn dropout(input: &Array2<f64>, dropout_rate: f64) -> Array2<f64> {
    let mut rng = rand::rng();
    let dropout_mask = Array2::from_shape_fn(input.dim(), |_| {
        if rng.random::<f64>() < dropout_rate {
            0.0
        } else {
            1.0
        }
    });
    input * dropout_mask
}

// Precision, Recall, F1 score
fn precision_recall_f1(
    y_true: &Array1<f64>,
    y_pred: &Array1<f64>,
    threshold: f64,
) -> (f64, f64, f64) {
    let mut true_positive = 0.0;
    let mut false_positive = 0.0;
    let mut false_negative = 0.0;

    for (&true_val, &pred_val) in y_true.iter().zip(y_pred.iter()) {
        let pred_class = if pred_val >= threshold { 1.0 } else { 0.0 };
        if pred_class == 1.0 && true_val == 1.0 {
            true_positive += 1.0;
        } else if pred_class == 1.0 && true_val == 0.0 {
            false_positive += 1.0;
        } else if pred_class == 0.0 && true_val == 1.0 {
            false_negative += 1.0;
        }
    }

    let precision = if true_positive + false_positive == 0.0 {
        0.0
    } else {
        true_positive / (true_positive + false_positive)
    };

    let recall = if true_positive + false_negative == 0.0 {
        0.0
    } else {
        true_positive / (true_positive + false_negative)
    };

    let f1 = if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * (precision * recall) / (precision + recall)
    };

    (precision, recall, f1)
}

// AUC-ROC calculation
fn auc_roc(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
    let mut sorted_pred: Vec<_> = y_pred.iter().zip(y_true.iter()).collect();
    sorted_pred.sort_by(|a, b| a.0.partial_cmp(b.0).unwrap());

    let mut tpr = vec![];
    let mut fpr = vec![];

    let total_positives = y_true.iter().filter(|&&y| y == 1.0).count() as f64;
    let total_negatives = y_true.iter().filter(|&&y| y == 0.0).count() as f64;

    let mut true_positives = 0.0;
    let mut false_positives = 0.0;

    for (_, &true_val) in sorted_pred {
        if true_val == 1.0 {
            true_positives += 1.0;
        } else {
            false_positives += 1.0;
        }

        tpr.push(true_positives / total_positives);
        fpr.push(false_positives / total_negatives);
    }

    let mut auc = 0.0;
    for i in 1..tpr.len() {
        let delta_x = fpr[i] - fpr[i - 1];
        let avg_y = (tpr[i] + tpr[i - 1]) / 2.0;
        auc += delta_x * avg_y;
    }

    auc
}

fn main() {
    let input_size = 3;
    let hidden_size = 5;
    let output_size = 1;
    let dropout_rate = 0.5;
    let lambda = 0.01;

    // Simulate Neural Network's input and output for demonstration purposes
    let mut rng = rand::rng();
    let input = Array2::from_shape_fn((1, input_size), |_| {
        rng.sample(Uniform::new(0.0, 1.0).unwrap())
    });
    let weights = Array2::from_shape_fn((hidden_size, output_size), |_| {
        rng.sample(Uniform::new(0.0, 1.0).unwrap())
    });

    // Apply L2 regularization
    let l2_loss = l2_regularization(&weights, lambda);
    println!("L2 Regularization Loss: {}", l2_loss);

    // Apply Dropout to the input
    let dropped_input = dropout(&input, dropout_rate);
    println!("Input after Dropout: {:?}", dropped_input);

    // Evaluation metrics
    let y_true = Array1::from(vec![1.0, 0.0, 1.0, 0.0, 1.0]);
    let y_pred = Array1::from(vec![0.9, 0.1, 0.8, 0.4, 0.6]);

    let (precision, recall, f1) = precision_recall_f1(&y_true, &y_pred, 0.5);
    println!(
        "Precision: {}, Recall: {}, F1 Score: {}",
        precision, recall, f1
    );

    // AUC-ROC Calculation
    let auc = auc_roc(&y_true, &y_pred);
    println!("AUC-ROC Score: {}", auc);
}
