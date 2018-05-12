use std::error::Error;
use std::fs::File;
use std::io::prelude::*;

use rand::{Rng, thread_rng};
use rayon::prelude::*;
use serde_json;

use dataset::Feature;
use dataset::Record;
use dataset::Dataset;
use loss::Loss;
use optimizer::{Optimizer};

#[derive(Serialize, Deserialize, Debug)]
pub struct Model {
    constant_weight: f32,
    linear_weights: Vec<f32>,
    pair_weights: Vec<Vec<f32>>,
    pub hash_bits: u32,
    factors_size: u32
}

impl Model {
    pub fn new(hash_bits: u32, factors_size: u32) -> Model {
        let mut rng = thread_rng();
        Model {
            constant_weight: 0.,
            linear_weights: (0..(1 << hash_bits)).map(|_| rng.gen_range(-0.01, 0.01)).collect(),
            pair_weights: (0..(1 << hash_bits)).map(|_| (0..factors_size).map(|_| rng.gen_range(-0.01, 0.01)).collect()).collect(),
            hash_bits,
            factors_size
        }
    }

    pub fn load(path: &str) -> Result<Model, Box<Error>> {
        let mut file = File::open(path)?;
        let mut content = String::new();
        file.read_to_string(&mut content)?;
        let model = serde_json::from_str(&content)?;
        Ok(model)
    }

    pub fn save(&self, path: &str) -> Result<(), Box<Error>> {
        let content = serde_json::to_string(&self)?;
        let mut file = File::create(&path)?;
        file.write_all(&content.as_bytes())?;
        Ok(())
    }

    pub fn get_parameters_number(&self) -> usize {
        (1 +
         (1 << self.hash_bits) + 
         self.factors_size * (1 << self.hash_bits)) as usize
    }

    fn get_linear_weight(&self, feature: &Feature) -> f32 {
        self.linear_weights[feature.name as usize]
    }

    fn get_linear_weight_ref(&mut self, feature: &Feature) -> &mut f32 {
        &mut self.linear_weights[feature.name as usize]
    }

    fn get_linear_weight_index(&self, feature: &Feature) -> usize {
        1 + feature.name as usize
    }

    fn get_pair_weight(&self, factor: u32, feature: &Feature) -> f32 {
        self.pair_weights[feature.name as usize][factor as usize]
    }

    fn get_pair_weight_ref(&mut self, factor: u32, feature: &Feature) -> &mut f32 {
        &mut self.pair_weights[feature.name as usize][factor as usize]
    }

    fn get_pair_weight_index(&self, factor: u32, feature: &Feature) -> usize {
        (1 +
         (1 << self.hash_bits) +
         feature.name * self.factors_size + factor) as usize
    }

    fn predict_record(&self, record: &Record) -> f32 {
        let linear_term = record.features.iter()
            .map(|f| self.get_linear_weight(f) * f.value).sum::<f32>();
        let mut pair_term: f32 = 0.;
        for factor in 0..self.factors_size {
            let sum_square = record.features.iter()
                .map(|f| self.get_pair_weight(factor, f) * f.value)
                .sum::<f32>().powi(2);
            let squares_sum = record.features.iter()
                .map(|f| (self.get_pair_weight(factor, f) * f.value).powi(2))
                .sum::<f32>();
            pair_term += sum_square - squares_sum;
        }
        self.constant_weight + linear_term + pair_term
    }

    pub fn predict(&self, dataset: &Dataset) -> Vec<f32> {
        dataset.records.par_iter().map(|r| self.predict_record(&r)).collect()
    }

    pub fn eval_loss(&self, dataset: &Dataset, loss: &Loss) -> f32 {
        dataset.records.par_iter()
            .map(|r| loss.loss(r.target, self.predict_record(&r))).sum::<f32>() / 
            (dataset.records.len() as f32)
    }

    pub fn fit<T: Optimizer>(&mut self, dataset: &Dataset, test_dataset_option: &Option<Dataset>, 
                             loss: &Loss, mut optimizer: T, epochs: i32, l2_regularization: f32) {
        let mut rng = thread_rng();
        for epoch in 0..epochs {
            optimizer.next_epoch();
            let mut record_indices: Vec<usize> = (0..dataset.records.len()).collect();
            rng.shuffle(&mut record_indices);

            unsafe {
                let model_ptr = self as *mut Model as usize;
                let optimizer_ptr = &mut optimizer as *mut T as usize;

                record_indices.par_iter().for_each(|record_index| {
                    let optimizer: &mut T = &mut *(optimizer_ptr as *mut T);
                    let self_model: &mut Model = &mut *(model_ptr as *mut Model);
                
                    optimizer.next_step();
                    let record = &dataset.records[*record_index];
                    let prediction = self_model.predict_record(&record);
                    let loss_grad = loss.loss_grad(record.target, prediction);
                    
                    self_model.constant_weight -= optimizer.get_step(
                        0, 1. * loss_grad + self_model.constant_weight * l2_regularization);
                    for feature in record.features.iter() {
                        *self_model.get_linear_weight_ref(feature) -= optimizer.get_step(
                            self_model.get_linear_weight_index(feature),
                            feature.value * loss_grad + 
                                self_model.get_linear_weight(feature) * l2_regularization);
                    }
                    for factor in 0..self_model.factors_size {
                        let weights_sum = record.features.iter()
                            .map(|feature| self_model.get_pair_weight(factor, feature) * feature.value).sum::<f32>();
                        for feature in record.features.iter() {
                            *self_model.get_pair_weight_ref(factor, feature) -= optimizer.get_step(
                                self_model.get_pair_weight_index(factor, feature),
                                (feature.value * weights_sum -
                                 self_model.get_pair_weight(factor, feature) * feature.value.powi(2)) * loss_grad +
                                self_model.get_pair_weight(factor, feature) * l2_regularization);
                        }
                    }
                });
            }
            if let &Some(ref test_dataset) = test_dataset_option {
                println!("Epoch: {}, loss on train: {}, loss on test: {}", epoch, 
                         self.eval_loss(&dataset, loss), self.eval_loss(&test_dataset, loss));
            } else {
                println!("Epoch: {}, loss on train: {}", epoch, self.eval_loss(&dataset, loss));
            }
        }
    }
}


#[derive(Clone)]
struct TransposedRecord {
    index: usize,
    value: f32,
    target: f32,
}

impl Model {
    fn get_transposed_dataset(&self, dataset: &Dataset) -> Vec<Vec<TransposedRecord>> {
        let mut result = vec![Vec::new(); 1 << self.hash_bits];
        for (index, record) in dataset.records.iter().enumerate() {
            for feature in record.features.iter() {
                result[feature.name as usize].push(TransposedRecord{index, value:feature.value, target: record.target});
            }
        }
        result
    }

    pub fn fit_als(&mut self, dataset: &Dataset, test_dataset_option: &Option<Dataset>, loss: &Loss, 
                   epochs: i32, l2_regularization: f32) {
        let transposed_dataset = self.get_transposed_dataset(&dataset);

        for epoch in 0..epochs {
            let mut predictions = self.predict(dataset);

            // theta = w_0
            // h_theta = 1
            {
                let grad_grad_sum = dataset.records.iter().zip(&predictions)
                    .map(|(r, p)| loss.loss_grad_grad(r.target, *p)).sum::<f32>();

                let grad_sum = dataset.records.iter().zip(&predictions)
                    .map(|(r, p)| loss.loss_grad(r.target, *p)).sum::<f32>();

                let new_constant_weight = (self.constant_weight * grad_grad_sum - grad_sum) / 
                    (grad_grad_sum + l2_regularization);

                predictions.iter_mut().for_each(|p| *p += new_constant_weight - self.constant_weight);

                self.constant_weight = new_constant_weight;
            }
           
            // theta = w_i
            // h_theta = x_i
            for feature in 0..(1 << self.hash_bits) {
                let mut grad_grad_sum = 0.;
                let mut grad_sum = 0.;
                for record in transposed_dataset[feature].iter() {
                    grad_grad_sum += 
                        record.value.powi(2) * 
                        loss.loss_grad_grad(record.target, predictions[record.index]);
                    grad_sum +=
                        record.value * 
                        loss.loss_grad(record.target, predictions[record.index]);
                }

                let new_linear_weight =
                    (self.linear_weights[feature as usize] * grad_grad_sum - grad_sum) / 
                    (grad_grad_sum + l2_regularization);
                
                for record in transposed_dataset[feature].iter() {
                    predictions[record.index] += 
                        (new_linear_weight - self.linear_weights[feature as usize]) * 
                        record.value;
                }

                self.linear_weights[feature as usize] = new_linear_weight;
            }
            
            // theta = v_{i,f}
            // h_theta = x_i (\sum_j v_{j,f} x_j - v_{i,f} x_i)
            for factor in 0..self.factors_size {
                let mut factors_sum: Vec<f32> = dataset.records.par_iter()
                    .map(|r| r.features.iter()
                         .map(|f| self.get_pair_weight(factor, f) * f.value).sum()).collect();

                unsafe {
                    let predictions_ptr = &mut predictions as *mut Vec<f32> as usize;
                    let factors_sum_ptr = &mut factors_sum as *mut Vec<f32> as usize;
                    let model_ptr = self as *mut Model as usize;

                    (0..(1 << self.hash_bits)).collect::<Vec<usize>>().par_iter().for_each(|&feature| {
                        let predictions: &mut Vec<f32> = &mut *(predictions_ptr as *mut Vec<f32>);
                        let factors_sum: &mut Vec<f32> = &mut *(factors_sum_ptr as *mut Vec<f32>);
                        let self_model: &mut Model = &mut *(model_ptr as *mut Model);

                        let mut grad_grad_sum = 0.;
                        let mut grad_sum = 0.;
                        for record in transposed_dataset[feature].iter() {

                            let mut prediction_grad = record.value * 
                                (factors_sum[record.index] - 
                                 self.pair_weights[feature as usize][factor as usize] * record.value);
                        
                            grad_grad_sum += prediction_grad.powi(2) * loss.loss_grad_grad(record.target, predictions[record.index]);
                            grad_sum += prediction_grad * loss.loss_grad(record.target, predictions[record.index]);
                        }

                        let new_pair_weight =
                            (self_model.pair_weights[feature as usize][factor as usize] * grad_grad_sum - grad_sum) / 
                            (grad_grad_sum + l2_regularization);

                        for record in transposed_dataset[feature].iter() {
                            let mut h = record.value * 
                                (factors_sum[record.index] - 
                                 self_model.pair_weights[feature as usize][factor as usize] * record.value);

                            predictions[record.index] += 2. * (new_pair_weight - self_model.pair_weights[feature as usize][factor as usize]) * h;

                            factors_sum[record.index] += 
                                (new_pair_weight - self_model.pair_weights[feature as usize][factor as usize]) * record.value;
                        }

                        self_model.pair_weights[feature as usize][factor as usize] = new_pair_weight;
                    });
                }
            }

            if let &Some(ref test_dataset) = test_dataset_option {
                println!("Epoch: {}, loss on train: {}, loss on test: {}", epoch, 
                         self.eval_loss(&dataset, loss), self.eval_loss(&test_dataset, loss));
            } else {
                println!("Epoch: {}, loss on train: {}", epoch, self.eval_loss(&dataset, loss));
            }
        }
    }
}
