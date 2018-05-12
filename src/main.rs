extern crate argparse;
extern crate factorization_machine;
extern crate rayon;

use std::fs::File;
use std::io::prelude::*;

use argparse::{ArgumentParser, Store, StoreOption, StoreTrue};

use factorization_machine::dataset::Dataset;
use factorization_machine::model::Model;
use factorization_machine::loss::{Loss, Logistic, Mae, Mse};
use factorization_machine::optimizer::{AdaGrad, Sgd};

fn main() {
    let mut dataset_path = "".to_string();
    let mut dataset_type = "libsvm".to_string();
    let mut target_column_opt: Option<String> = None;
    let mut model_path = "".to_string();
    let mut output_path_opt: Option<String> = None;
    let mut test_dataset_path: Option<String> = None;
    let mut bits = 18;
    let mut factors_number = 10;
    let mut optimizer_name = "adagrad".to_string();
    let mut loss_name = "mse".to_string();
    let mut epochs = 10;
    let mut lr = 0.1;
    let mut lr_decay = 1.0;
    let mut l2_regularization = 1e-5;
    let mut predict = false;
    let mut jobs = 1;

    {
        let mut ap = ArgumentParser::new();
        ap.refer(&mut predict)
            .add_option(&["-p", "--predict"], StoreTrue, "Predict");
        ap.refer(&mut dataset_path)
            .add_option(&["-d", "--data"], Store, "Dataset path").required();
        ap.refer(&mut test_dataset_path)
            .add_option(&["--test_data"], StoreOption, "Test dataset path");
        ap.refer(&mut dataset_type)
            .add_option(&["--data_type"], Store, 
                        "Dataset type. Possible values: csv, libsvm. Default value: libsvm");
        ap.refer(&mut target_column_opt)
            .add_option(&["-t", "--target"], StoreOption, "Target column for csv dataset");
        ap.refer(&mut model_path)
            .add_option(&["-m", "--model"], Store, "Model path").required();
        ap.refer(&mut output_path_opt)
            .add_option(&["-o", "--output"], StoreOption, "Output path");
        ap.refer(&mut optimizer_name)
            .add_option(&["--opt"], Store, 
                        "Optimizer. Possible values: sgd, adagrad, als. Default value: adagrad");
        ap.refer(&mut loss_name)
            .add_option(&["--loss"], Store,
                        "Loss. Possible values: mse, logistic, mae. Default value: mse");
        ap.refer(&mut epochs)
            .add_option(&["-i", "--iterations"], Store, "Number of epochs");
        ap.refer(&mut bits)
            .add_option(&["-b", "--bits"], Store, "Number of bits in feature");
        ap.refer(&mut factors_number)
            .add_option(&["-k", "--factors_number"], Store, "Factors number");
        ap.refer(&mut l2_regularization)
            .add_option(&["--l2"], Store, "l2 regularization");
        ap.refer(&mut lr)
            .add_option(&["--lr"], Store, "Learning rate");
        ap.refer(&mut lr_decay)
            .add_option(&["--decay"], Store, "Learning rate decay");
        ap.refer(&mut jobs)
            .add_option(&["-j", "--jobs"], Store, "Threads number");
        ap.parse_args_or_exit();
    }
    
    rayon::ThreadPoolBuilder::new().num_threads(jobs).build_global().unwrap();

    let mut loss: Box<Loss>;
    loss_name = loss_name.to_lowercase();
    if loss_name == "mse" {
        loss = Box::new(Mse{});
    } else if loss_name == "mae" {
        loss = Box::new(Mae{});
    } else if loss_name == "logistic" {
        loss = Box::new(Logistic{});
    } else {
        panic!("Unknown loss name: {}", loss_name);
    }

    if predict {
        let model = Model::load(&model_path).expect("Can't load model");

        dataset_type = dataset_type.to_lowercase();
        let dataset: Dataset;
        if dataset_type == "libsvm" {
            dataset = Dataset::from_libsvm(&dataset_path, bits).expect("Can't load dataset");
        } else if dataset_type == "csv" {
            let target_column = target_column_opt.expect(
                "Target column must be specified for csv dataset");
            dataset = Dataset::from_csv(&dataset_path, &target_column, model.hash_bits).expect("Can't load dataset");
        } else {
            panic!("Unknown dataset type: {}", dataset_type);
        }

        let prediction = model.predict(&dataset);
        if let Some(output_path) = output_path_opt {
            let mut file = File::create(output_path).expect("Can't create output file");
            for x in prediction {
                file.write_all((x.to_string() + "\n").as_bytes()).expect("Can't write line to output file");
            }
        }
        println!("Loss on test: {}", model.eval_loss(&dataset, &mut *loss));
    } else {
        let mut model = Model::new(bits, factors_number);

        dataset_type = dataset_type.to_lowercase();
        let dataset: Dataset;
        let test_dataset: Option<Dataset>;
        if dataset_type == "libsvm" {
            dataset = Dataset::from_libsvm(&dataset_path, bits).expect("Can't load dataset");
            test_dataset = match test_dataset_path {
                Some(path) => Some(Dataset::from_libsvm(&path, bits).expect("Can't load test dataset")),
                None => None
            }
        } else if dataset_type == "csv" {
            let target_column = target_column_opt.expect(
                "Target column must be specified for csv dataset");
            dataset = Dataset::from_csv(&dataset_path, &target_column, bits).expect("Can't load dataset");
            test_dataset = match test_dataset_path {
                Some(path) => Some(Dataset::from_csv(&path, &target_column, bits).expect("Can't load dataset")),
                None => None
            }
        } else {
            panic!("Unknown dataset type: {}", dataset_type);
        }

        if optimizer_name == "als" {
            model.fit_als(&dataset, &test_dataset, &mut *loss, epochs, l2_regularization);
        } else if optimizer_name == "adagrad" {
            let optimizer = AdaGrad::new(model.get_parameters_number(), lr, lr_decay);
            model.fit(&dataset, &test_dataset, &mut *loss, optimizer, epochs, l2_regularization);
        } else if optimizer_name == "sgd" {
            let optimizer = Sgd::new(lr, lr_decay);
            model.fit(&dataset, &test_dataset, &mut *loss, optimizer, epochs, l2_regularization);
        } else {
            panic!("Unknown optimizer name: {}", optimizer_name);
        }
        model.save(&model_path).expect("Can't save model");
    }
}
