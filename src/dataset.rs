use std::collections::hash_map::DefaultHasher;
use std::error::Error;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::prelude::*;
use std::io::BufReader;

use csv;

#[derive(Debug)]
pub struct Feature {
    pub name: u32,
    pub value: f32
}

impl Feature {
    fn new(name: &str, value: f32, bits: u32) -> Feature {
        let mut s = DefaultHasher::new();
        name.hash(&mut s);
        Feature {
            name: (s.finish() as u32) & ((1 << bits) - 1),
            value 
        }
    }
}

#[derive(Debug)]
pub struct Record {
    pub target: f32,
    pub features: Vec<Feature>
}

impl Record {
    pub fn new() -> Record {
        Record {
            target: 0.,
            features: Vec::new()
        }
    }
}

#[derive(Debug)]
pub struct Dataset {
    pub records: Vec<Record>
}

impl Dataset {
    pub fn new() -> Dataset {
        Dataset {
            records: Vec::new()
        }
    }

    pub fn from_libsvm(path: &str, bits: u32) -> Result<Dataset, Box<Error>> {
        let f = match File::open(path) {
            Ok(v) => v,
            Err(e) => return Err(From::from(e))
        };
        let file = BufReader::new(&f);

        let mut dataset = Dataset::new();
        for line in file.lines() {
            let l = line.unwrap();
            let mut tokens = l.split_whitespace();

            let mut record = Record::new();
            match tokens.next() {
                Some(v) => record.target = v.parse()?,
                None => continue
            }

            for feature in tokens {
                let mut feature_tokens = feature.split(":");
                let name = feature_tokens.next().unwrap();
                let value = match feature_tokens.next() {
                    Some(v) => v.parse()?,
                    None => 1.
                };
                let feature_object = Feature::new(name, value, bits);
                if let Some(position) = record.features.iter()
                        .position(|ref r| r.name == feature_object.name) {
                    record.features[position].value += value;
                } else {
                    record.features.push(feature_object);
                }
            }
            dataset.records.push(record);
        }
        Ok(dataset)
    }

    pub fn from_csv(path: &str, target_field: &str, bits: u32) -> Result<(Dataset), Box<Error>> {
        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(false)
            .from_path(path)?;
        let header = rdr.records().next().ok_or("Empty csv")??;

        let mut dataset = Dataset::new();
        for result in rdr.records() {
            let mut record = Record::new();
            let csv_line = result?;
            for (value, name) in csv_line.iter().zip(&header) {
                if name == target_field {
                    record.target = value.parse()?;
                } else {
                    let feature_object = Feature::new(&(name.to_owned() + "^" + value), 1., bits);
                    if let Some(position) = record.features.iter()
                            .position(|ref r| r.name == feature_object.name) {
                        record.features[position].value += 1.;
                    } else {
                        record.features.push(feature_object);
                    }
                }
            }
            dataset.records.push(record)
        }
        Ok(dataset)
    }
}
