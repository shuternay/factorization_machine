pub trait Optimizer {
    fn get_step(&mut self, parameter: usize, deriv: f32) -> f32;
    fn next_step(&mut self);
    fn next_epoch(&mut self);
}


#[derive(Debug)]
pub struct Sgd {
    lr: f32,
    decay: f32,
    epoch: i32,
    pub step: i32
}

impl Sgd {
    pub fn new(lr: f32, decay: f32) -> Sgd {
        Sgd{lr, decay, epoch: 0, step: 0}
    }
}

impl Optimizer for Sgd {
    fn get_step(&mut self, _parameter: usize, deriv: f32) -> f32 {
        self.lr * self.decay.powi(self.epoch) * deriv / (1. + self.step as f32)
    }

    fn next_step(&mut self) {
        self.step += 1
    }

    fn next_epoch(&mut self) {
        self.epoch += 1
    }
}


#[derive(Debug)]
pub struct AdaGrad {
    parameter_derivs_sum: Vec<f32>,
    lr: f32,
    decay: f32,
    epoch: i32,
}

impl AdaGrad {
    pub fn new(parameters_number: usize, lr: f32, decay: f32) -> AdaGrad {
        AdaGrad {
            parameter_derivs_sum: vec![1e-8; parameters_number],
            lr, 
            decay, 
            epoch: 0, 
        }
    }
}

impl Optimizer for AdaGrad {
    fn get_step(&mut self, parameter: usize, deriv: f32) -> f32 {
        self.parameter_derivs_sum[parameter] += deriv.powi(2);
        self.lr * self.decay.powi(self.epoch) * deriv / self.parameter_derivs_sum[parameter].sqrt()
    }

    fn next_step(&mut self) {
    }

    fn next_epoch(&mut self) {
        self.epoch += 1
    }
}
