pub trait Loss : Send + Sync {
    fn loss(&self, target: f32, prediction: f32) -> f32;
    fn loss_grad(&self, target: f32, prediction: f32) -> f32;
    fn loss_grad_grad(&self, target: f32, prediction: f32) -> f32;
}


pub struct Mse;

impl Loss for Mse {
    fn loss(&self, target: f32, prediction: f32) -> f32 {
        (prediction - target).powi(2)
    }

    fn loss_grad(&self, target: f32, prediction: f32) -> f32 {
        2. * (prediction - target)
    }

    fn loss_grad_grad(&self, _target: f32, _prediction: f32) -> f32 {
        2.
    }
}


pub struct Mae;

impl Loss for Mae {
    fn loss(&self, target: f32, prediction: f32) -> f32 {
        (prediction - target).abs()
    }

    fn loss_grad(&self, target: f32, prediction: f32) -> f32 {
        (prediction - target).signum()
    }

    fn loss_grad_grad(&self, _target: f32, _prediction: f32) -> f32 {
        2.
    }
}


pub struct Logistic;

impl Loss for Logistic {
    fn loss(&self, target: f32, prediction: f32) -> f32 {
        (1. + (-target * prediction).exp()).ln()
    }

    fn loss_grad(&self, target: f32, prediction: f32) -> f32 {
        -target * (1. - 1. / (1. + (-target * prediction).exp()))
    }

    fn loss_grad_grad(&self, target: f32, prediction: f32) -> f32 {
        let result = target.powi(2) * (-target * prediction).exp() / (1. + (-target * prediction).exp()).powi(2);
        println!("t: {}, p: {}", target, prediction);
        if result.is_nan() {
            0.1
        } else {
            result.max(0.1)
        }
    }
}
