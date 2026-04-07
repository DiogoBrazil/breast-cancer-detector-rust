use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{BatchNorm, BatchNormConfig, PaddingConfig2d};
use burn::tensor::{Tensor, backend::Backend};

/// Conv2d(3x3) + BN + ReLU + Conv2d(1x1) para uma branch de detecção.
#[derive(Module, Debug)]
pub struct HeadBranch<B: Backend> {
    conv1: Conv2d<B>,
    bn1: BatchNorm<B>,
    conv2: Conv2d<B>,
}

impl<B: Backend> HeadBranch<B> {
    pub fn new(device: &B::Device, in_ch: usize, hidden_ch: usize, out_ch: usize) -> Self {
        Self {
            conv1: Conv2dConfig::new([in_ch, hidden_ch], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(device),
            bn1: BatchNormConfig::new(hidden_ch).init(device),
            conv2: Conv2dConfig::new([hidden_ch, out_ch], [1, 1]).init(device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv1.forward(x);
        let x = self.bn1.forward(x);
        let x = burn::tensor::activation::relu(x);
        self.conv2.forward(x)
    }
}

/// Saída do detector com as predições brutas (logits).
pub struct DetectionOutput<B: Backend> {
    /// Logits de classificação [B, 1, H, W] — sigmoid aplicado na loss/inferência
    pub cls_logits: Tensor<B, 4>,
    /// Regressão de offsets [B, 4, H, W] — distâncias ltrb de cada célula ao bbox
    pub reg_pred: Tensor<B, 4>,
}

/// Detection head com 2 branches sobre o feature map do backbone.
#[derive(Module, Debug)]
pub struct DetectionHead<B: Backend> {
    cls_branch: HeadBranch<B>,
    reg_branch: HeadBranch<B>,
}

impl<B: Backend> DetectionHead<B> {
    pub fn new(device: &B::Device, in_ch: usize, num_classes: usize) -> Self {
        Self {
            cls_branch: HeadBranch::new(device, in_ch, 128, num_classes),
            reg_branch: HeadBranch::new(device, in_ch, 128, 4),
        }
    }

    pub fn forward(&self, features: Tensor<B, 4>) -> DetectionOutput<B> {
        DetectionOutput {
            cls_logits: self.cls_branch.forward(features.clone()),
            reg_pred: self.reg_branch.forward(features),
        }
    }
}
