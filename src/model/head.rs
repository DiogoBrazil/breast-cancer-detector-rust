use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{BatchNorm, BatchNormConfig, Dropout, DropoutConfig, PaddingConfig2d};
use burn::tensor::{Tensor, backend::Backend};

/// Stack de N camadas Conv3x3+BN+ReLU+Dropout + Conv1x1 final.
#[derive(Module, Debug)]
pub struct HeadBranch<B: Backend> {
    convs: Vec<Conv2d<B>>,
    bns: Vec<BatchNorm<B>>,
    dropouts: Vec<Dropout>,
    out_conv: Conv2d<B>,
}

impl<B: Backend> HeadBranch<B> {
    pub fn new(
        device: &B::Device,
        in_ch: usize,
        hidden_ch: usize,
        out_ch: usize,
        num_layers: usize,
        dropout_rate: f64,
    ) -> Self {
        let mut convs = Vec::new();
        let mut bns = Vec::new();
        let mut dropouts = Vec::new();

        for i in 0..num_layers {
            let ch_in = if i == 0 { in_ch } else { hidden_ch };
            convs.push(
                Conv2dConfig::new([ch_in, hidden_ch], [3, 3])
                    .with_padding(PaddingConfig2d::Explicit(1, 1))
                    .init(device),
            );
            bns.push(BatchNormConfig::new(hidden_ch).init(device));
            dropouts.push(DropoutConfig::new(dropout_rate).init());
        }

        let out_conv = Conv2dConfig::new([hidden_ch, out_ch], [1, 1]).init(device);

        Self { convs, bns, dropouts, out_conv }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut x = x;
        for i in 0..self.convs.len() {
            x = self.convs[i].forward(x);
            x = self.bns[i].forward(x);
            x = burn::tensor::activation::relu(x);
            x = self.dropouts[i].forward(x);
        }
        self.out_conv.forward(x)
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
            cls_branch: HeadBranch::new(device, in_ch, 128, num_classes, 4, 0.1),
            reg_branch: HeadBranch::new(device, in_ch, 128, 4, 4, 0.1),
        }
    }

    pub fn forward(&self, features: Tensor<B, 4>) -> DetectionOutput<B> {
        DetectionOutput {
            cls_logits: self.cls_branch.forward(features.clone()),
            reg_pred: self.reg_branch.forward(features).exp(),
        }
    }
}
