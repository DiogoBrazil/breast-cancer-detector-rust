use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::pool::{MaxPool2d, MaxPool2dConfig};
use burn::nn::{BatchNorm, BatchNormConfig, PaddingConfig2d};
use burn::tensor::{Tensor, backend::Backend};

/// Conv2d + BatchNorm + ReLU
#[derive(Module, Debug)]
pub struct ConvBlock<B: Backend> {
    conv: Conv2d<B>,
    bn: BatchNorm<B>,
}

impl<B: Backend> ConvBlock<B> {
    pub fn new(device: &B::Device, in_ch: usize, out_ch: usize) -> Self {
        Self {
            conv: Conv2dConfig::new([in_ch, out_ch], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(device),
            bn: BatchNormConfig::new(out_ch).init(device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv.forward(x);
        let x = self.bn.forward(x);
        burn::tensor::activation::relu(x)
    }
}

/// 5 blocos convolucionais com MaxPool2d(2x2) entre cada.
/// Input:  [B, 3, 416, 416]
/// Output: [B, 256, 13, 13]  (downsampling 32x)
#[derive(Module, Debug)]
pub struct Backbone<B: Backend> {
    block1: ConvBlock<B>, // 3   -> 32,  /2 -> 208
    block2: ConvBlock<B>, // 32  -> 64,  /2 -> 104
    block3: ConvBlock<B>, // 64  -> 128, /2 -> 52
    block4: ConvBlock<B>, // 128 -> 256, /2 -> 26
    block5: ConvBlock<B>, // 256 -> 256, /2 -> 13
    pool: MaxPool2d,
}

impl<B: Backend> Backbone<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            block1: ConvBlock::new(device, 3, 32),
            block2: ConvBlock::new(device, 32, 64),
            block3: ConvBlock::new(device, 64, 128),
            block4: ConvBlock::new(device, 128, 256),
            block5: ConvBlock::new(device, 256, 256),
            pool: MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init(),
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.pool.forward(self.block1.forward(x));
        let x = self.pool.forward(self.block2.forward(x));
        let x = self.pool.forward(self.block3.forward(x));
        let x = self.pool.forward(self.block4.forward(x));
        self.pool.forward(self.block5.forward(x))
    }

    pub fn out_channels(&self) -> usize {
        256
    }
}
