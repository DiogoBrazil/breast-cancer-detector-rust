use std::path::PathBuf;

use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::pool::{MaxPool2d, MaxPool2dConfig};
use burn::nn::{BatchNorm, BatchNormConfig, PaddingConfig2d};
use burn::tensor::{Tensor, backend::Backend};

/// Stride total do backbone ResNet-18 até layer3 (2×2×1×2×2 = 16).
pub const BACKBONE_STRIDE: usize = 16;

// ---------------------------------------------------------------------------
// Download de pesos pré-treinados
// ---------------------------------------------------------------------------

const RESNET18_URL: &str = "https://download.pytorch.org/models/resnet18-f37072fd.pth";
const RESNET18_EXPECTED_SIZE: u64 = 46_830_571;

/// Retorna o caminho dos pesos ResNet-18, baixando do PyTorch se necessário.
pub fn ensure_resnet18_weights() -> anyhow::Result<PathBuf> {
    let home =
        std::env::var("HOME").map_err(|_| anyhow::anyhow!("HOME env var não definida"))?;
    let cache_dir = PathBuf::from(home)
        .join(".cache")
        .join("breast-cancer-detector");
    std::fs::create_dir_all(&cache_dir)?;

    let weights_path = cache_dir.join("resnet18-f37072fd.pth");

    if weights_path.exists() {
        let metadata = std::fs::metadata(&weights_path)?;
        if metadata.len() == RESNET18_EXPECTED_SIZE {
            println!("Usando pesos em cache: {}", weights_path.display());
            return Ok(weights_path);
        }
        println!("Cache corrompido (tamanho incorreto), re-baixando...");
        std::fs::remove_file(&weights_path)?;
    }

    println!("Baixando pesos pretrained ResNet-18 (~44MB)...");
    let response = ureq::get(RESNET18_URL)
        .call()
        .map_err(|e| anyhow::anyhow!("Erro no download: {}", e))?;

    let mut file = std::fs::File::create(&weights_path)?;
    std::io::copy(&mut response.into_reader(), &mut file)?;

    let downloaded_size = std::fs::metadata(&weights_path)?.len();
    if downloaded_size != RESNET18_EXPECTED_SIZE {
        std::fs::remove_file(&weights_path)?;
        anyhow::bail!(
            "Download incompleto: {} bytes (esperado {})",
            downloaded_size,
            RESNET18_EXPECTED_SIZE
        );
    }

    println!("Pesos salvos em: {}", weights_path.display());
    Ok(weights_path)
}

// ---------------------------------------------------------------------------
// Downsample (projeção 1x1 para residual connections)
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct Downsample<B: Backend> {
    conv: Conv2d<B>,
    bn: BatchNorm<B>,
}

impl<B: Backend> Downsample<B> {
    pub fn new(device: &B::Device, in_ch: usize, out_ch: usize, stride: usize) -> Self {
        Self {
            conv: Conv2dConfig::new([in_ch, out_ch], [1, 1])
                .with_stride([stride, stride])
                .init(device),
            bn: BatchNormConfig::new(out_ch).init(device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        self.bn.forward(self.conv.forward(x))
    }
}

// ---------------------------------------------------------------------------
// BasicBlock (ResNet-18/34)
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct BasicBlock<B: Backend> {
    conv1: Conv2d<B>,
    bn1: BatchNorm<B>,
    conv2: Conv2d<B>,
    bn2: BatchNorm<B>,
    downsample: Option<Downsample<B>>,
}

impl<B: Backend> BasicBlock<B> {
    pub fn new(device: &B::Device, in_ch: usize, out_ch: usize, stride: usize) -> Self {
        let downsample = if stride != 1 || in_ch != out_ch {
            Some(Downsample::new(device, in_ch, out_ch, stride))
        } else {
            None
        };

        Self {
            conv1: Conv2dConfig::new([in_ch, out_ch], [3, 3])
                .with_stride([stride, stride])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(device),
            bn1: BatchNormConfig::new(out_ch).init(device),
            conv2: Conv2dConfig::new([out_ch, out_ch], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(device),
            bn2: BatchNormConfig::new(out_ch).init(device),
            downsample,
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let identity = match &self.downsample {
            Some(ds) => ds.forward(x.clone()),
            None => x.clone(),
        };

        let out = self.conv1.forward(x);
        let out = self.bn1.forward(out);
        let out = burn::tensor::activation::relu(out);

        let out = self.conv2.forward(out);
        let out = self.bn2.forward(out);

        burn::tensor::activation::relu(out + identity)
    }
}

// ---------------------------------------------------------------------------
// LayerBlock (sequência de BasicBlocks)
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct LayerBlock<B: Backend> {
    blocks: Vec<BasicBlock<B>>,
}

impl<B: Backend> LayerBlock<B> {
    pub fn new(
        device: &B::Device,
        num_blocks: usize,
        in_ch: usize,
        out_ch: usize,
        stride: usize,
    ) -> Self {
        let mut blocks = Vec::with_capacity(num_blocks);
        // Primeiro bloco: pode ter stride > 1 e downsample
        blocks.push(BasicBlock::new(device, in_ch, out_ch, stride));
        // Blocos restantes: stride 1, sem downsample
        for _ in 1..num_blocks {
            blocks.push(BasicBlock::new(device, out_ch, out_ch, 1));
        }
        Self { blocks }
    }

    pub fn forward(&self, mut x: Tensor<B, 4>) -> Tensor<B, 4> {
        for block in &self.blocks {
            x = block.forward(x);
        }
        x
    }
}

// ---------------------------------------------------------------------------
// ResNetBackbone (ResNet-18 até layer3)
// ---------------------------------------------------------------------------

/// Backbone ResNet-18 truncado em layer3.
///
/// Input:  [B, 3, 416, 416] (pixels normalizados [0, 1])
/// Output: [B, 256, 26, 26] (stride 16)
///
/// A normalização ImageNet (mean/std por canal) é aplicada automaticamente
/// no forward — o caller entrega pixels [0, 1] e o backbone cuida do resto.
#[derive(Module, Debug)]
pub struct ResNetBackbone<B: Backend> {
    conv1: Conv2d<B>,
    bn1: BatchNorm<B>,
    maxpool: MaxPool2d,
    layer1: LayerBlock<B>,
    layer2: LayerBlock<B>,
    layer3: LayerBlock<B>,
}

impl<B: Backend> ResNetBackbone<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            conv1: Conv2dConfig::new([3, 64], [7, 7])
                .with_stride([2, 2])
                .with_padding(PaddingConfig2d::Explicit(3, 3))
                .init(device),
            bn1: BatchNormConfig::new(64).init(device),
            maxpool: MaxPool2dConfig::new([3, 3])
                .with_strides([2, 2])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(),
            layer1: LayerBlock::new(device, 2, 64, 64, 1),
            layer2: LayerBlock::new(device, 2, 64, 128, 2),
            layer3: LayerBlock::new(device, 2, 128, 256, 2),
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        // Normalização ImageNet (input esperado [0, 1])
        let device = x.device();
        let mean =
            Tensor::<B, 4>::from_floats([[[[0.485]], [[0.456]], [[0.406]]]], &device);
        let std =
            Tensor::<B, 4>::from_floats([[[[0.229]], [[0.224]], [[0.225]]]], &device);
        let x = (x - mean) / std;

        // Stem
        let x = self.conv1.forward(x);
        let x = self.bn1.forward(x);
        let x = burn::tensor::activation::relu(x);
        let x = self.maxpool.forward(x);

        // Stages (layer4 omitido — stride 16, output 26x26 para input 416)
        let x = self.layer1.forward(x);
        let x = self.layer2.forward(x);
        self.layer3.forward(x)
    }

    pub fn out_channels(&self) -> usize {
        256
    }

    /// Carrega pesos pré-treinados de um arquivo PyTorch (.pth).
    /// Tensors de layer4 e fc são ignorados (allow_partial).
    pub fn load_pretrained(&mut self, weights_path: &str) -> anyhow::Result<()> {
        use burn_store::{ModuleSnapshot, PytorchStore};

        let mut store = PytorchStore::from_file(weights_path)
            .with_key_remapping(r"(.+)\.downsample\.0\.(.+)", "$1.downsample.conv.$2")
            .with_key_remapping(r"(.+)\.downsample\.1\.(.+)", "$1.downsample.bn.$2")
            .with_key_remapping(r"(layer[1-3])\.([0-9]+)\.(.+)", "$1.blocks.$2.$3")
            .allow_partial(true);

        let result = self
            .load_from(&mut store)
            .map_err(|e| anyhow::anyhow!("Erro ao carregar pesos pretrained: {}", e))?;

        println!("Pesos pretrained carregados:");
        println!("  Aplicados: {} tensors", result.applied.len());
        if !result.missing.is_empty() {
            println!(
                "  Não usados (layer4/fc): {} tensors",
                result.missing.len()
            );
        }
        Ok(())
    }
}
