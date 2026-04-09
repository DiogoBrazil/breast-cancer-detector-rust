use burn::tensor::{Tensor, backend::Backend};

use super::target::GridTargets;
use crate::model::DetectionOutput;

pub struct LossOutput<B: Backend> {
    pub total: Tensor<B, 1>,
    pub cls_loss: Tensor<B, 1>,
    pub reg_loss: Tensor<B, 1>,
}

/// Focal Loss with logits (RetinaNet, α=0.25, γ=2.0).
/// Reduz o peso dos negativos fáceis, crucial para grids com poucos positivos.
/// Numericamente estável via log-sigmoid.
fn focal_loss_with_logits<B: Backend>(logits: Tensor<B, 4>, targets: Tensor<B, 4>) -> Tensor<B, 1> {
    let alpha = 0.25f32;
    let gamma = 2.0f32;

    // p = sigmoid(logits), calculado de forma estável
    let p = burn::tensor::activation::sigmoid(logits.clone());

    // log(p) estável = logits - relu(logits) - log(1 + exp(-|logits|))
    // log(1-p) estável = -relu(logits) - log(1 + exp(-|logits|))
    let abs_logits = logits.clone().abs();
    let log_stable = (abs_logits.neg().exp() + 1.0).log(); // log(1 + exp(-|x|))
    let log_p = logits.clone() - logits.clone().clamp_min(0.0) - log_stable.clone();
    let log_1_minus_p = logits.clamp_min(0.0).neg() - log_stable;

    let num_pos = targets.clone().sum().clamp_min(1.0);

    // Focal weight e loss por elemento
    let p_t = targets.clone() * p.clone() + (targets.clone().neg() + 1.0) * (p.clone().neg() + 1.0);
    let focal_weight = (p_t.neg() + 1.0).powf_scalar(gamma);
    let alpha_t = targets.clone() * alpha + (targets.clone().neg() + 1.0) * (1.0 - alpha);

    let ce = targets.clone().neg() * log_p + (targets.neg() + 1.0).neg() * log_1_minus_p;

    let loss = alpha_t * focal_weight * ce;

    loss.sum() / num_pos
}

/// Smooth L1 loss aplicada apenas nas posições positivas (onde há objeto).
fn smooth_l1_masked<B: Backend>(
    pred: Tensor<B, 4>,
    target: Tensor<B, 4>,
    mask: Tensor<B, 4>,
) -> Tensor<B, 1> {
    let diff = (pred - target).abs();

    // smooth_l1: 0.5*x² se |x|<1, |x|-0.5 caso contrário
    let ones = diff.clone().ones_like();
    let lt_one = diff.clone().lower(ones);
    let lt_one_f = lt_one.float();

    let quadratic = diff.clone().powf_scalar(2.0) * 0.5;
    let linear = diff - 0.5;

    let loss = lt_one_f.clone() * quadratic + (lt_one_f.neg() + 1.0) * linear;

    // Expandir máscara [B,1,H,W] -> [B,4,H,W] para casar com loss
    let mask4 = mask.clone().repeat_dim(1, 4);
    let masked_loss = loss * mask4;

    // Dividir pelo número de positivos (mask=1), clamp para evitar div/0
    let num_pos = mask.sum().clamp_min(1.0);
    masked_loss.sum() / num_pos
}

/// Calcula a loss total do detector.
/// cls_loss: BCE with logits sobre todo o grid.
/// reg_loss: Smooth L1 apenas nas posições positivas.
pub fn detection_loss<B: Backend>(
    output: &DetectionOutput<B>,
    targets: &GridTargets<B>,
    reg_weight: f32,
) -> LossOutput<B> {
    let cls_loss = focal_loss_with_logits(output.cls_logits.clone(), targets.cls_target.clone());

    let reg_loss = smooth_l1_masked(
        output.reg_pred.clone(),
        targets.reg_target.clone(),
        targets.pos_mask.clone(),
    );

    let total = cls_loss.clone() + reg_loss.clone() * reg_weight;

    LossOutput {
        total,
        cls_loss,
        reg_loss,
    }
}
