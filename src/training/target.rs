use burn::tensor::{Tensor, backend::Backend};

use crate::data::batcher::PreparedSample;

/// Targets codificados no grid para o detector anchor-free (FCOS-like).
pub struct GridTargets<B: Backend> {
    /// 1.0 onde há objeto, 0.0 onde não há. [B, 1, grid_h, grid_w]
    pub cls_target: Tensor<B, 4>,
    /// Distâncias ltrb normalizadas pelo stride. [B, 4, grid_h, grid_w]
    pub reg_target: Tensor<B, 4>,
    /// Máscara dos positivos (igual a cls_target). [B, 1, grid_h, grid_w]
    pub pos_mask: Tensor<B, 4>,
}

/// Tenta atribuir uma célula (gx, gy) para uma box.
/// Retorna true se atribuiu (célula livre ou box atual é menor que a ocupante).
#[allow(clippy::too_many_arguments)]
fn assign_cell(
    b: usize,
    gx: usize,
    gy: usize,
    grid_size: usize,
    stride: f32,
    x_min: f32,
    y_min: f32,
    x_max: f32,
    y_max: f32,
    area: f32,
    clamp_targets: bool,
    cls_data: &mut [f32],
    reg_data: &mut [f32],
    assigned_area: &mut [f32],
) -> bool {
    let grid_cells = grid_size * grid_size;
    let idx = b * grid_cells + gy * grid_size + gx;

    if area >= assigned_area[idx] {
        return false;
    }

    assigned_area[idx] = area;
    cls_data[idx] = 1.0;

    let cx = gx as f32 * stride + stride / 2.0;
    let cy = gy as f32 * stride + stride / 2.0;

    let mut l = (cx - x_min) / stride;
    let mut t = (cy - y_min) / stride;
    let mut r = (x_max - cx) / stride;
    let mut bot = (y_max - cy) / stride;

    if clamp_targets {
        l = l.max(0.0);
        t = t.max(0.0);
        r = r.max(0.0);
        bot = bot.max(0.0);
    }

    let reg_base = b * 4 * grid_cells;
    reg_data[reg_base + gy * grid_size + gx] = l;
    reg_data[reg_base + grid_cells + gy * grid_size + gx] = t;
    reg_data[reg_base + 2 * grid_cells + gy * grid_size + gx] = r;
    reg_data[reg_base + 3 * grid_cells + gy * grid_size + gx] = bot;

    true
}

/// Encontra a célula do grid mais próxima do centro da box que esteja
/// livre ou ocupada por box de área maior (substituível).
fn find_fallback_cell(
    b: usize,
    box_cx: f32,
    box_cy: f32,
    area: f32,
    grid_size: usize,
    stride: f32,
    assigned_area: &[f32],
) -> Option<(usize, usize)> {
    let grid_cells = grid_size * grid_size;
    let mut best: Option<(usize, usize, f32)> = None;

    for gy in 0..grid_size {
        for gx in 0..grid_size {
            let idx = b * grid_cells + gy * grid_size + gx;
            if area < assigned_area[idx] {
                let cx = gx as f32 * stride + stride / 2.0;
                let cy = gy as f32 * stride + stride / 2.0;
                let dist = (cx - box_cx).powi(2) + (cy - box_cy).powi(2);

                if best.is_none() || dist < best.unwrap().2 {
                    best = Some((gx, gy, dist));
                }
            }
        }
    }

    best.map(|(gx, gy, _)| (gx, gy))
}

/// Converte bounding boxes (xyxy em coords da imagem) em targets no grid.
///
/// Para cada célula do grid, verifica se o centro dela cai dentro de algum GT box.
/// Se cair em múltiplos boxes, atribui ao menor (por área).
/// Se nenhuma célula cair dentro de uma box, uma célula fallback é atribuída:
/// a célula mais próxima do centro da box que esteja livre ou substituível.
/// Os targets de regressão do fallback são clampados para >= 0.0
/// (compatível com exp() no head).
/// Os targets de regressão são distâncias ltrb (left, top, right, bottom)
/// do centro da célula até as bordas do box, normalizadas pelo stride.
pub fn encode_targets<B: Backend>(
    samples: &[PreparedSample],
    image_size: usize,
    grid_size: usize,
    device: &B::Device,
) -> GridTargets<B> {
    let stride = image_size as f32 / grid_size as f32;
    let batch_size = samples.len();
    let grid_cells = grid_size * grid_size;

    let mut cls_data = vec![0.0f32; batch_size * grid_cells];
    let mut reg_data = vec![0.0f32; batch_size * 4 * grid_cells];
    // Área do box atribuído a cada célula (para resolver conflitos)
    let mut assigned_area = vec![f32::MAX; batch_size * grid_cells];

    for (b, sample) in samples.iter().enumerate() {
        for bbox in &sample.boxes {
            let (x_min, y_min, x_max, y_max) = (bbox[0], bbox[1], bbox[2], bbox[3]);
            let area = (x_max - x_min) * (y_max - y_min);

            // Regra normal: marcar células cujo centro cai dentro da box
            let gx_start = ((x_min / stride).floor() as usize).min(grid_size - 1);
            let gx_end = ((x_max / stride).ceil() as usize).min(grid_size);
            let gy_start = ((y_min / stride).floor() as usize).min(grid_size - 1);
            let gy_end = ((y_max / stride).ceil() as usize).min(grid_size);

            let mut assigned_count = 0usize;

            for gy in gy_start..gy_end {
                for gx in gx_start..gx_end {
                    let cx = gx as f32 * stride + stride / 2.0;
                    let cy = gy as f32 * stride + stride / 2.0;

                    if cx >= x_min && cx <= x_max && cy >= y_min && cy <= y_max
                        && assign_cell(
                            b, gx, gy, grid_size, stride,
                            x_min, y_min, x_max, y_max, area,
                            false,
                            &mut cls_data, &mut reg_data, &mut assigned_area,
                        )
                    {
                        assigned_count += 1;
                    }
                }
            }

            // Fallback: se nenhuma célula foi atribuída, forçar a mais próxima
            if assigned_count == 0 {
                let box_cx = (x_min + x_max) * 0.5;
                let box_cy = (y_min + y_max) * 0.5;

                if let Some((gx, gy)) = find_fallback_cell(
                    b, box_cx, box_cy, area, grid_size, stride, &assigned_area,
                ) {
                    assign_cell(
                        b, gx, gy, grid_size, stride,
                        x_min, y_min, x_max, y_max, area,
                        true,
                        &mut cls_data, &mut reg_data, &mut assigned_area,
                    );
                }
            }
        }
    }

    let cls_target = Tensor::<B, 1>::from_floats(cls_data.as_slice(), device)
        .reshape([batch_size, 1, grid_size, grid_size]);

    let reg_target = Tensor::<B, 1>::from_floats(reg_data.as_slice(), device)
        .reshape([batch_size, 4, grid_size, grid_size]);

    let pos_mask = cls_target.clone();

    GridTargets {
        cls_target,
        reg_target,
        pos_mask,
    }
}
