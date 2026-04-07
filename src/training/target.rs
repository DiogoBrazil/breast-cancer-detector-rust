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

/// Converte bounding boxes (xyxy em coords da imagem) em targets no grid.
///
/// Para cada célula do grid, verifica se o centro dela cai dentro de algum GT box.
/// Se cair em múltiplos boxes, atribui ao menor (por área).
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

            // Quais células do grid podem ser afetadas por este box
            let gx_start = ((x_min / stride).floor() as usize).min(grid_size - 1);
            let gx_end = ((x_max / stride).ceil() as usize).min(grid_size);
            let gy_start = ((y_min / stride).floor() as usize).min(grid_size - 1);
            let gy_end = ((y_max / stride).ceil() as usize).min(grid_size);

            for gy in gy_start..gy_end {
                for gx in gx_start..gx_end {
                    let cx = gx as f32 * stride + stride / 2.0;
                    let cy = gy as f32 * stride + stride / 2.0;

                    // Centro da célula está dentro do GT box?
                    if cx >= x_min && cx <= x_max && cy >= y_min && cy <= y_max {
                        let idx = b * grid_cells + gy * grid_size + gx;

                        // Se múltiplos boxes, atribui ao menor
                        if area < assigned_area[idx] {
                            assigned_area[idx] = area;
                            cls_data[idx] = 1.0;

                            let l = (cx - x_min) / stride;
                            let t = (cy - y_min) / stride;
                            let r = (x_max - cx) / stride;
                            let bot = (y_max - cy) / stride;

                            let reg_base = b * 4 * grid_cells;
                            reg_data[reg_base + gy * grid_size + gx] = l;
                            reg_data[reg_base + grid_cells + gy * grid_size + gx] = t;
                            reg_data[reg_base + 2 * grid_cells + gy * grid_size + gx] = r;
                            reg_data[reg_base + 3 * grid_cells + gy * grid_size + gx] = bot;
                        }
                    }
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
