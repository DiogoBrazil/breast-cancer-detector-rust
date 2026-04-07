use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::Path;

use burn::data::dataset::Dataset;

use breast_cancer_detector_rust::data::{dataset::MammogramDataset, image_ops};
use breast_cancer_detector_rust::inference::draw::save_preprocessed_sample_with_boxes;
use breast_cancer_detector_rust::training::{TrainConfig, run_train_loop};
use breast_cancer_detector_rust::utils::print_header;

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    Inspect,
    Train,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let config = TrainConfig::default();

    match cli.command {
        Commands::Inspect => {
            print_header("INSPECT DATASET");

            let train_dataset = MammogramDataset::new(format!("{}/train", config.dataset_root))?;
            let valid_dataset = MammogramDataset::new(format!("{}/valid", config.dataset_root))?;
            let test_dataset = MammogramDataset::new(format!("{}/test", config.dataset_root))?;

            println!("Train samples: {}", train_dataset.len());
            println!("Valid samples: {}", valid_dataset.len());
            println!("Test samples: {}", test_dataset.len());

            if let Some(sample) = train_dataset.get(0) {
                println!("First sample image: {}", sample.image_name);
                println!("Image path: {}", sample.image_path.display());
                println!("Width: {}", sample.width);
                println!("Height: {}", sample.height);
                println!("Boxes: {}", sample.boxes.len());

                for (i, bbox) in sample.boxes.iter().enumerate() {
                    println!(
                        "Box {} => x_min={}, y_min={}, x_max={}, y_max={}, class_id={}",
                        i, bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max, bbox.class_id
                    );
                }

                let image = image_ops::load_grayscale_image(&sample.image_path)?;
                let proc = image_ops::preprocess_image_and_boxes(
                    &image,
                    &sample.boxes,
                    config.image_size,
                )?;

                println!("Processed image_size: {}", proc.image_size);
                println!("Processed boxes: {}", proc.boxes.len());
                println!("Scale: {}", proc.scale);
                println!("Pad X: {}", proc.pad_x);
                println!("Pad Y: {}", proc.pad_y);
                println!("Resized W: {}", proc.resized_width);
                println!("Resized H: {}", proc.resized_height);

                // 🔥 NOVO: salvar imagem com bounding boxes
                let output_path = Path::new("outputs/inspect_first_sample.png");

                save_preprocessed_sample_with_boxes(
                    &sample.image_path,
                    &sample.boxes,
                    config.image_size,
                    output_path,
                )?;

                println!("Preview image saved at: {}", output_path.display());
            } else {
                println!("Dataset vazio!");
            }
        }

        Commands::Train => {
            print_header("TRAIN PIPELINE");
            run_train_loop(&config)?;
        }
    }

    Ok(())
}
