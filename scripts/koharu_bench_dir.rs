// Detector bench: run koharu's default layout model over every slice listed in index.json.
use std::{path::PathBuf, time::Instant};

use anyhow::Result;
use koharu_ml::koharu_layout_rfdetr_seg_2xl::KoharuLayoutRFDetrSeg2XL;

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let input = PathBuf::from(&args[1]);
    let output = PathBuf::from(&args[2]);
    std::fs::create_dir_all(&output)?;
    koharu_ml::init().await?;
    let model = KoharuLayoutRFDetrSeg2XL::load(koharu_ml::device(true)).await?;
    let thresholds = model.recommended_thresholds();
    let index: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(input.join("index.json"))?)?;
    for entry in index.as_array().into_iter().flatten() {
        let Some(file) = entry["file"].as_str() else { continue };
        let image = image::open(input.join(file))?;
        let started = Instant::now();
        let detections = model.inference_with_thresholds(&image, thresholds)?;
        let seconds = started.elapsed().as_secs_f64();
        let body = serde_json::json!({"seconds": seconds, "detections": detections.detections});
        std::fs::write(output.join(format!("{}.json", file.replace('/', "__"))), body.to_string())?;
        println!("{file}: {} detections in {seconds:.1} s", detections.detections.len());
    }
    Ok(())
}
