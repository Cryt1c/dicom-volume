//! Temporary: cold-process break-even, i.e. counting GPU upload + first-call
//! initialization as part of the measurement. Run once per N:
//!   cargo run --release --example cold_break_even -- <N>

use std::path::PathBuf;

use dicom_volume::{
    enums::{Interpolation, Orientation, SortBy},
    gpu_interpolator::GpuInterpolator,
    volume_loader::VolumeLoader,
};

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|value| value.parse().ok())
        .unwrap_or(64);

    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("failed to build tokio runtime");
    let volume = VolumeLoader::load_from_directory(PathBuf::from("dicom"), SortBy::InstanceNumber)
        .expect("should have loaded files from directory");
    let max_index = volume.dim().1;
    let n = n.min(max_index);
    // Centered sampling of `n` evenly spread slices.
    let indices: Vec<usize> = (0..n)
        .map(|i| (i * max_index + max_index / 2) / n)
        .collect();

    // CPU path, same cold process.
    let start = web_time::Instant::now();
    let cpu = runtime.block_on(volume.get_images_from_axis(
        &indices,
        Orientation::Coronal,
        Interpolation::Linear,
        None,
    ));
    let cpu_ms = start.elapsed().as_secs_f64() * 1000.0;
    std::hint::black_box(cpu);

    // GPU path: timing starts before the interpolator exists, so the one-time
    // volume upload is included. First-call pipeline/driver init lands in
    // `extract_ms` and is therefore included as well.
    let start = web_time::Instant::now();
    let mut gpu = runtime
        .block_on(GpuInterpolator::try_new(&volume.data, volume.spacing))
        .expect("no GPU adapter supporting 16-bit normalized textures");
    let upload_ms = start.elapsed().as_secs_f64() * 1000.0;

    let start = web_time::Instant::now();
    let gpu_result = runtime.block_on(volume.get_images_from_axis(
        &indices,
        Orientation::Coronal,
        Interpolation::Linear,
        Some(&mut gpu),
    ));
    let extract_ms = start.elapsed().as_secs_f64() * 1000.0;
    std::hint::black_box(gpu_result);

    let gpu_total = upload_ms + extract_ms;
    println!(
        "RESULT n={n:>3} cpu={cpu_ms:>8.2} upload={upload_ms:>7.2} extract={extract_ms:>8.2} gpu_total={gpu_total:>8.2} winner={}",
        if gpu_total < cpu_ms { "gpu" } else { "cpu" }
    );
}
