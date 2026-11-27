//! Temporary: cost of the very first GPU extraction in a fresh process
//! (driver/shader init + clock ramp) versus the warmed-up steady state.

use std::path::PathBuf;

use dicom_volume::{
    enums::{Interpolation, Orientation, SortBy},
    gpu_interpolator::GpuInterpolator,
    volume_loader::VolumeLoader,
};

fn main() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("failed to build tokio runtime");
    let volume = VolumeLoader::load_from_directory(PathBuf::from("dicom"), SortBy::InstanceNumber)
        .expect("should have loaded files from directory");

    // CPU first, to get its cold number out of the way too.
    let start = web_time::Instant::now();
    let cpu = runtime.block_on(volume.get_images_from_axis(
        &[256],
        Orientation::Coronal,
        Interpolation::Linear,
        None,
    ));
    println!(
        "cpu cold single slice: {:.2} ms",
        start.elapsed().as_secs_f64() * 1000.0
    );
    std::hint::black_box(cpu);

    let start = web_time::Instant::now();
    let mut gpu = runtime
        .block_on(GpuInterpolator::try_new(&volume.data, volume.spacing))
        .expect("no GPU adapter supporting 16-bit normalized textures");
    println!(
        "gpu upload (try_new): {:.2} ms",
        start.elapsed().as_secs_f64() * 1000.0
    );

    let mut timings = Vec::new();
    for i in 0..12 {
        let start = web_time::Instant::now();
        let result = runtime.block_on(volume.get_images_from_axis(
            &[256],
            Orientation::Coronal,
            Interpolation::Linear,
            Some(&mut gpu),
        ));
        timings.push(start.elapsed().as_secs_f64() * 1000.0);
        std::hint::black_box(result);
        println!("gpu single slice call {i:>2}: {:>8.2} ms", timings[i]);
    }

    let start = web_time::Instant::now();
    let indices: Vec<usize> = (0..volume.dim().1).collect();
    let result = runtime.block_on(volume.get_images_from_axis(
        &indices,
        Orientation::Coronal,
        Interpolation::Linear,
        Some(&mut gpu),
    ));
    println!(
        "gpu 512 slices after warmup: {:.2} ms ({:.2} ms/slice)",
        start.elapsed().as_secs_f64() * 1000.0,
        start.elapsed().as_secs_f64() * 1000.0 / 512.0
    );
    std::hint::black_box(result);
}
