use std::path::{Path, PathBuf};

use dicom_volume::{
    enums::{Interpolation, Orientation, SortBy},
    gpu_interpolator::GpuInterpolator,
    volume_loader::VolumeLoader,
};
use image::{ImageBuffer, Luma};

type SliceImage = ImageBuffer<Luma<u8>, Vec<u8>>;

fn save_all(dir: &Path, images: &[Option<SliceImage>]) {
    std::fs::create_dir_all(dir)
        .unwrap_or_else(|err| panic!("failed to create {}: {err}", dir.display()));
    for (index, image) in images.iter().enumerate() {
        let Some(image) = image else { continue };
        let path = dir.join(format!("coronal_{index:04}.png"));
        image
            .save(&path)
            .unwrap_or_else(|err| panic!("failed to save {}: {err}", path.display()));
    }
    println!("saved {} slices to {}", images.len(), dir.display());
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let volume = VolumeLoader::load_from_directory(PathBuf::from("dicom"), SortBy::InstanceNumber)
        .expect("should have loaded files from directory");
    let mut gpu_interpolator = GpuInterpolator::try_new(&volume.data, volume.spacing)
        .await
        .expect("no GPU adapter available");

    // Coronal slices run along the height axis, so there is one per row of the
    // volume. Extract all of them in one call so the GPU path can batch.
    let indices: Vec<usize> = (0..volume.dim().1).collect();
    println!(
        "volume dim={:?} spacing={:?} -> extracting {} coronal slices",
        volume.dim(),
        volume.spacing,
        indices.len()
    );

    let start = web_time::Instant::now();
    let images_gpu = volume
        .get_images_from_axis(
            &indices,
            Orientation::Coronal,
            Interpolation::Linear,
            Some(&mut gpu_interpolator),
        )
        .await;
    let gpu_elapsed = start.elapsed();

    let start = web_time::Instant::now();
    let images_cpu = volume
        .get_images_from_axis(&indices, Orientation::Coronal, Interpolation::Linear, None)
        .await;
    let cpu_elapsed = start.elapsed();

    let slices = indices.len() as f64;
    println!(
        "gpu: {:>8.2} ms total, {:>7.2} ms/slice ({} slices)",
        gpu_elapsed.as_secs_f64() * 1000.0,
        gpu_elapsed.as_secs_f64() * 1000.0 / slices,
        images_gpu.iter().filter(|image| image.is_some()).count()
    );
    println!(
        "cpu: {:>8.2} ms total, {:>7.2} ms/slice ({} slices)",
        cpu_elapsed.as_secs_f64() * 1000.0,
        cpu_elapsed.as_secs_f64() * 1000.0 / slices,
        images_cpu.iter().filter(|image| image.is_some()).count()
    );

    // Persist every extracted slice for a side-by-side comparison.
    save_all(Path::new("dicom/gpu"), &images_gpu);
    save_all(Path::new("dicom/cpu"), &images_cpu);
}
