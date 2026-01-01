use std::path::PathBuf;

use dicom_volume::{
    enums::{Interpolation, Orientation, SortBy},
    gpu_interpolator::GpuInterpolator,
    volume_loader::VolumeLoader,
};

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let start = web_time::Instant::now();
    let volume = VolumeLoader::load_from_directory(&PathBuf::from("dicom"), SortBy::InstanceNumber)
        .expect("should have loaded files from directory");
    let gpu_interpolator = GpuInterpolator::new(&volume.data, volume.spacing).await;

    println!("request_device: {}ms", start.elapsed().as_millis());
    let start = web_time::Instant::now();
    let image = volume
        .get_image_from_axis(
            volume.dim().2 / 2,
            Orientation::Coronal,
            Interpolation::Linear,
            Some(&gpu_interpolator),
        )
        .await
        .expect("should have returned image at center of volume");

    println!("gpu: {}ms", start.elapsed().as_millis());
    let start = web_time::Instant::now();
    let image_cpu = volume
        .get_image_from_axis(
            volume.dim().2 / 2,
            Orientation::Coronal,
            Interpolation::Linear,
            None,
        )
        .await
        .expect("should have returned image at center of volume");
    println!("cpu: {}ms", start.elapsed().as_millis());
    let _ = image.save("result_gpu.png");
    let _ = image_cpu.save("result_cpu.png");
}
