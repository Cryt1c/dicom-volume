use std::path::PathBuf;

use dicom_volume::{
    enums::{Interpolation, Orientation, SortBy},
    volume::WGPU,
    volume_loader::VolumeLoader,
};

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let mut volume =
        VolumeLoader::load_from_directory(&PathBuf::from("dicom"), SortBy::InstanceNumber)
            .expect("should have loaded files from directory");
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        ..Default::default()
    });

    // Request adapter
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            ..Default::default()
        })
        .await
        .expect("Failed to find an appropriate adapter");

    // Request device and queue
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            ..Default::default()
        })
        .await
        .expect("Failed to create device");

    let wgpu = WGPU { device, queue };

    let image = volume
        .get_image_from_axis(
            volume.dim().2 / 2,
            Orientation::Coronal,
            Interpolation::Linear,
            Some(wgpu),
        )
        .await
        .expect("should have returned image at center of volume");
    let image_cpu = volume
        .get_image_from_axis(
            volume.dim().2 / 2,
            Orientation::Coronal,
            dicom_volume::enums::Interpolation::Linear,
            None,
        )
        .await
        .expect("should have returned image at center of volume");
    let _ = image.save("result_gpu.png");
    let _ = image_cpu.save("result_cpu.png");
}
