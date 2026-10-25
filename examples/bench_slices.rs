use std::path::PathBuf;

use dicom_volume::{
    enums::{Interpolation, Orientation, SortBy},
    gpu_interpolator::GpuInterpolator,
    volume_loader::VolumeLoader,
};

fn stats(mut v: Vec<u128>) -> (u128, u128, u128) {
    v.sort_unstable();
    let min = v[0];
    let max = *v.last().unwrap();
    let avg = v.iter().sum::<u128>() / v.len() as u128;
    (min, avg, max)
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let volume = VolumeLoader::load_from_directory(&PathBuf::from("dicom"), SortBy::InstanceNumber)
        .expect("load");
    let gpu = GpuInterpolator::new(&volume.data, volume.spacing).await;

    let (_, _, w) = volume.dim();
    let index = w / 2;
    let n = 20;

    // Warm up both paths.
    for _ in 0..2 {
        let _ = volume
            .get_image_from_axis(index, Orientation::Coronal, Interpolation::Linear, Some(&gpu))
            .await;
        let _ = volume
            .get_image_from_axis(index, Orientation::Coronal, Interpolation::Linear, None)
            .await;
    }

    for orientation in [Orientation::Coronal, Orientation::Sagittal] {
        let mut gpu_t = Vec::new();
        let mut cpu_t = Vec::new();
        for _ in 0..n {
            let t = web_time::Instant::now();
            let _ = volume
                .get_image_from_axis(index, orientation, Interpolation::Linear, Some(&gpu))
                .await;
            gpu_t.push(t.elapsed().as_micros());

            let t = web_time::Instant::now();
            let _ = volume
                .get_image_from_axis(index, orientation, Interpolation::Linear, None)
                .await;
            cpu_t.push(t.elapsed().as_micros());
        }
        let name = match orientation {
            Orientation::Coronal => "coronal",
            Orientation::Sagittal => "sagittal",
            Orientation::Axial => "axial",
        };
        println!("{name} gpu min/avg/max us: {:?}", stats(gpu_t));
        println!("{name} cpu min/avg/max us: {:?}", stats(cpu_t));
    }
}
