//! Compares the GPU and CPU slice-extraction paths.
//!
//! Run with `cargo bench`. If a `dicom/` directory with DICOM files exists it
//! is used as the benchmark volume, otherwise a synthetic volume with the same
//! dimensions is generated. Without a usable GPU adapter only the CPU path is
//! benchmarked.

use std::path::PathBuf;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use dicom_volume::{
    enums::{Interpolation, Orientation, SortBy},
    gpu_interpolator::GpuInterpolator,
    volume::Volume,
    volume_loader::VolumeLoader,
};
use ndarray::Array3;

/// Number of slices per batched extraction.
const BATCH: usize = 24;

fn load_volume() -> (Volume, String) {
    match VolumeLoader::load_from_directory(PathBuf::from("dicom"), SortBy::InstanceNumber) {
        Ok(volume) => (volume, "dicom/".to_string()),
        Err(err) => {
            eprintln!("could not load dicom/ ({err}); using a synthetic volume instead");
            (synthetic_volume(), "synthetic".to_string())
        }
    }
}

fn synthetic_volume() -> Volume {
    let (depth, height, width) = (24, 512, 512);
    let data = Array3::from_shape_fn((depth, height, width), |(z, y, x)| {
        let cx = x as f32 - width as f32 / 2.0;
        let cy = y as f32 - height as f32 / 2.0;
        let r = (cx * cx + cy * cy).sqrt();
        let v = ((r * 0.04).sin() * 0.5 + 0.5) * 0.6 + (z as f32 / depth as f32) * 0.4;
        (v.clamp(0.0, 1.0) * 65535.0) as u16
    });
    Volume::new(data, (0.332, 0.332, 4.0))
}

fn orientation_name(orientation: Orientation) -> &'static str {
    match orientation {
        Orientation::Axial => "axial",
        Orientation::Coronal => "coronal",
        Orientation::Sagittal => "sagittal",
    }
}

fn max_index(volume: &Volume, orientation: Orientation) -> usize {
    let (depth, height, width) = volume.dim();
    match orientation {
        Orientation::Axial => depth,
        Orientation::Coronal => height,
        Orientation::Sagittal => width,
    }
}

fn benchmark_slices(c: &mut Criterion) {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("failed to build tokio runtime");
    let (volume, source) = load_volume();
    let mut gpu = runtime.block_on(GpuInterpolator::try_new(&volume.data, volume.spacing));
    let has_gpu = gpu.is_some();

    println!(
        "benchmark volume: {source} dim={:?} spacing={:?} interpolated_dim={:?} gpu={}",
        volume.dim(),
        volume.spacing,
        volume.interpolated_dim,
        if has_gpu {
            "enabled"
        } else {
            "unavailable (CPU only)"
        }
    );

    for orientation in [Orientation::Coronal, Orientation::Sagittal] {
        let name = orientation_name(orientation);
        let last = max_index(&volume, orientation).max(2) - 1;
        let center = last / 2;
        let indices: Vec<usize> = (0..BATCH).map(|i| i * last / BATCH).collect();

        let mut group = c.benchmark_group(format!("slice/{name}"));
        group.throughput(Throughput::Elements(1));
        group.sample_size(50);
        group.bench_function("cpu", |b| {
            b.iter(|| {
                runtime.block_on(volume.get_image_from_axis(
                    center,
                    orientation,
                    Interpolation::Linear,
                    None,
                ))
            });
        });
        if let Some(gpu) = gpu.as_mut() {
            group.bench_function("gpu", |b| {
                b.iter(|| {
                    runtime.block_on(volume.get_image_from_axis(
                        center,
                        orientation,
                        Interpolation::Linear,
                        Some(&mut *gpu),
                    ))
                });
            });
        }
        group.finish();

        let mut group = c.benchmark_group(format!("batch/{name}"));
        group.throughput(Throughput::Elements(BATCH as u64));
        group.sample_size(20);
        group.measurement_time(Duration::from_secs(5));
        group.bench_with_input(BenchmarkId::new("cpu", BATCH), &indices, |b, indices| {
            b.iter(|| {
                runtime.block_on(volume.get_images_from_axis(
                    indices,
                    orientation,
                    Interpolation::Linear,
                    None,
                ))
            });
        });
        if let Some(gpu) = gpu.as_mut() {
            group.bench_with_input(BenchmarkId::new("gpu", BATCH), &indices, |b, indices| {
                b.iter(|| {
                    runtime.block_on(volume.get_images_from_axis(
                        indices,
                        orientation,
                        Interpolation::Linear,
                        Some(&mut *gpu),
                    ))
                });
            });
        }
        group.finish();
    }
}

criterion_group!(benches, benchmark_slices);
criterion_main!(benches);
