//! Checks that the batched GPU path produces the same output as the
//! single-slice GPU path, and that it stays close to the CPU reference.

use dicom_volume::{
    enums::{Interpolation, Orientation},
    gpu_interpolator::GpuInterpolator,
    volume::Volume,
};
use ndarray::Array3;

fn synthetic_volume() -> Volume {
    // Odd dimensions on purpose: the GPU packs four pixels per output word, so
    // rows need padding whenever the width is not a multiple of four.
    let (depth, height, width) = (8, 30, 45);
    let data = Array3::from_shape_fn((depth, height, width), |(z, y, x)| {
        ((z * 7919 + y * 131 + x) as u16).wrapping_mul(271)
    });
    Volume::new(data, (1.0, 1.0, 2.0))
}

fn name(orientation: Orientation) -> &'static str {
    match orientation {
        Orientation::Axial => "axial",
        Orientation::Coronal => "coronal",
        Orientation::Sagittal => "sagittal",
    }
}

#[test]
fn batched_gpu_matches_single_gpu_and_cpu() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("failed to build tokio runtime");
    let volume = synthetic_volume();
    let Some(mut gpu) = runtime.block_on(GpuInterpolator::try_new(&volume.data, volume.spacing))
    else {
        eprintln!("no GPU adapter available, skipping");
        return;
    };

    for orientation in [Orientation::Coronal, Orientation::Sagittal] {
        let indices: Vec<usize> = (0..15).collect();
        let batched = runtime.block_on(volume.get_images_from_axis(
            &indices,
            orientation,
            Interpolation::Linear,
            Some(&mut gpu),
        ));
        assert_eq!(batched.len(), indices.len());

        for (slot, &index) in indices.iter().enumerate() {
            let single = runtime
                .block_on(volume.get_image_from_axis(
                    index,
                    orientation,
                    Interpolation::Linear,
                    Some(&mut gpu),
                ))
                .expect("single GPU slice");
            let cpu = runtime
                .block_on(volume.get_image_from_axis(
                    index,
                    orientation,
                    Interpolation::Linear,
                    None,
                ))
                .expect("CPU slice");
            let batched = batched[slot].as_ref().expect("batched GPU slice");

            assert_eq!(batched.dimensions(), single.dimensions());
            assert_eq!(
                batched.as_raw(),
                single.as_raw(),
                "batched and single GPU results differ for {} index {index}",
                name(orientation)
            );

            let diff: Vec<i32> = batched
                .as_raw()
                .iter()
                .zip(cpu.as_raw())
                .map(|(gpu, cpu)| (*gpu as i32 - *cpu as i32).abs())
                .collect();
            let max = diff.iter().copied().max().unwrap_or(0);
            let mean = diff.iter().map(|d| *d as f64).sum::<f64>() / diff.len() as f64;
            println!(
                "{} index {index}: mean abs diff vs CPU {mean:.2}, max {max}",
                name(orientation)
            );
            assert!(
                mean < 5.0,
                "{} index {index}: mean diff {mean}",
                name(orientation)
            );
            assert!(
                max < 20,
                "{} index {index}: max diff {max}",
                name(orientation)
            );
        }

        // Out-of-range indices map to `None` without disturbing valid ones.
        let mixed = runtime.block_on(volume.get_images_from_axis(
            &[usize::MAX, 3],
            orientation,
            Interpolation::Linear,
            Some(&mut gpu),
        ));
        assert!(mixed[0].is_none());
        assert!(mixed[1].is_some());
    }
}

/// A volume large enough that the texture upload spans several
/// `write_texture` chunks (128 layers per 64 MiB chunk: 300 layers -> 3).
#[test]
fn gpu_upload_spanning_multiple_chunks_matches_cpu() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("failed to build tokio runtime");

    let (depth, height, width) = (300usize, 512usize, 512usize);
    let data = Array3::from_shape_fn((depth, height, width), |(z, y, x)| {
        ((z * 11 + y * 3 + x) % u16::MAX as usize) as u16
    });
    let volume = Volume::new(data, (1.0, 1.0, 1.0));

    let Some(mut gpu) = runtime.block_on(GpuInterpolator::try_new(&volume.data, volume.spacing))
    else {
        eprintln!("no GPU adapter available, skipping");
        return;
    };

    // Coronal slices span all depth rows, so every uploaded chunk is read.
    for index in [0, depth / 2, depth - 1] {
        let gpu_image = runtime
            .block_on(volume.get_image_from_axis(
                index,
                Orientation::Coronal,
                Interpolation::Linear,
                Some(&mut gpu),
            ))
            .expect("GPU slice");
        let cpu_image = runtime
            .block_on(volume.get_image_from_axis(
                index,
                Orientation::Coronal,
                Interpolation::Linear,
                None,
            ))
            .expect("CPU slice");

        assert_eq!(gpu_image.dimensions(), cpu_image.dimensions());
        let diff: Vec<i32> = gpu_image
            .as_raw()
            .iter()
            .zip(cpu_image.as_raw())
            .map(|(gpu, cpu)| (*gpu as i32 - *cpu as i32).abs())
            .collect();
        let max = diff.iter().copied().max().unwrap_or(0);
        let mean = diff.iter().map(|d| *d as f64).sum::<f64>() / diff.len() as f64;
        assert!(mean < 5.0, "coronal index {index}: mean diff {mean}");
        assert!(max < 20, "coronal index {index}: max diff {max}");
    }
}
