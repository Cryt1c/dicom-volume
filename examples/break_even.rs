//! Temporary benchmark: where does the GPU path overtake the CPU path?
//!
//! Extracts N coronal slices with each backend for a range of N, reports the
//! median wall-clock time, and fits `time = a + b*N` to locate the break-even.

use std::path::PathBuf;

use dicom_volume::{
    enums::{Interpolation, Orientation, SortBy},
    gpu_interpolator::GpuInterpolator,
    volume_loader::VolumeLoader,
};

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    values[values.len() / 2]
}

/// Ordinary least squares fit of `y = a + b*x`.
fn fit(xs: &[f64], ys: &[f64]) -> (f64, f64) {
    let n = xs.len() as f64;
    let sx: f64 = xs.iter().sum();
    let sy: f64 = ys.iter().sum();
    let sxx: f64 = xs.iter().map(|x| x * x).sum();
    let sxy: f64 = xs.iter().zip(ys).map(|(x, y)| x * y).sum();
    let b = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    let a = (sy - b * sx) / n;
    (a, b)
}

fn main() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("failed to build tokio runtime");
    let volume = VolumeLoader::load_from_directory(PathBuf::from("dicom"), SortBy::InstanceNumber)
        .expect("should have loaded files from directory");
    let mut gpu = runtime
        .block_on(GpuInterpolator::try_new(&volume.data, volume.spacing))
        .expect("no GPU adapter supporting 16-bit normalized textures");

    let max_index = volume.dim().1;
    println!(
        "volume dim={:?} spacing={:?} coronal slices={max_index}\n",
        volume.dim(),
        volume.spacing
    );

    // Warm up both paths so shader compilation / rayon pool startup do not
    // leak into the first measurement.
    let warmup: Vec<usize> = (0..8).map(|i| i * max_index / 8).collect();
    for _ in 0..3 {
        runtime.block_on(volume.get_images_from_axis(
            &warmup,
            Orientation::Coronal,
            Interpolation::Linear,
            None,
        ));
        runtime.block_on(volume.get_images_from_axis(
            &warmup,
            Orientation::Coronal,
            Interpolation::Linear,
            Some(&mut gpu),
        ));
    }

    let sizes = [
        1usize, 2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 512,
    ];
    let mut rows = Vec::new();
    let mut fit_x = Vec::new();
    let mut fit_cpu = Vec::new();
    let mut fit_gpu = Vec::new();

    println!(
        "{:>5} {:>10} {:>10} {:>10} {:>10} {:>8}",
        "N", "cpu med", "gpu med", "cpu/slice", "gpu/slice", "winner"
    );
    for &n in &sizes {
        let n = n.min(max_index);
        let indices: Vec<usize> = (0..n).map(|i| i * max_index / n).collect();
        let reps = match n {
            0..=4 => 40,
            5..=16 => 20,
            17..=64 => 10,
            65..=256 => 4,
            _ => 3,
        };

        let mut cpu_times = Vec::with_capacity(reps);
        let mut gpu_times = Vec::with_capacity(reps);
        for _ in 0..reps {
            let start = web_time::Instant::now();
            let cpu = runtime.block_on(volume.get_images_from_axis(
                &indices,
                Orientation::Coronal,
                Interpolation::Linear,
                None,
            ));
            cpu_times.push(start.elapsed().as_secs_f64() * 1000.0);
            std::hint::black_box(cpu);

            let start = web_time::Instant::now();
            let gpu_result = runtime.block_on(volume.get_images_from_axis(
                &indices,
                Orientation::Coronal,
                Interpolation::Linear,
                Some(&mut gpu),
            ));
            gpu_times.push(start.elapsed().as_secs_f64() * 1000.0);
            std::hint::black_box(gpu_result);
        }

        let cpu_ms = median(&mut cpu_times);
        let gpu_ms = median(&mut gpu_times);
        let winner = if gpu_ms < cpu_ms { "gpu" } else { "cpu" };
        println!(
            "{n:>5} {cpu_ms:>9.2}ms {gpu_ms:>9.2}ms {:>9.3}ms {:>9.3}ms {winner:>8}",
            cpu_ms / n as f64,
            gpu_ms / n as f64
        );

        if n <= 64 {
            fit_x.push(n as f64);
            fit_cpu.push(cpu_ms);
            fit_gpu.push(gpu_ms);
        }
        rows.push((n, cpu_ms, gpu_ms));
    }

    let (cpu_a, cpu_b) = fit(&fit_x, &fit_cpu);
    let (gpu_a, gpu_b) = fit(&fit_x, &fit_gpu);
    println!(
        "\nfit (N <= 64): cpu = {cpu_a:.3} + {cpu_b:.3}*N ms, gpu = {gpu_a:.3} + {gpu_b:.3}*N ms"
    );
    if gpu_b < cpu_b {
        let break_even = (cpu_a - gpu_a) / (gpu_b - cpu_b);
        println!("estimated break-even: N ~= {break_even:.2} slices");
    } else {
        println!("gpu marginal cost is not below cpu; no break-even");
    }
}
