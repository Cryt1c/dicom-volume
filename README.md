# dicom-volume
A high-level API for handling multiple DICOM files as volumes

## Installation

```sh
cargo add dicom-volume
```

## Usage

Load every DICOM file from a directory, sorted by Instance Number, then extract
the slice at the center of the volume on the Sagittal axis (interpolated on the
CPU) and save it as a PNG:

```rust,no_run
use std::path::PathBuf;

use dicom_volume::{
    enums::{Interpolation, Orientation, SortBy},
    volume_loader::VolumeLoader,
};

#[tokio::main]
async fn main() {
    let volume = VolumeLoader::load_from_directory(PathBuf::from("dicom"), SortBy::InstanceNumber)
        .expect("should have loaded files from directory");
    let image = volume
        .get_image_from_axis(
            volume.dim().2 / 2,
            Orientation::Sagittal,
            Interpolation::Linear,
            None,
        )
        .await
        .expect("should have returned image at center of volume");
    image.save("result.png").unwrap();
}
```

Slice extraction is `async`, so the example needs a runtime such as
[tokio](https://crates.io/crates/tokio).

## GPU acceleration

Coronal and sagittal slices can be interpolated on the GPU through
`GpuInterpolator`, which uses WGPU compute shaders. The volume is uploaded as
an `R16Float` 3D texture, so interpolation runs in hardware on native backends
and WebGPU alike.

```rust,no_run
use dicom_volume::{gpu_interpolator::GpuInterpolator, volume_loader::VolumeLoader};
use std::path::PathBuf;

#[tokio::main]
async fn main() {
    let volume = VolumeLoader::load_from_directory(PathBuf::from("dicom"), SortBy::InstanceNumber)
        .expect("should have loaded files from directory");
    let mut gpu = GpuInterpolator::try_new(&volume.data, volume.spacing)
        .await
        .expect("should have found a GPU adapter");

    // One GPU submission for many slices.
    let indices: Vec<usize> = (0..volume.dim().2).collect();
    let images = volume
        .get_images_from_axis(&indices, dicom_volume::enums::Orientation::Sagittal,
                              dicom_volume::enums::Interpolation::Linear, Some(&mut gpu))
        .await;
}
```

A single slice is dominated by the submit/read-back round-trip, so extracting
several slices at once with `Volume::get_images_from_axis` is considerably
faster than calling `Volume::get_image_from_axis` in a loop.

When no GPU adapter is available, `GpuInterpolator::try_new` returns `None`, so
callers can fall back to the CPU.

## Input assumptions

DICOM files are assumed to have the following attributes:

- Axial data set (only Coronal and Sagittal axes are interpolated)
- No multiframe (always the first frame is used)
- Images from the same series (Series Instance UID) and acquisition
  (Acquisition Number)

## Benchmarks

`cargo bench` compares the CPU and GPU slice-extraction paths. If a `dicom/`
directory exists at the repository root it is used as the benchmark volume;
otherwise a synthetic volume of the same dimensions is generated instead.

To benchmark against your own data:

1. Copy your `.dcm` files into `dicom/` at the repository root
   (the folder is gitignored, so it stays local).
2. Run:

   ```sh
   cargo bench
   ```

Files must meet the [input assumptions](#input-assumptions) and are sorted by
Instance Number. Without a usable GPU adapter only the CPU path is benchmarked.

## Examples

Runnable examples live in `examples/`:

| Example              | Description                                                                     |
| -------------------- | ------------------------------------------------------------------------------- |
| `cold_start.rs`      | Cost of the first GPU extraction in a fresh process versus the warmed-up steady state |
| `break_even.rs`      | Where the GPU path overtakes the CPU path, extracted over a range of slice counts |
| `cold_break_even.rs` | Same measurement including GPU upload and first-call initialization            |
| `inspect.rs`         | Group the DICOM files in `dicom/` by identifying tags                           |

## License

Licensed under [Apache-2.0](http://www.apache.org/licenses/LICENSE-2.0).
