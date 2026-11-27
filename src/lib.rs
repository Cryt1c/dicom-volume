//! # DICOM-volume library
#![warn(missing_docs)]
//!
//! This crate serves a high-level API for handling multiple DICOM files as
//! volumes

//!
//! This library is part of the dicom-rs ecosystem and leverages its component
//! to provide a volume representation of multiple DICOM files.
//! Volumes can either be loaded from multiple [`FileDicomObject<InMemDicomObject>`] or from a
//! specified folder where each ".dcm" file is read from.
//! The DICOM files are loaded in parallel using rayon. The volume can be
//! sliced in the three different medical axes:
//!  - Axial
//!  - Coronal
//!  - Sagittal
//!
//!  Library consumers can chose whether the Coronal and Sagittal slices
//!  should be interpolated to preserve the aspect ratios between of the
//!  images. DICOM files are assumed to have the following attributes:
//!   - Axial data set (Only Coronal and Sagittal axes are interpolated)
//!   - No multiframe (always the first frame is used)
//!   - Images from the same series (Series Instance UID) and acquisition
//!     (Acquisition Number)
//!
//!   Contributions are highly welcome!
//!
//! # Roadmap
//!
//!  - Trilinear interpolation
//!  - Cubic interpolation
//!  - Caching of images
//!
//! # GPU acceleration
//!
//! Coronal and sagittal slices can be interpolated on the GPU through
//! [`GpuInterpolator`], which uses WGPU compute shaders. A single slice is
//! dominated by the submit/read-back round-trip, so extracting several slices
//! at once with [`Volume::get_images_from_axis`] is considerably faster than
//! calling [`Volume::get_image_from_axis`] in a loop.
//!
//! The volume is stored as an `R16Float` 3D texture: the raw `u16` voxels are
//! normalized to `[0, 1]` and converted to `f16` before upload, and the sampler
//! interpolates them in hardware. This format is supported by native backends
//! and by WebGPU, so the same path works in the browser. When no adapter is
//! available, [`GpuInterpolator::try_new`] returns `None` so callers can fall
//! back to the CPU.
//!
//! [`GpuInterpolator`]: gpu_interpolator::GpuInterpolator
//! [`GpuInterpolator::try_new`]: gpu_interpolator::GpuInterpolator::try_new
//! [`Volume::get_images_from_axis`]: volume::Volume::get_images_from_axis
//! [`Volume::get_image_from_axis`]: volume::Volume::get_image_from_axis
//!
//! # Examples
//!
//! ## Reading multiple DICOM files into a volume
//!
//! To read all DICOM files from the dicom/ directory, sort them by
//! InstanceNumber. Then get the image at the center of the volume in the
//! Sagittal axis, interpolated on the CPU.
//!
//! ```no_run
//! # use dicom_volume::{
//! #     enums::{Interpolation, Orientation, SortBy},
//! #     volume_loader::VolumeLoader,
//! # };
//! # use std::path::PathBuf;
//! # #[tokio::main(flavor = "current_thread")]
//! # async fn main() {
//! let volume = VolumeLoader::load_from_directory(PathBuf::from("dicom"), SortBy::InstanceNumber)
//!     .expect("should have loaded files from directory");
//! let image = volume
//!     .get_image_from_axis(
//!         volume.dim().2 / 2,
//!         Orientation::Sagittal,
//!         Interpolation::Linear,
//!         None,
//!     )
//!     .await
//!     .expect("should have returned image at center of volume");
//! image.save("result.png").unwrap();
//! # }
//! ```
//!
//! ## Interpolating on the GPU
//!
//! ```no_run
//! # use dicom_volume::{
//! #     enums::{Interpolation, Orientation, SortBy},
//! #     gpu_interpolator::GpuInterpolator,
//! #     volume_loader::VolumeLoader,
//! # };
//! # use std::path::PathBuf;
//! # #[tokio::main(flavor = "current_thread")]
//! # async fn main() {
//! let volume = VolumeLoader::load_from_directory(PathBuf::from("dicom"), SortBy::InstanceNumber)
//!     .expect("should have loaded files from directory");
//! let mut gpu = GpuInterpolator::try_new(&volume.data, volume.spacing)
//!     .await
//!     .expect("should have found a GPU adapter");
//!
//! // One GPU submission for many slices.
//! let indices: Vec<usize> = (0..volume.dim().2).collect();
//! let images = volume
//!     .get_images_from_axis(&indices, Orientation::Sagittal, Interpolation::Linear, Some(&mut gpu))
//!     .await;
//! # }
//! ```
//!
//! [`FileDicomObject<InMemDicomObject>`]: https://docs.rs/dicom-object/latest/dicom_object/struct.FileDicomObject.html

/// Enumerations used throughout the crate.
pub mod enums;
/// GPU-accelerated slice interpolation through WGPU compute shaders.
pub mod gpu_interpolator;
mod interpolator;
/// The [`Volume`](volume::Volume) type and slice extraction.
pub mod volume;
/// Loading DICOM files and objects into a volume.
pub mod volume_loader;
