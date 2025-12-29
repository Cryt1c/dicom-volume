//! # DICOM-volume library
//!
//! This crate serves a high-level API for handling multiple DICOM files as
//! volumes

//!
//! This library is part of the dicom-rs ecosystem and leverages its component
//! to provide a volume representation of multiple DICOM files.
//! Volumes can either be loaded from multiple [`FileDicomObject<InMemDicomObject>`] or from a
//! specified folder where each ".dcm" file is read from.
//! If the environment supports it the DICOM files are loaded in parallel
//! using rayon. The volume can be sliced in the three different medical axes:
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
//!  - GPU processor for interpolation using WGPU and compute shaders
//!  - Trilinear interpolation
//!  - Cubic interpolation
//!  - Caching of images
//!
//! # Examples
//!
//! ## Reading multiple DICOM files into a volume
//!
//! To read all DICOM files from the dicom/ directory, sort them by
//! InstanceNumber. Then get the image at the center of the volume in the
//! Sagittal axis.
//!
//! ```no_run
//! # use dicom_volume::{VolumeLoader, Orientation, Interpolation, Processor, SortBy};
//! # use std::path::PathBuf;
//! let volume = VolumeLoader::load_from_directory(&PathBuf::from("dicom"), SortBy::InstanceNumber)
//!     .expect("should have loaded files from directory");
//! let image = volume
//!     .get_image_from_axis(
//!         volume.dim().2 / 2,
//!         Orientation::Sagittal,
//!         Interpolation::Bilinear(Processor::CPU),
//!     )
//!     .expect("should have returned image at center of volume");
//! image.save("result.png");
//! ```
//!
//! [`FileDicomObject<InMemDicomObject>`]: https://docs.rs/dicom-object/latest/dicom_object/struct.FileDicomObject.html

pub mod enums;
mod interpolator;
pub mod volume;
pub mod volume_loader;
