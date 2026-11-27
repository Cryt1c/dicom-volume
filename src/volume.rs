use crate::enums::Interpolation;
use crate::enums::Orientation;
use crate::gpu_interpolator::GpuInterpolator;
use crate::interpolator::Interpolator;

use image::ImageBuffer;
use image::Luma;
use ndarray::Array3;
use ndarray::ArrayView2;
use ndarray::s;
use rayon::prelude::*;

/// A voxel volume with its spacing information.
///
/// The data is stored as a `(depth, height, width)` `Array3<u16>`.
#[derive(Default)]
pub struct Volume {
    /// Voxel data as a `(depth, height, width)` array.
    pub data: Array3<u16>,
    /// Voxel spacing in millimetres as `(x, y, z)`.
    pub spacing: (f32, f32, f32),
    /// Dimensions of the volume resampled to isotropic voxel spacing.
    pub interpolated_dim: (u32, u32, u32),
}

impl Volume {
    /// Create a volume from voxel data and voxel spacing.
    pub fn new(data: Array3<u16>, spacing: (f32, f32, f32)) -> Self {
        let original_dim = data.dim();
        Self {
            data,
            spacing,
            interpolated_dim: Interpolator::get_isotropic_dimensions(spacing, original_dim),
        }
    }

    /// Get the dimensions of the volume (depth, height, width)
    pub fn dim(&self) -> (usize, usize, usize) {
        self.data.dim()
    }

    /// Get a reference to the underlying data
    pub fn data(&self) -> &Array3<u16> {
        &self.data
    }

    /// Get a mutable reference to the underlying data
    pub fn data_mut(&mut self) -> &mut Array3<u16> {
        &mut self.data
    }

    #[inline]
    fn normalize_to_u8(value: u16) -> u8 {
        ((value as f32 / 65535.0) * 255.0).clamp(0.0, 255.0) as u8
    }

    /// Get a raw, uninterpolated slice of the volume along the given
    /// orientation.
    ///
    /// Returns `None` when the index is out of bounds.
    pub fn get_slice_from_axis(
        &self,
        index: usize,
        orientation: &Orientation,
    ) -> Option<ArrayView2<'_, u16>> {
        let slice_result = match orientation {
            Orientation::Axial => self.data().slice(s![index, .., ..]),
            Orientation::Coronal => self.data().slice(s![.., index, ..]),
            Orientation::Sagittal => self.data().slice(s![.., .., index]),
        };
        Some(slice_result)
    }

    fn get_output_dimensions(&self, orientation: &Orientation) -> (u32, u32) {
        // Always return (width, height) - standard image convention
        match orientation {
            Orientation::Axial => {
                // Looking down Z-axis: X is width, Y is height
                (self.interpolated_dim.2, self.interpolated_dim.1)
            }
            Orientation::Coronal => {
                // Looking down Y-axis: X is width, Z is height
                (self.interpolated_dim.2, self.interpolated_dim.0)
            }
            Orientation::Sagittal => {
                // Looking down X-axis: Y is width, Z is height
                (self.interpolated_dim.1, self.interpolated_dim.0)
            }
        }
    }

    // Extract slice to image conversion
    fn slice_to_image(slice: &ArrayView2<'_, u16>) -> Option<ImageBuffer<Luma<u8>, Vec<u8>>> {
        let (height, width) = slice.dim();
        let pixel_data: Vec<u8> = slice
            .into_par_iter()
            .map(|&v| Self::normalize_to_u8(v))
            .collect();
        ImageBuffer::from_raw(width as u32, height as u32, pixel_data)
    }

    /// Extract a single slice as an 8-bit grayscale image.
    ///
    /// Axial slices are returned uninterpolated; Coronal and Sagittal slices
    /// are resampled to isotropic dimensions, on the GPU when a
    /// [`GpuInterpolator`] is supplied together with [`Interpolation::Linear`],
    /// and on the CPU otherwise. Returns `None` for out-of-bounds indices.
    pub async fn get_image_from_axis(
        &self,
        index: usize,
        orientation: Orientation,
        interpolation: Interpolation,
        gpu_interpolator: Option<&mut GpuInterpolator>,
    ) -> Option<ImageBuffer<Luma<u8>, Vec<u8>>> {
        if !self.is_valid_index(index, &orientation) {
            return None;
        }

        match (interpolation, orientation, gpu_interpolator) {
            // both None and Linear+Axial use the same CPU slice path
            (Interpolation::None | Interpolation::Linear, Orientation::Axial, _) => {
                let slice = self.get_slice_from_axis(index, &orientation)?;
                Self::slice_to_image(&slice)
            }

            // Linear + GPU available => use GPU
            (Interpolation::Linear, _, Some(gpu_interpolator)) => {
                let (width, height) = self.get_output_dimensions(&orientation);
                let pixel_data = gpu_interpolator
                    .extract_slice(index, orientation, width, height)
                    .await;
                ImageBuffer::from_raw(width, height, pixel_data)
            }

            // Linear + no GPU => fallback to CPU interpolation
            (Interpolation::Linear, _, None) => {
                let (width, height) = self.get_output_dimensions(&orientation);
                let slice = self.get_slice_from_axis(index, &orientation)?;
                let pixel_data = self.interpolate_slice(&slice, width, height);
                ImageBuffer::from_raw(width, height, pixel_data)
            }

            // Unhandled combinations (e.g. a future interpolation mode) yield
            // no image instead of panicking.
            _ => None,
        }
    }

    /// Extract several slices of the same axis at once.
    ///
    /// When a [`GpuInterpolator`] is supplied together with
    /// [`Interpolation::Linear`], all requested slices are uploaded, dispatched
    /// and read back in a single batched submission. This is significantly
    /// faster than calling [`Volume::get_image_from_axis`] in a loop, because
    /// the GPU round-trip is paid once for the whole batch.
    ///
    /// Invalid indices yield `None` in the corresponding output position.
    pub async fn get_images_from_axis(
        &self,
        indices: &[usize],
        orientation: Orientation,
        interpolation: Interpolation,
        mut gpu_interpolator: Option<&mut GpuInterpolator>,
    ) -> Vec<Option<ImageBuffer<Luma<u8>, Vec<u8>>>> {
        let gpu_interpolation = matches!(&interpolation, Interpolation::Linear)
            && !matches!(orientation, Orientation::Axial);

        if let (true, Some(gpu_interpolator)) = (gpu_interpolation, gpu_interpolator.as_mut()) {
            let mut output: Vec<Option<ImageBuffer<Luma<u8>, Vec<u8>>>> =
                (0..indices.len()).map(|_| None).collect();
            let valid: Vec<(usize, usize)> = indices
                .iter()
                .copied()
                .enumerate()
                .filter(|(_, index)| self.is_valid_index(*index, &orientation))
                .collect();

            if !valid.is_empty() {
                let valid_indices: Vec<usize> = valid.iter().map(|(_, index)| *index).collect();
                let (width, height) = self.get_output_dimensions(&orientation);
                let slices = gpu_interpolator
                    .extract_slices(&valid_indices, orientation, width, height)
                    .await;
                for ((position, _), pixel_data) in valid.into_iter().zip(slices) {
                    output[position] = ImageBuffer::from_raw(width, height, pixel_data);
                }
            }

            return output;
        }

        let mut output = Vec::with_capacity(indices.len());
        for &index in indices {
            output.push(
                self.get_image_from_axis(index, orientation, interpolation, None)
                    .await,
            );
        }
        output
    }

    fn interpolate_slice(&self, slice: &ArrayView2<'_, u16>, width: u32, height: u32) -> Vec<u8> {
        let (slice_height, slice_width) = slice.dim();

        let pixel_data: Vec<u8> = (0..height)
            .into_par_iter()
            .flat_map(|y| {
                (0..width)
                    .map(|x| {
                        // Match GPU: use normalized coordinates with half-pixel offset
                        let norm_x = (x as f32 + 0.5) / width as f32;
                        let norm_y = (y as f32 + 0.5) / height as f32;

                        // Convert back to source coordinates
                        let src_x = norm_x * slice_width as f32 - 0.5;
                        let src_y = norm_y * slice_height as f32 - 0.5;

                        // Clamp to valid range
                        let src_x = src_x.max(0.0).min((slice_width - 1) as f32);
                        let src_y = src_y.max(0.0).min((slice_height - 1) as f32);

                        let value = Interpolator::bilinear_interpolate(slice, src_y, src_x);
                        Self::normalize_to_u8(value)
                    })
                    .collect::<Vec<u8>>()
            })
            .collect();

        pixel_data
    }

    fn is_valid_index(&self, index: usize, orientation: &Orientation) -> bool {
        let dim = self.data.dim();
        let max_index = match orientation {
            Orientation::Axial => dim.0,
            Orientation::Coronal => dim.1,
            Orientation::Sagittal => dim.2,
        };
        index < max_index
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn volume() -> Volume {
        // (depth, height, width) = (4, 8, 16) with distinct voxel values.
        let data = Array3::from_shape_fn((4, 8, 16), |(z, y, x)| (z * 128 + y * 16 + x) as u16);
        Volume::new(data, (1.0, 1.0, 2.0))
    }

    #[test]
    fn is_valid_index_per_orientation() {
        let v = volume();
        assert!(v.is_valid_index(3, &Orientation::Axial));
        assert!(!v.is_valid_index(4, &Orientation::Axial));
        assert!(v.is_valid_index(7, &Orientation::Coronal));
        assert!(!v.is_valid_index(8, &Orientation::Coronal));
        assert!(v.is_valid_index(15, &Orientation::Sagittal));
        assert!(!v.is_valid_index(16, &Orientation::Sagittal));
    }

    #[test]
    fn get_slice_from_axis_returns_expected_dims() {
        let v = volume();
        assert_eq!(
            v.get_slice_from_axis(0, &Orientation::Axial).unwrap().dim(),
            (8, 16)
        );
        assert_eq!(
            v.get_slice_from_axis(0, &Orientation::Coronal)
                .unwrap()
                .dim(),
            (4, 16)
        );
        assert_eq!(
            v.get_slice_from_axis(0, &Orientation::Sagittal)
                .unwrap()
                .dim(),
            (4, 8)
        );
    }

    #[test]
    fn get_output_dimensions_uses_interpolated_dim() {
        let v = volume();
        // spacing (1.0, 1.0, 2.0) resamples depth by 2: (8, 8, 16).
        assert_eq!(v.interpolated_dim, (8, 8, 16));
        assert_eq!(v.get_output_dimensions(&Orientation::Axial), (16, 8));
        assert_eq!(v.get_output_dimensions(&Orientation::Coronal), (16, 8));
        assert_eq!(v.get_output_dimensions(&Orientation::Sagittal), (8, 8));
    }

    #[test]
    fn slice_to_image_normalizes_full_range() {
        let data = Array2::from_shape_vec((1, 3), vec![0u16, 32768, 65535]).unwrap();
        let img = Volume::slice_to_image(&data.view()).expect("2-pixel row should convert");
        assert_eq!((img.width(), img.height()), (3, 1));
        assert_eq!(img.get_pixel(0, 0).0[0], 0);
        assert_eq!(img.get_pixel(1, 0).0[0], 127);
        assert_eq!(img.get_pixel(2, 0).0[0], 255);
    }

    #[test]
    fn get_image_from_axis_axial_and_invalid_index() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("failed to build tokio runtime");
        let v = volume();
        runtime.block_on(async {
            let img = v
                .get_image_from_axis(1, Orientation::Axial, Interpolation::None, None)
                .await
                .expect("axial slice should convert");
            // Axial slices are returned at original resolution.
            assert_eq!((img.width(), img.height()), (16, 8));
            assert!(
                v.get_image_from_axis(99, Orientation::Axial, Interpolation::None, None)
                    .await
                    .is_none()
            );
        });
    }

    #[test]
    fn get_image_from_axis_coronal_cpu_fallback() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("failed to build tokio runtime");
        let v = volume();
        runtime.block_on(async {
            // No GPU interpolator: Linear must fall back to CPU bilinear.
            let img = v
                .get_image_from_axis(2, Orientation::Coronal, Interpolation::Linear, None)
                .await
                .expect("coronal slice should convert");
            assert_eq!((img.width(), img.height()), (16, 8));
        });
    }
}
