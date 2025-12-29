use crate::enums::Interpolation;
use crate::enums::Orientation;
use crate::interpolator::Interpolator;

use image::ImageBuffer;
use image::Luma;
use ndarray::Array3;
use ndarray::ArrayView2;
use ndarray::s;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;

#[derive(Default)]
pub struct Volume {
    pub data: Array3<f32>,
    pub spacing: (f32, f32, f32),
    pub interpolated_dim: (u32, u32, u32),
}

impl Volume {
    pub fn new(data: Array3<f32>, spacing: (f32, f32, f32)) -> Self {
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
    pub fn data(&self) -> &Array3<f32> {
        &self.data
    }

    /// Get a mutable reference to the underlying data
    pub fn data_mut(&mut self) -> &mut Array3<f32> {
        &mut self.data
    }

    #[inline]
    fn normalize_to_u8(value: f32) -> u8 {
        ((value / 65535.0) * 255.0).clamp(0.0, 255.0) as u8
    }

    pub fn get_slice_from_axis(
        &self,
        index: usize,
        orientation: &Orientation,
    ) -> Option<ArrayView2<'_, f32>> {
        let dim = self.data.dim();
        let max_index = match orientation {
            Orientation::Axial => dim.0 - 1,
            Orientation::Coronal => dim.1 - 1,
            Orientation::Sagittal => dim.2 - 1,
        };

        if index > max_index {
            return None;
        }

        let slice_result = match orientation {
            Orientation::Axial => self.data().slice(s![index, .., ..]),
            Orientation::Coronal => self.data().slice(s![.., index, ..]),
            Orientation::Sagittal => self.data().slice(s![.., .., index]),
        };
        Some(slice_result)
    }

    fn get_plane_spacing(&self, orientation: &Orientation) -> (u32, u32) {
        match orientation {
            Orientation::Axial => (self.interpolated_dim.1, self.interpolated_dim.2), // (height, width)
            Orientation::Coronal => (self.interpolated_dim.1, self.interpolated_dim.0), // (height, depth)
            Orientation::Sagittal => (self.interpolated_dim.2, self.interpolated_dim.0), // (width, depth)
        }
    }
    // Extract slice to image conversion
    fn slice_to_image(slice: &ArrayView2<'_, f32>) -> Option<ImageBuffer<Luma<u8>, Vec<u8>>> {
        let (height, width) = slice.dim();
        let pixel_data: Vec<u8> = slice
            .into_par_iter()
            .map(|&v| Self::normalize_to_u8(v))
            .collect();
        ImageBuffer::from_raw(width as u32, height as u32, pixel_data)
    }

    // Simplified get_image_from_axis
    pub fn get_image_from_axis(
        &self,
        index: usize,
        orientation: Orientation,
        interpolation: Interpolation,
    ) -> Option<ImageBuffer<Luma<u8>, Vec<u8>>> {
        let slice = self.get_slice_from_axis(index, &orientation)?;

        match interpolation {
            Interpolation::None => Self::slice_to_image(&slice),
            Interpolation::Bilinear(_) => {
                // Axial doesn't need interpolation (already isotropic in-plane)
                if matches!(orientation, Orientation::Axial) {
                    return Self::slice_to_image(&slice);
                }

                let (target_width, target_height) = self.get_plane_spacing(&orientation);
                self.interpolate_slice(&slice, target_width, target_height)
            }
        }
    }

    fn interpolate_slice(
        &self,
        slice: &ArrayView2<'_, f32>,
        target_width: u32,
        target_height: u32,
    ) -> Option<ImageBuffer<Luma<u8>, Vec<u8>>> {
        let (height, width) = slice.dim();
        let scale_x = (width - 1) as f32 / (target_width - 1).max(1) as f32;
        let scale_y = (height - 1) as f32 / (target_height - 1).max(1) as f32;

        let pixel_data: Vec<u8> = (0..target_height)
            .into_par_iter()
            .flat_map(|row| {
                (0..target_width)
                    .map(|col| {
                        let src_y = row as f32 * scale_y;
                        let src_x = col as f32 * scale_x;
                        let value = Interpolator::bilinear_interpolate(slice, src_y, src_x);
                        Self::normalize_to_u8(value)
                    })
                    .collect::<Vec<u8>>()
            })
            .collect();

        ImageBuffer::from_raw(target_width, target_height, pixel_data)
    }
}
