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
use wgpu::Device;
use wgpu::Queue;

#[derive(Default)]
pub struct Volume {
    pub data: Array3<u16>,
    pub spacing: (f32, f32, f32),
    pub interpolated_dim: (u32, u32, u32),
    pub gpu_interpolator: Option<GpuInterpolator>,
}

pub struct WGPU {
    pub device: Device,
    pub queue: Queue,
}

impl Volume {
    pub fn new(data: Array3<u16>, spacing: (f32, f32, f32)) -> Self {
        let original_dim = data.dim();
        Self {
            data,
            spacing,
            interpolated_dim: Interpolator::get_isotropic_dimensions(spacing, original_dim),
            gpu_interpolator: None,
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

    // Simplified get_image_from_axis
    pub async fn get_image_from_axis(
        &self,
        index: usize,
        orientation: Orientation,
        interpolation: Interpolation,
        wgpu: Option<WGPU>,
    ) -> Option<ImageBuffer<Luma<u8>, Vec<u8>>> {
        if !self.is_valid_index(index, &orientation) {
            return None;
        }
        let slice = self.get_slice_from_axis(index, &orientation)?;

        match interpolation {
            Interpolation::None => Self::slice_to_image(&slice),
            Interpolation::Linear => {
                // Axial doesn't need interpolation (already isotropic in-plane)
                if matches!(orientation, Orientation::Axial) {
                    return Self::slice_to_image(&slice);
                }

                match wgpu {
                    Some(wgpu) => {
                        let gpu_interpolator =
                            GpuInterpolator::new(&self.data, self.spacing, wgpu).await;
                        let (width, height) = self.get_output_dimensions(&orientation);
                        let pixel_data = gpu_interpolator
                            .extract_slice(index, orientation, width, height)
                            .await;

                        ImageBuffer::from_raw(width, height, pixel_data)
                    }
                    None => {
                        let (width, height) = self.get_output_dimensions(&orientation);
                        self.interpolate_slice(&slice, width, height)
                    }
                }
            }
        }
    }

    fn interpolate_slice(
        &self,
        slice: &ArrayView2<'_, u16>,
        width: u32,
        height: u32,
    ) -> Option<ImageBuffer<Luma<u8>, Vec<u8>>> {
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

        ImageBuffer::from_raw(width, height, pixel_data)
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
