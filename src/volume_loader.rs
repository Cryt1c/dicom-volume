use crate::{enums::SortBy, volume::Volume};

use dicom::{
    object::{FileDicomObject, InMemDicomObject, open_file},
    pixeldata::{ConvertOptions, PixelDecoder, VoiLutOption},
};
use dicom_dictionary_std::tags;
use ndarray::{Array2, Array3, s};
use std::{fs, path::Path};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum VolumeLoaderError {
    #[error("No valid DICOM images found")]
    NoValidImages,

    #[error("Inconsistent image dimensions")]
    InconsistentDimensions,

    #[error("Missing spacing information")]
    MissingSpacing,

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("DICOM error: {0}")]
    Dicom(#[from] dicom::object::ReadError),
}

pub struct VolumeLoader;

impl VolumeLoader {
    /// Load a volume from DICOM objects
    ///
    /// # Arguments
    ///
    /// * `dicom_objects` - Slice of DICOM file objects
    /// * `sort_by` - Method to sort the slices
    ///
    /// # Errors
    ///
    /// Returns error if no valid images found or dimensions are inconsistent
    pub fn load_from_dicom_objects(
        dicom_objects: &[FileDicomObject<InMemDicomObject>],
        sort_by: SortBy,
    ) -> Result<Volume, VolumeLoaderError> {
        let mut images_with_order: Vec<_> = dicom_objects
            .iter()
            .filter_map(|dicom_object| Self::extract_image_with_order(dicom_object, &sort_by))
            .collect();

        if images_with_order.is_empty() {
            return Err(VolumeLoaderError::NoValidImages);
        }

        Self::sort_images(&mut images_with_order, sort_by);

        let images: Vec<_> = images_with_order
            .into_iter()
            .map(|(_, image)| image)
            .collect();

        Self::validate_dimensions(&images)?;

        let volume_array = Self::build_volume_array(&images);
        let spacing = Self::get_spacing(dicom_objects).ok_or(VolumeLoaderError::MissingSpacing)?;

        Ok(Volume::new(volume_array, spacing))
    }

    /// Load a volume from file paths
    pub fn load_from_file_paths(
        paths: &[impl AsRef<Path>],
        sort_by: SortBy,
    ) -> Result<Volume, VolumeLoaderError> {
        let objects: Result<Vec<_>, _> =
            paths.iter().map(|path| open_file(path.as_ref())).collect();

        Self::load_from_dicom_objects(&objects?, sort_by)
    }

    /// Load a volume from a directory containing .dcm files
    pub fn load_from_directory(
        path: impl AsRef<Path>,
        sort_by: SortBy,
    ) -> Result<Volume, VolumeLoaderError> {
        let paths: Vec<_> = fs::read_dir(path.as_ref())?
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| {
                path.extension()
                    .and_then(|s| s.to_str())
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("dcm"))
            })
            .collect();

        if paths.is_empty() {
            return Err(VolumeLoaderError::NoValidImages);
        }

        Self::load_from_file_paths(&paths, sort_by)
    }

    fn extract_image_with_order(
        dicom_object: &FileDicomObject<InMemDicomObject>,
        sort_by: &SortBy,
    ) -> Option<(Option<f32>, Array2<u16>)> {
        let order = Self::get_sort_order(dicom_object, sort_by)?;
        let image_2d = Self::decode_image(dicom_object)?;
        Some((order, image_2d))
    }

    fn get_sort_order(
        dicom_object: &FileDicomObject<InMemDicomObject>,
        sort_by: &SortBy,
    ) -> Option<Option<f32>> {
        match sort_by {
            SortBy::ImagePositionPatient => {
                let pos = dicom_object
                    .element(tags::IMAGE_POSITION_PATIENT)
                    .ok()?
                    .to_multi_float32()
                    .ok()?;
                Some(pos.get(2).copied())
            }
            SortBy::TablePosition => {
                let pos = dicom_object
                    .element(tags::TABLE_POSITION)
                    .ok()?
                    .to_float32()
                    .ok();
                Some(pos)
            }
            SortBy::InstanceNumber => {
                let num = dicom_object
                    .element(tags::INSTANCE_NUMBER)
                    .ok()?
                    .to_int::<i32>()
                    .ok()
                    .map(|n| n as f32);
                Some(num)
            }
            SortBy::None => Some(Some(0.0)),
        }
    }

    fn decode_image(
        dicom_object: &FileDicomObject<InMemDicomObject>,
    ) -> Option<ndarray::ArrayBase<ndarray::OwnedRepr<u16>, ndarray::Dim<[usize; 2]>>> {
        let pixel_data = dicom_object.decode_pixel_data().ok()?;
        let options = ConvertOptions::new().with_voi_lut(VoiLutOption::First);
        pixel_data
            .to_ndarray_with_options::<u16>(&options)
            .ok()
            .map(|arr| arr.slice_move(s![0, .., .., 0]))
    }

    fn sort_images(images_with_order: &mut [(Option<f32>, Array2<u16>)], sort_by: SortBy) {
        if !matches!(sort_by, SortBy::None) {
            images_with_order
                .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        }

        if matches!(sort_by, SortBy::ImagePositionPatient) {
            images_with_order.reverse();
        }
    }

    fn validate_dimensions(
        images: &[ndarray::ArrayBase<ndarray::OwnedRepr<u16>, ndarray::Dim<[usize; 2]>>],
    ) -> Result<(), VolumeLoaderError> {
        let first_dim = images[0].dim();
        if images.iter().any(|img| img.dim() != first_dim) {
            return Err(VolumeLoaderError::InconsistentDimensions);
        }
        Ok(())
    }

    fn build_volume_array(
        images: &[ndarray::ArrayBase<ndarray::OwnedRepr<u16>, ndarray::Dim<[usize; 2]>>],
    ) -> Array3<u16> {
        let (height, width) = images[0].dim();
        let depth = images.len();
        let mut volume = Array3::<u16>::zeros((depth, height, width));

        for (i, image) in images.iter().enumerate() {
            volume.slice_mut(s![i, .., ..]).assign(image);
        }

        volume
    }

    fn get_spacing(dicom_objects: &[FileDicomObject<InMemDicomObject>]) -> Option<(f32, f32, f32)> {
        dicom_objects.iter().find_map(|dicom_object| {
            let pixel_spacing = dicom_object
                .element(tags::PIXEL_SPACING)
                .ok()?
                .to_multi_float32()
                .ok()?;

            let slice_thickness = dicom_object
                .element(tags::SLICE_THICKNESS)
                .ok()?
                .to_float32()
                .ok()?;

            Some((pixel_spacing[0], pixel_spacing[1], slice_thickness))
        })
    }
}
