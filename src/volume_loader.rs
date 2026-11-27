use crate::{enums::SortBy, volume::Volume};

use dicom::{
    object::{FileDicomObject, InMemDicomObject, open_file},
    pixeldata::{ConvertOptions, PixelDecoder, VoiLutOption},
};
use dicom_dictionary_std::tags;
use ndarray::{Array2, Array3, s};
use rayon::prelude::*;
use std::{fs, path::Path};
use thiserror::Error;

/// Errors that can occur while loading a volume.
#[derive(Debug, Error)]
pub enum VolumeLoaderError {
    /// No valid DICOM images were found in the input.
    #[error("No valid DICOM images found")]
    NoValidImages,

    /// The loaded images do not all share the same pixel dimensions.
    #[error("Inconsistent image dimensions")]
    InconsistentDimensions,

    /// No Pixel Spacing and Slice Thickness attribute was found.
    #[error("Missing spacing information")]
    MissingSpacing,

    /// Reading a file from disk failed.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Parsing or decoding a DICOM file failed.
    #[error("DICOM error: {0}")]
    Dicom(#[from] dicom::object::ReadError),
}

/// Builds a [`Volume`](crate::volume::Volume) from DICOM objects, file paths,
/// or a directory of `.dcm` files.
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
            .par_iter()
            .filter_map(|dicom_object| Self::extract_image_with_order(dicom_object, &sort_by))
            .collect();

        let skipped = dicom_objects.len() - images_with_order.len();
        if skipped > 0 {
            eprintln!(
                "skipped {skipped} of {} DICOM objects (missing pixel data, unsupported encoding, or missing sort tag)",
                dicom_objects.len()
            );
        }

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
        paths: &[impl AsRef<Path> + Sync],
        sort_by: SortBy,
    ) -> Result<Volume, VolumeLoaderError> {
        let objects: Result<Vec<_>, _> = paths
            .par_iter()
            .map(|path| open_file(path.as_ref()))
            .collect();

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

#[cfg(test)]
mod tests {
    use super::*;
    use dicom::core::DataElement;
    use dicom::core::PrimitiveValue;
    use dicom::core::VR;
    use dicom::object::StandardDataDictionary;
    use dicom::object::mem::InMemElement;
    use dicom::object::meta::FileMetaTableBuilder;
    use dicom_dictionary_std::uids;

    /// A `FileDicomObject` carrying only the given elements (no pixel data).
    fn file_obj(
        elements: Vec<InMemElement<StandardDataDictionary>>,
    ) -> FileDicomObject<InMemDicomObject> {
        InMemDicomObject::from_element_iter(elements).with_exact_meta(
            FileMetaTableBuilder::new()
                .transfer_syntax(uids::EXPLICIT_VR_LITTLE_ENDIAN)
                .build()
                .expect("static meta table should build"),
        )
    }

    fn image(h: usize, w: usize, fill: u16) -> Array2<u16> {
        Array2::from_elem((h, w), fill)
    }

    fn obj_with_instance_number(n: &str) -> FileDicomObject<InMemDicomObject> {
        file_obj(vec![DataElement::new(tags::INSTANCE_NUMBER, VR::IS, n)])
    }

    #[test]
    fn sort_images_orders_ascending() {
        let mut images = vec![
            (Some(3.0), image(1, 1, 3)),
            (Some(1.0), image(1, 1, 1)),
            (Some(2.0), image(1, 1, 2)),
        ];
        VolumeLoader::sort_images(&mut images, SortBy::InstanceNumber);
        let values: Vec<u16> = images.iter().map(|(_, img)| img[[0, 0]]).collect();
        assert_eq!(values, [1, 2, 3]);
    }

    #[test]
    fn sort_images_reverses_image_position_patient() {
        let mut images = vec![
            (Some(3.0), image(1, 1, 3)),
            (Some(1.0), image(1, 1, 1)),
            (Some(2.0), image(1, 1, 2)),
        ];
        VolumeLoader::sort_images(&mut images, SortBy::ImagePositionPatient);
        let values: Vec<u16> = images.iter().map(|(_, img)| img[[0, 0]]).collect();
        assert_eq!(values, [3, 2, 1]);
    }

    #[test]
    fn sort_images_keeps_order_for_none() {
        let mut images = vec![
            (Some(3.0), image(1, 1, 3)),
            (Some(1.0), image(1, 1, 1)),
            (Some(2.0), image(1, 1, 2)),
        ];
        VolumeLoader::sort_images(&mut images, SortBy::None);
        let values: Vec<u16> = images.iter().map(|(_, img)| img[[0, 0]]).collect();
        assert_eq!(values, [3, 1, 2]);
    }

    #[test]
    fn get_sort_order_reads_instance_number() {
        let obj = obj_with_instance_number("7");
        assert_eq!(
            VolumeLoader::get_sort_order(&obj, &SortBy::InstanceNumber),
            Some(Some(7.0))
        );
    }

    #[test]
    fn get_sort_order_reads_z_of_image_position_patient() {
        let obj = file_obj(vec![DataElement::new(
            tags::IMAGE_POSITION_PATIENT,
            VR::DS,
            PrimitiveValue::Strs(vec!["1.0".to_owned(), "2.0".to_owned(), "3.5".to_owned()].into()),
        )]);
        assert_eq!(
            VolumeLoader::get_sort_order(&obj, &SortBy::ImagePositionPatient),
            Some(Some(3.5))
        );
    }

    #[test]
    fn get_sort_order_reads_table_position() {
        let obj = file_obj(vec![DataElement::new(tags::TABLE_POSITION, VR::DS, "42.0")]);
        assert_eq!(
            VolumeLoader::get_sort_order(&obj, &SortBy::TablePosition),
            Some(Some(42.0))
        );
    }

    #[test]
    fn get_sort_order_is_zero_for_none() {
        let obj = file_obj(vec![]);
        assert_eq!(
            VolumeLoader::get_sort_order(&obj, &SortBy::None),
            Some(Some(0.0))
        );
    }

    #[test]
    fn get_sort_order_is_none_without_tag() {
        let obj = file_obj(vec![]);
        assert_eq!(
            VolumeLoader::get_sort_order(&obj, &SortBy::InstanceNumber),
            None
        );
    }

    #[test]
    fn validate_dimensions_accepts_uniform() {
        let images = vec![image(2, 3, 0), image(2, 3, 1), image(2, 3, 2)];
        assert!(VolumeLoader::validate_dimensions(&images).is_ok());
    }

    #[test]
    fn validate_dimensions_rejects_inconsistent() {
        let images = vec![image(2, 3, 0), image(2, 2, 1)];
        assert!(VolumeLoader::validate_dimensions(&images).is_err());
    }

    #[test]
    fn get_spacing_reads_pixel_spacing_and_slice_thickness() {
        let obj = file_obj(vec![
            DataElement::new(
                tags::PIXEL_SPACING,
                VR::DS,
                PrimitiveValue::Strs(vec!["0.5".to_owned(), "0.75".to_owned()].into()),
            ),
            DataElement::new(tags::SLICE_THICKNESS, VR::DS, "1.25"),
        ]);
        assert_eq!(VolumeLoader::get_spacing(&[obj]), Some((0.5, 0.75, 1.25)));
    }

    #[test]
    fn get_spacing_is_none_without_tags() {
        let obj = file_obj(vec![]);
        assert_eq!(VolumeLoader::get_spacing(&[obj]), None);
    }

    #[test]
    fn load_errors_when_no_object_has_pixel_data() {
        // Objects with sort tags but no decodable pixel data are skipped;
        // if all of them are skipped the load fails.
        let objects = vec![obj_with_instance_number("1"), obj_with_instance_number("2")];
        let result = VolumeLoader::load_from_dicom_objects(&objects, SortBy::InstanceNumber);
        assert!(matches!(result, Err(VolumeLoaderError::NoValidImages)));
    }

    #[test]
    fn load_from_file_paths_errors_on_missing_file() {
        let result = VolumeLoader::load_from_file_paths(
            &[std::path::Path::new("/nonexistent/image-000001.dcm")],
            SortBy::InstanceNumber,
        );
        assert!(matches!(result, Err(VolumeLoaderError::Dicom(_))));
    }
}
