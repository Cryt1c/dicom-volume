use std::path::PathBuf;

use dicom_volume::{
    enums::{Orientation, SortBy},
    volume_loader::VolumeLoader,
};

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let mut volume =
        VolumeLoader::load_from_directory(&PathBuf::from("dicom"), SortBy::InstanceNumber)
            .expect("should have loaded files from directory");
    let image = volume
        .get_image_from_axis_gpu(volume.dim().2 / 2, Orientation::Coronal)
        .await
        .expect("should have returned image at center of volume");
    image.save("result.png");
}
