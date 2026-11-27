use half::f16;
use ndarray::Array3;
use rayon::prelude::*;
use std::borrow::Cow;
use wgpu::PollType;

use crate::enums::Orientation;

/// Maximum number of slices dispatched per submission.
///
/// Batching amortizes the cost of the GPU round-trip (submit -> wait -> read
/// back), which otherwise dominates the actual interpolation work.
const MAX_BATCH: usize = 64;

/// Workgroup size declared in the compute shader.
const WORKGROUP_SIZE: u32 = 8;

/// Memory-backed initial size for the output/staging buffers.
const INITIAL_BUFFER_BYTES: u64 = 256;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    slice_index: u32,
    orientation: u32,
    output_width: u32,
    output_height: u32,
    volume_width: u32,
    volume_height: u32,
    volume_depth: u32,
    /// Word offset of this slice inside the shared output buffer.
    output_offset: u32,
    /// Output row stride in words (each word holds four pixels).
    output_stride: u32,
    _padding: [u32; 3],
}

impl Uniforms {
    const SIZE: u64 = std::mem::size_of::<Self>() as u64;

    #[allow(clippy::too_many_arguments)]
    fn new(
        slice_index: u32,
        orientation: u32,
        output_width: u32,
        output_height: u32,
        volume_width: u32,
        volume_height: u32,
        volume_depth: u32,
        output_offset: u32,
        output_stride: u32,
    ) -> Self {
        Self {
            slice_index,
            orientation,
            output_width,
            output_height,
            volume_width,
            volume_height,
            volume_depth,
            output_offset,
            output_stride,
            _padding: [0; 3],
        }
    }
}

/// Interpolates slices of a volume on the GPU through a WGPU compute shader.
///
/// Upload the volume once with [`GpuInterpolator::try_new`], then pass the
/// interpolator to [`Volume::get_image_from_axis`](crate::volume::Volume::get_image_from_axis)
/// or [`Volume::get_images_from_axis`](crate::volume::Volume::get_images_from_axis).
pub struct GpuInterpolator {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    /// Kept alive for as long as `volume_view` exists.
    #[allow(dead_code)]
    volume_texture: wgpu::Texture,
    volume_view: wgpu::TextureView,
    sampler: wgpu::Sampler,
    dimensions: (u32, u32, u32), // (depth, height, width)
    #[allow(dead_code)]
    spacing: (f32, f32, f32),
    /// Holds one [`Uniforms`] per slice in the current batch.
    uniform_buffer: wgpu::Buffer,
    /// Alignment required between two dynamically offset uniform bindings.
    uniform_stride: u64,
    output_buffer: wgpu::Buffer,
    staging_buffer: wgpu::Buffer,
    /// Capacity of `output_buffer`/`staging_buffer` in bytes.
    capacity_bytes: u64,
}

/// Maximum bytes a single internal upload staging buffer may use.
const MAX_UPLOAD_CHUNK_BYTES: u64 = 64 * 1024 * 1024;

/// Number of 3D-texture layers to upload per `write_texture` call.
///
/// Backends such as Dawn stage each upload internally in a buffer limited by
/// `max_buffer_size` (256 MiB by default), so uploading a large volume in one
/// call fails with a buffer-size validation error (e.g. a 512x512x568 f16
/// upload needs a ~284 MiB staging buffer). Chunking by texture layers keeps
/// every internal staging buffer small, independent of the adapter's limits.
fn upload_layers_per_chunk(max_buffer_size: u64, f16_bytes_per_layer: u64) -> u32 {
    let max_chunk_bytes = max_buffer_size.min(MAX_UPLOAD_CHUNK_BYTES);
    (max_chunk_bytes / f16_bytes_per_layer).max(1) as u32
}

impl GpuInterpolator {
    /// Creates the interpolator, panicking when no GPU adapter is available.
    ///
    /// # Panics
    ///
    /// Panics if no suitable GPU adapter can be found.
    #[deprecated(
        since = "0.1.4",
        note = "use `try_new` instead; this panics when no GPU adapter is available"
    )]
    pub async fn new(volume_data: &Array3<u16>, spacing: (f32, f32, f32)) -> Self {
        Self::try_new(volume_data, spacing)
            .await
            .expect("Failed to find a GPU adapter")
    }

    /// Like [`GpuInterpolator::new`], but returns `None` when no suitable GPU
    /// adapter is available instead of panicking.
    pub async fn try_new(volume_data: &Array3<u16>, spacing: (f32, f32, f32)) -> Option<Self> {
        let (depth, height, width) = volume_data.dim();
        let (depth, height, width) = (depth as u32, height as u32, width as u32);
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            ..Default::default()
        });

        // Request adapter
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                ..Default::default()
            })
            .await
            .ok()?;

        // Request device and queue, asking for the adapter's full
        // `max_buffer_size` so large internal staging allocations stay valid
        // even when the default limit (256 MiB) is too small.
        let mut limits = wgpu::Limits::default();
        limits.max_buffer_size = adapter.limits().max_buffer_size;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_limits: limits,
                ..Default::default()
            })
            .await
            .ok()?;

        // Create 3D texture
        let texture_size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: depth,
        };

        let volume_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Volume 3D Texture"),
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            // `R16Unorm` is a native-only wgpu feature and unavailable in
            // WebGPU, so store the voxels as normalized `f16` instead. The
            // sampler returns them in [0, 1], matching the CPU path's
            // `value / u16::MAX` normalization.
            format: wgpu::TextureFormat::R16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // Normalize the raw `u16` voxels to [0, 1] and store them as `f16`.
        // An 8-bit output only needs ~8 bits of precision, so the 11-bit
        // mantissa is more than enough.
        let volume_bits: Vec<u16> = volume_data
            .as_slice()
            .expect("Volume must be contiguous")
            .par_iter()
            .map(|&value| f16::from_f32(value as f32 / u16::MAX as f32).to_bits())
            .collect();

        // Upload the volume in depth chunks via `upload_layers_per_chunk`,
        // so no backend-internal staging buffer exceeds `max_buffer_size`.
        let f16_bytes_per_layer = width as u64 * height as u64 * std::mem::size_of::<u16>() as u64;
        let layers_per_chunk =
            upload_layers_per_chunk(device.limits().max_buffer_size, f16_bytes_per_layer);

        for start in (0..depth).step_by(layers_per_chunk as usize) {
            let layers = layers_per_chunk.min(depth - start);
            let layer_texels = width as usize * height as usize;
            // The data slice must cover exactly the uploaded chunk. On the
            // web the slice becomes a view over WASM memory, and Dawn sizes
            // its internal upload staging buffer from the view's byte
            // length; passing the rest of the volume would still allocate
            // one giant staging buffer and trip `max_buffer_size`.
            let begin = start as usize * layer_texels;
            let end = (start + layers) as usize * layer_texels;
            queue.write_texture(
                wgpu::TexelCopyTextureInfoBase {
                    texture: &volume_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        z: start,
                        ..Default::default()
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                bytemuck::cast_slice(&volume_bits[begin..end]),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(width * std::mem::size_of::<u16>() as u32),
                    rows_per_image: Some(height),
                },
                wgpu::Extent3d {
                    depth_or_array_layers: layers,
                    ..texture_size
                },
            );
        }

        let volume_view = volume_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Create sampler with linear filtering for bilinear interpolation
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Volume Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Volume Slice Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "shaders/volume_slice_unorm.wgsl"
            ))),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Volume Slice Bind Group Layout"),
            entries: &[
                // 3D texture
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                // Sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // Output buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Uniforms (one entry per slice, addressed with a dynamic offset)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(Uniforms::SIZE),
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Volume Slice Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Volume Slice Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Allocate every device-side buffer once; they are reused (and grown
        // when necessary) instead of being recreated for every slice.
        let uniform_stride =
            (device.limits().min_uniform_buffer_offset_alignment as u64).max(Uniforms::SIZE);
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Uniform Buffer"),
            size: uniform_stride * MAX_BATCH as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let output_buffer = Self::create_output_buffer(&device, INITIAL_BUFFER_BYTES);
        let staging_buffer = Self::create_staging_buffer(&device, INITIAL_BUFFER_BYTES);
        let bind_group = Self::create_bind_group(
            &device,
            &bind_group_layout,
            &volume_view,
            &sampler,
            &output_buffer,
            &uniform_buffer,
        );

        Some(Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
            bind_group,
            volume_texture,
            volume_view,
            sampler,
            dimensions: (depth, height, width),
            spacing,
            uniform_buffer,
            uniform_stride,
            output_buffer,
            staging_buffer,
            capacity_bytes: INITIAL_BUFFER_BYTES,
        })
    }

    fn create_output_buffer(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    fn create_staging_buffer(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn create_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        volume_view: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
        output_buffer: &wgpu::Buffer,
        uniform_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Volume Slice Bind Group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(volume_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: uniform_buffer,
                        offset: 0,
                        size: wgpu::BufferSize::new(Uniforms::SIZE),
                    }),
                },
            ],
        })
    }

    /// Grow the output/staging buffers (and rebind them) when a batch needs
    /// more room than the current allocation provides.
    fn ensure_capacity(&mut self, bytes: u64) {
        if bytes <= self.capacity_bytes {
            return;
        }

        self.output_buffer = Self::create_output_buffer(&self.device, bytes);
        self.staging_buffer = Self::create_staging_buffer(&self.device, bytes);
        self.bind_group = Self::create_bind_group(
            &self.device,
            &self.bind_group_layout,
            &self.volume_view,
            &self.sampler,
            &self.output_buffer,
            &self.uniform_buffer,
        );
        self.capacity_bytes = bytes;
    }

    /// Extract a single slice.
    pub async fn extract_slice(
        &mut self,
        slice_index: usize,
        orientation: Orientation,
        target_width: u32,
        target_height: u32,
    ) -> Vec<u8> {
        self.extract_slices(&[slice_index], orientation, target_width, target_height)
            .await
            .pop()
            .expect("one slice was requested")
    }

    /// Extract several slices of the same orientation with a single GPU
    /// submission and a single read-back.
    ///
    /// Batching is what makes the GPU path worthwhile: a single slice is
    /// dominated by the submit/wait/read-back round-trip, while a batch only
    /// pays that cost once for all slices.
    pub async fn extract_slices(
        &mut self,
        slice_indices: &[usize],
        orientation: Orientation,
        target_width: u32,
        target_height: u32,
    ) -> Vec<Vec<u8>> {
        let mut result = Vec::with_capacity(slice_indices.len());
        for chunk in slice_indices.chunks(MAX_BATCH) {
            result.extend(
                self.extract_chunk(chunk, orientation, target_width, target_height)
                    .await,
            );
        }
        result
    }

    async fn extract_chunk(
        &mut self,
        slice_indices: &[usize],
        orientation: Orientation,
        target_width: u32,
        target_height: u32,
    ) -> Vec<Vec<u8>> {
        debug_assert!(!slice_indices.is_empty());

        // The shader packs four pixels into every output word, so the host only
        // has to read one byte per pixel back over the (slow) device-to-host
        // path instead of one u32.
        let words_per_row = target_width.div_ceil(4);
        let slice_words = words_per_row as u64 * target_height as u64;
        let slice_bytes = slice_words * std::mem::size_of::<u32>() as u64;
        let total_bytes = slice_bytes * slice_indices.len() as u64;
        self.ensure_capacity(total_bytes);

        for (i, &slice_index) in slice_indices.iter().enumerate() {
            let uniforms = Uniforms::new(
                slice_index as u32,
                orientation as u32,
                target_width,
                target_height,
                self.dimensions.2,
                self.dimensions.1,
                self.dimensions.0,
                (i as u64 * slice_words) as u32,
                words_per_row,
            );
            self.queue.write_buffer(
                &self.uniform_buffer,
                i as u64 * self.uniform_stride,
                bytemuck::bytes_of(&uniforms),
            );
        }

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Volume Slice Encoder"),
            });
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Volume Slice Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.pipeline);
            let dispatch_x = words_per_row.div_ceil(WORKGROUP_SIZE);
            let dispatch_y = target_height.div_ceil(WORKGROUP_SIZE);
            for i in 0..slice_indices.len() {
                let offset = i as u64 * self.uniform_stride;
                compute_pass.set_bind_group(0, &self.bind_group, &[offset as u32]);
                compute_pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
            }
        }
        encoder.copy_buffer_to_buffer(&self.output_buffer, 0, &self.staging_buffer, 0, total_bytes);

        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = self.staging_buffer.slice(..total_bytes);
        let (sender, receiver) = futures::channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        let _ = self.device.poll(PollType::Wait {
            submission_index: None,
            timeout: None,
        });

        receiver
            .await
            .expect("map_async callback was dropped")
            .expect("failed to map staging buffer");
        let data = buffer_slice.get_mapped_range();
        let stride_bytes = words_per_row as usize * std::mem::size_of::<u32>();
        let row_bytes = target_width as usize;
        let result = data
            .chunks_exact(slice_bytes as usize)
            .map(|slice| {
                if stride_bytes == row_bytes {
                    slice[..row_bytes * target_height as usize].to_vec()
                } else {
                    // Rows are padded to whole words; strip the padding.
                    let mut image = Vec::with_capacity(row_bytes * target_height as usize);
                    for row in slice.chunks_exact(stride_bytes) {
                        image.extend_from_slice(&row[..row_bytes]);
                    }
                    image
                }
            })
            .collect();
        drop(data);
        self.staging_buffer.unmap();

        result
    }
}

#[cfg(test)]
mod tests {
    use super::upload_layers_per_chunk;

    const MIB: u64 = 1024 * 1024;

    #[test]
    fn many_layers_fit_per_chunk() {
        // A 512x512 f16 layer is 512 KiB; 64 MiB / 512 KiB = 128 layers.
        assert_eq!(upload_layers_per_chunk(256 * MIB, 512 * 512 * 2), 128);
    }

    #[test]
    fn chunk_size_is_capped_by_upload_limit_not_device_limit() {
        // An adapter with a 1 GiB max_buffer_size still chunks at 64 MiB.
        assert_eq!(upload_layers_per_chunk(1024 * MIB, 512 * 512 * 2), 128);
    }

    #[test]
    fn huge_single_layer_falls_back_to_one_layer() {
        // A layer larger than the chunk cap still uploads one at a time.
        assert_eq!(upload_layers_per_chunk(256 * MIB, 128 * MIB), 1);
    }

    #[test]
    fn tiny_layer_bulks_up() {
        // An 8x16 f16 layer is 256 bytes -> as many as fit in 64 MiB.
        assert_eq!(
            upload_layers_per_chunk(256 * MIB, 8 * 16 * 2),
            ((64 * MIB) / 256) as u32
        );
    }
}
