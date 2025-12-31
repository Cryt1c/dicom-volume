use ndarray::Array3;
use std::borrow::Cow;
use wgpu::{PollType, util::DeviceExt};

use crate::{enums::Orientation, volume::WGPU};

pub struct GpuInterpolator {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    volume_texture: wgpu::Texture,
    volume_view: wgpu::TextureView,
    sampler: wgpu::Sampler,
    dimensions: (u32, u32, u32), // (depth, height, width)
    spacing: (f32, f32, f32),
}

impl GpuInterpolator {
    pub async fn new(volume_data: &Array3<u16>, spacing: (f32, f32, f32), wgpu: WGPU) -> Self {
        let (depth, height, width) = volume_data.dim();
        let (depth, height, width) = (depth as u32, height as u32, width as u32);
        let WGPU { device, queue } = wgpu;

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
            format: wgpu::TextureFormat::Rg8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // Upload volume data
        let data_slice = volume_data.as_slice().expect("Volume must be contiguous");
        queue.write_texture(
            wgpu::TexelCopyTextureInfoBase {
                texture: &volume_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&data_slice),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(2 * width * std::mem::size_of::<u8>() as u32),
                rows_per_image: Some(height),
            },
            texture_size,
        );

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
                "shaders/volume_slice.wgsl"
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
                        sample_type: wgpu::TextureSampleType::Float { filterable: true }, // Changed from Uint
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
                // Uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
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

        Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
            volume_texture,
            volume_view,
            sampler,
            dimensions: (depth, height, width),
            spacing,
        }
    }

    pub async fn extract_slice(
        &self,
        slice_index: usize,
        orientation: Orientation,
        target_width: u32,
        target_height: u32,
    ) -> Vec<u8> {
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
            _padding: u32,
        }
        let uniforms = Uniforms {
            slice_index: slice_index as u32,
            orientation: orientation as u32,
            output_width: target_width,
            output_height: target_height,
            volume_width: self.dimensions.2,
            volume_height: self.dimensions.1,
            volume_depth: self.dimensions.0,
            _padding: 0,
        };
        let uniform_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Uniform Buffer"),
                contents: bytemuck::bytes_of(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let output_size = (target_width * target_height) as usize;
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size: (output_size * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: (output_size * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Volume Slice Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.volume_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });
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
            compute_pass.set_bind_group(0, &bind_group, &[]);
            let workgroup_size = 8;
            let dispatch_x = (target_width + workgroup_size - 1) / workgroup_size;
            let dispatch_y = (target_height + workgroup_size - 1) / workgroup_size;
            compute_pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        }
        encoder.copy_buffer_to_buffer(
            &output_buffer,
            0,
            &staging_buffer,
            0,
            (output_size * std::mem::size_of::<u32>()) as u64,
        );
        self.queue.submit(Some(encoder.finish()));
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures::channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        let _ = self.device.poll(PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        receiver.await.unwrap().unwrap();
        let data = buffer_slice.get_mapped_range();
        let u32_data: &[u32] = bytemuck::cast_slice(&data);
        let result: Vec<u8> = u32_data.iter().map(|&v| v as u8).collect();

        drop(data);
        staging_buffer.unmap();
        result
    }
}
