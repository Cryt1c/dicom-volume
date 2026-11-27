struct Uniforms {
    slice_index: u32,
    orientation: u32,
    output_width: u32,
    output_height: u32,
    volume_width: u32,
    volume_height: u32,
    volume_depth: u32,
    // Word offset of this slice inside the shared output buffer.
    output_offset: u32,
    // Output row stride in words; every word holds four pixels.
    output_stride: u32,
    _padding0: u32,
    _padding1: u32,
    _padding2: u32,
}

@group(0) @binding(0) var volume_texture: texture_3d<f32>;
@group(0) @binding(1) var volume_sampler: sampler;
@group(0) @binding(2) var<storage, read_write> output_data: array<u32>;
@group(0) @binding(3) var<uniform> uniforms: Uniforms;

fn sample_dicom_value(coords: vec3<f32>) -> f32 {
    return textureSampleLevel(volume_texture, volume_sampler, coords, 0.0).r;
}

fn get_uvw_coords(x: u32, y: u32) -> vec3<f32> {
    let inv_output_width = 1.0 / f32(uniforms.output_width);
    let inv_output_height = 1.0 / f32(uniforms.output_height);
    
    let fx = (f32(x) + 0.5) * inv_output_width;
    let fy = (f32(y) + 0.5) * inv_output_height;
    let slice_norm = f32(uniforms.slice_index) + 0.5;
    
    switch uniforms.orientation {
        case 0u: {  // Axial
            let w = slice_norm / f32(uniforms.volume_depth);
            return vec3<f32>(fx, fy, w);
        }
        case 1u: {  // Coronal
            let v = slice_norm / f32(uniforms.volume_height);
            return vec3<f32>(fx, v, fy);
        }
        default: {  // Sagittal
            let u = slice_norm / f32(uniforms.volume_width);
            return vec3<f32>(u, fx, fy);
        }
    }
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Each invocation packs four horizontally adjacent pixels into one word.
    let word_x = global_id.x;
    let y = global_id.y;
    let x0 = word_x * 4u;

    if (x0 >= uniforms.output_width || y >= uniforms.output_height) {
        return;
    }

    var packed: u32 = 0u;
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let x = x0 + i;
        if (x < uniforms.output_width) {
            let uvw = get_uvw_coords(x, y);
            let sampled_value = sample_dicom_value(uvw);
            let u8_value = u32(clamp(sampled_value * 255.0, 0.0, 255.0));
            packed = packed | (u8_value << (i * 8u));
        }
    }

    output_data[uniforms.output_offset + y * uniforms.output_stride + word_x] = packed;
}
