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

@group(0) @binding(0) var volume_texture: texture_3d<f32>;
@group(0) @binding(1) var volume_sampler: sampler;
@group(0) @binding(2) var<storage, read_write> output_data: array<u32>;
@group(0) @binding(3) var<uniform> uniforms: Uniforms;

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

fn sample_dicom_value(coords: vec3<f32>) -> f32 {
    let sample = textureSampleLevel(volume_texture, volume_sampler, coords, 0.0);
    let low_byte = sample.r * 255.0;
    let high_byte = sample.g * 255.0;
    return (low_byte + high_byte * 256.0) / 65535.0;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    
    if (x >= uniforms.output_width || y >= uniforms.output_height) {
        return;
    }
    
    let uvw = get_uvw_coords(x, y);
    let sampled_value = sample_dicom_value(uvw);
    
    let u8_value = u32(clamp(sampled_value * 255.0, 0.0, 255.0));
    
    output_data[y * uniforms.output_width + x] = u8_value;
}
