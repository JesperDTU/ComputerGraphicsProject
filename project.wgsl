struct Uniforms {
    mvp     : mat4x4<f32>,
    invProj : mat4x4<f32>, 
    invViewRot : mat4x4<f32>,
    projF   : f32,
    aspect  : f32,
    mode    : f32,
    _pad    : f32,
    eyePos  : vec4<f32>,
    reflective : vec4<f32>,
    blurLevel : f32,
};

@group(0) @binding(0) var<uniform> uniforms : Uniforms;
@group(0) @binding(1) var CubeSampler : sampler;
@group(0) @binding(2) var CubeTexture : texture_cube<f32>;
@group(0) @binding(3) var NormalSampler : sampler;
@group(0) @binding(4) var NormalTexture : texture_2d<f32>;
@group(0) @binding(5) var<uniform> diffuseColor : vec4<f32>;

fn rotate_to_normal(n_in: vec3<f32>, v: vec3<f32>) -> vec3<f32> {
    let sgn_nz = sign(n_in.z + 1.0e-16);
    let a = -1.0/(1.0 + abs(n_in.z));
    let b = n_in.x * n_in.y * a;
    let row0 = vec3<f32>(1.0 + n_in.x*n_in.x*a, b, -sgn_nz*n_in.x);
    let row1 = vec3<f32>(sgn_nz*b, sgn_nz*(1.0 + n_in.y*n_in.y*a), -n_in.y);
    return row0 * v.x + row1 * v.y + n_in * v.z;
}

struct VSOut {
    @builtin(position) position : vec4<f32>,
    @location(0) normal : vec3<f32>, 
    @location(1) posModel : vec3<f32>, 
};

@vertex
fn main_vs(@location(0) inPos : vec4<f32>, @location(2) inNormal : vec3<f32>) -> VSOut {
    var out : VSOut;
    if (uniforms.mode < 0.5) {
        out.position = uniforms.mvp * vec4<f32>(inPos.xyz, 1.0);
        out.normal   = inNormal;
        out.posModel = inPos.xyz;
    } else {
        out.position = inPos;
        // Reconstruct camera-space ray from NDC using fov/aspect 
        let ndc = inPos.xy / inPos.w;
        let x_cam = ndc.x * uniforms.aspect / uniforms.projF;
        let y_cam = ndc.y / uniforms.projF;
        let camDir = normalize(vec3<f32>(x_cam, y_cam, -1.0));
        let worldDir4 = uniforms.invViewRot * vec4<f32>(camDir, 0.0);
        out.normal = normalize(worldDir4.xyz);
        out.posModel = vec3<f32>(0.0, 0.0, 0.0);
    }

    return out;
}

@fragment
fn main_fs(@location(0) normal_in : vec3<f32>, @location(1) posModel : vec3<f32>) -> @location(0) vec4<f32> {
    let n = normalize(normal_in);
    var bumped_n : vec3<f32> = n;
    if (uniforms.mode < 0.5) {
        // Spherical inverse mapping to compute UV from the surface normal
        let PI : f32 = 3.14159265359;
        let u : f32 = 0.5 - atan2(n.z, n.x) / (2.0 * PI);
        let v : f32 = 1.0 - (acos(n.y) / PI);
        let uv : vec2<f32> = vec2<f32>(u, v);

        // Sample normal map (tangent-space normal) with mipmap selection, convert from [0,1] to [-1,1]
        let nm_rgb : vec3<f32> = textureSample(NormalTexture, NormalSampler, uv).rgb;
        let nm_ts  : vec3<f32> = nm_rgb * 2.0 - vec3<f32>(1.0, 1.0, 1.0);

        // rotate tangent-space normal into world/model space and normalize
        bumped_n = normalize(rotate_to_normal(n, nm_ts));
    }
  
    var sampleDir : vec3<f32>;
    if (uniforms.mode < 0.5) {
        let v = normalize(uniforms.eyePos.xyz - posModel);
        let world_n = normalize((uniforms.invViewRot * vec4<f32>(bumped_n, 0.0)).xyz);
        let world_v = normalize((uniforms.invViewRot * vec4<f32>(v, 0.0)).xyz);
        let incident = -world_v;
        let r = normalize(reflect(incident, world_n));
        let useReflect = uniforms.reflective.x > 0.5;
        sampleDir = select(world_n, r, useReflect);
    } else {
        let v = normalize(uniforms.eyePos.xyz - posModel);
        let incident = -v;
        let r = normalize(reflect(incident, bumped_n));
        let useReflect = uniforms.reflective.x > 0.5;
        sampleDir = select(bumped_n, r, useReflect);
    }

    // Sample base mip level (LOD 0) to avoid blurry mipmap selection
    let color_rgb : vec3<f32> = textureSampleLevel(CubeTexture, CubeSampler, sampleDir, uniforms.blurLevel).rgb;
    // Apply user-controlled diffuse color only for object path (mode < 0.5)
    var final_rgb : vec3<f32> = color_rgb;
    if (uniforms.mode < 0.5) {
        // Use tint strength 0.8 (0.0 = no tint, 1.0 = full multiply)
        let tint = mix(vec3<f32>(1.0), diffuseColor.rgb, 0.8);
        final_rgb = color_rgb * tint;
    }
    return vec4<f32>(final_rgb, 1.0);
}

@group(1) @binding(0) var mipSampler : sampler;
@group(1) @binding(1) var mipTexture : texture_2d<f32>;

struct MipVSOut {
    @builtin(position) position : vec4<f32>,
    @location(0) texcoord : vec2<f32>,
};

@vertex
fn mip_vs(@builtin(vertex_index) vertexIndex : u32) -> MipVSOut {
    var out : MipVSOut;
    let pos = array<vec2<f32>, 4>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(1.0, 0.0)
    );
    let xy = pos[vertexIndex];
    out.position = vec4<f32>(xy * 2.0 - vec2<f32>(1.0, 1.0), 0.0, 1.0);
    out.texcoord = vec2<f32>(xy.x, 1.0 - xy.y);
    return out;
}

@fragment
fn mip_fs(@location(0) texcoord : vec2<f32>) -> @location(0) vec4<f32> {
    return textureSample(mipTexture, mipSampler, texcoord);
}