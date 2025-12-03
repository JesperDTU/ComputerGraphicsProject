"use strict";

// Restore UI state (reads localStorage and updates controls). Run on DOM ready.
function restoreUIState() {
    // Restore previously chosen object (if any) so a reload keeps the selection
    const sel = document.getElementById('objectSelect');
    const saved = window.localStorage.getItem('selectedObject');
    if (sel && saved) {
        try { sel.value = saved; } catch (e) { /* ignore if option not present */ }
    }
    // Restore other UI state so switching objects preserves user choices
    try {
        const savedColor = window.localStorage.getItem('diffuseColor');
        if (savedColor) {
            const ci = document.getElementById('diffuseColor');
            if (ci) ci.value = savedColor;
        }
        const savedEnv = window.localStorage.getItem('Environment');
        if (savedEnv) {
            const env = document.getElementById('Environment');
            if (env) env.value = savedEnv;
        }
        const savedBlur = window.localStorage.getItem('blurLevel');
        if (savedBlur) {
            const bs = document.getElementById('blurSlider');
            if (bs) bs.value = savedBlur;
        }
        const savedOrbit = window.localStorage.getItem('orbitSpeed');
        if (savedOrbit) {
            const ss = document.getElementById('orbitSpeedSlider');
            if (ss) ss.value = savedOrbit;
        }
    } catch (e) { }
}

// Small math / utility helpers used by multiple places
function invViewRotation(viewMat) {
    const r00 = viewMat[0][0], r01 = viewMat[0][1], r02 = viewMat[0][2];
    const r10 = viewMat[1][0], r11 = viewMat[1][1], r12 = viewMat[1][2];
    const r20 = viewMat[2][0], r21 = viewMat[2][1], r22 = viewMat[2][2];
    return mat4(
        r00, r10, r20, 0.0,
        r01, r11, r21, 0.0,
        r02, r12, r22, 0.0,
        0.0, 0.0, 0.0, 1.0
    );
}

function normalizeVec(v) {
    const l = Math.hypot(v[0], v[1], v[2]) || 1.0;
    return [v[0]/l, v[1]/l, v[2]/l];
}

function computeProjF(deg) { return 1.0 / Math.tan(radians(deg) / 2.0); }

// Start when DOM is ready
document.addEventListener('DOMContentLoaded', () => { restoreUIState(); main(); });
// Helper to load an image file into a WebGPU 2D texture
async function loadTexture(device, url) {
    const response = await fetch(url);
    const blob = await response.blob();
    const image = await createImageBitmap(blob, { colorSpaceConversion: "none" });
    const mipLevels = numMipLevels(image.width, image.height);
    const texture = device.createTexture({
        size: [image.width, image.height, 1],
        format: "rgba8unorm",
        mipLevelCount: mipLevels,
        usage: GPUTextureUsage.TEXTURE_BINDING |
                GPUTextureUsage.COPY_DST |
                GPUTextureUsage.RENDER_ATTACHMENT
    });

    device.queue.copyExternalImageToTexture(
        { source: image, flipY: true },
        { texture: texture },
        { width: image.width, height: image.height }
    );
    generateMipmap(device, texture);

    return texture;
}

// Helper to load six images into a cube texture
async function loadCubeTexture(device, urls) {
    // load all images
    const imgs = [];
    for (let url of urls) {
        const resp = await fetch(url);
        const blob = await resp.blob();
        const img = await createImageBitmap(blob, { colorSpaceConversion: "none" });
        imgs.push(img);
    }
    const w = imgs[0].width, h = imgs[0].height;

    const texture = device.createTexture({
        size: [w, h, 6],
        format: "rgba8unorm",
        // Adding RENDER_ATTACHMENT so future render/copy/mipmap ops succeed
        mipLevelCount: numMipLevels(w, h),
        usage: GPUTextureUsage.TEXTURE_BINDING |
               GPUTextureUsage.COPY_DST |
               GPUTextureUsage.RENDER_ATTACHMENT
    });

    // copy each face into the corresponding array layer
    for (let i = 0; i < imgs.length; ++i) {
        device.queue.copyExternalImageToTexture(
            { source: imgs[i], flipY: false },
            { texture: texture, origin: { x: 0, y: 0, z: i } },
            { width: w, height: h, depthOrArrayLayers: 1 }
        );  
    }

    // Generate mipmaps for each cube face so LOD sampling (textureSampleLevel)
    // can access higher, progressively-blurred mip levels.
    // The shared `genmipmap.js` helper only handles 2D textures; generate
    // per-face mipmaps here by rendering each face/mip as a 2D target.
    if (texture.mipLevelCount > 1) {
        // Use the shader module already stored in `project.wgsl` by fetching it
        // (we're inside a top-level helper so we fetch the file here).
        const mipWGSL = await fetch('project.wgsl', { cache: 'reload' }).then(r => r.text());
        const mipmodule = device.createShaderModule({ code: mipWGSL });
        const mipPipeline = device.createRenderPipeline({
            layout: 'auto',
            vertex: { module: mipmodule, entryPoint: 'mip_vs' },
            fragment: { module: mipmodule, entryPoint: 'mip_fs', targets: [{ format: texture.format }] },
            primitive: { topology: 'triangle-strip' },
        });
        const mipSampler = device.createSampler({ minFilter: 'linear' });

        const encoder = device.createCommandEncoder();
        const layers = 6; // cubemap has 6 faces (array layers)
        for (let layer = 0; layer < layers; ++layer) {
            for (let level = 1; level < texture.mipLevelCount; ++level) {
                const srcView = texture.createView({
                    dimension: '2d',
                    baseMipLevel: level - 1,
                    mipLevelCount: 1,
                    baseArrayLayer: layer,
                    arrayLayerCount: 1,
                });
                const dstView = texture.createView({
                    dimension: '2d',
                    baseMipLevel: level,
                    mipLevelCount: 1,
                    baseArrayLayer: layer,
                    arrayLayerCount: 1,
                });
                const bindGroup = device.createBindGroup({
                    layout: mipPipeline.getBindGroupLayout(1),
                    entries: [
                        { binding: 0, resource: mipSampler },
                        { binding: 1, resource: srcView },
                    ],
                });
                const pass = encoder.beginRenderPass({
                    colorAttachments: [{
                        view: dstView,
                        loadOp: 'clear',
                        storeOp: 'store',
                    }]
                });
                pass.setPipeline(mipPipeline);
                pass.setBindGroup(1, bindGroup);
                pass.draw(4);
                pass.end();
            }
        }
        device.queue.submit([encoder.finish()]);
    }

    return texture;
}

async function main()
{
    // --- Initialize WebGPU --- 
    const gpu = navigator.gpu;
    const adapter = await gpu.requestAdapter();
    const device = await adapter.requestDevice();

    // --- Configure canvas context for WebGPU ---
    const canvas = document.getElementById('my-canvas');
    const context = canvas.getContext('webgpu');
    const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
    context.configure({
        device: device,
        format: canvasFormat,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
        alphaMode: "opaque"
    });

    // --- Load WGSL shader code from external file ---
    const wgslfile = "project.wgsl";
    const wgslcode = await fetch(wgslfile, {cache: "reload"}).then(r => r.text());
    const wgsl = device.createShaderModule({ code: wgslcode });




    // --- Load OBJ model ---
    // Read selection from HTML (default to monkey_with_hat)
    const selElem = document.getElementById('objectSelect');
    // If the select exists but its value is empty (possible via saved localStorage),
    // fall back to the default object name so we don't end up with ".obj".
    const selectedObject = (selElem && selElem.value) ? selElem.value : 'monkey_with_hat';
    // Map selection to file name. Add new entries here to support more objects.
    const objectMap = {
        'monkey_with_hat': 'Objects/monkey_with_hat.obj',
        'Tree': 'Objects/Tree.obj',
        'Bunny': 'Objects/bunny.obj',
        'Donut': 'Objects/donut.obj',
        'Sphere': 'Objects/Sphere.obj',
        'Teapot': 'Objects/teapot.obj',
    };
    const objFile = objectMap[selectedObject] || (selectedObject ? (selectedObject + '.obj') : 'monkey_with_hat.obj');
    const mesh = await readOBJFile(objFile, 1.0, false);
    // mesh.vertices and mesh.normals are Float32Array with 4 floats per vertex
    const numVertices = mesh.vertices.length / 4;

    // Build interleaved buffer: vec4 position (x,y,z,w) followed by vec3 normal (x,y,z)
    const VERT_STRIDE_FLOATS = 4 + 3; // 7 floats: pos(vec4) + normal(vec3)
    const vertexData = new Float32Array(numVertices * VERT_STRIDE_FLOATS);
    for (let i = 0; i < numVertices; ++i) {
        const vOff = i * 4;
        const dstOff = i * VERT_STRIDE_FLOATS;
        // position (vec4)
        vertexData[dstOff + 0] = mesh.vertices[vOff + 0];
        vertexData[dstOff + 1] = mesh.vertices[vOff + 1];
        vertexData[dstOff + 2] = mesh.vertices[vOff + 2];
        vertexData[dstOff + 3] = mesh.vertices[vOff + 3];
        // normal (vec3) - mesh.normals has a vec4 per-vertex, use first 3
        vertexData[dstOff + 4] = mesh.normals[vOff + 0];
        vertexData[dstOff + 5] = mesh.normals[vOff + 1];
        vertexData[dstOff + 6] = mesh.normals[vOff + 2];
    }

    const positionBuffer = device.createBuffer({
        size: vertexData.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(positionBuffer, 0, vertexData);

    // Create index buffer
    const indexBuffer = device.createBuffer({
        size: mesh.indices.byteLength,
        usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(indexBuffer, 0, mesh.indices);

    // Describe interleaved position(vec4) + normal(vec3) layout (shaderLocation 0 and 2)
    const positionBufferLayout = {
        arrayStride: VERT_STRIDE_FLOATS * 4, // bytes
        attributes: [
            { shaderLocation: 0, format: 'float32x4', offset: 0 },       // pos
            { shaderLocation: 2, format: 'float32x3', offset: 16 },      // normal (matches @location(2) in WGSL)
        ]
    };


    // MVP matrix setup //
    // Define the Model matrix (manipulations)
    // Choose per-object model transform so objects are centered/visible
    // (scale first, then translate: translate * scale)
    // Per-object visual adjustments (scale/offset). Extend to support more objects.
    const objectParams = {
        'monkey_with_hat': { scale: 0.5, yOffset: -0.3 },
        'Tree': { scale: 1, yOffset: -0.2 },
        'Bunny': { scale: 0.5, yOffset: -0.5 },
        'Donut': { scale: 0.7, yOffset: 0.0 },
        'Sphere': { scale: 0.5, yOffset: 0.0 },
        'Teapot': { scale: 0.25, yOffset: -0.4 },
    };
    const params = objectParams[selectedObject] || { scale: 1.0, yOffset: -0.6 };
    const M = mult(translate(0.0, params.yOffset, 0.0), scalem(params.scale, params.scale, params.scale));

    // Define view matrix (isometric view)
        // Keep the camera position in `currentEye` so switching modes is smooth
        let currentEye = vec3(0, 0, 3);
    const at = vec3(0, 0, 0);
    const up = vec3(0, 1, 0);
        const V = lookAt(currentEye, at, up);

    // Define projection matrix (use actual canvas aspect ratio)
    const aspect = canvas.width / canvas.height;
    const P = perspective(45, aspect, 0.1, 10);  // 45° vertical field of view
    
    // Correction matrix for WebGPU's clip space
        // The correction matrix ensures the OpenGL-style ortho matrices from MV.js produce the correct depth in WebGPU. Without it, the cube might vanish or render weirdly because z-values don’t land in [0,1].
    const Mst = mat4(
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 0.5, 0.5,
        0, 0, 0, 1
    );

    // Combine to get Model-View-Projection matrix
    const MVP = mult(Mst, mult(P, mult(V, M)));

    // inverse model matrix (used to transform eye into model-space for object)
    const invM = inverse(M);
    // eye in model-space (used by object shader path to compute view vector)
    const eyeModel4 = mult(invM, vec4(currentEye[0], currentEye[1], currentEye[2], 1.0));

    // --- Uniform buffers ---
    const identityMat = mat4(
        1,0,0,0,
        0,1,0,0,
        0,0,1,0,
        0,0,0,1
    );

    // compute invProjFull = inv(P) * inv(Mst) to transform clip -> camera
    const invP = inverse(P);
    const invMst = inverse(Mst);
    const invProjFull = mult(invP, invMst);
    // Alternative ordering (try invMst * invP) in case correction ordering differs
    const invProjFullAlt = mult(invMst, invP);

    const invViewRot = invViewRotation(V);
    // object uniform buffer: mvp, invProj (identity), invView (identity), mode = 0
    const objectUniformBuffer = device.createBuffer({
        size: 256,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    // projF = 1 / tan(fovy/2); aspect = canvas.width/canvas.height
    let projF = computeProjF(45.0); // object uses fixed 45° fov
    // Background uses fixed 90° as requested
    let projFQuad = computeProjF(90.0);
    const aspectUniform = aspect;
    const objectInit = new Float32Array([
        ...flatten(MVP),           // mvp
        ...flatten(identityMat),   // invProj = identity for object (not used)
        ...flatten(identityMat),   // invViewRot = identity for object
        projF,                     // projF for object (unused for object path)
        aspectUniform,             // aspect
        0.0,                       // mode
        0.0,                       // padding
        ...flatten(eyeModel4),     // eyePos (model-space)
        1.0, 0.0, 0.0, 0.0         // reflective vec4.x = 1 => reflective
    ]);
    device.queue.writeBuffer(objectUniformBuffer, 0, objectInit);

    // background quad uniform buffer: mvp = identity (vertices are in clip space),
    // invProj = invProjFull, invView = inverse(V) (camera->world), mode = 1
    const quadUniformBuffer = device.createBuffer({
        size: 256,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const quadInit = new Float32Array([
        ...flatten(identityMat),   // mvp for quad
        ...flatten(invProjFull),   // invProj for quad
        ...flatten(invViewRot),    // invViewRot for quad
        projFQuad,                 // projF (from bg slider)
        aspectUniform,             // aspect
        1.0,                       // mode
        0.0,                      // padding
           currentEye[0], currentEye[1], currentEye[2], 1.0,// eyePos (world-space)
        0.0, 0.0, 0.0, 0.0         // reflective = false
    ]);
    device.queue.writeBuffer(quadUniformBuffer, 0, quadInit);
    // More diagnostics: compute reconstructed directions for the quad corners
    const quadClips = [
        vec4(-1.0, -1.0, 0.999, 1.0),
        vec4( 1.0, -1.0, 0.999, 1.0),
        vec4(-1.0,  1.0, 0.999, 1.0),
        vec4( 1.0,  1.0, 0.999, 1.0)
    ];
    for (let i = 0; i < quadClips.length; ++i) {
        const clip = quadClips[i];
        const camH = mult(invProjFull, clip); // homogeneous camera-space (ordering invP*invMst)
        const w = Math.max(Math.abs(camH[3]), 1e-6);
        const cam = [camH[0]/w, camH[1]/w, camH[2]/w];
        const dir4 = mult(invViewRot, vec4(cam[0], cam[1], cam[2], 0.0));
        const dir = normalizeVec([dir4[0], dir4[1], dir4[2]]);
        // try alternative ordering
        const camHalt = mult(invProjFullAlt, clip);
        const walt = Math.max(Math.abs(camHalt[3]), 1e-6);
        const camalt = [camHalt[0]/walt, camHalt[1]/walt, camHalt[2]/walt];
        const dir4alt = mult(invViewRot, vec4(camalt[0], camalt[1], camalt[2], 0.0));
        const diralt = normalizeVec([dir4alt[0], dir4alt[1], dir4alt[2]]);
    }


    // --- Orbiting setup ---
    // Separate angles for object and camera so they don't overwrite each other
    let objAlpha = 0.0; // object rotation angle (radians)
    // Initialize camera angle/radius from current eye position so mode switches start smoothly
    let camAlpha = Math.atan2(currentEye[0], currentEye[2]);
    let radius = Math.hypot(currentEye[0], currentEye[2]);
    let angularSpeed = 0.005;
    // orbit mode: true => camera orbits, false => object orbits
    let orbitCamera = true;



    // Depth buffer
    const depthTexture = device.createTexture({
        size: { width: canvas.width, height: canvas.height },
        format: 'depth24plus',
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });




    
    // --- Create an explicit bind group layout matching the WGSL bindings ---
    const bindGroupLayout0 = device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
            { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
            { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: 'cube' } },
            { binding: 3, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
            { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d' } },
            { binding: 5, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        ]
    });

    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout0] });

    // --- Pipeline (shaders + vertex layout + render state) ---
    const pipeline = device.createRenderPipeline({
        layout: pipelineLayout,
        vertex: { module: wgsl, entryPoint: 'main_vs', buffers: [positionBufferLayout] },
        fragment: { module: wgsl, entryPoint: 'main_fs', targets: [{ format: canvasFormat }] },
        primitive: { topology: 'triangle-list', cullMode: 'back' },
        depthStencil: {
            format: 'depth24plus',
            depthWriteEnabled: true,
            depthCompare: 'less',
        },
    });

    // Background pipeline: same shaders/layout but disable depth writes so
    // a quad at the far plane can be drawn reliably. Use less-equal so
    // fragments at depth==1.0 are accepted.
    const bgPipeline = device.createRenderPipeline({
        layout: pipelineLayout,
        vertex: { module: wgsl, entryPoint: 'main_vs', buffers: [positionBufferLayout] },
        fragment: { module: wgsl, entryPoint: 'main_fs', targets: [{ format: canvasFormat }] },
        primitive: { topology: 'triangle-list', cullMode: 'back' },
        depthStencil: {
            format: 'depth24plus',
            depthWriteEnabled: false,
            depthCompare: 'less-equal',
        },
    });


    // --- Load cubemap and create sampler ---
    // We'll load the cubemap based on the `#Environment` select value.
    let CubeTexture = null;
    let CubeSampler = null;
    let quadBindGroup = null;
    let objectBindGroup = null;

    function getCubemapFolderForEnv(env) {
        // Map the HTML select values to folder names under `cubemaps/`
        const map = {
            'Autumn': 'autumn_cubemap',
            'Brightday': 'brightday2_cubemap',
            'CloudyHills': 'cloudyhills_cubemap',
            'GreenHill': 'greenhill_cubemap',
            'Terrain': 'terrain_cubemap',
        };
        return map[env] || 'autumn_cubemap';
    }

    function makeCubemapURLs(folder) {
        // derive base name from folder (remove trailing '_cubemap')
        const base = folder.replace(/_cubemap$/i, '');
        // Most folders use PNG, but cloudyhills uses JPG. Hardcode by folder.
        const ext = (folder === 'cloudyhills_cubemap') ? 'jpg' : 'png';
        return [
            `cubemaps/${folder}/${base}_posx.${ext}`,
            `cubemaps/${folder}/${base}_negx.${ext}`,
            `cubemaps/${folder}/${base}_posy.${ext}`,
            `cubemaps/${folder}/${base}_negy.${ext}`,
            `cubemaps/${folder}/${base}_posz.${ext}`,
            `cubemaps/${folder}/${base}_negz.${ext}`,
        ];
    }

    // NOTE: `updateEnvironment` relocated later so it can reference NormalTexture/NormalSampler/layout0

    // --- Create a neutral (flat) normal texture so the surface is smooth ---
    // This replaces the normal map to remove bumpiness. The pixel value
    // [128,128,255] corresponds to the (0,0,1) normal in tangent-space.
    const neutralImage = new ImageData(new Uint8ClampedArray([128,128,255,255]), 1, 1);
    const neutralBitmap = await createImageBitmap(neutralImage, { colorSpaceConversion: 'none' });
    const NormalTexture = device.createTexture({
        size: [1, 1, 1],
        format: 'rgba8unorm',
        mipLevelCount: 1,
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT
    });
    device.queue.copyExternalImageToTexture(
        { source: neutralBitmap, flipY: true },
        { texture: NormalTexture },
        { width: 1, height: 1 }
    );
    const NormalSampler = device.createSampler({
        addressModeU: 'repeat',
        addressModeV: 'repeat',
        magFilter: 'linear',
        minFilter: 'linear',
        mipmapFilter: 'linear',
    });

    // --- Diffuse color uniform buffer (controlled by HTML color input) ---
    const diffuseUniformBuffer = device.createBuffer({
        size: 16, // vec4<f32>
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    // initialize to white
    device.queue.writeBuffer(diffuseUniformBuffer, 0, new Float32Array([1.0, 1.0, 1.0, 0.5]));

    // --- Environment update helper (defined after NormalTexture/NormalSampler/diffuse buffer exist)
    async function updateEnvironment(env) {
        const folder = getCubemapFolderForEnv(env);
        const urls = makeCubemapURLs(folder);
        // load the 6 faces into a cube texture
        const tex = await loadCubeTexture(device, urls);
        CubeTexture = tex;
        CubeSampler = device.createSampler({
            addressModeU: 'clamp-to-edge',
            addressModeV: 'clamp-to-edge',
            addressModeW: 'clamp-to-edge',
            magFilter: 'linear',
            minFilter: 'linear',
            mipmapFilter: 'linear',
        });
        // recreate bind groups to point at the new cubemap texture
        quadBindGroup = device.createBindGroup({
            layout: layout0,
            entries: [
                { binding: 0, resource: { buffer: quadUniformBuffer } },
                { binding: 1, resource: CubeSampler },
                { binding: 2, resource: CubeTexture.createView({ dimension: 'cube' }) },
                { binding: 3, resource: NormalSampler },
                { binding: 4, resource: NormalTexture.createView() },
                { binding: 5, resource: { buffer: diffuseUniformBuffer } },
            ]
        });
        objectBindGroup = device.createBindGroup({
            layout: layout0,
            entries: [
                { binding: 0, resource: { buffer: objectUniformBuffer } },
                { binding: 1, resource: CubeSampler },
                { binding: 2, resource: CubeTexture.createView({ dimension: 'cube' }) },
                { binding: 3, resource: NormalSampler },
                { binding: 4, resource: NormalTexture.createView() },
                { binding: 5, resource: { buffer: diffuseUniformBuffer } },
            ]
        });
    }

    // --- Two bind groups: one for background quad, one for object ---
    const layout0 = bindGroupLayout0;
    // Load the initial environment and create bind groups. When the user
    // changes the `#Environment` select, `updateEnvironment` will be called
    // to recreate the bind groups with the new cubemap.
    const envSelect = document.getElementById('Environment');
    const initialEnv = envSelect ? envSelect.value : 'Autumn';
    await updateEnvironment(initialEnv);
    if (envSelect) {
        envSelect.onchange = () => { try { window.localStorage.setItem('Environment', envSelect.value); } catch (e) {} ; updateEnvironment(envSelect.value); };
    }


    // --- Clip-space quad placed near far plane (z = 0.999) ---
    // Build quad buffer with same interleaved layout: pos(vec4) + normal(vec3)
    // Each quad vertex gets a vec4 position followed by a zero normal
    const quadVerts = [
        [-1.0, -1.0,  1.0, 1.0],
        [ 1.0, -1.0,  1.0, 1.0],
        [-1.0,  1.0,  1.0, 1.0],
        [ 1.0, -1.0,  1.0, 1.0],
        [ 1.0,  1.0,  1.0, 1.0],
        [-1.0,  1.0,  1.0, 1.0],
    ];
    const quadPositions = new Float32Array(quadVerts.length * VERT_STRIDE_FLOATS);
    for (let i = 0; i < quadVerts.length; ++i) {
        const base = i * VERT_STRIDE_FLOATS;
        quadPositions[base + 0] = quadVerts[i][0];
        quadPositions[base + 1] = quadVerts[i][1];
        quadPositions[base + 2] = quadVerts[i][2];
        quadPositions[base + 3] = quadVerts[i][3];
        quadPositions[base + 4] = 0.0; // normal x
        quadPositions[base + 5] = 0.0; // normal y
        quadPositions[base + 6] = 0.0; // normal z
    }
    const quadBuffer = device.createBuffer({
        size: quadPositions.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(quadBuffer, 0, quadPositions);


    // --- Initialization-time event wiring (do not attach per-frame) ---
    // Wire up orbit speed slider (live updates while dragging)
    const speedSlider = document.getElementById('orbitSpeedSlider');
    const speedValue = document.getElementById('orbitSpeedValue');
    if (speedSlider) {
        // internal maximum angular speed (0..0.5) - increased so the orbit can run much faster
        const internalMax = 0.5;
        // derive display max from the slider element so HTML can change it freely
        const displayMax = parseFloat(speedSlider.max) || 100;
        // initialize slider to saved display value if present, otherwise use current speed
        let savedDisplay = null;
        try { savedDisplay = window.localStorage.getItem('orbitSpeed'); } catch (e) { savedDisplay = null; }
        if (savedDisplay !== null) {
            speedSlider.value = savedDisplay;
            if (speedValue) speedValue.textContent = savedDisplay;
            angularSpeed = (parseFloat(savedDisplay) / displayMax) * internalMax;
        } else {
            speedSlider.value = ((angularSpeed / internalMax) * displayMax).toFixed(0);
            if (speedValue) speedValue.textContent = ((angularSpeed / internalMax) * displayMax).toFixed(0);
        }
        speedSlider.addEventListener('input', (ev) => {
            const v = parseFloat(ev.target.value); // 0..displayMax
            if (!isNaN(v)) angularSpeed = (v / displayMax) * internalMax; // convert display -> internal
            if (speedValue) speedValue.textContent = v.toFixed(0);
            try { window.localStorage.setItem('orbitSpeed', v.toFixed(0)); } catch (e) { }
        });
    }
    // Orbit mode wiring (Camera vs Object) using two buttons
    const orbitCameraBtn = document.getElementById('orbitCameraBtn');
    const orbitObjectBtn = document.getElementById('orbitObjectBtn');
    try {
        const saved = window.localStorage.getItem('orbitMode');
        if (saved !== null) orbitCamera = (saved === 'camera');
    } catch (e) { }
    function updateOrbitButtons() {
        if (orbitCameraBtn) {
            if (orbitCamera) orbitCameraBtn.classList.add('active'); else orbitCameraBtn.classList.remove('active');
        }
        if (orbitObjectBtn) {
            if (!orbitCamera) orbitObjectBtn.classList.add('active'); else orbitObjectBtn.classList.remove('active');
        }
    }
    if (orbitCameraBtn) {
        orbitCameraBtn.addEventListener('click', () => {
            orbitCamera = true;
            // Recompute camAlpha/radius from the current camera world position so
            // the camera orbit starts from wherever the camera currently is.
            try {
                camAlpha = Math.atan2(currentEye[0], currentEye[2]);
                radius = Math.hypot(currentEye[0], currentEye[2]);
            } catch (e) { }
            // debug removed
            try { window.localStorage.setItem('orbitMode', 'camera'); } catch (e) { }
            updateOrbitButtons();
        });
    }
    if (orbitObjectBtn) {
        orbitObjectBtn.addEventListener('click', () => {
            orbitCamera = false;
            // debug removed
            try { window.localStorage.setItem('orbitMode', 'object'); } catch (e) { }
            updateOrbitButtons();
        });
    }
    // initialize button state
    updateOrbitButtons();
    // If user changes selection, save it and reload the page to reinitialize with new model
    if (selElem) {
        selElem.onchange = () => {
            try { window.localStorage.setItem('selectedObject', selElem.value); } catch (e) { }
            window.location.reload();
        };
    }

    // Diffuse color input wiring: update the small uniform buffer when the user picks a color
    const colorInput = document.getElementById('diffuseColor');
    if (colorInput) {
        function hexToRgbNormalized(hex) {
            const v = (hex[0] === '#') ? hex.slice(1) : hex;
            const r = parseInt(v.slice(0,2), 16) / 255.0;
            const g = parseInt(v.slice(2,4), 16) / 255.0;
            const b = parseInt(v.slice(4,6), 16) / 255.0;
            return [r,g,b];
        }
        // write initial color
        const initial = colorInput.value || '#ffffff';
        const rgbInit = hexToRgbNormalized(initial);
        device.queue.writeBuffer(diffuseUniformBuffer, 0, new Float32Array([rgbInit[0], rgbInit[1], rgbInit[2], 1.0]));
        colorInput.addEventListener('input', (ev) => {
            const hex = ev.target.value;
            const rgb = hexToRgbNormalized(hex);
            device.queue.writeBuffer(diffuseUniformBuffer, 0, new Float32Array([rgb[0], rgb[1], rgb[2], 1.0]));
            try { window.localStorage.setItem('diffuseColor', hex); } catch (e) { }
        });
    }

    // Persist blur slider changes so the value remains when switching objects
    try {
        const blurControl = document.getElementById('blurSlider');
        if (blurControl) {
            blurControl.addEventListener('input', (ev) => {
                try { window.localStorage.setItem('blurLevel', ev.target.value); } catch (e) { }
            });
        }
    } catch (e) { }




    // --- Render Pass ---
    // Draw function
    function draw() {
        // --- UPDATE BLUR LEVEL UNIFORM ---------------------------------
        // read slider
        const blurSlider = document.getElementById("blurSlider");
        const blurValue = parseFloat(blurSlider ? blurSlider.value : 0);

        // write blur to object buffer (offset 240 bytes)
        device.queue.writeBuffer(
            objectUniformBuffer,
            240,
            new Float32Array([blurValue])
        );
        // ---------------------------------------------------------------
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginRenderPass({
            colorAttachments: [{
                view: context.getCurrentTexture().createView(),
                loadOp: 'clear',
                storeOp: 'store',
                clearValue: { r: 0, g: 0, b: 0, a: 1.0 },
            }],
            depthStencilAttachment: {
        view: depthTexture.createView(),
        depthClearValue: 1.0,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
        },
        });
        // Draw background quad using the background pipeline (no depth write)
        pass.setPipeline(bgPipeline);
        pass.setVertexBuffer(0, quadBuffer);
        pass.setBindGroup(0, quadBindGroup);
        pass.draw(6);

        // Draw model with the main pipeline (depth writes enabled)
        pass.setPipeline(pipeline);
        pass.setVertexBuffer(0, positionBuffer);
        pass.setIndexBuffer(indexBuffer, 'uint32');
        pass.setBindGroup(0, objectBindGroup);
        pass.drawIndexed(mesh.indices.length);
         pass.end();
         device.queue.submit([encoder.finish()]);
     }
 
     // animation loop
        function animate() {
        if (angularSpeed !== 0) {
        if (orbitCamera) {
            // Camera orbits around the origin (existing behavior)
            // Use the current camera height so toggling modes preserves vertical position
            camAlpha += angularSpeed;
            const eye = vec3(radius * Math.sin(camAlpha), currentEye[1], radius * Math.cos(camAlpha));
             const at = vec3(0, 0, 0);
             const up = vec3(0, 1, 0);
             const Vnew = lookAt(eye, at, up);
                // compute rotated model from object angle so object keeps its orientation when modes switch
                const objAlphaDeg = objAlpha * 180.0 / Math.PI;
                const rotatedModel = mult(rotateY(objAlphaDeg), M);
                const newMVP = mult(Mst, mult(P, mult(Vnew, rotatedModel)));
                // update object MVP
               const invRotated = inverse(rotatedModel);
               const eyeModelNew = mult(invRotated, vec4(eye[0], eye[1], eye[2], 1.0));
               const objectUpdate = new Float32Array([
                   ...flatten(newMVP),
                   ...flatten(identityMat), // invProj = identity for object
                   ...flatten(rotatedModel), // invViewRot: provide model->world rotation so reflections stay view-fixed
                   projF,
                   aspectUniform,
                   0.0,
                   0.0,
                   ...flatten(eyeModelNew),  // eyePos (model-space)
                   1.0, 0.0, 0.0, 0.0        // reflective = true
               ]);
               device.queue.writeBuffer(objectUniformBuffer, 0, objectUpdate);
            // update quad mtex (camera rotation inverse * invProjFull)
            const invViewRotNew = invViewRotation(Vnew);
            const quadUpdate = new Float32Array([
                ...flatten(identityMat),
                ...flatten(invProjFull),
                ...flatten(invViewRotNew),
                projFQuad,
                aspectUniform,
                1.0,
                0.0,
                    eye[0], eye[1], eye[2], 1.0, // eyePos (world-space)
                0.0, 0.0, 0.0, 0.0           // reflective = false
            ]);
            device.queue.writeBuffer(quadUniformBuffer, 0, quadUpdate);
                // remember the camera position so mode switches can resume smoothly
                currentEye = eye;
         } else {
             // Object orbits around the origin; camera stays fixed
                 const eye = currentEye; // preserve whatever camera position we have
             // advance object rotation only when in object mode
             objAlpha += angularSpeed;
             const objAlphaDeg = objAlpha * 180.0 / Math.PI;
             const rotatedModel = mult(rotateY(objAlphaDeg), M);
             // Recompute view matrix from the current camera position so switching
             // between modes doesn't jump (V may be stale from initialization).
             const Vcur = lookAt(currentEye, at, up);
             const newMVP = mult(Mst, mult(P, mult(Vcur, rotatedModel)));
             // update object MVP using inverse of rotated model
            const invMrot = inverse(rotatedModel);
            const eyeModelNew = mult(invMrot, vec4(eye[0], eye[1], eye[2], 1.0));
            const objectUpdate = new Float32Array([
                ...flatten(newMVP),
                ...flatten(identityMat), // invProj = identity for object
                ...flatten(rotatedModel), // invViewRot: provide model->world rotation so reflections stay view-fixed
                projF,
                aspectUniform,
                0.0,
                0.0,
                ...flatten(eyeModelNew),  // eyePos (model-space)
                1.0, 0.0, 0.0, 0.0        // reflective = true
            ]);
            device.queue.writeBuffer(objectUniformBuffer, 0, objectUpdate);
            // Note: quad (background) uses the fixed camera; no update needed
         }
     }
 
    // end of animate body: draw and schedule next frame
    draw();
    requestAnimationFrame(animate);
    }
animate();
}