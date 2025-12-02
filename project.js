"use strict";
window.onload = function() {
    // Restore previously chosen object (if any) so a reload keeps the selection
    const sel = document.getElementById('objectSelect');
    const saved = window.localStorage.getItem('selectedObject');
    if (sel && saved) {
        try { sel.value = saved; } catch (e) { /* ignore if option not present */ }
    }
    main();
}
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
        mipLevelCount: 1,
        usage: GPUTextureUsage.TEXTURE_BINDING |
               GPUTextureUsage.COPY_DST |
               GPUTextureUsage.RENDER_ATTACHMENT
    });

    // copy each face into the corresponding array layer
    for (let i = 0; i < imgs.length; ++i) {
        device.queue.copyExternalImageToTexture(
            { source: imgs[i], flipY: true },
            { texture: texture, origin: { x: 0, y: 0, z: i } },
            { width: w, height: h, depthOrArrayLayers: 1 }
        );
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
    const selectedObject = selElem ? selElem.value : 'monkey_with_hat';
    // Map selection to file name. Add new entries here to support more objects.
    const objectMap = {
        'monkey_with_hat': 'monkey_with_hat.obj',
        'Tree': 'Tree.obj',
        'Statue': 'Statue.obj',

    };
    const objFile = objectMap[selectedObject] || (selectedObject + '.obj');
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
        'Statue': { scale: 0.6, yOffset: -0.9 },
    };
    const params = objectParams[selectedObject] || { scale: 1.0, yOffset: -0.6 };
    const M = mult(translate(0.0, params.yOffset, 0.0), scalem(params.scale, params.scale, params.scale));

    // Define view matrix (isometric view)
    const eye = vec3(0, 0, 3);
    const at = vec3(0, 0, 0);
    const up = vec3(0, 1, 0);
    const V = lookAt(eye, at, up);

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
    const eyeModel4 = mult(invM, vec4(eye[0], eye[1], eye[2], 1.0));

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

    const invViewRot = invViewRotation(V);

    // object uniform buffer: mvp, invProj (identity), invView (identity), mode = 0
    const objectUniformBuffer = device.createBuffer({
        size: 256,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    // projF = 1 / tan(fovy/2); aspect = canvas.width/canvas.height
    function computeProjF(deg) { return 1.0 / Math.tan(radians(deg) / 2.0); }
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
        eye[0], eye[1], eye[2], 1.0,// eyePos (world-space)
        0.0, 0.0, 0.0, 0.0         // reflective = false
    ]);
    device.queue.writeBuffer(quadUniformBuffer, 0, quadInit);
    // Debug output removed
    // More diagnostics: compute reconstructed directions for the quad corners
    function normalizeVec(v) {
        const l = Math.hypot(v[0], v[1], v[2]) || 1.0;
        return [v[0]/l, v[1]/l, v[2]/l];
    }
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
    let orbitOn = false;
    let alpha = 0.0;
    const radius = 3.0;
    const angularSpeed = 0.01;



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
    const cubemap = [
        'textures/cm_left.png',   // POSITIVE_X
        'textures/cm_right.png',  // NEGATIVE_X
        'textures/cm_top.png',    // POSITIVE_Y
        'textures/cm_bottom.png', // NEGATIVE_Y
        'textures/cm_back.png',   // POSITIVE_Z
        'textures/cm_front.png'   // NEGATIVE_Z
    ];

    const CubeTexture = await loadCubeTexture(device, cubemap);
    const CubeSampler = device.createSampler({
        addressModeU: 'clamp-to-edge',
        addressModeV: 'clamp-to-edge',
        addressModeW: 'clamp-to-edge',
        magFilter: 'linear',
        minFilter: 'linear',
        mipmapFilter: 'nearest',
    });

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

    // --- Two bind groups: one for background quad, one for object ---
    const layout0 = bindGroupLayout0;
    const quadBindGroup = device.createBindGroup({
        layout: layout0,
        entries: [
            { binding: 0, resource: { buffer: quadUniformBuffer } },
            { binding: 1, resource: CubeSampler },
            { binding: 2, resource: CubeTexture.createView({ dimension: 'cube' }) },
            { binding: 3, resource: NormalSampler },
            { binding: 4, resource: NormalTexture.createView() },
        ]
    });
    const objectBindGroup = device.createBindGroup({
        layout: layout0,
        entries: [
            { binding: 0, resource: { buffer: objectUniformBuffer } },
            { binding: 1, resource: CubeSampler },
            { binding: 2, resource: CubeTexture.createView({ dimension: 'cube' }) },
            { binding: 3, resource: NormalSampler },
            { binding: 4, resource: NormalTexture.createView() },
        ]
    });


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


    // --- Render Pass ---
    // Draw function
    function draw() {
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
     if (orbitOn) {
         alpha += angularSpeed;
         const eye = vec3(radius * Math.sin(alpha), 0, radius * Math.cos(alpha));
         const at = vec3(0, 0, 0);
         const up = vec3(0, 1, 0);
         const Vnew = lookAt(eye, at, up);
         const newMVP = mult(Mst, mult(P, mult(Vnew, M)));
         // update object MVP
        const eyeModelNew = mult(invM, vec4(eye[0], eye[1], eye[2], 1.0));
        const objectUpdate = new Float32Array([
            ...flatten(newMVP),
            ...flatten(identityMat), // invProj = identity for object
            ...flatten(identityMat), // invViewRot = identity for object
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
     }
 
     // --- Button event ---
     document.getElementById("OrbitToggle").onclick = () => {
         orbitOn = !orbitOn;
     };
    // If user changes selection, save it and reload the page to reinitialize with new model
    if (selElem) {
        selElem.onchange = () => {
            try { window.localStorage.setItem('selectedObject', selElem.value); } catch (e) { }
            window.location.reload();
        };
    }
    draw();
    requestAnimationFrame(animate);
    }
animate();
}