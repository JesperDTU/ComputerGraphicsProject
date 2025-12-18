## Computer Graphics Project:
# Combining Diffuse Color and Glossy Reflections with Environment Maps

A small WebGPU-based computer graphics demo using JavaScript and WGSL shaders.

Features
- Real-time rendering using `project.wgsl` shader code.
- Simple object loading and wireframe support via `OBJParser.js` and `OBJParser_wire.js`.
- Environment cubemaps located in the `cubemaps/` folder.

Requirements
- A Chromium-based browser with WebGPU support (Chrome/Edge). If WebGPU is disabled, enable it in browser flags or use a recent Canary build.
- A local HTTP server to serve files (browsers may block `file://` WebGPU resources).

Run (from repository root)
Open a terminal and run a simple HTTP server, then open `project.html` in your browser:

```
python -m http.server 8000
```

Then visit:

```
http://localhost:8000/project.html
```

Project layout (key files)
- `project.html` — demo page
- `project.js` — main app and WebGPU setup
- `project.wgsl` — WGSL shader code
- `OBJParser.js`, `OBJParser_wire.js` — OBJ loading utilities
- `MV.js`, `genmipmap.js` — math / helper utilities
- `cubemaps/` — environment textures (cubemap folders)
- `Objects/` — 3D model assets



Author
- Repository: `ComputerGraphicsProject` 
- Writers: Chelina Emilie Risager & Jesper Berg Lund
                  s214594               s214639    

