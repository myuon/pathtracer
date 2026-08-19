struct Uniforms {
  camPos: vec3f,
  lensRadius: f32,
  camU: vec3f,
  focusDist: f32,
  camV: vec3f,
  tanHalfFov: f32,
  camW: vec3f,
  aspect: f32,
  width: u32,
  height: u32,
  frameIndex: u32,
  sppPerFrame: u32,
  maxBounces: u32,
  sphereCount: u32,
  samplesBefore: u32,
  samplesAfter: u32,
  quadCount: u32,
  env: u32,
  lightCount: u32,
  nee: u32,
  mis: u32,
};

@group(0) @binding(0) var<uniform> U: Uniforms;
@group(0) @binding(1) var<storage, read> accum: array<vec4f>;

struct VsOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
};

@vertex
fn vsMain(@builtin(vertex_index) vi: u32) -> VsOut {
  // フルスクリーン三角形
  var p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  let xy = p[vi];
  var out: VsOut;
  out.pos = vec4f(xy, 0.0, 1.0);
  out.uv = vec2f(xy.x * 0.5 + 0.5, 0.5 - xy.y * 0.5);
  return out;
}

// Narkowicz の ACES 近似
fn acesFilm(x: vec3f) -> vec3f {
  let a = 2.51;
  let b = 0.03;
  let c = 2.43;
  let d = 0.59;
  let e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3f(0.0), vec3f(1.0));
}

@fragment
fn fsMain(in: VsOut) -> @location(0) vec4f {
  let x = min(u32(in.uv.x * f32(U.width)), U.width - 1u);
  let y = min(u32(in.uv.y * f32(U.height)), U.height - 1u);
  let raw = accum[y * U.width + x].rgb;
  let n = max(f32(U.samplesAfter), 1.0);
  let color = acesFilm(raw / n);
  return vec4f(pow(color, vec3f(1.0 / 2.2)), 1.0);
}
