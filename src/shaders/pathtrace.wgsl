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
};

@group(0) @binding(0) var<uniform> U: Uniforms;
@group(0) @binding(1) var<storage, read_write> accum: array<vec4f>;

const MAT_LAMBERT: u32 = 0u;
const MAT_METAL: u32 = 1u;
const MAT_DIELECTRIC: u32 = 2u;

struct Sphere {
  center: vec3f,
  radius: f32,
  albedo: vec3f,
  fuzz: f32,
  material: u32,
  ior: f32,
  _pad: vec2f,
};

@group(0) @binding(2) var<storage, read> spheres: array<Sphere>;

// ---------------------------------------------------------------- random
// PCG hash ベースなので状態バッファは不要。seed は毎回呼び出し側で進める。
fn pcg(v: u32) -> u32 {
  let state = v * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

var<private> rngState: u32;

fn randU32() -> u32 {
  rngState = pcg(rngState);
  return rngState;
}

fn rand() -> f32 {
  return f32(randU32()) * (1.0 / 4294967296.0);
}

// ---------------------------------------------------------------- sky
fn skyColor(dir: vec3f) -> vec3f {
  let t = 0.5 * (normalize(dir).y + 1.0);
  return mix(vec3f(1.0, 1.0, 1.0), vec3f(0.5, 0.7, 1.0), t);
}

// ---------------------------------------------------------------- camera
struct Ray {
  origin: vec3f,
  dir: vec3f,
};

fn makeRay(px: f32, py: f32) -> Ray {
  // px, py は [-1, 1] のスクリーン座標 (py は上が +1)
  let d = U.camU * (px * U.tanHalfFov * U.aspect)
        + U.camV * (py * U.tanHalfFov)
        + U.camW;
  let focusPoint = U.camPos + U.focusDist * d;
  var origin = U.camPos;
  if (U.lensRadius > 0.0) {
    // 単位円内の一様サンプル
    let r = U.lensRadius * sqrt(rand());
    let theta = rand() * 6.2831853;
    origin = origin + U.camU * (r * cos(theta)) + U.camV * (r * sin(theta));
  }
  return Ray(origin, normalize(focusPoint - origin));
}

// ---------------------------------------------------------------- sampling
fn randUnitVec3() -> vec3f {
  // 球面上の一様サンプル
  let z = rand() * 2.0 - 1.0;
  let a = rand() * 6.2831853;
  let r = sqrt(max(0.0, 1.0 - z * z));
  return vec3f(r * cos(a), r * sin(a), z);
}

// ---------------------------------------------------------------- intersect
struct Hit {
  t: f32,
  p: vec3f,
  normal: vec3f,
  frontFace: bool,
  index: u32,
};

fn hitScene(ray: Ray, tMin: f32, tMax: f32, hit: ptr<function, Hit>) -> bool {
  var closest = tMax;
  var found = false;
  for (var i = 0u; i < U.sphereCount; i = i + 1u) {
    let s = spheres[i];
    let oc = ray.origin - s.center;
    let halfB = dot(oc, ray.dir);
    let c = dot(oc, oc) - s.radius * s.radius;
    let disc = halfB * halfB - c;
    if (disc < 0.0) {
      continue;
    }
    let sq = sqrt(disc);
    var t = -halfB - sq;
    if (t < tMin || t > closest) {
      t = -halfB + sq;
      if (t < tMin || t > closest) {
        continue;
      }
    }
    closest = t;
    found = true;
    let p = ray.origin + t * ray.dir;
    let outward = (p - s.center) / s.radius;
    let front = dot(ray.dir, outward) < 0.0;
    (*hit).t = t;
    (*hit).p = p;
    (*hit).normal = select(-outward, outward, front);
    (*hit).frontFace = front;
    (*hit).index = i;
  }
  return found;
}

// ---------------------------------------------------------------- material
fn schlick(cosine: f32, refIdx: f32) -> f32 {
  var r0 = (1.0 - refIdx) / (1.0 + refIdx);
  r0 = r0 * r0;
  return r0 + (1.0 - r0) * pow(1.0 - cosine, 5.0);
}

/// 散乱方向とアルベドを返す。false なら吸収 (打ち切り)
fn scatter(
  ray: Ray,
  hit: Hit,
  attenuation: ptr<function, vec3f>,
  scattered: ptr<function, Ray>,
) -> bool {
  let s = spheres[hit.index];

  if (s.material == MAT_LAMBERT) {
    var dir = hit.normal + randUnitVec3();
    if (dot(dir, dir) < 1e-8) {
      dir = hit.normal;
    }
    *attenuation = s.albedo;
    *scattered = Ray(hit.p + hit.normal * 1e-4, normalize(dir));
    return true;
  }

  if (s.material == MAT_METAL) {
    let reflected = reflect(ray.dir, hit.normal);
    let dir = normalize(reflected + s.fuzz * randUnitVec3());
    if (dot(dir, hit.normal) <= 0.0) {
      return false;
    }
    *attenuation = s.albedo;
    *scattered = Ray(hit.p + hit.normal * 1e-4, dir);
    return true;
  }

  // MAT_DIELECTRIC
  let ratio = select(s.ior, 1.0 / s.ior, hit.frontFace);
  let cosTheta = min(dot(-ray.dir, hit.normal), 1.0);
  let sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
  var dir: vec3f;
  if (ratio * sinTheta > 1.0 || schlick(cosTheta, ratio) > rand()) {
    dir = reflect(ray.dir, hit.normal);
  } else {
    dir = refract(ray.dir, hit.normal, ratio);
  }
  *attenuation = s.albedo;
  *scattered = Ray(hit.p + sign(dot(dir, hit.normal)) * hit.normal * 1e-4, normalize(dir));
  return true;
}

// ---------------------------------------------------------------- trace
fn trace(primary: Ray) -> vec3f {
  var ray = primary;
  var throughput = vec3f(1.0);
  var radiance = vec3f(0.0);

  for (var depth = 0u; depth < U.maxBounces; depth = depth + 1u) {
    var hit: Hit;
    if (!hitScene(ray, 1e-3, 1e30, &hit)) {
      radiance = radiance + throughput * skyColor(ray.dir);
      break;
    }
    var attenuation: vec3f;
    var scattered: Ray;
    if (!scatter(ray, hit, &attenuation, &scattered)) {
      break;
    }
    throughput = throughput * attenuation;
    ray = scattered;

    // ロシアンルーレット (4 バウンス目以降)
    if (depth >= 3u) {
      let q = max(throughput.r, max(throughput.g, throughput.b));
      if (rand() > q) {
        break;
      }
      throughput = throughput / max(q, 1e-4);
    }
  }
  return radiance;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= U.width || gid.y >= U.height) {
    return;
  }
  let pixel = gid.y * U.width + gid.x;
  rngState = pcg(pixel * 9781u + U.frameIndex * 6271u + 1u);

  var sum = vec3f(0.0);
  for (var s = 0u; s < U.sppPerFrame; s = s + 1u) {
    let jx = rand();
    let jy = rand();
    let px = (f32(gid.x) + jx) / f32(U.width) * 2.0 - 1.0;
    let py = 1.0 - (f32(gid.y) + jy) / f32(U.height) * 2.0;
    sum = sum + trace(makeRay(px, py));
  }

  if (U.samplesBefore == 0u) {
    accum[pixel] = vec4f(sum, 1.0);
  } else {
    accum[pixel] = accum[pixel] + vec4f(sum, 1.0);
  }
}
