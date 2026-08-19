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
@group(0) @binding(1) var<storage, read_write> accum: array<vec4f>;

const MAT_LAMBERT: u32 = 0u;
/// GGX マイクロファセット (導体)。旧 metal と旧 glossy を統合したもの
const MAT_GGX: u32 = 1u;
const MAT_DIELECTRIC: u32 = 2u;
const MAT_EMISSIVE: u32 = 3u;

const ENV_SKY: u32 = 0u;
const ENV_BLACK: u32 = 1u;

const PI: f32 = 3.14159265;
const INV_PI: f32 = 0.31830988;

/// 球と quad で共有するマテリアル (48 バイト)
struct Material {
  /// lambert の反射率 / GGX の垂直入射反射率 F0 / dielectric の減衰色
  albedo: vec3f,
  /// GGX の粗さ (知覚的)。alpha = roughness^2
  roughness: f32,
  emission: vec3f,
  ior: f32,
  kind: u32,
};

struct Sphere {
  center: vec3f,
  radius: f32,
  mat: Material,
};

/// 角 q と 2 辺 u, v が張る平行四辺形
struct Quad {
  q: vec3f,
  _p0: f32,
  u: vec3f,
  _p1: f32,
  v: vec3f,
  _p2: f32,
  mat: Material,
};

@group(0) @binding(2) var<storage, read> spheres: array<Sphere>;
@group(0) @binding(3) var<storage, read> quads: array<Quad>;
/// NEE でサンプルする面光源。quads へのインデックス列
@group(0) @binding(4) var<storage, read> lights: array<u32>;

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
fn envColor(dir: vec3f) -> vec3f {
  if (U.env == ENV_BLACK) {
    return vec3f(0.0);
  }
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
  mat: Material,
  /// NEE でサンプルできる面光源ならその面積、できないなら 0。MIS 重みの計算に使う
  lightArea: f32,
};

/// outward は正規化済みの外向き法線
fn fillHit(hit: ptr<function, Hit>, ray: Ray, t: f32, outward: vec3f, mat: Material, lightArea: f32) {
  let front = dot(ray.dir, outward) < 0.0;
  (*hit).t = t;
  (*hit).p = ray.origin + t * ray.dir;
  (*hit).normal = select(-outward, outward, front);
  (*hit).frontFace = front;
  (*hit).mat = mat;
  (*hit).lightArea = lightArea;
}

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
    fillHit(hit, ray, t, (p - s.center) / s.radius, s.mat, 0.0);
  }

  for (var i = 0u; i < U.quadCount; i = i + 1u) {
    let quad = quads[i];
    let n = cross(quad.u, quad.v);
    let denom = dot(n, ray.dir);
    if (abs(denom) < 1e-8) {
      continue;
    }
    let t = dot(n, quad.q - ray.origin) / denom;
    if (t < tMin || t > closest) {
      continue;
    }
    // 平面上の点を u, v 基底で表したときの係数が両方 [0, 1] なら内側
    let planar = ray.origin + t * ray.dir - quad.q;
    let w = n / dot(n, n);
    let alpha = dot(w, cross(planar, quad.v));
    let beta = dot(w, cross(quad.u, planar));
    if (alpha < 0.0 || alpha > 1.0 || beta < 0.0 || beta > 1.0) {
      continue;
    }
    closest = t;
    found = true;
    // cross(u, v) の長さがそのまま quad の面積になる
    let ln = length(n);
    fillHit(hit, ray, t, n / ln, quad.mat, ln);
  }

  return found;
}

/// 影レイ用。最初の 1 個で打ち切るので hitScene より速い。
/// ガラスも遮蔽物として扱うので、屈折で回り込む光は NEE では拾えない
/// (その経路はスペキュラ連鎖として BSDF サンプリング側が拾う)
fn occluded(origin: vec3f, dir: vec3f, maxT: f32) -> bool {
  for (var i = 0u; i < U.sphereCount; i = i + 1u) {
    let sp = spheres[i];
    let oc = origin - sp.center;
    let halfB = dot(oc, dir);
    let c = dot(oc, oc) - sp.radius * sp.radius;
    let disc = halfB * halfB - c;
    if (disc < 0.0) {
      continue;
    }
    let sq = sqrt(disc);
    let t0 = -halfB - sq;
    let t1 = -halfB + sq;
    if ((t0 > 1e-4 && t0 < maxT) || (t1 > 1e-4 && t1 < maxT)) {
      return true;
    }
  }
  for (var i = 0u; i < U.quadCount; i = i + 1u) {
    let quad = quads[i];
    let n = cross(quad.u, quad.v);
    let denom = dot(n, dir);
    if (abs(denom) < 1e-8) {
      continue;
    }
    let t = dot(n, quad.q - origin) / denom;
    if (t <= 1e-4 || t >= maxT) {
      continue;
    }
    let planar = origin + t * dir - quad.q;
    let w = n / dot(n, n);
    let alpha = dot(w, cross(planar, quad.v));
    let beta = dot(w, cross(quad.u, planar));
    if (alpha >= 0.0 && alpha <= 1.0 && beta >= 0.0 && beta <= 1.0) {
      return true;
    }
  }
  return false;
}

// ---------------------------------------------------------------- BSDF
/// n を z 軸とする正規直交基底 (Duff et al. の分岐なし版)
fn onb(n: vec3f) -> mat3x3f {
  let sg = select(-1.0, 1.0, n.z >= 0.0);
  let a = -1.0 / (sg + n.z);
  let b = n.x * n.y * a;
  return mat3x3f(
    vec3f(1.0 + sg * n.x * n.x * a, sg * b, -sg * n.x),
    vec3f(b, sg + n.y * n.y * a, -n.y),
    n,
  );
}

/// 知覚的な roughness から GGX の alpha へ。完全な鏡面は数値的に扱えないので下限を切る
fn ggxAlpha(roughness: f32) -> f32 {
  return clamp(roughness * roughness, 1e-4, 1.0);
}

/// GGX の法線分布関数
fn ggxD(nDotH: f32, a: f32) -> f32 {
  let a2 = a * a;
  let d = nDotH * nDotH * (a2 - 1.0) + 1.0;
  return a2 / max(PI * d * d, 1e-9);
}

/// Smith の遮蔽・陰影項 (片側)
fn ggxG1(nDotX: f32, a: f32) -> f32 {
  if (nDotX <= 0.0) {
    return 0.0;
  }
  let a2 = a * a;
  return 2.0 * nDotX / (nDotX + sqrt(a2 + (1.0 - a2) * nDotX * nDotX));
}

fn fresnelSchlick(cosTheta: f32, f0: vec3f) -> vec3f {
  return f0 + (vec3f(1.0) - f0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

/// Heitz 2018 の可視法線 (VNDF) サンプリング。ve は法線を +z とするローカル座標系
fn sampleGgxVndf(ve: vec3f, a: f32, u1: f32, u2: f32) -> vec3f {
  let vh = normalize(vec3f(a * ve.x, a * ve.y, ve.z));
  let lensq = vh.x * vh.x + vh.y * vh.y;
  var t1 = vec3f(1.0, 0.0, 0.0);
  if (lensq > 0.0) {
    t1 = vec3f(-vh.y, vh.x, 0.0) * inverseSqrt(lensq);
  }
  let t2 = cross(vh, t1);
  let r = sqrt(u1);
  let phi = 2.0 * PI * u2;
  let px = r * cos(phi);
  let pyRaw = r * sin(phi);
  let sc = 0.5 * (1.0 + vh.z);
  let py = (1.0 - sc) * sqrt(max(0.0, 1.0 - px * px)) + sc * pyRaw;
  let nh = px * t1 + py * t2 + sqrt(max(0.0, 1.0 - px * px - py * py)) * vh;
  return normalize(vec3f(a * nh.x, a * nh.y, max(1e-6, nh.z)));
}

/// 光源サンプリングできるのは pdf を評価できるマテリアルだけ。
/// dielectric はデルタ分布なので対象外
fn isDiffuseLike(kind: u32) -> bool {
  return kind == MAT_LAMBERT || kind == MAT_GGX;
}

/// BRDF * cos(theta_i)。デルタ分布のマテリアルでは 0 を返す
fn bsdfEval(hit: Hit, rayDir: vec3f, wi: vec3f) -> vec3f {
  let cosI = dot(hit.normal, wi);
  if (cosI <= 0.0) {
    return vec3f(0.0);
  }
  if (hit.mat.kind == MAT_LAMBERT) {
    return hit.mat.albedo * INV_PI * cosI;
  }
  if (hit.mat.kind == MAT_GGX) {
    let v = -rayDir;
    let nDotV = dot(hit.normal, v);
    if (nDotV <= 0.0) {
      return vec3f(0.0);
    }
    let h = normalize(v + wi);
    let a = ggxAlpha(hit.mat.roughness);
    let g2 = ggxG1(nDotV, a) * ggxG1(cosI, a);
    let f = fresnelSchlick(max(dot(v, h), 0.0), hit.mat.albedo);
    // BRDF * cos(theta_i) = D * G2 * F / (4 * nDotV) (cos は約分されている)
    return f * (ggxD(dot(hit.normal, h), a) * g2 / (4.0 * nDotV));
  }
  return vec3f(0.0);
}

/// bsdfEval と同じ方向に対する立体角 pdf
fn bsdfPdfFor(hit: Hit, rayDir: vec3f, wi: vec3f) -> f32 {
  let cosI = dot(hit.normal, wi);
  if (cosI <= 0.0) {
    return 0.0;
  }
  if (hit.mat.kind == MAT_LAMBERT) {
    return cosI * INV_PI;
  }
  if (hit.mat.kind == MAT_GGX) {
    let v = -rayDir;
    let nDotV = dot(hit.normal, v);
    if (nDotV <= 0.0) {
      return 0.0;
    }
    let h = normalize(v + wi);
    let a = ggxAlpha(hit.mat.roughness);
    // VNDF サンプリングの pdf。D_v / (4 (v.h)) を整理すると G1(v) * D / (4 nDotV)
    return ggxG1(nDotV, a) * ggxD(dot(hit.normal, h), a) / (4.0 * nDotV);
  }
  return 0.0;
}

// ---------------------------------------------------------------- NEE
/// Veach の power heuristic (beta = 2)。
/// balance heuristic (pA / (pA + pB)) より重みが優れた戦略へ鋭く寄るので、
/// 片方の戦略が明らかに良い領域で「劣る側に重みを配ってしまう」損が小さい
fn misWeight(pA: f32, pB: f32) -> f32 {
  let a = pA * pA;
  let b = pB * pB;
  return a / (a + b);
}

/// 面光源を 1 つ一様に選んで 1 点サンプルし、放射照度を返す。
/// BRDF (拡散なら albedo / PI) は呼び出し側で掛ける
fn sampleDirectLight(hit: Hit, rayDir: vec3f) -> vec3f {
  let pick = min(u32(rand() * f32(U.lightCount)), U.lightCount - 1u);
  let light = quads[lights[pick]];

  // 光源上の一様サンプル
  let onLight = light.q + light.u * rand() + light.v * rand();
  let toLight = onLight - hit.p;
  let dist2 = dot(toLight, toLight);
  let dist = sqrt(dist2);
  let wi = toLight / dist;

  let fcos = bsdfEval(hit, rayDir, wi);
  if (all(fcos <= vec3f(0.0))) {
    return vec3f(0.0);
  }
  let ln = cross(light.u, light.v);
  let area = length(ln);
  let cosLight = abs(dot(ln / area, wi));
  if (cosLight <= 1e-6) {
    return vec3f(0.0);
  }
  if (occluded(hit.p + hit.normal * 1e-4, wi, dist - 1e-3)) {
    return vec3f(0.0);
  }

  // 面積についての pdf 1 / (lightCount * area) を立体角に変換したもの
  let pL = dist2 / (cosLight * area * f32(U.lightCount));

  // MIS。BSDF サンプリングでも作りやすい方向ほど寄与を下げる
  var weight = 1.0;
  if (U.mis != 0u) {
    weight = misWeight(pL, bsdfPdfFor(hit, rayDir, wi));
  }
  return light.mat.emission * fcos * weight / pL;
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
  let m = hit.mat;

  if (m.kind == MAT_EMISSIVE) {
    // 放射は trace 側で足しているので、ここでは打ち切るだけ
    return false;
  }

  if (m.kind == MAT_LAMBERT) {
    var dir = hit.normal + randUnitVec3();
    if (dot(dir, dir) < 1e-8) {
      dir = hit.normal;
    }
    *attenuation = m.albedo;
    *scattered = Ray(hit.p + hit.normal * 1e-4, normalize(dir));
    return true;
  }

  if (m.kind == MAT_GGX) {
    let a = ggxAlpha(m.roughness);
    let basis = onb(hit.normal);
    let v = -ray.dir;
    // 法線を +z とするローカル座標へ移してから可視法線をサンプルする
    let vl = vec3f(dot(v, basis[0]), dot(v, basis[1]), dot(v, basis[2]));
    if (vl.z <= 0.0) {
      return false;
    }
    let h = basis * sampleGgxVndf(vl, a, rand(), rand());
    let dir = reflect(ray.dir, h);
    let cosI = dot(hit.normal, dir);
    if (cosI <= 0.0) {
      // 面の裏に潜ったサンプルは捨てる
      return false;
    }
    // f * cos / pdf を整理すると F * G2 / G1(v) = F * G1(l)
    *attenuation = fresnelSchlick(max(dot(v, h), 0.0), m.albedo) * ggxG1(cosI, a);
    *scattered = Ray(hit.p + hit.normal * 1e-4, dir);
    return true;
  }

  // MAT_DIELECTRIC
  let ratio = select(m.ior, 1.0 / m.ior, hit.frontFace);
  let cosTheta = min(dot(-ray.dir, hit.normal), 1.0);
  let sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
  var dir: vec3f;
  if (ratio * sinTheta > 1.0 || schlick(cosTheta, ratio) > rand()) {
    dir = reflect(ray.dir, hit.normal);
  } else {
    dir = refract(ray.dir, hit.normal, ratio);
  }
  *attenuation = m.albedo;
  *scattered = Ray(hit.p + sign(dot(dir, hit.normal)) * hit.normal * 1e-4, normalize(dir));
  return true;
}

// ---------------------------------------------------------------- trace
fn trace(primary: Ray) -> vec3f {
  var ray = primary;
  var throughput = vec3f(1.0);
  var radiance = vec3f(0.0);
  // 直前の頂点で BSDF サンプリングした方向の立体角 pdf。
  // 負ならカメラレイかスペキュラ反射で、光源サンプリングでは作れない方向なので重みは 1
  var bsdfPdf = -1.0;
  let useNee = U.nee != 0u && U.lightCount > 0u;

  for (var depth = 0u; depth < U.maxBounces; depth = depth + 1u) {
    var hit: Hit;
    if (!hitScene(ray, 1e-3, 1e30, &hit)) {
      // 環境光は NEE の対象外なので常に足す
      radiance = radiance + throughput * envColor(ray.dir);
      break;
    }

    // 光源に当たったときの放射。NEE と重複するぶんを MIS 重みで削る
    var weight = 1.0;
    if (useNee && bsdfPdf > 0.0 && hit.lightArea > 0.0) {
      if (U.mis != 0u) {
        // この方向を光源サンプリングで作る場合の pdf。sampleDirectLight と同じ式
        let cosLight = max(abs(dot(hit.normal, ray.dir)), 1e-6);
        let pL = hit.t * hit.t / (cosLight * hit.lightArea * f32(U.lightCount));
        weight = misWeight(bsdfPdf, pL);
      } else {
        // MIS なしなら NEE 側に完全に任せる (二重計上の防止)
        weight = 0.0;
      }
    }
    radiance = radiance + throughput * hit.mat.emission * weight;

    // pdf を評価できる面だけ光源を直接サンプルする。デルタ面は BSDF サンプリングに任せる
    if (useNee && isDiffuseLike(hit.mat.kind)) {
      radiance = radiance + throughput * sampleDirectLight(hit, ray.dir);
    }

    var attenuation: vec3f;
    var scattered: Ray;
    if (!scatter(ray, hit, &attenuation, &scattered)) {
      break;
    }
    // 次の頂点で MIS 重みを計算するために pdf を持ち回る。
    // デルタ分布のマテリアルは光源サンプリングで作れない方向なので負にしておく
    if (isDiffuseLike(hit.mat.kind)) {
      bsdfPdf = max(bsdfPdfFor(hit, ray.dir, scattered.dir), 1e-5);
    } else {
      bsdfPdf = -1.0;
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
