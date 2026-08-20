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
  /// スクランブル済み Sobol (0,2) 列を使うか
  qmc: u32,
  bvhNodeCount: u32,
  /// 環境マップを光源としてサンプルするか
  envIs: u32,
  envWidth: u32,
  envHeight: u32,
  /// pdf_omega = max(luminance, 1e-6) * envPdfScale
  envPdfScale: f32,
  /// 1 フレーム前のカメラ。累積を新しい視点へ投影し直すのに使う
  prevCamPos: vec3f,
  prevTanHalfFov: f32,
  prevCamU: vec3f,
  prevAspect: f32,
  prevCamV: vec3f,
  reproject: u32,
  prevCamW: vec3f,
  _pad: u32,
  /// 0 なら通常描画、それ以外は中間量を疑似カラーで出す
  debugMode: u32,
  /// 乱数の種をフレーム番号ではなく累積サンプル数から作る (A/B を再現可能にする)
  fixedSeed: u32,
};

@group(0) @binding(0) var<uniform> U: Uniforms;
/// 1 画素あたり 2 要素。[0] = 放射輝度の和と画素ごとのサンプル数、[1] = 1 次交差の世界座標
@group(0) @binding(1) var<storage, read_write> histWrite: array<vec4f>;
/// 前フレームの同じもの (ping-pong)
@group(0) @binding(6) var<storage, read> histRead: array<vec4f>;

/// 再投影で引き継ぐサンプル数の上限。古い情報が居座らないように抑える
const MAX_HISTORY: f32 = 48.0;

const MAT_LAMBERT: u32 = 0u;
/// GGX マイクロファセット (導体)。旧 metal と旧 glossy を統合したもの
const MAT_GGX: u32 = 1u;
const MAT_DIELECTRIC: u32 = 2u;
const MAT_EMISSIVE: u32 = 3u;

const ENV_SKY: u32 = 0u;
const ENV_BLACK: u32 = 1u;
/// 太陽つきの空を焼いた lat-long マップ
const ENV_HDRI: u32 = 2u;

/// CDF の重みが 0 になる方向を作らないための下限。TS 側と一致させること
const ENV_MIN_WEIGHT: f32 = 1e-6;

const PI: f32 = 3.14159265;
/// このバウンス数までは低食い違い列に次元を割り当てる。
/// 深いバウンスまで広げると経路ごとの次元のずれで逆に悪化するため 1 に留める
const QMC_DEPTH: u32 = 1u;
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
/// [面光源の quad インデックス][BVH のプリミティブ参照] を連結した配列。
/// storage buffer の本数に上限があるので 1 本にまとめている
@group(0) @binding(4) var<storage, read> indices: array<u32>;

/// 中央値分割の BVH。左の子は常に自分の直後、右の子は leftFirst にある
struct BvhNode {
  bmin: vec3f,
  /// 葉なら bvhRefs の開始位置、内部ノードなら右の子のノード番号
  leftFirst: u32,
  bmax: vec3f,
  /// 0 なら内部ノード
  count: u32,
};

@group(0) @binding(5) var<storage, read> bvh: array<BvhNode>;
/// 三角形。辺は v0 を原点とする差分で持つ
struct Triangle {
  v0: vec3f,
  _p0: f32,
  e1: vec3f,
  _p1: f32,
  e2: vec3f,
  _p2: f32,
  n0: vec3f,
  _p3: f32,
  n1: vec3f,
  _p4: f32,
  n2: vec3f,
  _p5: f32,
  mat: Material,
};

@group(0) @binding(7) var<storage, read> triangles: array<Triangle>;

/// [テクセル rgba][周辺 CDF][条件付き CDF] を連結した 1 本の配列
@group(0) @binding(8) var<storage, read> envData: array<f32>;

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

// -------------------------------------------------------- 低食い違い列
// 画素ごとに独立にスクランブルした Sobol (0,2) 列。
// 累積サンプル番号で引くので、プログレッシブ描画とそのまま噛み合う。
var<private> sampleIdx: u32;
var<private> pixelSeed: u32;

/// 基数 2 の van der Corput 列 (ビット反転)
fn vanDerCorput(nIn: u32, scramble: u32) -> f32 {
  var n = nIn;
  n = (n << 16u) | (n >> 16u);
  n = ((n & 0x00ff00ffu) << 8u) | ((n & 0xff00ff00u) >> 8u);
  n = ((n & 0x0f0f0f0fu) << 4u) | ((n & 0xf0f0f0f0u) >> 4u);
  n = ((n & 0x33333333u) << 2u) | ((n & 0xccccccccu) >> 2u);
  n = ((n & 0x55555555u) << 1u) | ((n & 0xaaaaaaaau) >> 1u);
  n = n ^ scramble;
  return f32(n) * 2.3283064365386963e-10;
}

/// Sobol 列の第 2 次元
fn sobol2(nIn: u32, scramble: u32) -> f32 {
  var n = nIn;
  var s = scramble;
  var v = 1u << 31u;
  for (var i = 0u; i < 32u; i = i + 1u) {
    if (n == 0u) {
      break;
    }
    if ((n & 1u) != 0u) {
      s = s ^ v;
    }
    v = v ^ (v >> 1u);
    n = n >> 1u;
  }
  return f32(s) * 2.3283064365386963e-10;
}

/// 次元 d に割り当てた 2 次元サンプル。次元ごとに画素依存のスクランブルをかける
fn stratified2d(d: u32) -> vec2f {
  let sc1 = pcg(pixelSeed ^ (d * 0x9e3779b9u));
  let sc2 = pcg(sc1 ^ 0x85ebca6bu);
  return vec2f(vanDerCorput(sampleIdx, sc1), sobol2(sampleIdx, sc2));
}

/// 低食い違い列を割り当てるのは浅いバウンスだけ。深いところは白色雑音に落とす
fn sample2d(d: u32, useQmc: bool) -> vec2f {
  if (useQmc && U.qmc != 0u) {
    return stratified2d(d);
  }
  return vec2f(rand(), rand());
}

/// コサイン重み付き半球サンプリング (pdf = cos / PI)
fn cosineHemisphere(u: vec2f) -> vec3f {
  let r = sqrt(u.x);
  let phi = 2.0 * PI * u.y;
  return vec3f(r * cos(phi), r * sin(phi), sqrt(max(0.0, 1.0 - u.x)));
}

// ---------------------------------------------------------------- 環境光
fn envLuminance(c: vec3f) -> f32 {
  return dot(c, vec3f(0.2126, 0.7152, 0.0722));
}

fn envMarginalOffset() -> u32 {
  return U.envWidth * U.envHeight * 4u;
}

fn envCondOffset() -> u32 {
  return envMarginalOffset() + U.envHeight + 1u;
}

fn envTexel(iu: u32, iv: u32) -> vec3f {
  let u = min(iu, U.envWidth - 1u);
  let v = min(iv, U.envHeight - 1u);
  let o = (v * U.envWidth + u) * 4u;
  return vec3f(envData[o], envData[o + 1u], envData[o + 2u]);
}

/// 方向を lat-long の [0,1]^2 座標へ
fn dirToUv(dir: vec3f) -> vec2f {
  let d = normalize(dir);
  let theta = acos(clamp(d.y, -1.0, 1.0));
  var phi = atan2(d.z, d.x);
  if (phi < 0.0) {
    phi = phi + 2.0 * PI;
  }
  return vec2f(phi / (2.0 * PI), theta / PI);
}

/// 双線形補間。経度方向は巻き戻し、緯度方向は端で留める
fn envSample(dir: vec3f) -> vec3f {
  let uv = dirToUv(dir);
  let fx = uv.x * f32(U.envWidth) - 0.5;
  let fy = uv.y * f32(U.envHeight) - 0.5;
  let ix = i32(floor(fx));
  let iy = i32(floor(fy));
  let tx = fx - floor(fx);
  let ty = fy - floor(fy);
  let w = i32(U.envWidth);
  let h = i32(U.envHeight);
  let x0 = u32(((ix % w) + w) % w);
  let x1 = u32((((ix + 1) % w) + w) % w);
  let y0 = u32(clamp(iy, 0, h - 1));
  let y1 = u32(clamp(iy + 1, 0, h - 1));
  let a = mix(envTexel(x0, y0), envTexel(x1, y0), tx);
  let b = mix(envTexel(x0, y1), envTexel(x1, y1), tx);
  return mix(a, b, ty);
}

fn envColor(dir: vec3f) -> vec3f {
  if (U.env == ENV_BLACK) {
    return vec3f(0.0);
  }
  if (U.env == ENV_HDRI) {
    return envSample(dir);
  }
  let t = 0.5 * (normalize(dir).y + 1.0);
  return mix(vec3f(1.0, 1.0, 1.0), vec3f(0.5, 0.7, 1.0), t);
}

/// この方向を環境マップからサンプルしたときの立体角 pdf。
/// CDF は最近傍テクセルの区分定数なので、pdf も最近傍で引く
fn envPdf(dir: vec3f) -> f32 {
  let uv = dirToUv(dir);
  let iu = min(u32(uv.x * f32(U.envWidth)), U.envWidth - 1u);
  let iv = min(u32(uv.y * f32(U.envHeight)), U.envHeight - 1u);
  return max(envLuminance(envTexel(iu, iv)), ENV_MIN_WEIGHT) * U.envPdfScale;
}

/// 正規化済み CDF の中から xi 以下の最大の区間を探す
fn cdfSearch(base: u32, count: u32, xi: f32) -> u32 {
  var lo = 0u;
  var hi = count;
  for (var i = 0u; i < 32u; i = i + 1u) {
    if (lo + 1u >= hi) {
      break;
    }
    let mid = (lo + hi) / 2u;
    if (envData[base + mid] <= xi) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return lo;
}

/// 環境マップから方向をサンプルする。xyz = 方向, w = 立体角 pdf
fn sampleEnvDir(xi: vec2f) -> vec4f {
  let mOff = envMarginalOffset();
  let cOff = envCondOffset();

  let iv = cdfSearch(mOff, U.envHeight, xi.x);
  let m0 = envData[mOff + iv];
  let m1 = envData[mOff + iv + 1u];
  var dv = 0.5;
  if (m1 > m0) {
    dv = (xi.x - m0) / (m1 - m0);
  }

  let rowBase = cOff + iv * (U.envWidth + 1u);
  let iu = cdfSearch(rowBase, U.envWidth, xi.y);
  let c0 = envData[rowBase + iu];
  let c1 = envData[rowBase + iu + 1u];
  var du = 0.5;
  if (c1 > c0) {
    du = (xi.y - c0) / (c1 - c0);
  }

  let theta = (f32(iv) + dv) / f32(U.envHeight) * PI;
  let phi = (f32(iu) + du) / f32(U.envWidth) * 2.0 * PI;
  let sinT = sin(theta);
  let dir = vec3f(sinT * cos(phi), cos(theta), sinT * sin(phi));
  let pdf = max(envLuminance(envTexel(iu, iv)), ENV_MIN_WEIGHT) * U.envPdfScale;
  return vec4f(dir, pdf);
}

/// 環境マップを光源として扱えるか
fn envIsActive() -> bool {
  return U.envIs != 0u && U.env == ENV_HDRI;
}

/// 一様に選ぶ光源の総数 (面光源 + 環境マップ)
fn lightSelectCount() -> u32 {
  return U.lightCount + select(0u, 1u, envIsActive());
}

// ---------------------------------------------------------------- camera
struct Ray {
  origin: vec3f,
  dir: vec3f,
};

fn makeRay(px: f32, py: f32, lensU: vec2f) -> Ray {
  // px, py は [-1, 1] のスクリーン座標 (py は上が +1)
  let d = U.camU * (px * U.tanHalfFov * U.aspect)
        + U.camV * (py * U.tanHalfFov)
        + U.camW;
  let focusPoint = U.camPos + U.focusDist * d;
  var origin = U.camPos;
  if (U.lensRadius > 0.0) {
    // 単位円内の一様サンプル
    let r = U.lensRadius * sqrt(lensU.x);
    let theta = lensU.y * 6.2831853;
    origin = origin + U.camU * (r * cos(theta)) + U.camV * (r * sin(theta));
  }
  return Ray(origin, normalize(focusPoint - origin));
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

fn aabbHit(bmin: vec3f, bmax: vec3f, o: vec3f, invD: vec3f, tMin: f32, tMax: f32) -> bool {
  let t0 = (bmin - o) * invD;
  let t1 = (bmax - o) * invD;
  let lo = min(t0, t1);
  let hi = max(t0, t1);
  let tn = max(max(lo.x, lo.y), max(lo.z, tMin));
  let tf = min(min(hi.x, hi.y), min(hi.z, tMax));
  return tn <= tf;
}

fn hitScene(ray: Ray, tMin: f32, tMax: f32, hit: ptr<function, Hit>) -> bool {
  var closest = tMax;
  var found = false;
  if (U.bvhNodeCount == 0u) {
    return false;
  }
  let invD = vec3f(1.0) / ray.dir;

  var stack: array<u32, 32>;
  var sp = 0u;
  var node = 0u;
  loop {
    let n = bvh[node];
    var descend = false;

    if (aabbHit(n.bmin, n.bmax, ray.origin, invD, tMin, closest)) {
      if (n.count == 0u) {
        descend = true;
      } else {
        for (var k = 0u; k < n.count; k = k + 1u) {
          let code = indices[U.lightCount + n.leftFirst + k];
          let idx = code & 0x3fffffffu;
          let kind = code >> 30u;

          if (kind == 2u) {
            // Moller-Trumbore
            let tri = triangles[idx];
            let pv = cross(ray.dir, tri.e2);
            let det = dot(tri.e1, pv);
            if (abs(det) < 1e-12) {
              continue;
            }
            let invDet = 1.0 / det;
            let tv = ray.origin - tri.v0;
            let bu = dot(tv, pv) * invDet;
            if (bu < 0.0 || bu > 1.0) {
              continue;
            }
            let qv = cross(tv, tri.e1);
            let bv = dot(ray.dir, qv) * invDet;
            if (bv < 0.0 || bu + bv > 1.0) {
              continue;
            }
            let t = dot(tri.e2, qv) * invDet;
            if (t < tMin || t > closest) {
              continue;
            }
            closest = t;
            found = true;
            // 頂点法線を重心座標で補間して滑らかにする
            let sm = normalize(tri.n0 * (1.0 - bu - bv) + tri.n1 * bu + tri.n2 * bv);
            fillHit(hit, ray, t, sm, tri.mat, 0.0);
          } else if (kind == 0u) {
            let sph = spheres[idx];
            let oc = ray.origin - sph.center;
            let halfB = dot(oc, ray.dir);
            let c = dot(oc, oc) - sph.radius * sph.radius;
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
            fillHit(hit, ray, t, (p - sph.center) / sph.radius, sph.mat, 0.0);
          } else {
            let quad = quads[idx];
            let nrm = cross(quad.u, quad.v);
            let denom = dot(nrm, ray.dir);
            if (abs(denom) < 1e-8) {
              continue;
            }
            let t = dot(nrm, quad.q - ray.origin) / denom;
            if (t < tMin || t > closest) {
              continue;
            }
            // 平面上の点を u, v 基底で表したときの係数が両方 [0, 1] なら内側
            let planar = ray.origin + t * ray.dir - quad.q;
            let w = nrm / dot(nrm, nrm);
            let alpha = dot(w, cross(planar, quad.v));
            let beta = dot(w, cross(quad.u, planar));
            if (alpha < 0.0 || alpha > 1.0 || beta < 0.0 || beta > 1.0) {
              continue;
            }
            closest = t;
            found = true;
            // cross(u, v) の長さがそのまま quad の面積になる
            let ln = length(nrm);
            fillHit(hit, ray, t, nrm / ln, quad.mat, ln);
          }
        }
      }
    }

    if (descend) {
      if (sp < 32u) {
        stack[sp] = n.leftFirst;
        sp = sp + 1u;
      }
      node = node + 1u;
    } else {
      if (sp == 0u) {
        break;
      }
      sp = sp - 1u;
      node = stack[sp];
    }
  }

  return found;
}

/// 影レイ用。最初の 1 個で打ち切るので hitScene より速い。
/// ガラスも遮蔽物として扱うので、屈折で回り込む光は NEE では拾えない
/// (その経路はスペキュラ連鎖として BSDF サンプリング側が拾う)
fn occluded(origin: vec3f, dir: vec3f, maxT: f32) -> bool {
  if (U.bvhNodeCount == 0u) {
    return false;
  }
  let invD = vec3f(1.0) / dir;

  var stack: array<u32, 32>;
  var sp = 0u;
  var node = 0u;
  loop {
    let n = bvh[node];
    var descend = false;

    if (aabbHit(n.bmin, n.bmax, origin, invD, 1e-4, maxT)) {
      if (n.count == 0u) {
        descend = true;
      } else {
        for (var k = 0u; k < n.count; k = k + 1u) {
          let code = indices[U.lightCount + n.leftFirst + k];
          let idx = code & 0x3fffffffu;
          let kind = code >> 30u;

          if (kind == 2u) {
            let tri = triangles[idx];
            let pv = cross(dir, tri.e2);
            let det = dot(tri.e1, pv);
            if (abs(det) < 1e-12) {
              continue;
            }
            let invDet = 1.0 / det;
            let tv = origin - tri.v0;
            let bu = dot(tv, pv) * invDet;
            if (bu < 0.0 || bu > 1.0) {
              continue;
            }
            let qv = cross(tv, tri.e1);
            let bv = dot(dir, qv) * invDet;
            if (bv < 0.0 || bu + bv > 1.0) {
              continue;
            }
            let t = dot(tri.e2, qv) * invDet;
            if (t > 1e-4 && t < maxT) {
              return true;
            }
          } else if (kind == 0u) {
            let sph = spheres[idx];
            let oc = origin - sph.center;
            let halfB = dot(oc, dir);
            let c = dot(oc, oc) - sph.radius * sph.radius;
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
          } else {
            let quad = quads[idx];
            let nrm = cross(quad.u, quad.v);
            let denom = dot(nrm, dir);
            if (abs(denom) < 1e-8) {
              continue;
            }
            let t = dot(nrm, quad.q - origin) / denom;
            if (t <= 1e-4 || t >= maxT) {
              continue;
            }
            let planar = origin + t * dir - quad.q;
            let w = nrm / dot(nrm, nrm);
            let alpha = dot(w, cross(planar, quad.v));
            let beta = dot(w, cross(quad.u, planar));
            if (alpha >= 0.0 && alpha <= 1.0 && beta >= 0.0 && beta <= 1.0) {
              return true;
            }
          }
        }
      }
    }

    if (descend) {
      if (sp < 32u) {
        stack[sp] = n.leftFirst;
        sp = sp + 1u;
      }
      node = node + 1u;
    } else {
      if (sp == 0u) {
        break;
      }
      sp = sp - 1u;
      node = stack[sp];
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
fn sampleDirectLight(hit: Hit, rayDir: vec3f, u: vec2f, misOut: ptr<function, f32>) -> vec3f {
  let n = lightSelectCount();
  if (n == 0u) {
    return vec3f(0.0);
  }
  let pick = min(u32(rand() * f32(n)), n - 1u);

  // 面光源の後ろに環境マップを 1 個ぶら下げて、一様に選ぶ
  if (pick >= U.lightCount) {
    let smp = sampleEnvDir(u);
    let wi = smp.xyz;
    let pL = smp.w / f32(n);
    if (pL <= 0.0) {
      return vec3f(0.0);
    }
    let fcos = bsdfEval(hit, rayDir, wi);
    if (all(fcos <= vec3f(0.0))) {
      return vec3f(0.0);
    }
    if (occluded(hit.p + hit.normal * 1e-4, wi, 1e30)) {
      return vec3f(0.0);
    }
    var w = 1.0;
    if (U.mis != 0u) {
      w = misWeight(pL, bsdfPdfFor(hit, rayDir, wi));
    }
    *misOut = w;
    return envColor(wi) * fcos * w / pL;
  }

  let light = quads[indices[pick]];

  // 光源上の一様サンプル
  let onLight = light.q + light.u * u.x + light.v * u.y;
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
  let pL = dist2 / (cosLight * area * f32(n));

  // MIS。BSDF サンプリングでも作りやすい方向ほど寄与を下げる
  var weight = 1.0;
  if (U.mis != 0u) {
    weight = misWeight(pL, bsdfPdfFor(hit, rayDir, wi));
  }
  *misOut = weight;
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
  u: vec2f,
  attenuation: ptr<function, vec3f>,
  scattered: ptr<function, Ray>,
) -> bool {
  let m = hit.mat;

  if (m.kind == MAT_EMISSIVE) {
    // 放射は trace 側で足しているので、ここでは打ち切るだけ
    return false;
  }

  if (m.kind == MAT_LAMBERT) {
    // pdf = cos / PI ちょうどなので f * cos / pdf = albedo
    let dir = normalize(onb(hit.normal) * cosineHemisphere(u));
    *attenuation = m.albedo;
    *scattered = Ray(hit.p + hit.normal * 1e-4, dir);
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
    let h = basis * sampleGgxVndf(vl, a, u.x, u.y);
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

// ---------------------------------------------------------------- デバッグ
/// 1 次交差まわりの中間量。debugMode が 0 でなければこれを画に出す
struct Aov {
  normal: vec3f,
  albedo: vec3f,
  dist: f32,
  bsdfPdf: f32,
  misWeight: f32,
  bounces: f32,
};

fn aovColor(a: Aov) -> vec3f {
  if (U.debugMode == 1u) {
    return a.normal * 0.5 + vec3f(0.5);
  }
  if (U.debugMode == 2u) {
    return a.albedo;
  }
  if (U.debugMode == 3u) {
    // 単調に [0,1) へ潰すだけ。絶対値ではなく分布を見るためのもの
    return vec3f(a.dist / (a.dist + 8.0));
  }
  if (U.debugMode == 4u) {
    return vec3f(a.bsdfPdf / (a.bsdfPdf + 1.0));
  }
  if (U.debugMode == 5u) {
    return vec3f(a.misWeight);
  }
  if (U.debugMode == 6u) {
    return vec3f(a.bounces / max(f32(U.maxBounces), 1.0));
  }
  return vec3f(0.0);
}

// ---------------------------------------------------------------- trace
fn trace(primary: Ray, firstHit: ptr<function, vec4f>, aov: ptr<function, Aov>) -> vec3f {
  var ray = primary;
  var throughput = vec3f(1.0);
  var radiance = vec3f(0.0);
  // 直前の頂点で BSDF サンプリングした方向の立体角 pdf。
  // 負ならカメラレイかスペキュラ反射で、光源サンプリングでは作れない方向なので重みは 1
  var bsdfPdf = -1.0;
  let useNee = U.nee != 0u && lightSelectCount() > 0u;
  let useEnvNee = U.nee != 0u && envIsActive();

  for (var depth = 0u; depth < U.maxBounces; depth = depth + 1u) {
    var hit: Hit;
    if (!hitScene(ray, 1e-3, 1e30, &hit)) {
      // 環境マップを光源としてサンプルしているなら、ここも二重計上になる
      var w = 1.0;
      if (useEnvNee && bsdfPdf > 0.0) {
        if (U.mis != 0u) {
          w = misWeight(bsdfPdf, envPdf(ray.dir) / f32(lightSelectCount()));
        } else {
          w = 0.0;
        }
      }
      radiance = radiance + throughput * envColor(ray.dir) * w;
      break;
    }
    if (depth == 0u) {
      *firstHit = vec4f(hit.p, 1.0);
      (*aov).normal = hit.normal;
      (*aov).albedo = hit.mat.albedo;
      (*aov).dist = hit.t;
    }
    (*aov).bounces = f32(depth + 1u);

    // 光源に当たったときの放射。NEE と重複するぶんを MIS 重みで削る
    var weight = 1.0;
    if (useNee && bsdfPdf > 0.0 && hit.lightArea > 0.0) {
      if (U.mis != 0u) {
        // この方向を光源サンプリングで作る場合の pdf。sampleDirectLight と同じ式
        let cosLight = max(abs(dot(hit.normal, ray.dir)), 1e-6);
        let pL = hit.t * hit.t / (cosLight * hit.lightArea * f32(lightSelectCount()));
        weight = misWeight(bsdfPdf, pL);
      } else {
        // MIS なしなら NEE 側に完全に任せる (二重計上の防止)
        weight = 0.0;
      }
    }
    radiance = radiance + throughput * hit.mat.emission * weight;

    // pdf を評価できる面だけ光源を直接サンプルする。デルタ面は BSDF サンプリングに任せる
    if (useNee && isDiffuseLike(hit.mat.kind)) {
      var mw = 1.0;
      radiance = radiance + throughput
        * sampleDirectLight(hit, ray.dir, sample2d(2u + depth * 2u, depth < QMC_DEPTH), &mw);
      if (depth == 0u) {
        (*aov).misWeight = mw;
      }
    }

    var attenuation: vec3f;
    var scattered: Ray;
    if (!scatter(ray, hit, sample2d(3u + depth * 2u, depth < QMC_DEPTH), &attenuation, &scattered)) {
      break;
    }
    // 次の頂点で MIS 重みを計算するために pdf を持ち回る。
    // デルタ分布のマテリアルは光源サンプリングで作れない方向なので負にしておく
    if (isDiffuseLike(hit.mat.kind)) {
      bsdfPdf = max(bsdfPdfFor(hit, ray.dir, scattered.dir), 1e-5);
    } else {
      bsdfPdf = -1.0;
    }
    if (depth == 0u) {
      (*aov).bsdfPdf = max(bsdfPdf, 0.0);
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
  // 固定 seed では累積サンプル数から種を作るので、同条件なら毎回同じ絵になる
  let seedBase = select(U.frameIndex, U.samplesBefore, U.fixedSeed != 0u);
  rngState = pcg(pixel * 9781u + seedBase * 6271u + 1u);
  pixelSeed = pcg(pixel * 26699u + 1u);

  var sum = vec3f(0.0);
  var firstHit = vec4f(0.0);
  for (var s = 0u; s < U.sppPerFrame; s = s + 1u) {
    // 累積サンプル番号で低食い違い列を引く
    sampleIdx = U.samplesBefore + s;
    let jitter = sample2d(0u, true);
    let px = (f32(gid.x) + jitter.x) / f32(U.width) * 2.0 - 1.0;
    let py = 1.0 - (f32(gid.y) + jitter.y) / f32(U.height) * 2.0;
    var fh = vec4f(0.0);
    var aov = Aov(vec3f(0.0), vec3f(0.0), 0.0, 0.0, 0.0, 0.0);
    let radiance = trace(makeRay(px, py, sample2d(1u, true)), &fh, &aov);
    if (U.debugMode == 0u) {
      sum = sum + radiance;
    } else {
      sum = sum + aovColor(aov);
    }
    if (s == 0u) {
      firstHit = fh;
    }
  }

  let o = pixel * 2u;
  var prevSum = vec3f(0.0);
  var prevCount = 0.0;

  if (U.reproject != 0u && firstHit.w > 0.5) {
    // 1 次交差の世界座標を前フレームのカメラで投影し直す
    let d = firstHit.xyz - U.prevCamPos;
    let z = dot(d, U.prevCamW);
    if (z > 1e-4) {
      let sx = (dot(d, U.prevCamU) / (z * U.prevTanHalfFov * U.prevAspect) * 0.5 + 0.5) * f32(U.width);
      let sy = (0.5 - dot(d, U.prevCamV) / (z * U.prevTanHalfFov) * 0.5) * f32(U.height);
      if (sx >= 0.0 && sy >= 0.0 && sx < f32(U.width) && sy < f32(U.height)) {
        let pp = (u32(sy) * U.width + u32(sx)) * 2u;
        let hp = histRead[pp + 1u];
        // 同じ面を見ているかを世界座標で確かめる (遮蔽が外れた画素を弾く)
        if (hp.w > 0.5 && distance(hp.xyz, firstHit.xyz) < 0.01 * z) {
          let hc = histRead[pp];
          if (hc.w > 0.0) {
            let capped = min(hc.w, MAX_HISTORY);
            prevSum = hc.rgb * (capped / hc.w);
            prevCount = capped;
          }
        }
      }
    }
  } else if (U.samplesBefore > 0u) {
    // カメラが動いていないので同じ画素からそのまま引き継ぐ
    let hc = histRead[o];
    prevSum = hc.rgb;
    prevCount = hc.w;
  }

  histWrite[o] = vec4f(prevSum + sum, prevCount + f32(U.sppPerFrame));
  histWrite[o + 1u] = firstHit;
}
