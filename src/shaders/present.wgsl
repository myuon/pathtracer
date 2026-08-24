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
  qmc: u32,
  bvhNodeCount: u32,
  envIs: u32,
  envWidth: u32,
  envHeight: u32,
  envPdfScale: f32,
  prevCamPos: vec3f,
  prevTanHalfFov: f32,
  prevCamU: vec3f,
  prevAspect: f32,
  prevCamV: vec3f,
  reproject: u32,
  prevCamW: vec3f,
  /// デノイザをかけるか
  denoise: u32,
  debugMode: u32,
  fixedSeed: u32,
  fogMin: vec3f,
  sigmaS: f32,
  fogMax: vec3f,
  sigmaA: f32,
  fogG: f32,
  fogEnabled: u32,
  sppm: u32,
  photonCount: u32,
  gridCells: u32,
  cellSize: f32,
  radius0: f32,
  photonsEmitted: f32,
  sceneCenter: vec3f,
  sceneRadius: f32,
};

@group(0) @binding(0) var<uniform> U: Uniforms;
/// 1 画素あたり 2 要素。[0] が放射輝度の和と画素ごとのサンプル数
@group(0) @binding(1) var<storage, read> hist: array<vec4f>;

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

// -------------------------------------------------------------- a-trous デノイザ
// bench/run.mjs --present で表示に出る絵の RMSE (0〜255) を測ったところ、
// 効きは spp とともに薄れるが一度も悪化しない:
//        cornell enclosed indirect glass
//   64 spp  1.22x   1.35x   1.44x   1.04x
//  256 spp  1.13x   1.26x   1.39x   1.03x
// 1024 spp  1.05x   1.12x   1.30x   1.02x
// SPPM の残差は「滑らかな壁の上に出る低周波の斑点」。半径ぶんのボケなので
// 細かい粒ではなく大きめの塊で出る。法線と距離が近いタップだけを混ぜる
// エッジ避け付きの広いガウシアンで消す。ストレージバッファ・compute
// パイプラインを増やせないので、present のフラグメントシェーダ 1 パスで完結させる。
// 本来の a-trous は「前段の結果をさらにフィルタする」ことで広い範囲を
// 少ないタップで覆うが、1 パスではそれができない。代わりにスケールを
// 細かく刻んで足す。粗いスケールだけにすると、タップが飛び飛びに効いて
// 輪郭に破線状のリンギングが出る
const KERNEL5 = array<f32, 5>(1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0);
/// 法線 (八面体 2 成分) の許容差
const SIGMA_N: f32 = 0.01;
/// 距離の許容差 (中心の距離に比例させる。遠景ほど緩く)
const SIGMA_Z: f32 = 0.015;
/// 直接光の輝度の許容差 (明るさに比例させる)。
/// 法線・距離だけだと、シルエットの縁 (箱の側面と、扉越しに見える明るい
/// 奥の壁) がたまたま近い法線・近い距離になったときに抜けてしまい、
/// タップの格子 (step=4,8) がそのまま輪郭に沿った破線状のエコーになって
/// 出る。直接光の明るさが桁違いに違えば別物として弾く
const SIGMA_L: f32 = 0.3;

/// 推定したばらつきの何倍まで、輝度の違うタップを混ぜてよいか。
///
/// これが無いと、低 spp では「輝度が違う = 別の物体」と
/// 「輝度が違う = まだノイズ」を区別できない。区別できないので安全側に
/// 倒すしかなく、結果としてフィルタがほとんど効かなくなっていた。
/// 画素ごとのばらつきが分かれば、その範囲内の違いはノイズとして
/// 混ぜてよい (Schied 2017 の SVGF が輝度シグマを分散で割るのと同じ狙い)
const SIGMA_L_NOISE: f32 = 8.0;
/// 相対ばらつきがこれを下回ったらフィルタを止める。収束した画素を
/// ぼかしても偏りが乗るだけで、実際 128 spp の veach では
/// 表示画像の RMSE が 0.67 倍に悪化していた
const NOISE_OFF: f32 = 0.02;
/// ここを超えたら全力で掛ける
const NOISE_FULL: f32 = 0.10;
/// ばらつきの推定値の下限を、サンプル数から決めるときの係数。
///
/// 8 サンプルから出した標本分散はそれ自体が当てにならず、たまたま
/// 小さく出ればノイズだらけの画素でもフィルタが止まってしまう。
/// 「サンプルが少ないうちは収束を主張できない」ぶんを下限にする
const NOISE_FLOOR_K: f32 = 0.5;
/// 粗いスケールを使い始めるばらつきの下限が、1 段ごとに何倍になるか。
///
/// フィルタの広がりはノイズの量に見合っているべきで、収束した画素まで
/// 遠くのタップと混ぜる理由はない。とくに veach の平らなプレートは
/// 法線も距離も一定なので、粗いタップが反射のグラデーションをそのまま
/// 舐めてしまう (128 spp で表示画像の RMSE が 0.72 倍に悪化していた)
const NOISE_SCALE_GROWTH: f32 = 3.0;

/// 画素ごとの相対的なばらつき (平均値の標準誤差 / 平均値) の見積もり。
///
/// SPPM の間接光は密度推定なので、集めた光子数から 1/sqrt(n) で見る。
/// パストレース側は輝度の 2 乗和を積んであるので、1 サンプルの分散から
/// 平均値の標準誤差を出せる
fn relNoise(o: u32) -> f32 {
  if (U.sppm != 0u) {
    let n = max(hist[o + 2u].w, 1.0);
    return 1.0 / sqrt(n);
  }
  let c = hist[o];
  let n = max(c.w, 1.0);
  let mu = luminance(c.rgb) / n;
  let varS = max(hist[o + 2u].x / n - mu * mu, 0.0);
  return max(sqrt(varS / n) / max(mu, 1e-4), NOISE_FLOOR_K / sqrt(n));
}

/// 輪郭らしさ。上下左右の隣とガイドを比べ、食い違う方向が多いほど 1 に近づく。
/// ガイド用のレイは画素の中心を通す一方、蓄積された色は画素内をばらけた
/// サンプルの平均なので、輪郭の画素ではこの 2 つが食い違う。そこで無理に
/// フィルタをかけると、隣の面の明るさがそのまま点になって乗る
fn edgeness(x: u32, y: u32, g0: vec4f) -> f32 {
  var bad = 0.0;
  for (var k = 0u; k < 4u; k = k + 1u) {
    let dx = select(select(0, 1, k == 1u), select(-1, 0, k >= 2u), k != 1u);
    let dy = select(0, select(-1, 1, k == 3u), k >= 2u);
    let tx = i32(x) + dx;
    let ty = i32(y) + dy;
    if (tx < 0 || tx >= i32(U.width) || ty < 0 || ty >= i32(U.height)) {
      continue;
    }
    let gt = hist[(u32(ty) * U.width + u32(tx)) * 4u + 3u];
    let dn = g0.yz - gt.yz;
    if (dot(dn, dn) > SIGMA_N || abs(g0.w - gt.w) > SIGMA_Z * max(g0.w, 1.0)) {
      bad = bad + 0.25;
    }
  }
  return bad;
}

/// 輪郭の画素では、ガイドが示す面と実際に蓄積された色 (画素内をばらけた
/// サンプルの平均) が食い違う。中心が暗い箱なのにガイドが明るい壁を指す、
/// といったことが起きて、結果が桁違いに明るくなる。元の値から大きく
/// 外れた結果は信用しない。斑点を均す用途には 4 倍あれば十分足りる
/// フィルタ結果を中心画素の値の近くへ引き戻す。
///
/// ばらつきが小さい画素ではきつく締める (ぼかしすぎを防ぐ) が、大きい
/// 画素ではほぼ素通しにする。低 spp では中心画素の値こそが当てにならず、
/// そこへ締め付けるとフィルタの意味が無くなる。とくに「まだ明るい経路を
/// 1 本も引けていない」画素は中心値が 0 なので、掛け算の下限では
/// [0, 0.02] に固定されてしまっていた
fn clampToCenter(v: vec3f, center: vec3f, rn: f32) -> vec3f {
  let slack = clamp(rn / NOISE_FULL, 0.0, 1.0);
  let lo = mix(center * 0.25, vec3f(0.0), slack);
  let hi = mix(center * 4.0 + vec3f(0.02), v, slack);
  return clamp(v, lo, hi);
}

fn luminance(c: vec3f) -> f32 {
  return dot(c, vec3f(0.2126, 0.7152, 0.0722));
}

/// SPPM は稀に強い外れ値 (fire fly) を出す。それが a-trous のタップとして
/// 遠くの画素にまで混ざると、タップの格子間隔 (4px, 8px, ...) に沿った
/// 破線・点線状の「エコー」になって非常に目立つ (法線・距離・輝度が
/// 近くても、値そのものが桁違いに大きいタップは重みだけでは弾き切れない)。
/// 直近 (step=1) の 5x5 だけを見た局所平均を基準に、タップの値をクランプする
const FIREFLY_MULT: f32 = 3.0;

fn localIndirectLuma(x: u32, y: u32, g0: vec4f) -> f32 {
  var sum = 0.0;
  var wsum = 0.0;
  for (var j = 0u; j < 5u; j = j + 1u) {
    let ty = i32(y) + i32(j) - 2;
    if (ty < 0 || ty >= i32(U.height)) { continue; }
    for (var i = 0u; i < 5u; i = i + 1u) {
      let tx = i32(x) + i32(i) - 2;
      if (tx < 0 || tx >= i32(U.width)) { continue; }
      let to = (u32(ty) * U.width + u32(tx)) * 4u;
      let gt = hist[to + 3u];
      let dn = g0.yz - gt.yz;
      let dz = g0.w - gt.w;
      let z0 = max(g0.w, 1.0);
      let wn = exp(-dot(dn, dn) / SIGMA_N);
      let wz = exp(-abs(dz) / (SIGMA_Z * z0));
      let w = KERNEL5[i] * KERNEL5[j] * wn * wz;
      let rt = max(gt.x, 1e-5);
      let lum = luminance(hist[to + 2u].rgb) / (3.14159265 * rt * rt * max(U.photonsEmitted, 1.0));
      sum = sum + lum * w;
      wsum = wsum + w;
    }
  }
  if (wsum < 1e-6) { return 0.0; }
  return sum / wsum;
}

/// タップ (i, j) の空間・法線・距離・輝度の重みの積。
/// g0/gt は hist[.+3] (半径, 法線oct, 距離)、l0/lt は中心・タップの直接光輝度
fn atrousTapWeight(i: u32, j: u32, g0: vec4f, gt: vec4f, l0: f32, lt: f32, sigL: f32) -> f32 {
  let dn = g0.yz - gt.yz;
  let dz = g0.w - gt.w;
  let z0 = max(g0.w, 1.0);
  let wn = exp(-dot(dn, dn) / SIGMA_N);
  let wz = exp(-abs(dz) / (SIGMA_Z * z0));
  // 許容差にばらつきのぶんを足す。ノイズで生じている差では弾かない
  let tol = SIGMA_L * max(max(l0, lt), 0.1) + SIGMA_L_NOISE * sigL;
  let wl = exp(-abs(l0 - lt) / max(tol, 1e-6));
  return KERNEL5[i] * KERNEL5[j] * wn * wz * wl;
}

/// SPPM の間接光にだけかける a-trous 風デノイザ。
/// 法線と距離が近いタップだけ混ぜることで、壁の低周波な斑点は消しつつ
/// 別の面や別の奥行きへにじむのを防ぐ
fn denoiseIndirect(x: u32, y: u32) -> vec3f {
  let o = (y * U.width + x) * 4u;
  let g0 = hist[o + 3u];
  let c0 = hist[o];
  let l0 = luminance(c0.rgb / max(c0.w, 1.0));
  let rn = relNoise(o);
  let sigL = rn * max(l0, 0.1);
  let localRef = localIndirectLuma(x, y, g0);
  var sum = vec3f(0.0);
  var wsum = 0.0;
  let steps = array<u32, 5>(1u, 2u, 4u, 8u, 16u);
  for (var s = 0u; s < 5u; s = s + 1u) {
    let step = steps[s];
    // このスケールを使ってよいだけのばらつきがあるか
    let g = pow(NOISE_SCALE_GROWTH, f32(s));
    let lw = smoothstep(NOISE_OFF * g, NOISE_FULL * g, rn);
    if (lw <= 0.0) { continue; }
    for (var j = 0u; j < 5u; j = j + 1u) {
      let ty = i32(y) + (i32(j) - 2) * i32(step);
      if (ty < 0 || ty >= i32(U.height)) { continue; }
      for (var i = 0u; i < 5u; i = i + 1u) {
        let tx = i32(x) + (i32(i) - 2) * i32(step);
        if (tx < 0 || tx >= i32(U.width)) { continue; }
        let to = (u32(ty) * U.width + u32(tx)) * 4u;
        let gt = hist[to + 3u];
        let ct = hist[to];
        let lt = luminance(ct.rgb / max(ct.w, 1.0));
        let w = atrousTapWeight(i, j, g0, gt, l0, lt, sigL) * lw;
        let rt = max(gt.x, 1e-5);
        var indirect = hist[to + 2u].rgb / (3.14159265 * rt * rt * max(U.photonsEmitted, 1.0));
        // fire fly クランプ。局所平均の何倍までしか許さない
        let lt2 = luminance(indirect);
        let cap = localRef * FIREFLY_MULT + 1e-4;
        if (lt2 > cap) {
          indirect = indirect * (cap / lt2);
        }
        sum = sum + indirect * w;
        wsum = wsum + w;
      }
    }
  }
  // 重みの合計は、全タップが通れば 5 (スケール 5 段 x カーネルの総和 1)。
  // これが小さい = 輪郭の上にいて、ごく少数のタップしか通っていない。
  // そのまま平均すると通ったタップがそのまま出て、輪郭に沿って斑点が並ぶ。
  // タップが少ないほど元の値に寄せる
  let r0 = max(g0.x, 1e-5);
  let center = hist[o + 2u].rgb / (3.14159265 * r0 * r0 * max(U.photonsEmitted, 1.0));
  if (wsum < 1e-6) {
    return center;
  }
  // ばらつきが小さい画素 = もう収束しているので触らない
  let t = clamp(wsum / 0.75, 0.0, 1.0) * (1.0 - edgeness(x, y, g0))
    * smoothstep(NOISE_OFF, NOISE_FULL, rn);
  return clampToCenter(mix(center, sum / wsum, t), center, rn);
}

/// SPPM off のとき用。蓄積色 (v そのもの) を同じ重みでぼかす汎用版
fn denoiseColor(x: u32, y: u32) -> vec3f {
  let o = (y * U.width + x) * 4u;
  let g0 = hist[o + 3u];
  let c0 = hist[o];
  let l0 = luminance(c0.rgb / max(c0.w, 1.0));
  let rn = relNoise(o);
  let sigL = rn * max(l0, 0.1);
  var sum = vec3f(0.0);
  var wsum = 0.0;
  let steps = array<u32, 5>(1u, 2u, 4u, 8u, 16u);
  for (var s = 0u; s < 5u; s = s + 1u) {
    let step = steps[s];
    // このスケールを使ってよいだけのばらつきがあるか
    let g = pow(NOISE_SCALE_GROWTH, f32(s));
    let lw = smoothstep(NOISE_OFF * g, NOISE_FULL * g, rn);
    if (lw <= 0.0) { continue; }
    for (var j = 0u; j < 5u; j = j + 1u) {
      let ty = i32(y) + (i32(j) - 2) * i32(step);
      if (ty < 0 || ty >= i32(U.height)) { continue; }
      for (var i = 0u; i < 5u; i = i + 1u) {
        let tx = i32(x) + (i32(i) - 2) * i32(step);
        if (tx < 0 || tx >= i32(U.width)) { continue; }
        let to = (u32(ty) * U.width + u32(tx)) * 4u;
        let gt = hist[to + 3u];
        let ct = hist[to];
        let ctv = ct.rgb / max(ct.w, 1.0);
        let lt = luminance(ctv);
        let w = atrousTapWeight(i, j, g0, gt, l0, lt, sigL) * lw;
        sum = sum + ctv * w;
        wsum = wsum + w;
      }
    }
  }
  let cc = hist[o];
  let center = cc.rgb / max(cc.w, 1.0);
  if (wsum < 1e-6) {
    return center;
  }
  let t = clamp(wsum / 0.75, 0.0, 1.0) * (1.0 - edgeness(x, y, g0))
    * smoothstep(NOISE_OFF, NOISE_FULL, rn);
  return clampToCenter(mix(center, sum / wsum, t), center, rn);
}

@fragment
fn fsMain(in: VsOut) -> @location(0) vec4f {
  let x = min(u32(in.uv.x * f32(U.width)), U.width - 1u);
  let y = min(u32(in.uv.y * f32(U.height)), U.height - 1u);
  let o = (y * U.width + x) * 4u;
  let c = hist[o];
  // 再投影で画素ごとにサンプル数が変わるので、一律の除算ではなく画素の値を使う
  var v = c.rgb / max(c.w, 1.0);

  if (U.debugMode == 0u && U.denoise != 0u) {
    if (U.sppm != 0u) {
      // 直接光は NEE で綺麗に出ているのでそのまま。斑点が出るのは
      // SPPM の間接光だけなので、そこだけフィルタする
      v = v + denoiseIndirect(x, y);
    } else {
      // SPPM off なら v がそのまま唯一の蓄積なので、まとめてフィルタする
      v = denoiseColor(x, y);
    }
  } else if (U.sppm != 0u) {
    // 間接光は「集めたフラックス / (pi r^2 * これまでに撒いた光子数)」
    let r = max(hist[o + 3u].x, 1e-5);
    v = v + hist[o + 2u].rgb / (3.14159265 * r * r * max(U.photonsEmitted, 1.0));
  }

  // 直接光と間接光を分けて見る。どちらがノイズを持ち込んでいるかの切り分け用。
  // 比べられるよう、通常描画と同じトーンマップを通す
  if (U.debugMode == 9u || U.debugMode == 10u) {
    let direct = c.rgb / max(c.w, 1.0);
    var part = direct;
    if (U.debugMode == 10u) {
      part = v - direct;
    }
    return vec4f(pow(acesFilm(part), vec3f(1.0 / 2.2)), 1.0);
  }
  if (U.debugMode != 0u) {
    // 中間量はそのまま見たいのでトーンマップもガンマも通さない
    return vec4f(clamp(v, vec3f(0.0), vec3f(1.0)), 1.0);
  }
  return vec4f(pow(acesFilm(v), vec3f(1.0 / 2.2)), 1.0);
}
