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
  /// デノイザをかけるか
  denoise: u32,
  /// 0 なら通常描画、それ以外は中間量を疑似カラーで出す
  debugMode: u32,
  /// 乱数の種をフレーム番号ではなく累積サンプル数から作る (A/B を再現可能にする)
  fixedSeed: u32,
  /// 霧の入っている軸並行な箱
  fogMin: vec3f,
  /// 散乱係数
  sigmaS: f32,
  fogMax: vec3f,
  /// 吸収係数
  sigmaA: f32,
  /// Henyey-Greenstein の非対称パラメータ (正で前方散乱)
  fogG: f32,
  fogEnabled: u32,
  /// SPPM を使うか
  sppm: u32,
  /// 1 反復あたりに撒く光子の数
  photonCount: u32,
  /// ハッシュグリッドのセル数
  gridCells: u32,
  cellSize: f32,
  /// 半径の初期値
  radius0: f32,
  /// これまでに撒いた光子の総数
  photonsEmitted: f32,
  /// シーンの外接球。環境マップから光子を撒くときの始点に使う
  sceneCenter: vec3f,
  sceneRadius: f32,
  /// カメラ側の頂点と光源側の頂点をつなぐ戦略 (VCM の vertex connection) を使うか
  vcm: u32,
  /// 空間 x 方向の分布を学習して BSDF サンプリングを寄せるか
  guide: u32,
  /// 収束した画素のサンプリングを止めるか
  adaptivePixels: u32,
  /// ロシアンルーレットの生存確率を、この先期待される放射輝度から決めるか
  ears: u32,
  /// 乱数と低食い違い列のスクランブルに混ぜる塩。
  /// 計測 (bench) で参照画像と検証画像に別の値を入れ、両者が同じ点列を
  /// 共有しないようにするためのもの。描画では 0 のまま
  salt: u32,
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
/// 媒質内の散乱点。表面ではないが、位相関数を BSDF と同じ枠で扱えるようにする
const MAT_PHASE: u32 = 4u;

const ENV_SKY: u32 = 0u;
const ENV_BLACK: u32 = 1u;
/// 太陽つきの空を焼いた lat-long マップ
const ENV_HDRI: u32 = 2u;

/// CDF の重みが 0 になる方向を作らないための下限。TS 側と一致させること
const ENV_MIN_WEIGHT: f32 = 1e-6;

const PI: f32 = 3.14159265;
/// このバウンス数までは低食い違い列に次元を割り当てる。
/// 深いバウンスまで広げると経路ごとの次元のずれで逆に悪化するため 1 に留める。
///
/// bench/run.mjs の 8 シーン / 1024 spp で測った結果 (depth 1 を 1.000 とし、
/// 1 より小さいほど悪い。素の relMSE / 上位 1% 除外 / 相対誤差の中央値):
///   QMC_DEPTH 2 -> 0.874 / 0.827 / 0.667
///   QMC_DEPTH 4 -> 0.816 / 0.798 / 0.615
/// 3 つの指標すべてで単調に悪くなる。スクランブルを XOR から Owen
/// (Burley 2020 のハッシュ版) に替えても 0.989 / 0.993 / 0.960 で変わらない。
/// サンプラは GPU 上で層化を直接測って両者とも完全に層化できていることを
/// 確認済みなので、実装の不具合ではなく、経路ごとに次元の意味がずれる
/// ことによる本質的な限界。ここは触らないのが正解。
///
/// なお素の relMSE だけで見ると veach が光沢板のシルエットに出る firefly
/// 十数個に支配されて「QMC を切った方が良い」という誤った結論になる。
/// 判断には trim か med を使うこと (bench/README.md を参照)
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
  /// 下位 8 ビットが葉のプリミティブ数 (0 なら内部ノード)。
  /// 内部ノードは bit 8-9 に分割軸が入っている
  count: u32,
};

/// レイの向きから、分割軸に沿って手前にある子を先に返す
fn childOrder(axis: u32, dir: vec3f, left: u32, right: u32) -> vec2u {
  let d = select(select(dir.x, dir.y, axis == 1u), dir.z, axis == 2u);
  if (d < 0.0) {
    return vec2u(right, left);
  }
  return vec2u(left, right);
}

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

// -------------------------------------------------------------- SPPM
/// 光子 1 個あたり vec4f 3 個 (位置 / 入射方向 / 出力)
@group(0) @binding(9) var<storage, read_write> photons: array<vec4f>;
/// [セルごとの個数][セルごとの光子インデックス] を連結したもの。
/// WGSL の atomic は u32 と i32 しかないので、すべて u32 で通す
@group(0) @binding(10) var<storage, read_write> grid: array<atomic<u32>>;

/// 1 セルに入る光子の上限。あふれた分は捨てる。
/// 捨てた分は反復を重ねても取り返せない = 消えないバイアスになるので、
/// 実測で真値との差が 0 に向かうところまで余裕を取ってある
/// (cap48 では -0.52% で頭打ちだったのが、ここでは +0.09% まで下がる)
const GRID_CAP: u32 = 96u;

/// BVH 走査のスタック段数。中央値分割なので木はほぼ平衡で、
/// 13.5 万三角形でも深さ 16 程度。スレッドごとの private メモリを
/// 食うので、余裕を見つつ小さくしておく
const BVH_STACK: u32 = 24u;

/// 光源側の経路頂点 1 個が使うスロット数 (vec4f 単位)。
/// [0] 位置 + dVCM / [1] 入射方向 + dVC / [2] 経路の重み + 放出方向のビン
/// [3] 法線 + 材質の種類 / [4] アルベド + 粗さ / [5] dVM
/// 後半 2 つは VCM の接続 (camera 側の頂点とつなぐ) で光源側の BSDF を
/// 評価するために要る。SPPM の集光だけなら [0..2] しか使わない
const VTX_SLOTS: u32 = 6u;

/// 光子側のロシアンルーレットを使うか
const PHOTON_RR: bool = true;
/// 光子のロシアンルーレットを始めるバウンス
const PHOTON_RR_START: u32 = 2u;

/// 集めた光子が実際にその点から見えるかを影レイで確かめるか。
///
/// 厚みゼロの衝立を越えて「同じ壁の明るい側」から漏れてくるぶんは、
/// 面の一致判定では弾けない (同じ面・同じ法線・同じ平面なので)。
/// 影レイは集光半径ぶんしか伸びないので BVH の走査もすぐ終わり、
/// 12 シーンの合計時間は 14.6s -> 21.1s (+45%) で済む。
///
/// 256 spp / 12 シーンの relMSE:
///   maze     1.40e+1 -> 2.43e-1  (57 倍)。対 PT 効率も 0.042 -> 5.20
///   indirect 5.96e-2 -> 1.99e-2  (3.0 倍)
///   spheres  6.65e-3 -> 6.15e-3  (1.08 倍)
///   残りは横ばい。悪化するシーンはない
const GATHER_VISIBILITY: bool = true;

/// 光子を集めるときに「同じ面か」を判定する閾値。
/// 法線の内積がこれを下回る光子は捨てる。曲面 (球) では 1 つの半径の中で
/// 法線が振れるので、あまり厳しくすると拾えるはずの光子まで落ちる
const GATHER_COS: f32 = 0.9;
/// 接平面からの距離の許容量 (半径に対する比)。向きが揃った平行な 2 面が
/// 半径より近いときに漏れるのを止める
const GATHER_PLANE: f32 = 0.3;

/// 光子 1 本が堆積できる回数の上限。白い部屋では光子が 3〜5 回跳ねるので、
/// 2 回で打ち切ると間接光がごっそり欠ける。水面越しの集光はさらに跳ねるため、
/// 6 -> 10 でノイズが 3.26 -> 1.23 まで落ちた
const MAX_DEPOSITS: u32 = 10u;

// -------------------------------------------- 光子の放出方向のガイディング
// 光源から無誘導に撒くと、狭い隙間の向こうにしか使い道がないシーンでは
// ほとんどの光子が無駄になる。実際に集光で使われた光子の放出方向を
// ヒストグラムに貯めておき、次のフレームはそちらへ優先的に撒く。
// pdf で割るので推定値は不偏のまま
/// 方向のヒストグラム。cos(theta) と phi で切ると 1 ビンが等立体角になる
const HIST_THETA: u32 = 16u;
const HIST_PHI: u32 = 32u;
const HIST_BINS: u32 = 512u;
/// 学習した分布を使う確率。残りは今までどおり余弦分布で撒く。
/// 混ぜておかないと学習が偏ったときに pdf が 0 の方向ができて破綻する
const GUIDE_MIX: f32 = 0.5;
/// 集光に使われた光子を記録する確率。全部記録すると atomic が詰まる。
/// 間引いても期待値は変わらないので分布の形は保たれる
const CREDIT_RATE: f32 = 1.0 / 32.0;

/// grid バッファの後ろに間借りしている。ストレージバッファの本数が
/// 上限に張り付いているので、専用のバッファを増やせない
fn histOff() -> u32 {
  return U.gridCells * (1u + GRID_CAP) + 1u;
}
fn cdfOff() -> u32 {
  return histOff() + HIST_BINS;
}

/// カメラ側が実際に集光しているセルの印。光子がそこへ届いたかを数えるのに使う。
/// カメラが止まっていれば印は変わらないので、消さずに貯め続ける
fn markOff() -> u32 {
  return cdfOff() + HIST_BINS + 1u;
}
/// [0] 今フレームの堆積の総数 / [1] そのうち集光している場所に入った数
/// [2] 集光の回数 / [3] 見つかった光子数の和 / [4] その 2 乗和
fn statOff() -> u32 {
  return markOff() + U.gridCells;
}

// ---------------------------------------------------- パスガイディング
// 「この辺りにいるときは、どの方向から光が来るか」を学習して、BSDF
// サンプリングをそちらへ寄せる。BSDF は「面がどの方向へ光を返しやすいか」
// しか知らないので、光源が狭い方向にしかないシーンでは当てが外れ続ける
/// 適応サンプリングを打ち切る相対標準誤差。これを下回った画素は撃たない
const ADAPTIVE_TOL: f32 = 0.004;

/// 空間の分割数 (1 辺)
const GUIDE_DIM: u32 = 16u;
const GUIDE_VOX: u32 = 4096u;
/// 方向の分割数。cos(theta) と phi で等立体角に切る。
/// 解像度は実測で決めた。空間を細かくしても効かず (16^3 と 32^3 で
/// glass 1.78x 対 1.76x)、方向だけが効く。ただし細かすぎると 1 ビンあたりの
/// データが薄まって学習した分布自体がノイジーになる
///   64 方向  glass 1.78x
///   256 方向 glass 2.37x  <- ここが最適
///   1024 方向 glass 1.50x
const GUIDE_TH: u32 = 16u;
const GUIDE_PH: u32 = 16u;
const GUIDE_BINS: u32 = 256u;
/// 学習した分布を使う確率。残りは今までどおり BSDF から引く。
/// 混ぜておかないと pdf が 0 の方向ができて破綻する
const GUIDE_MIX_C: f32 = 0.5;
/// 教師データとして覚えておく頂点の数。深い頂点ほど寄与が小さいので浅い側だけ
const GUIDE_REC: u32 = 4u;

fn guideOff() -> u32 {
  return statOff() + 8u;
}
fn guideCdfOff() -> u32 {
  return guideOff() + GUIDE_VOX * GUIDE_BINS;
}

/// 位置からボクセル番号。シーンの外接球で正規化する
fn guideVoxel(p: vec3f) -> u32 {
  let t = (p - U.sceneCenter) / max(U.sceneRadius, 1e-4) * 0.5 + vec3f(0.5);
  let c = clamp(vec3u(t * f32(GUIDE_DIM)), vec3u(0u), vec3u(GUIDE_DIM - 1u));
  return (c.z * GUIDE_DIM + c.y) * GUIDE_DIM + c.x;
}

// ------------------------------------------------ 効率を考えたロシアンルーレット
// 今のロシアンルーレットは生存確率をスループットだけで決めている。
// これは「この先どれだけ光が返ってくるか」を一切見ていないので、
// 明るい間接光が待っている経路を切り、真っ暗な方向へ無駄に伸ばす。
//
// ADRRS (Vorba & Krivanek 2016) は、その地点から先に期待される放射輝度を
// 覚えておき、「この経路が画素の目標値にどれだけ届きそうか」で生存確率を
// 決める。ガイディングと同じ 16^3 ボクセルに、方向を潰したスカラーの
// 平均放射輝度だけを貯める
/// 1 ボクセルあたりの記録数の上限。ここに達したら凍結する。
/// u32 の足し込みが桁あふれするのを防ぐのと、十分溜まった推定値を
/// これ以上動かす必要がないのと、両方の理由
const EARS_CAP: u32 = 1u << 20u;
/// 記録する値の固定小数の倍率と上限
const EARS_SCALE: f32 = 16.0;
const EARS_MAX: f32 = 65535.0;
/// 生存確率の下限。推定が 0 に振れた場所で経路が全滅すると、そこには
/// 二度とデータが溜まらず推定も直らない (自己強化して直らなくなる)
const EARS_MIN_Q: f32 = 0.05;
/// ロシアンルーレットを始めるバウンス。スループットだけで決めていた頃は
/// 早く始めると効く経路まで切ってしまうので 3 に置いていたが、ADRRS は
/// 「この先どれだけ返ってきそうか」を見られるので早く始められるはず
const RR_START: u32 = 3u;

/// [vox] = 放射輝度の和 (固定小数) / [GUIDE_VOX + vox] = 記録数
///
/// 同じディスパッチの中で他のスレッドが書いている途中の値を読むので、
/// これを有効にすると固定 seed でも絵が完全には再現しなくなる
/// (実測で relMSE が 0.5% 程度ぶれる。計測のばらつき 6% よりは十分小さい)。
/// パスガイディングも同じ性質を持っている
fn earsOff() -> u32 {
  return guideCdfOff() + GUIDE_VOX * (GUIDE_BINS + 1u);
}

/// そのボクセルで期待される入射放射輝度。データがなければ負を返す
fn earsMean(vox: u32) -> f32 {
  let n = atomicLoad(&grid[earsOff() + GUIDE_VOX + vox]);
  if (n == 0u) {
    return -1.0;
  }
  return f32(atomicLoad(&grid[earsOff() + vox])) / (EARS_SCALE * f32(n));
}

fn earsRecord(p: vec3f, li: f32) {
  let vox = guideVoxel(p);
  if (atomicLoad(&grid[earsOff() + GUIDE_VOX + vox]) >= EARS_CAP) {
    return;
  }
  atomicAdd(&grid[earsOff() + vox], u32(clamp(li * EARS_SCALE, 0.0, EARS_MAX)));
  atomicAdd(&grid[earsOff() + GUIDE_VOX + vox], 1u);
}

/// ADRRS の生存確率。target はこの画素が目指している値 (これまでの平均)。
/// 「今のスループット x この先期待される放射輝度」が目標値に対して
/// どれだけの割合かで決める。1 を超えるなら必ず生き残らせる
fn earsSurvival(thrLum: f32, vox: u32, aim: f32) -> f32 {
  let e = earsMean(vox);
  if (e < 0.0 || aim <= 0.0) {
    return -1.0;
  }
  return clamp(thrLum * e / aim, 0.0, 1.0);
}


/// 世界座標の方向を、面の法線を +z とする局所座標へ
fn toLocal(basis: mat3x3f, d: vec3f) -> vec3f {
  return vec3f(dot(d, basis[0]), dot(d, basis[1]), dot(d, basis[2]));
}

/// 局所座標の方向 -> ビン。cos(theta) を [0, 1] で等分し phi も等分するので、
/// 1 ビンが等立体角 (2pi / GUIDE_BINS) になる。
///
/// 分布を「世界座標の全球」ではなく「面の法線まわりの半球」で持つのが肝。
/// 全球で持つと、面の裏へ出る方向にも学習した確率が乗る。そこは BSDF が
/// 0 なので経路を打ち切るしかなく、混合比 0.5 なら全サンプルの 1/4 が
/// その場で死ぬ。実測でも混合比を上げるほど単調に悪化していた
/// (0.25 -> 0.887x / 0.5 -> 0.848x / 0.75 -> 0.714x)。
/// 半球で持てば無駄はゼロになり、同じビン数で方向の分解能も 2 倍になる
fn guideBinLocal(l: vec3f) -> u32 {
  let it = min(u32(clamp(l.z, 0.0, 1.0) * f32(GUIDE_TH)), GUIDE_TH - 1u);
  let ip = min(u32((atan2(l.y, l.x) / (2.0 * PI) + 0.5) * f32(GUIDE_PH)), GUIDE_PH - 1u);
  return it * GUIDE_PH + ip;
}

/// ビンの中を一様に引く (局所座標)
fn guideBinDirLocal(bin: u32, u: vec2f) -> vec3f {
  let cosT = (f32(bin / GUIDE_PH) + u.x) / f32(GUIDE_TH);
  let phi = ((f32(bin % GUIDE_PH) + u.y) / f32(GUIDE_PH) - 0.5) * 2.0 * PI;
  let sinT = sqrt(max(0.0, 1.0 - cosT * cosT));
  return vec3f(sinT * cos(phi), sinT * sin(phi), cosT);
}

/// 立体角についての pdf。1 ビンの立体角は 2pi / GUIDE_BINS
fn guidePdfLocal(vox: u32, l: vec3f) -> f32 {
  if (l.z <= 0.0) {
    return 0.0;
  }
  let base = guideCdfOff() + vox * (GUIDE_BINS + 1u);
  let total = f32(atomicLoad(&grid[base + GUIDE_BINS]));
  if (total <= 0.0) {
    return 0.0;
  }
  let b = guideBinLocal(l);
  let lo = f32(atomicLoad(&grid[base + b]));
  let hi = f32(atomicLoad(&grid[base + b + 1u]));
  return ((hi - lo) / total) * (f32(GUIDE_BINS) / (2.0 * PI));
}

fn guideHasData(vox: u32) -> bool {
  return atomicLoad(&grid[guideCdfOff() + vox * (GUIDE_BINS + 1u) + GUIDE_BINS]) > 0u;
}

/// 学習した分布から局所座標の方向を 1 つ引く
fn guideSampleLocal(vox: u32, u: f32, uv: vec2f) -> vec3f {
  let base = guideCdfOff() + vox * (GUIDE_BINS + 1u);
  let pickAt = u32(u * f32(atomicLoad(&grid[base + GUIDE_BINS])));
  var lo = 0u;
  var hi = GUIDE_BINS;
  loop {
    if (lo + 1u >= hi) {
      break;
    }
    let mid = (lo + hi) / 2u;
    if (atomicLoad(&grid[base + mid]) <= pickAt) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return guideBinDirLocal(lo, uv);
}

/// 学習値の上限。これを超えたら全ビンを半分にする。
/// u32 に足し続けると桁があふれて CDF が壊れ、しかも反復を重ねるほど
/// 悪化する (誤差が spp とともに拡大するという妙な挙動になる)
const GUIDE_CAP: u32 = 1u << 28u;

/// 光が来た方向を記録する。方向は面の法線を +z とする局所座標で渡すこと。
/// 1 回の記録が大きくなりすぎないよう抑える
fn guideRecord(p: vec3f, l: vec3f, lum: f32) {
  if (lum <= 0.0 || l.z <= 0.0) {
    return;
  }
  atomicAdd(&grid[guideOff() + guideVoxel(p) * GUIDE_BINS + guideBinLocal(l)],
    u32(clamp(lum * 8.0, 0.0, 4096.0)) + 1u);
}

fn dirToBin(d: vec3f) -> u32 {
  let it = min(u32((d.y * 0.5 + 0.5) * f32(HIST_THETA)), HIST_THETA - 1u);
  let ip = min(u32((atan2(d.z, d.x) / (2.0 * PI) + 0.5) * f32(HIST_PHI)), HIST_PHI - 1u);
  return it * HIST_PHI + ip;
}

/// ビンの中を一様に引く
fn binToDir(bin: u32, u: vec2f) -> vec3f {
  let cosT = ((f32(bin / HIST_PHI) + u.x) / f32(HIST_THETA)) * 2.0 - 1.0;
  let phi = ((f32(bin % HIST_PHI) + u.y) / f32(HIST_PHI) - 0.5) * 2.0 * PI;
  let sinT = sqrt(max(0.0, 1.0 - cosT * cosT));
  return vec3f(sinT * cos(phi), cosT, sinT * sin(phi));
}

fn histTotal() -> u32 {
  return atomicLoad(&grid[cdfOff() + HIST_BINS]);
}

/// 立体角についての pdf。1 ビンの立体角は 4pi / HIST_BINS
fn histPdf(bin: u32) -> f32 {
  let total = f32(histTotal());
  if (total <= 0.0) {
    return 0.0;
  }
  let lo = f32(atomicLoad(&grid[cdfOff() + bin]));
  let hi = f32(atomicLoad(&grid[cdfOff() + bin + 1u]));
  return ((hi - lo) / total) * (f32(HIST_BINS) / (4.0 * PI));
}

fn sampleHistBin(u: f32) -> u32 {
  let pickAt = u32(u * f32(histTotal()));
  var lo = 0u;
  var hi = HIST_BINS;
  loop {
    if (lo + 1u >= hi) {
      break;
    }
    let mid = (lo + hi) / 2u;
    if (atomicLoad(&grid[cdfOff() + mid]) <= pickAt) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return lo;
}

fn gridHash(ix: i32, iy: i32, iz: i32) -> u32 {
  let h = (u32(ix) * 73856093u) ^ (u32(iy) * 19349663u) ^ (u32(iz) * 83492791u);
  return h % U.gridCells;
}

fn gridCoord(p: vec3f) -> vec3i {
  return vec3i(floor(p / U.cellSize));
}

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

/// 累積に入れてよい値か。NaN との比較は必ず false になるので、
/// これで NaN も Inf もまとめて弾ける。
///
/// 一度でも NaN が累積バッファに入ると、以後その画素は何サンプル積んでも
/// NaN のままで二度と戻らない。実際 spheres の 1 画素 (123, 124) が
/// 32768 spp の参照画像で死んでいた。落とすぶんの偏りは 10 万分の 1 程度
/// firefly の抑制。画素のこれまでの平均輝度 x FIREFLY_K x sqrt(サンプル数)
/// を超えたサンプルは、そこまで押し下げてから積む。0 で無効。
///
/// 閾値をサンプル数の平方根に比例させるのが肝。1 本の外れ値が N サンプルの
/// 推定値を動かす量は L / N で、通常のばらつきは sigma / sqrt(N) なので、
/// 「悪さをする外れ値」の境目は L ~ sigma * sqrt(N) にある。ここを固定値に
/// すると spp をいくら積んでも押し下げが残り、明るい集光が暗いまま
/// 収束しなくなる。
///
/// 係数は bench/run.mjs で振って決めた。全 12 シーン / 1024 spp では
/// K = 64 で relMSE の幾何平均が 1.52x、効率も 1.51x、しかも 12 シーン
/// すべてで悪化なし (最小 1.01x)。
///
/// K = 16 の方が低 spp では強いが、収束先がずれるので採らない
/// (cornell/enclosed/maze/ajar/water/glass の 6 シーンで比較):
///            1024 spp   4096 spp   1024->4096 の誤差の減り
///   K = 16     1.43x      1.07x     1.5 〜 3.2 (無効時は 1.8 〜 3.5)
///   K = 64     1.23x      1.08x     1.8 〜 3.2
/// K = 16 は 4096 spp で water 0.82x / enclosed 0.98x と無効時に負ける。
/// 押し下げの偏りが変動の減りを食い潰している。K = 64 はどちらの予算でも
/// 1.00x を下回らない
const FIREFLY_K: f32 = 64.0;

fn accumulable(c: vec3f) -> bool {
  return c.x >= 0.0 && c.y >= 0.0 && c.z >= 0.0
    && c.x < 1e30 && c.y < 1e30 && c.z < 1e30;
}

/// コサイン重み付き半球サンプリング (pdf = cos / PI)
fn cosineHemisphere(u: vec2f) -> vec3f {
  let r = sqrt(u.x);
  let phi = 2.0 * PI * u.y;
  return vec3f(r * cos(phi), r * sin(phi), sqrt(max(0.0, 1.0 - u.x)));
}

// ---------------------------------------------------------------- 環境光
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
/// 重みは alpha に入っている近傍 3x3 の最大輝度。色をバイリニアで引くので、
/// 最近傍の輝度をそのまま使うと太陽の縁で pdf が足りずに斑点が出る
fn envWeight(iu: u32, iv: u32) -> f32 {
  let u = min(iu, U.envWidth - 1u);
  let v = min(iv, U.envHeight - 1u);
  return envData[(v * U.envWidth + u) * 4u + 3u];
}

fn envPdf(dir: vec3f) -> f32 {
  let uv = dirToUv(dir);
  let iu = min(u32(uv.x * f32(U.envWidth)), U.envWidth - 1u);
  let iv = min(u32(uv.y * f32(U.envHeight)), U.envHeight - 1u);
  return envWeight(iu, iv) * U.envPdfScale;
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
  let pdf = envWeight(iu, iv) * U.envPdfScale;
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
  /// 面光源だったときの quad の番号。立体角サンプリングの pdf を
  /// 後から計算し直すのに要る
  lightQuad: u32,
};

/// outward は正規化済みの外向き法線
fn fillHit(hit: ptr<function, Hit>, ray: Ray, t: f32, outward: vec3f, mat: Material, lightArea: f32, lightQuad: u32) {
  let front = dot(ray.dir, outward) < 0.0;
  (*hit).t = t;
  (*hit).p = ray.origin + t * ray.dir;
  (*hit).normal = select(-outward, outward, front);
  (*hit).frontFace = front;
  (*hit).mat = mat;
  (*hit).lightArea = lightArea;
  (*hit).lightQuad = lightQuad;
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

  var stack: array<u32, BVH_STACK>;
  var sp = 0u;
  var node = 0u;
  loop {
    let n = bvh[node];
    var descend = false;

    let cnt = n.count & 0xffu;
    if (aabbHit(n.bmin, n.bmax, ray.origin, invD, tMin, closest)) {
      if (cnt == 0u) {
        descend = true;
      } else {
        for (var k = 0u; k < cnt; k = k + 1u) {
          let code = indices[U.lightCount + n.leftFirst + k];
          let idx = code & 0x3fffffffu;
          let kind = code >> 30u;

          if (kind == 2u) {
            // Moller-Trumbore。交差が確定するまでは頂点と辺しか読まない
            // (Triangle は 144 バイトあるが判定に要るのは 48 バイト)
            let e2 = triangles[idx].e2;
            let pv = cross(ray.dir, e2);
            let e1 = triangles[idx].e1;
            let det = dot(e1, pv);
            if (abs(det) < 1e-12) {
              continue;
            }
            let invDet = 1.0 / det;
            let tv = ray.origin - triangles[idx].v0;
            let bu = dot(tv, pv) * invDet;
            if (bu < 0.0 || bu > 1.0) {
              continue;
            }
            let qv = cross(tv, e1);
            let bv = dot(ray.dir, qv) * invDet;
            if (bv < 0.0 || bu + bv > 1.0) {
              continue;
            }
            let t = dot(e2, qv) * invDet;
            if (t < tMin || t > closest) {
              continue;
            }
            closest = t;
            found = true;
            // ここまで来たら法線とマテリアルも読む
            let tri = triangles[idx];
            let sm = normalize(tri.n0 * (1.0 - bu - bv) + tri.n1 * bu + tri.n2 * bv);
            fillHit(hit, ray, t, sm, tri.mat, 0.0, 0u);
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
            fillHit(hit, ray, t, (p - sph.center) / sph.radius, sph.mat, 0.0, 0u);
          } else {
            let qu = quads[idx].u;
            let qv2 = quads[idx].v;
            let nrm = cross(qu, qv2);
            let denom = dot(nrm, ray.dir);
            if (abs(denom) < 1e-8) {
              continue;
            }
            let t = dot(nrm, quads[idx].q - ray.origin) / denom;
            if (t < tMin || t > closest) {
              continue;
            }
            // 平面上の点を u, v 基底で表したときの係数が両方 [0, 1] なら内側
            let planar = ray.origin + t * ray.dir - quads[idx].q;
            let w = nrm / dot(nrm, nrm);
            let alpha = dot(w, cross(planar, qv2));
            let beta = dot(w, cross(qu, planar));
            if (alpha < 0.0 || alpha > 1.0 || beta < 0.0 || beta > 1.0) {
              continue;
            }
            closest = t;
            found = true;
            // cross(u, v) の長さがそのまま quad の面積になる
            let ln = length(nrm);
            fillHit(hit, ray, t, nrm / ln, quads[idx].mat, ln, idx);
          }
        }
      }
    }

    if (descend) {
      // 手前の子から降りると、遠い側を早く枝刈りできる
      let order = childOrder((n.count >> 8u) & 3u, ray.dir, node + 1u, n.leftFirst);
      if (sp < BVH_STACK) {
        stack[sp] = order.y;
        sp = sp + 1u;
      }
      node = order.x;
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

fn octEncode(n: vec3f) -> vec2f {
  let p = n.xy / (abs(n.x) + abs(n.y) + abs(n.z));
  if (n.z >= 0.0) { return p; }
  return (1.0 - abs(p.yx)) * vec2f(select(-1.0, 1.0, p.x >= 0.0), select(-1.0, 1.0, p.y >= 0.0));
}

/// デノイザ用の手がかり。画素の中心を通るレイを1本だけ撃って、最初に
/// 当たった面の法線と距離を返す。フィルタがこれを見て、別の面や
/// 別の奥行きへにじむのを防ぐ
fn guideFor(px: f32, py: f32) -> vec3f {  // (oct.x, oct.y, distance)
  let ray = makeRay(px, py, vec2f(0.5, 0.5));
  var hit: Hit;
  if (hitScene(ray, 1e-4, 1e30, &hit)) {
    let oct = octEncode(hit.normal);
    return vec3f(oct.x, oct.y, hit.t);
  }
  return vec3f(0.0, 0.0, 1e30);
}

/// 影レイ用。最初の 1 個で打ち切るので hitScene より速い。
/// ガラスも遮蔽物として扱うので、屈折で回り込む光は NEE では拾えない
/// (その経路はスペキュラ連鎖として BSDF サンプリング側が拾う)
fn occluded(origin: vec3f, dir: vec3f, maxT: f32) -> bool {
  if (U.bvhNodeCount == 0u) {
    return false;
  }
  let invD = vec3f(1.0) / dir;

  var stack: array<u32, BVH_STACK>;
  var sp = 0u;
  var node = 0u;
  loop {
    let n = bvh[node];
    var descend = false;

    let cnt = n.count & 0xffu;
    if (aabbHit(n.bmin, n.bmax, origin, invD, 1e-4, maxT)) {
      if (cnt == 0u) {
        descend = true;
      } else {
        for (var k = 0u; k < cnt; k = k + 1u) {
          let code = indices[U.lightCount + n.leftFirst + k];
          let idx = code & 0x3fffffffu;
          let kind = code >> 30u;

          if (kind == 2u) {
            let e2 = triangles[idx].e2;
            let pv = cross(dir, e2);
            let e1 = triangles[idx].e1;
            let det = dot(e1, pv);
            if (abs(det) < 1e-12) {
              continue;
            }
            let invDet = 1.0 / det;
            let tv = origin - triangles[idx].v0;
            let bu = dot(tv, pv) * invDet;
            if (bu < 0.0 || bu > 1.0) {
              continue;
            }
            let qv = cross(tv, e1);
            let bv = dot(dir, qv) * invDet;
            if (bv < 0.0 || bu + bv > 1.0) {
              continue;
            }
            let t = dot(e2, qv) * invDet;
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
            let qu = quads[idx].u;
            let qv2 = quads[idx].v;
            let nrm = cross(qu, qv2);
            let denom = dot(nrm, dir);
            if (abs(denom) < 1e-8) {
              continue;
            }
            let t = dot(nrm, quads[idx].q - origin) / denom;
            if (t <= 1e-4 || t >= maxT) {
              continue;
            }
            let planar = origin + t * dir - quads[idx].q;
            let w = nrm / dot(nrm, nrm);
            let alpha = dot(w, cross(planar, qv2));
            let beta = dot(w, cross(qu, planar));
            if (alpha >= 0.0 && alpha <= 1.0 && beta >= 0.0 && beta <= 1.0) {
              return true;
            }
          }
        }
      }
    }

    if (descend) {
      // 手前の子から降りると、遠い側を早く枝刈りできる
      let order = childOrder((n.count >> 8u) & 3u, dir, node + 1u, n.leftFirst);
      if (sp < BVH_STACK) {
        stack[sp] = order.y;
        sp = sp + 1u;
      }
      node = order.x;
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

// ---------------------------------------------------------------- 参加媒質
fn fogActive() -> bool {
  return U.fogEnabled != 0u;
}

fn sigmaT() -> f32 {
  return U.sigmaS + U.sigmaA;
}

/// レイと霧の箱が重なる区間 [t0, t1]。重なりがなければ y <= x になる
fn fogRange(origin: vec3f, dir: vec3f, tMax: f32) -> vec2f {
  let invD = vec3f(1.0) / dir;
  let a = (U.fogMin - origin) * invD;
  let b = (U.fogMax - origin) * invD;
  let lo = min(a, b);
  let hi = max(a, b);
  return vec2f(
    max(max(lo.x, lo.y), max(lo.z, 0.0)),
    min(min(hi.x, hi.y), min(hi.z, tMax)),
  );
}

/// その区間を通り抜ける透過率。影レイの減衰に使う
fn fogTransmittance(origin: vec3f, dir: vec3f, tMax: f32) -> f32 {
  if (!fogActive()) {
    return 1.0;
  }
  let r = fogRange(origin, dir, tMax);
  if (r.y <= r.x) {
    return 1.0;
  }
  return exp(-sigmaT() * (r.y - r.x));
}

/// Henyey-Greenstein 位相関数。cosT は進行方向と出射方向の内積
fn hgPhase(cosT: f32, g: f32) -> f32 {
  let d = max(1.0 + g * g - 2.0 * g * cosT, 1e-6);
  return (1.0 - g * g) / (4.0 * PI * d * sqrt(d));
}

/// hgPhase に対応する逆関数サンプリング。dir は進行方向
fn sampleHg(dir: vec3f, g: f32, u: vec2f) -> vec3f {
  var cosT: f32;
  if (abs(g) < 1e-3) {
    cosT = 1.0 - 2.0 * u.x;
  } else {
    let sq = (1.0 - g * g) / (1.0 - g + 2.0 * g * u.x);
    cosT = (1.0 + g * g - sq * sq) / (2.0 * g);
  }
  cosT = clamp(cosT, -1.0, 1.0);
  let sinT = sqrt(max(0.0, 1.0 - cosT * cosT));
  let phi = 2.0 * PI * u.y;
  return normalize(onb(dir) * vec3f(sinT * cos(phi), sinT * sin(phi), cosT));
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

/// Smith の遮蔽・陰影項 (両側、高さ相関型)。
/// G1(v) * G1(l) の分離型は遮蔽を過大評価して粗い面が暗くなる。
/// **eval と scatter の重みは必ずこれで揃えること** (pdf は G1(v) のままで
/// 変わらないので MIS には影響しない)
fn ggxG2(cosO: f32, cosI: f32, a: f32) -> f32 {
  if (cosO <= 0.0 || cosI <= 0.0) {
    return 0.0;
  }
  let a2 = a * a;
  let lo = cosI * sqrt(a2 + (1.0 - a2) * cosO * cosO);
  let li = cosO * sqrt(a2 + (1.0 - a2) * cosI * cosI);
  return 2.0 * cosO * cosI / max(lo + li, 1e-9);
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
/// 誘電体は粗さがある場合のみ対象にする (ほぼ鏡面だと pdf が極端に大きく、
/// 影レイを飛ばしても寄与がほぼ 0 で無駄になるため)
/// 発光しているか。hit.lightArea は発光していない quad でも正の値を持つので
/// 判定に使えない。使うと壁に当たるたびに立体角の計算 (acos を 4 回) が走る
fn isEmissive(m: Material) -> bool {
  return m.emission.r > 0.0 || m.emission.g > 0.0 || m.emission.b > 0.0;
}

/// SPPM / VCM で光子を集められる面か。
///
/// 鏡面に近い GGX を対象にすると、BRDF がほぼデルタなので半径内の光子の
/// ほとんどが寄与 0、一部だけ巨大という分布になり、分散もバイアスも大きい。
/// 誘電体と同じく通り抜けて次を探すべき。**堆積側と集光側で必ず同じ判定を
/// 使うこと** (食い違うと片方だけ光子が入って偏る)
fn gatherableMat(m: Material) -> bool {
  return m.kind == MAT_LAMBERT || (m.kind == MAT_GGX && m.roughness > 0.3);
}

fn isDiffuseLike(m: Material) -> bool {
  return m.kind == MAT_LAMBERT || m.kind == MAT_GGX || m.kind == MAT_PHASE
    || (m.kind == MAT_DIELECTRIC && m.roughness > 0.02);
}

/// 誘電体の相対屈折率 nt / ni。hit.normal は視線側を向いている
fn dielectricEtaRel(hit: Hit) -> f32 {
  return select(1.0 / hit.mat.ior, hit.mat.ior, hit.frontFace);
}

/// 誘電体のマイクロファセット半ベクトルとフレネル、屈折のヤコビアン分母
struct DiTerms {
  valid: bool,
  isReflect: bool,
  h: vec3f,
  denom: f32,
  fr: f32,
};

fn dielectricTerms(hit: Hit, rayDir: vec3f, wi: vec3f) -> DiTerms {
  var t: DiTerms;
  t.valid = false;
  t.isReflect = false;
  t.h = hit.normal;
  t.denom = 1.0;
  t.fr = 1.0;

  let v = -rayDir;
  let cosO = dot(hit.normal, v);
  let cosI = dot(hit.normal, wi);
  if (cosO <= 1e-6 || abs(cosI) < 1e-6) {
    return t;
  }
  t.isReflect = cosI > 0.0;
  let etaRel = dielectricEtaRel(hit);

  // 屈折側は Walter の一般化半ベクトル
  var h = select(normalize(wi * etaRel + v), normalize(v + wi), t.isReflect);
  if (dot(h, hit.normal) < 0.0) {
    h = -h;
  }
  let dotVH = dot(v, h);
  if (dotVH <= 1e-6) {
    return t;
  }
  let dotIH = dot(wi, h);

  // 全反射の領域では透過が起きない
  let eta = 1.0 / etaRel;
  let sin2 = eta * eta * max(0.0, 1.0 - dotVH * dotVH);
  t.fr = select(schlick(min(dotVH, 1.0), eta), 1.0, sin2 > 1.0);

  let dn = dotIH + dotVH / etaRel;
  t.denom = max(dn * dn, 1e-12);
  t.h = h;
  t.valid = true;
  return t;
}

/// 誘電体の BSDF * |cos(theta_i)|
fn dielectricEvalCos(hit: Hit, rayDir: vec3f, wi: vec3f) -> f32 {
  let t = dielectricTerms(hit, rayDir, wi);
  if (!t.valid) {
    return 0.0;
  }
  let a = ggxAlpha(hit.mat.roughness);
  let v = -rayDir;
  let cosO = dot(hit.normal, v);
  let cosI = dot(hit.normal, wi);
  let d = ggxD(dot(hit.normal, t.h), a);
  let g2 = ggxG2(cosO, abs(cosI), a);
  if (!t.isReflect) {
    // 透過方向は直線の影レイが必ずガラス自身に遮られるので、光源サンプリングの
    // 担当外にする。BSDF サンプリング側が重み 1 で受け持つ
    return 0.0;
  }
  return d * g2 * t.fr / (4.0 * cosO);
}

/// dielectricEvalCos と同じ方向に対する立体角 pdf
fn dielectricPdf(hit: Hit, rayDir: vec3f, wi: vec3f) -> f32 {
  let t = dielectricTerms(hit, rayDir, wi);
  if (!t.valid) {
    return 0.0;
  }
  let a = ggxAlpha(hit.mat.roughness);
  let v = -rayDir;
  let cosO = dot(hit.normal, v);
  let dotVH = dot(v, t.h);
  // 可視法線分布の密度
  let dv = ggxG1(cosO, a) * dotVH * ggxD(dot(hit.normal, t.h), a) / cosO;
  if (!t.isReflect) {
    return 0.0;
  }
  return t.fr * dv / (4.0 * dotVH);
}

/// BRDF * cos(theta_i)。デルタ分布のマテリアルでは 0 を返す
fn bsdfEval(hit: Hit, rayDir: vec3f, wi: vec3f) -> vec3f {
  if (hit.mat.kind == MAT_DIELECTRIC) {
    return vec3f(dielectricEvalCos(hit, rayDir, wi));
  }
  if (hit.mat.kind == MAT_PHASE) {
    // 媒質にはコサイン項がないので位相関数そのもの
    return vec3f(hgPhase(dot(rayDir, wi), U.fogG));
  }
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
    let g2 = ggxG2(nDotV, cosI, a);
    let f = fresnelSchlick(max(dot(v, h), 0.0), hit.mat.albedo);
    // BRDF * cos(theta_i) = D * G2 * F / (4 * nDotV) (cos は約分されている)
    return f * (ggxD(dot(hit.normal, h), a) * g2 / (4.0 * nDotV));
  }
  return vec3f(0.0);
}

/// bsdfEval と同じ方向に対する立体角 pdf
fn bsdfPdfFor(hit: Hit, rayDir: vec3f, wi: vec3f) -> f32 {
  if (hit.mat.kind == MAT_DIELECTRIC) {
    return dielectricPdf(hit, rayDir, wi);
  }
  if (hit.mat.kind == MAT_PHASE) {
    return hgPhase(dot(rayDir, wi), U.fogG);
  }
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
/// BSDF サンプリング戦略の pdf。ガイディングを使っているときは混合分布の
/// pdf を返す。MIS の重みはここと trace 側で必ず同じ式を使うこと。
/// 片方だけ差し替えると重みの和が 1 にならず偏る
/// (VCM の MIS もこれを使うこと。BSDF の pdf のままにすると、ガイディングと
///  併用したときに戦略ごとの前提がずれて偏る)
fn samplingPdf(hit: Hit, rayDir: vec3f, wi: vec3f) -> f32 {
  let pb = bsdfPdfFor(hit, rayDir, wi);
  if (U.guide == 0u || hit.mat.kind != MAT_LAMBERT) {
    return pb;
  }
  let vox = guideVoxel(hit.p);
  if (!guideHasData(vox)) {
    return pb;
  }
  return (1.0 - GUIDE_MIX_C) * pb
    + GUIDE_MIX_C * guidePdfLocal(vox, toLocal(onb(hit.normal), wi));
}

fn misWeight(pA: f32, pB: f32) -> f32 {
  let a = pA * pA;
  let b = pB * pB;
  return a / (a + b);
}

/// 面光源を 1 つ一様に選んで 1 点サンプルし、放射照度を返す。
/// BRDF (拡散なら albedo / PI) は呼び出し側で掛ける。
///
/// ここを RIS や ReSTIR で賢くする案は 計測の結果として見送った。
/// 1 頂点あたりの光源サンプルを 1 本から 4 本に増やして「直接光の
/// サンプリングを完璧にしたときの上限」を測ったところ、bench/run.mjs の
/// 12 シーン / 1024 spp で relMSE の幾何平均が 1.07x にしかならなかった
/// (一番効いた enclosed でも 1.25x)。影レイを 4 倍払っての 1.07x なので、
/// 1 本のまま候補を選び直す RIS の取り分はそれ以下にしかならない。
/// このリポジトリのシーンは光源が 1〜2 個で、しかも quad を立体角について
/// 一様にサンプルしているため、既に取りこぼしが小さい。
/// 残っている誤差は間接光の側にある
/// 矩形を立体角について一様にサンプルするための下ごしらえ (Urena et al. 2013)。
/// 面上を一様に引く素朴な方法は、光源が大きく近いほど分散が増える。
/// 立体角で引けばその依存が消える
struct SphQuad {
  o: vec3f, xa: vec3f, ya: vec3f, za: vec3f,
  z0: f32, z0sq: f32,
  x0: f32, y0: f32, y0sq: f32,
  x1: f32, y1: f32, y1sq: f32,
  b0: f32, b1: f32, b0sq: f32, k: f32,
  solid: f32,
};

fn sphQuadInit(sPos: vec3f, ex: vec3f, ey: vec3f, o: vec3f) -> SphQuad {
  var q: SphQuad;
  q.o = o;
  let exl = length(ex);
  let eyl = length(ey);
  q.xa = ex / exl;
  q.ya = ey / eyl;
  q.za = cross(q.xa, q.ya);
  let d = sPos - o;
  q.z0 = dot(d, q.za);
  if (q.z0 > 0.0) {
    q.za = -q.za;
    q.z0 = -q.z0;
  }
  q.z0sq = q.z0 * q.z0;
  q.x0 = dot(d, q.xa);
  q.y0 = dot(d, q.ya);
  q.x1 = q.x0 + exl;
  q.y1 = q.y0 + eyl;
  q.y0sq = q.y0 * q.y0;
  q.y1sq = q.y1 * q.y1;
  let v00 = vec3f(q.x0, q.y0, q.z0);
  let v01 = vec3f(q.x0, q.y1, q.z0);
  let v10 = vec3f(q.x1, q.y0, q.z0);
  let v11 = vec3f(q.x1, q.y1, q.z0);
  let n0 = normalize(cross(v00, v10));
  let n1 = normalize(cross(v10, v11));
  let n2 = normalize(cross(v11, v01));
  let n3 = normalize(cross(v01, v00));
  let g0 = acos(clamp(-dot(n0, n1), -1.0, 1.0));
  let g1 = acos(clamp(-dot(n1, n2), -1.0, 1.0));
  let g2 = acos(clamp(-dot(n2, n3), -1.0, 1.0));
  let g3 = acos(clamp(-dot(n3, n0), -1.0, 1.0));
  q.b0 = n0.z;
  q.b1 = n2.z;
  q.b0sq = q.b0 * q.b0;
  q.k = 2.0 * PI - g2 - g3;
  q.solid = g0 + g1 - q.k;
  return q;
}

fn sphQuadSample(q: SphQuad, u: f32, v: f32) -> vec3f {
  let au = u * q.solid + q.k;
  let su = sin(au);
  let fu = (cos(au) * q.b0 - q.b1) / select(su, 1e-7, abs(su) < 1e-7);
  var cu = 1.0 / sqrt(fu * fu + q.b0sq) * select(-1.0, 1.0, fu > 0.0);
  cu = clamp(cu, -1.0, 1.0);
  var xu = -(cu * q.z0) / max(sqrt(1.0 - cu * cu), 1e-7);
  xu = clamp(xu, q.x0, q.x1);
  let d2 = xu * xu + q.z0sq;
  let h0 = q.y0 / sqrt(d2 + q.y0sq);
  let h1 = q.y1 / sqrt(d2 + q.y1sq);
  let hv = h0 + v * (h1 - h0);
  let hv2 = hv * hv;
  let yv = select(q.y1, (hv * sqrt(d2)) / sqrt(max(1.0 - hv2, 1e-7)), hv2 < 1.0 - 1e-6);
  return q.o + xu * q.xa + yv * q.ya + q.z0 * q.za;
}

/// 立体角サンプリングを使うか。VCM の MIS は面積測度の項を持っていて
/// そちらの差し替えが要るので、VCM のときは従来どおり面上で引く
fn useSolidAngle() -> bool {
  return U.vcm == 0u;
}

/// NEE でこの方向をサンプルする確率 (立体角について)。
///
/// **BSDF サンプリングで光源に当たったときの MIS 重みは必ずこれを使うこと。**
/// sampleDirectLight のサンプリング方法と食い違うと、重みの和が 1 にならず
/// 偏る。以前は呼び出し側で式を直接書いていて、立体角サンプリングを
/// 入れたときに traceSppm 側の 2 か所を更新し忘れていた
fn neePdfW(lightQuad: u32, viewFrom: vec3f, dist: f32, cosLight: f32, area: f32) -> f32 {
  let n = f32(lightSelectCount());
  if (useSolidAngle()) {
    let lq = quads[lightQuad];
    let sq = sphQuadInit(lq.q, lq.u, lq.v, viewFrom);
    if (sq.solid > 1e-5) {
      return 1.0 / (sq.solid * n);
    }
  }
  return dist * dist / (max(cosLight, 1e-6) * area * n);
}

fn sampleDirectLight(
  hit: Hit,
  rayDir: vec3f,
  u: vec2f,
  misOut: ptr<function, f32>,
  dirOut: ptr<function, vec3f>,
  useVcm: bool,
  dVCMc: f32,
  dVCc: f32,
) -> vec3f {
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
    let tr = fogTransmittance(hit.p, wi, 1e30);
    var w = 1.0;
    if (U.mis != 0u) {
      w = misWeight(pL, samplingPdf(hit, rayDir, wi));
    }
    *misOut = w;
    *dirOut = wi;
    return envColor(wi) * fcos * tr * w / pL;
  }

  let light = quads[indices[pick]];

  // 光源をサンプルする。立体角について一様に引けるならそちらを使う
  let sq = sphQuadInit(light.q, light.u, light.v, hit.p);
  let useSA = useSolidAngle() && sq.solid > 1e-5;
  var onLight = light.q + light.u * u.x + light.v * u.y;
  if (useSA) {
    onLight = sphQuadSample(sq, u.x, u.y);
  }
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
  let trq = fogTransmittance(hit.p, wi, dist);

  // 立体角で引いたなら pdf はそのまま 1 / (光源の選択数 * 立体角)。
  // 面上で引いたときは面積の pdf を立体角に変換する
  var pL = dist2 / (cosLight * area * f32(n));
  if (useSA) {
    pL = 1.0 / (sq.solid * f32(n));
  }

  // MIS。BSDF サンプリングでも作りやすい方向ほど寄与を下げる
  var weight = 1.0;
  if (useVcm) {
    // 接続戦略も同じ経路を作れるので、3 戦略で和が 1 になる形にする。
    // power heuristic の 2 戦略版と混ぜると和が 1 を超えて二重計上になる
    let cosToLight = abs(dot(hit.normal, wi));
    let emissionPdfW = emissionPdfDir(ln / area, -wi) / (area * f32(n));
    let wLight = samplingPdf(hit, rayDir, wi) / pL;
    let wCamera = emissionPdfW * cosToLight / (pL * cosLight)
      * (etaVcm() + dVCMc + dVCc * samplingPdf(hit, -wi, -rayDir));
    weight = 1.0 / (wLight + 1.0 + wCamera);
  } else if (U.mis != 0u) {
    weight = misWeight(pL, samplingPdf(hit, rayDir, wi));
  }
  *misOut = weight;
  *dirOut = wi;
  return light.mat.emission * fcos * trq * weight / pL;
}

// ---------------------------------------------------------------- material
fn schlick(cosine: f32, refIdx: f32) -> f32 {
  var r0 = (1.0 - refIdx) / (1.0 + refIdx);
  r0 = r0 * r0;
  return r0 + (1.0 - r0) * pow(1.0 - cosine, 5.0);
}

/// 散乱方向とアルベドを返す。false なら吸収 (打ち切り)
/// radianceTransport は「放射輝度を運んでいるか」。カメラ側の経路なら true、
/// 光子 (仕事率) を撒く側なら false。屈折の扱いだけが変わる
fn scatter(
  ray: Ray,
  hit: Hit,
  u: vec2f,
  attenuation: ptr<function, vec3f>,
  scattered: ptr<function, Ray>,
  radianceTransport: bool,
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

  if (m.kind == MAT_PHASE) {
    let dir = sampleHg(ray.dir, U.fogG, u);
    // 位相関数からサンプルしているので f / pdf = 1
    *attenuation = vec3f(1.0);
    *scattered = Ray(hit.p, dir);
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
    // f * cos / pdf = F * G2 / G1(v)。eval 側と同じ G2 を使うこと
    let nDotV = dot(hit.normal, v);
    *attenuation = fresnelSchlick(max(dot(v, h), 0.0), m.albedo)
      * (ggxG2(nDotV, cosI, a) / max(ggxG1(nDotV, a), 1e-6));
    *scattered = Ray(hit.p + hit.normal * 1e-4, dir);
    return true;
  }

  // MAT_DIELECTRIC
  let eta = select(m.ior, 1.0 / m.ior, hit.frontFace);  // ni / nt
  let a = ggxAlpha(m.roughness);

  // 粗さがあればマイクロファセット法線を引く。0 なら幾何法線そのままで完全鏡面
  var h = hit.normal;
  if (m.roughness > 0.0) {
    let basis = onb(hit.normal);
    let vv = -ray.dir;
    let vl = vec3f(dot(vv, basis[0]), dot(vv, basis[1]), dot(vv, basis[2]));
    if (vl.z <= 0.0) {
      return false;
    }
    h = basis * sampleGgxVndf(vl, a, u.x, u.y);
  }

  let cosVH = min(dot(-ray.dir, h), 1.0);
  let sin2 = eta * eta * max(0.0, 1.0 - cosVH * cosVH);
  var dir: vec3f;
  var refracted = false;
  if (sin2 > 1.0 || schlick(cosVH, eta) > rand()) {
    dir = reflect(ray.dir, h);
    if (dot(dir, hit.normal) <= 0.0) {
      // マイクロファセットの裏に回った分は捨てる
      return false;
    }
  } else {
    dir = refract(ray.dir, h, eta);
    if (dot(dir, hit.normal) >= 0.0) {
      return false;
    }
    refracted = true;
  }
  dir = normalize(dir);

  // Fresnel は反射/屈折の確率的な選択で消化済みなので、残る重みは G2 / G1(v) = G1(l)
  var weight = vec3f(1.0);
  if (m.roughness > 0.0) {
    let cO = dot(hit.normal, -ray.dir);
    let cI = abs(dot(hit.normal, dir));
    weight = vec3f(ggxG2(cO, cI, a) / max(ggxG1(cO, a), 1e-6));
  }
  // 屈折は放射輝度を保存しない。媒質が変わると立体角が圧縮され、
  // 保存量は L ではなく L / eta^2 になる (Veach 5.2 の非対称性)。
  // 放射輝度を運ぶ側ではこの分を掛ける必要があり、仕事率を運ぶ光子側では
  // 掛けてはいけない。
  //
  // パストレースだけなら、入るときの eta^2 と出るときの 1/eta^2 が
  // 経路の中で必ず対になって打ち消し合うので、抜けていても絵は合う。
  // SPPM のようにカメラ側と光源側で経路を分ける方式で初めて露呈する
  if (refracted && radianceTransport) {
    weight = weight * (eta * eta);
  }

  // 内側から当たったなら、その区間ぶんだけ媒質を通ってきている (Beer-Lambert)
  if (!hit.frontFace) {
    weight = weight * pow(max(m.albedo, vec3f(1e-4)), vec3f(hit.t));
  }

  *attenuation = weight;
  *scattered = Ray(hit.p + sign(dot(dir, hit.normal)) * hit.normal * 1e-4, dir);
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
  /// VCM の接続の内訳。効かない理由を推測で決めないための計測用。
  /// x: 生存判定を通った候補の割合 / y: 遮蔽を抜けた割合 / z: 寄与の輝度
  connStat: vec3f,
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
  if (U.debugMode == 8u) {
    // 画面全体を単色で塗る。赤 = 集光している場所に届いた光子の割合
    let tot = f32(atomicLoad(&grid[statOff()]));
    let hit = f32(atomicLoad(&grid[statOff() + 1u]));
    let n = f32(atomicLoad(&grid[statOff() + 2u]));
    let sm = f32(atomicLoad(&grid[statOff() + 3u]));
    let s2 = f32(atomicLoad(&grid[statOff() + 4u]));
    let mean = select(0.0, sm / n, n > 0.0);
    let varm = max(select(0.0, s2 / n - mean * mean, n > 0.0), 0.0);
    // 赤: 有効な光子の割合 / 緑: 集光で見つかる光子数の平均 (1/256 に潰す)
    // 青: そのばらつき (変動係数、1/4 に潰す)
    return vec3f(
      select(0.0, hit / tot, tot > 0.0),
      mean / 256.0,
      // 青: 光子ごとの寄与の変動係数 (1/8 に潰す)
      select(0.0,
        f32(atomicLoad(&grid[statOff() + 6u]))
          / (100.0 * f32(max(atomicLoad(&grid[statOff() + 5u]), 1u))),
        true) / 8.0,
    );
  }
  if (U.debugMode == 7u) {
    // 赤: 枠に今フレームの頂点が入っていた割合
    // 緑: そのうち遮蔽を抜けた割合
    // 青: 実際の寄与 (見やすいように潰してある)
    return vec3f(a.connStat.x, a.connStat.y, a.connStat.z / (a.connStat.z + 0.01));
  }
  return vec3f(0.0);
}

// ---------------------------------------------------------------- trace
fn trace(primary: Ray, pixelTarget: f32, firstHit: ptr<function, vec4f>, aov: ptr<function, Aov>) -> vec3f {
  var ray = primary;
  var throughput = vec3f(1.0);
  var radiance = vec3f(0.0);
  // 直前の頂点で BSDF サンプリングした方向の立体角 pdf。
  // 負ならカメラレイかスペキュラ反射で、光源サンプリングでは作れない方向なので重みは 1
  var bsdfPdf = -1.0;
  let useNee = U.nee != 0u && lightSelectCount() > 0u;
  let useEnvNee = U.nee != 0u && envIsActive();
  // VCM の接続戦略用。光源側と対になる漸化式をカメラ側でも回す。
  // 初期値が 0 なのは「光源側の経路をカメラに直接つなぐ」戦略 (light tracing)
  // を実装していないため。使っていない戦略は重みから外す
  let useVcm = U.vcm != 0u && U.photonCount > 0u;
  // カメラ側 dVCM の初期値。「光源側の経路の本数 / カメラの立体角 pdf」で、
  // カメラの pdf は 1 / (画素が張る面積 * cos^3)。
  //
  // ここを 0 にしていたのが merging を入れたときの +2.30% の原因だった。
  // 「light tracing を実装していないから、その戦略の項は外してよい」と
  // 考えたが、この項は戦略の有無だけでなく、光源側と camera 側の量を
  // 同じ尺度に揃える役目も持っている。merging の正規化が光源側の経路の
  // 本数で割っているので、そこと辻褄が合わなくなる。
  // 接続だけのときは影響が出ず (+0.62%)、merging を入れて初めて露呈した
  var dVCMc = 0.0;
  // merging を使うときだけ入れる。この項は light tracing の戦略ぶんであると
  // 同時に、光源側と camera 側の量を同じ尺度に揃える役目も持っている。
  // その尺度合わせが必要なのは merging の正規化が光源側の経路の本数で
  // 割っているからで、接続だけの構成では逆に入れると偏る (-6.6%)
  if (useVcm && VCM_MERGE) {
    let pixArea = 4.0 * U.tanHalfFov * U.tanHalfFov * U.aspect
      / (f32(U.width) * f32(U.height));
    let cosT = max(abs(dot(primary.dir, normalize(U.camW))), 1e-6);
    dVCMc = f32(U.photonCount) * pixArea * cosT * cosT * cosT;
  }
  var dVCc = 0.0;
  var dVMc = 0.0;
  let useGuide = U.guide != 0u;
  let useEars = U.ears != 0u;
  // 経路の終わりに書き戻す仕組みはガイディングと ADRRS で共通
  let useRec = useGuide || useEars;
  var mdir = vec3f(0.0);
  // ガイディングの教師データ。各頂点で「サンプルした方向から実際に
  // どれだけの放射輝度が返ってきたか」を、経路を最後までたどってから
  // 書き戻す。NEE の寄与を教師にすると「NEE が既に拾えている方向」を
  // 学習するだけで冗長になり、分散が減らない
  var gN = 0u;
  var gPos = array<vec3f, GUIDE_REC>();
  /// 面の法線を +z とする局所座標での方向
  var gDir = array<vec3f, GUIDE_REC>();
  /// 散乱後のスループットの輝度
  var gThr = array<f32, GUIDE_REC>();
  /// その方向をサンプルした立体角 pdf。ガイディングは L / pdf を貯めるので要る
  var gPdf = array<f32, GUIDE_REC>();
  var gRad = array<f32, GUIDE_REC>();

  for (var depth = 0u; depth < U.maxBounces; depth = depth + 1u) {
    var hit: Hit;
    let hitSurface = hitScene(ray, 1e-3, 1e30, &hit);

    // 表面に届く前に媒質で散乱するかを先に決める
    if (fogActive()) {
      let tSurf = select(1e30, hit.t, hitSurface);
      let r = fogRange(ray.origin, ray.dir, tSurf);
      if (r.y > r.x) {
        let d = -log(max(1.0 - rand(), 1e-9)) / sigmaT();
        if (d < r.y - r.x) {
          // 媒質内で散乱した。透過率と pdf が約分され、残るのは単散乱アルベド
          throughput = throughput * (U.sigmaS / sigmaT());

          var mhit: Hit;
          mhit.t = r.x + d;
          mhit.p = ray.origin + mhit.t * ray.dir;
          mhit.normal = vec3f(0.0, 1.0, 0.0);  // 位相関数では使わない
          mhit.frontFace = true;
          mhit.lightArea = 0.0;
          mhit.mat = Material(vec3f(0.0), 0.0, vec3f(0.0), 1.0, MAT_PHASE);

          if (useNee) {
            var mw = 1.0;
            radiance = radiance + throughput
              * sampleDirectLight(mhit, ray.dir, sample2d(2u + depth * 2u, depth < QMC_DEPTH), &mw, &mdir,
          useVcm, dVCMc, dVCc);
            if (depth == 0u) {
              (*aov).misWeight = mw;
            }
          }

          let inDir = ray.dir;
          let outDir = sampleHg(inDir, U.fogG, sample2d(3u + depth * 2u, depth < QMC_DEPTH));
          bsdfPdf = max(hgPhase(dot(inDir, outDir), U.fogG), 1e-8);
          ray = Ray(mhit.p, outDir);
          (*aov).bounces = f32(depth + 1u);

          // ロシアンルーレット (4 バウンス目以降)
          // VCM のときはロシアンルーレットを使わない。打ち切る確率を pdf に
    // 入れないと戦略ごとに前提がずれて偏る。逆向きの pdf にも入れる必要が
    // あって厄介なので、経路長の上限だけで打ち切る
    if (!useVcm && depth >= RR_START) {
            // 生存確率は 1 で頭打ちにする。1 を超えたまま割ると、経路は必ず
              // 生き残るのにスループットだけが減ってエネルギーを失う。
              // ガイディングは bsdfEval / pdf が 1 を超えやすいので踏みやすい
              let q = min(max(throughput.r, max(throughput.g, throughput.b)), 1.0);
            if (rand() > q) {
              break;
            }
            throughput = throughput / max(q, 1e-4);
          }
          continue;
        }
      }
    }

    if (!hitSurface) {
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

    // 面積についての pdf に直す補正。光源側と同じ形
    let cosInC = abs(dot(hit.normal, ray.dir));
    if (useVcm && cosInC > 1e-6) {
      dVCMc = dVCMc * hit.t * hit.t / cosInC;
      dVCc = dVCc / cosInC;
      dVMc = dVMc / cosInC;
    }

    // 光源に当たったときの放射。NEE と重複するぶんを MIS 重みで削る
    var weight = 1.0;
    if (useVcm && bsdfPdf > 0.0 && isEmissive(hit.mat)) {
      // 3 戦略版。dVCMc が 0 のカメラ 1 頂点目では重み 1 になり、
      // 「カメラから光源が直接見えている」場合に正しく全部拾える
      let nl = f32(lightSelectCount());
      let directPdfA = 1.0 / (hit.lightArea * nl);
      let emissionPdfW = emissionPdfDir(hit.normal, -ray.dir) / (hit.lightArea * nl);
      weight = 1.0 / (1.0 + directPdfA * dVCMc + emissionPdfW * dVCc);
    } else if (useNee && bsdfPdf > 0.0 && isEmissive(hit.mat)) {
      if (U.mis != 0u) {
        // この方向を光源サンプリングで作る場合の pdf。sampleDirectLight と同じ式
        let cosLight = abs(dot(hit.normal, ray.dir));
        let pL = neePdfW(hit.lightQuad, ray.origin, hit.t, cosLight, hit.lightArea);
        weight = misWeight(bsdfPdf, pL);
      } else {
        // MIS なしなら NEE 側に完全に任せる (二重計上の防止)
        weight = 0.0;
      }
    }
    radiance = radiance + throughput * hit.mat.emission * weight;

    // pdf を評価できる面だけ光源を直接サンプルする。デルタ面は BSDF サンプリングに任せる
    if (useNee && isDiffuseLike(hit.mat)) {
      var mw = 1.0;
      let nee = throughput
        * sampleDirectLight(hit, ray.dir, sample2d(2u + depth * 2u, depth < QMC_DEPTH), &mw, &mdir,
          useVcm, dVCMc, dVCc);
      radiance = radiance + nee;

      if (depth == 0u) {
        (*aov).misWeight = mw;
      }
    }

    // 光源側の経路の頂点とつなぐ。隙間の向こうまで届いた光源側の頂点を
    // 仮想的な光源として使い回せるので、NEE がほぼ効かないシーンで効く
    if (useVcm && isDiffuseLike(hit.mat)) {
      var st = vec3f(0.0);
      radiance = radiance + throughput * connectToLightVertex(hit, ray.dir, dVCMc, dVCc, &st);
      if (VCM_MERGE) {
        radiance = radiance + throughput * mergeAtVertex(hit, ray.dir, dVCMc, dVCc, dVMc);
      }
      if (depth == 0u) {
        (*aov).connStat = st;
      }
    }

    var attenuation: vec3f;
    var scattered: Ray;
    // 学習した分布と BSDF の混合から方向を引く。混ぜてあるので pdf が 0 の
    // 方向はできない = 不偏性が保たれる。拡散面だけを対象にする
    var guidedPdf = -1.0;
    let guideHere = useGuide && hit.mat.kind == MAT_LAMBERT
      && guideHasData(guideVoxel(hit.p));
    if (guideHere) {
      let vox = guideVoxel(hit.p);
      // 低食い違い列を通す。ここを rand() にしていると、ガイドを有効に
      // した瞬間に 1 バウンス目の層化が失われる (ガイドの利得と相殺していた)
      let gu = sample2d(3u + depth * 2u, depth < QMC_DEPTH);
      let gb = onb(hit.normal);
      var wl: vec3f;
      if (rand() < GUIDE_MIX_C) {
        // ビンの選択に層化した次元を割り当てる。ビン内は素の乱数で足りる
        wl = guideSampleLocal(vox, gu.x, vec2f(gu.y, rand()));
      } else {
        wl = cosineHemisphere(gu);
      }
      let wi = normalize(gb * wl);
      let cosI = dot(hit.normal, wi);
      // 分布が半球に閉じているのでここへは落ちないはずだが、数値誤差の保険
      if (cosI <= 1e-6) {
        break;
      }
      let pd = (1.0 - GUIDE_MIX_C) * cosI * INV_PI + GUIDE_MIX_C * guidePdfLocal(vox, wl);
      if (pd <= 0.0) {
        break;
      }
      attenuation = bsdfEval(hit, ray.dir, wi) / pd;
      scattered = Ray(hit.p + hit.normal * 1e-4, wi);
      guidedPdf = pd;
    }
    if (!guideHere) {
      if (!scatter(ray, hit, sample2d(3u + depth * 2u, depth < QMC_DEPTH), &attenuation, &scattered, true)) {
        break;
      }
    }
    // 次の頂点で MIS 重みを計算するために pdf を持ち回る。
    // デルタ分布のマテリアルは光源サンプリングで作れない方向なので負にしておく
    if (guidedPdf > 0.0) {
      // ガイドしたときは MIS もこの pdf を使う。BSDF の pdf のままにすると
      // 重みがずれて偏る
      bsdfPdf = guidedPdf;
    } else if (isDiffuseLike(hit.mat)) {
      let pv = bsdfPdfFor(hit, ray.dir, scattered.dir);
      // pdf が 0 の方向 (誘電体の透過ローブ) は光源サンプリングで作れないので
      // 負にしておき、MIS 重み 1 で扱う
      bsdfPdf = select(-1.0, max(pv, 1e-8), pv > 0.0);
    } else {
      bsdfPdf = -1.0;
    }
    if (depth == 0u) {
      (*aov).bsdfPdf = max(bsdfPdf, 0.0);
    }

    // VCM の漸化式。光源側とまったく同じ形
    if (useVcm) {
      let pF = samplingPdf(hit, ray.dir, scattered.dir);
      if (pF > 0.0) {
        let pR = samplingPdf(hit, -scattered.dir, -ray.dir);
        let cosOut = abs(dot(hit.normal, scattered.dir));
        let eta = etaVcm();
        dVCc = (cosOut / pF) * (dVCc * pR + dVCMc + eta);
        dVMc = (cosOut / pF) * (dVMc * pR + dVCMc / eta + 1.0);
        dVCMc = 1.0 / pF;
      } else {
        // デルタ的な散乱。光源側とまったく同じ扱いにする。
        // 全部 0 にすると MIS の重みが 1 に張り付いて過剰計上になる
        let cosOut = abs(dot(hit.normal, scattered.dir));
        dVCc = dVCc * cosOut;
        dVMc = dVMc * cosOut;
        dVCMc = 0.0;
      }
    }

    throughput = throughput * attenuation;
    // この頂点でサンプルした方向を覚えておく。返ってきた放射輝度は
    // 経路が終わってから (最終値 - ここまでの値) / スループット で出る
    if (useRec && gN < GUIDE_REC && hit.mat.kind == MAT_LAMBERT) {
      gPos[gN] = hit.p;
      // 局所座標に移してから覚える。法線を持ち回らずに済む
      gDir[gN] = toLocal(onb(hit.normal), scattered.dir);
      gThr[gN] = max(luminanceOf(throughput), 1e-6);
      // MAT_LAMBERT はデルタ分布ではないので bsdfPdf は必ず正
      gPdf[gN] = max(bsdfPdf, 1e-6);
      gRad[gN] = luminanceOf(radiance);
      gN = gN + 1u;
    }
    ray = scattered;

    // ロシアンルーレット (4 バウンス目以降)
    if (!useVcm && depth >= RR_START) {
      // 生存確率は 1 で頭打ちにする。1 を超えたまま割ると、経路は必ず
      // 生き残るのにスループットだけが減ってエネルギーを失う。
      // ガイディングは bsdfEval / pdf が 1 を超えやすいので踏みやすい
      var q = min(max(throughput.r, max(throughput.g, throughput.b)), 1.0);
      if (useEars) {
        // 「この先どれだけ返ってきそうか」を見て決め直す。
        // キャッシュが空のうちは従来どおりスループットだけで決める
        let adr = earsSurvival(max(throughput.r, max(throughput.g, throughput.b)),
          guideVoxel(hit.p), pixelTarget);
        if (adr >= 0.0) {
          // 生存確率を「上げる」方向にだけ使う。
          //
          // 下げる方向にも使うと、この先が暗いと推定した場所で経路を余計に
          // 切ることになり、スループットだけで決めていた頃より荒れる。
          // 実測でも spheres 0.94x / glass 0.82x と簡単なシーンで悪化した。
          // 上げる方向だけなら、間接光が奥にあるシーンで経路を生かしつつ、
          // それ以外では従来とまったく同じ挙動になる
          q = max(q, max(adr, EARS_MIN_Q));
        }
      }
      if (rand() > q) {
        break;
      }
      throughput = throughput / max(q, 1e-4);
    }
  }
  // 覚えておいた各頂点に、その方向から返ってきた放射輝度を書き戻す
  if (useRec) {
    let lf = luminanceOf(radiance);
    for (var i = 0u; i < gN; i = i + 1u) {
      // その方向から実際に返ってきた入射放射輝度の推定値
      let li = max(lf - gRad[i], 0.0) / gThr[i];
      if (useGuide) {
        // ガイディングが欲しいのは「ビンの立体角で積分した放射輝度」なので
        // pdf で割る。割らずに L のまま貯めると、方向は混合分布 p_mix から
        // 引いているため学習されるのは L ではなく p_mix * L になる。
        // 法線近傍が過大評価されるうえ、一度多く撒いた方向がさらに増える
        // 正のフィードバックが掛かる
        guideRecord(gPos[i], gDir[i], li / gPdf[i]);
      }
      if (useEars) {
        // ADRRS が欲しいのは素の期待放射輝度なので、こちらは割らない
        earsRecord(gPos[i], li);
      }
    }
  }
  return radiance;
}

@compute @workgroup_size(8, 8, 1)
fn sppmMain(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= U.width || gid.y >= U.height) {
    return;
  }
  let pixel = gid.y * U.width + gid.x;
  let o = pixel * 4u;
  rngState = pcg(pixel + pcg(U.frameIndex * 6271u + U.salt * 0x9e3779b9u + 1u));
  pixelSeed = pcg(pixel * 26699u + U.salt * 0x85ebca6bu + 1u);
  sampleIdx = U.samplesBefore;

  // ping-pong は使わず、同じバッファを読み書きして状態を持ち越す
  var flux = vec3f(0.0);
  var nAcc = 0.0;
  var radius = U.radius0;
  var direct = vec3f(0.0);
  var count = 0.0;
  if (U.samplesBefore > 0u) {
    let s2 = histWrite[o + 2u];
    flux = s2.rgb;
    nAcc = s2.w;
    radius = max(histWrite[o + 3u].x, 1e-5);
    let s0 = histWrite[o];
    direct = s0.rgb;
    count = s0.w;
  }

  let jitter = sample2d(0u, true);
  let px = (f32(gid.x) + jitter.x) / f32(U.width) * 2.0 - 1.0;
  let py = 1.0 - (f32(gid.y) + jitter.y) / f32(U.height) * 2.0;

  var newFlux = vec3f(0.0);
  var m = 0.0;
  var aov = Aov(vec3f(0.0), vec3f(0.0), 0.0, 0.0, 0.0, 0.0, vec3f(0.0));
  let d = traceSppm(makeRay(px, py, sample2d(1u, true)), radius, &newFlux, &m, &aov);

  // デバッグ表示。main しか描いていなかったので、SPPM を有効にすると
  // どの表示も普通の絵のままだった。光子の統計 (debug 8) に至っては
  // 「SPPM のときしか値が埋まらないのに SPPM では見られない」状態だった。
  // pdf と MIS 重み (4, 5) と VCM の接続 (7) はこの経路にない概念なので 0 になる
  if (U.debugMode != 0u) {
    histWrite[o] = vec4f(aovColor(aov), 1.0);
    histWrite[o + 2u] = vec4f(0.0, 0.0, 0.0, 0.0);
    histWrite[o + 3u] = vec4f(radius, 0.0, 0.0, 0.0);
    return;
  }

  // SPPM の半径・フラックス更新 (Hachisuka & Jensen)
  if (nAcc + m > 0.0) {
    let ratio = (nAcc + ALPHA * m) / (nAcc + m);
    radius = radius * sqrt(ratio);
    flux = (flux + newFlux) * ratio;
    nAcc = nAcc + ALPHA * m;
  }

  // firefly の押し下げ。PT 側と同じで、この画素のこれまでの平均から作った
  // 閾値を超えた寄与を落とす。光子で求めている間接光は密度推定なので
  // 元から滑らかで、跳ねるのは NEE と 1 本伸ばした BSDF サンプリングの側
  var dc = select(vec3f(0.0), d, accumulable(d));
  if (FIREFLY_K > 0.0 && count > 0.0) {
    let thr = (luminanceOf(direct) / count) * FIREFLY_K * sqrt(count);
    let l = luminanceOf(dc);
    if (thr > 0.0 && l > thr) {
      dc = dc * (thr / l);
    }
  }
  histWrite[o] = vec4f(direct + dc, count + 1.0);
  histWrite[o + 2u] = vec4f(select(vec3f(0.0), flux, accumulable(flux)), nAcc);

  // デノイザの手がかり (法線・距離) は画素中心のレイで取り直す。
  // jitter 済みの px, py をそのまま使うとサブピクセルごとにガイドが
  // 揺れてフィルタの重みが安定しない
  var guide = vec3f(0.0, 0.0, 0.0);
  if (U.denoise != 0u) {
    let pxC = (f32(gid.x) + 0.5) / f32(U.width) * 2.0 - 1.0;
    let pyC = 1.0 - (f32(gid.y) + 0.5) / f32(U.height) * 2.0;
    guide = guideFor(pxC, pyC);
  }
  histWrite[o + 3u] = vec4f(radius, guide.x, guide.y, guide.z);
}

// -------------------------------------------------------------- SPPM ギャザー
/// 半径 r 以内の光子を集め、BRDF を掛けたフラックスの和と個数を返す。
///
/// 光子がセルのどの枠に入るかは atomicAdd の順で決まるので、同じ設定でも
/// 走らせるたびに足す順番が変わる。浮動小数の加算は結合則を満たさないので、
/// 固定 seed にしても SPPM の絵は完全には再現しない。同じコード・同じ設定を
/// 2 回走らせて relMSE が最大 1.8% ぶれるのを実測している (enclosed)。
/// A/B を取るときはこの幅より小さい差を読まないこと。
/// パストレース側 (main) はビット単位で再現する
fn gatherPhotons(hit: Hit, rayDir: vec3f, r: f32, found: ptr<function, f32>) -> vec3f {
  var sum = vec3f(0.0);
  var m = 0.0;
  // 放出方向の学習用。サンプラの列を乱さないよう、rand() ではなく
  // 位置とフレーム番号から作った別のハッシュを使う
  var creditBin = -1.0;
  var seen = 0u;
  // 1 回の集光の中で、光子ごとの寄与がどれだけばらついているか。
  // 同じ数の光子を集めても、運ぶエネルギーがばらけていれば絵は荒れる
  var powSum = 0.0;
  var powSq = 0.0;
  let seed = pcg(bitcast<u32>(hit.p.x) ^ bitcast<u32>(hit.p.z) ^ U.frameIndex);
  let r2 = r * r;
  let c = gridCoord(hit.p);
  // ここが「カメラが集光している場所」。印を付けておき、次のフレームの
  // 光子がここへ届いたかを数えて、無駄になっている割合を出す
  atomicStore(&grid[markOff() + gridHash(c.x, c.y, c.z)], 1u);
  for (var dz = -1; dz <= 1; dz = dz + 1) {
    for (var dy = -1; dy <= 1; dy = dy + 1) {
      for (var dx = -1; dx <= 1; dx = dx + 1) {
        let cell = gridHash(c.x + dx, c.y + dy, c.z + dz);
        let cnt = min(atomicLoad(&grid[cell]), GRID_CAP);
        for (var k = 0u; k < cnt; k = k + 1u) {
          let pi = atomicLoad(&grid[U.gridCells + cell * GRID_CAP + k]);
          let d = photons[pi * VTX_SLOTS + 0u].xyz - hit.p;
          if (dot(d, d) > r2) {
            continue;
          }
          // 光子が「同じ面」に載っているかを確かめる。
          //
          // 距離だけで拾うと、半径の中に入っている別の面 (薄い衝立の裏側、
          // 隅で直交する壁、光源を囲う箱の外側) の光子まで集めてしまう。
          // 明るい面の光子が暗い面へ漏れるので、桁違いに明るくなる。
          // maze と enclosed で衝立や箱の輪郭に沿って 500 倍・70 倍の
          // 明るさが出ていたのはこれ。法線は光子を撒くときに書いてある
          let pn = photons[pi * VTX_SLOTS + 3u].xyz;
          if (dot(pn, hit.normal) < GATHER_COS) {
            continue;
          }
          // 向きが揃っていても、平行な 2 面が半径より近いと漏れる。
          // 接平面からの距離でも切る
          if (abs(dot(d, hit.normal)) > GATHER_PLANE * r) {
            continue;
          }
          let wi = -photons[pi * VTX_SLOTS + 1u].xyz;
          let cosI = dot(hit.normal, wi);
          if (cosI <= 1e-4) {
            continue;
          }
          // 実際にその光子が見えるかを確かめる。半径ぶんの短い影レイなので
          // BVH の走査もすぐ終わる。
          //
          // 両端を法線方向へ逃がすのが肝。曲面 (球) では同じ面の 2 点を結ぶ
          // 線分が内側を通るので、逃がさないと正当な光子まで自己遮蔽で
          // 落ちる (spheres で 2 倍悪化した)。逃がす量は集光半径に比例させる
          if (GATHER_VISIBILITY) {
            let eps = max(1e-4, r * 0.04);
            let o = hit.p + hit.normal * eps;
            let seg = (photons[pi * VTX_SLOTS + 0u].xyz + pn * eps) - o;
            let dist = length(seg);
            if (dist > 1e-6 && occluded(o, seg / dist, dist - eps)) {
              continue;
            }
          }
          // bsdfEval は f * cos を返すので、余弦で割って裸の BRDF に戻す
          let c1 = (bsdfEval(hit, rayDir, wi) / cosI) * photons[pi * VTX_SLOTS + 2u].xyz;
          sum = sum + c1;
          m = m + 1.0;
          let l1 = luminanceOf(c1);
          powSum = powSum + l1;
          powSq = powSq + l1 * l1;

          // 使われた光子の放出方向を 1 個だけ選んで覚えておく (貯留サンプリング)。
          // ここで毎回 atomicAdd すると 1 画素あたり数百回になり、512 個の
          // アドレスに集中して桁違いに遅くなる。1 回の集光につき 1 個で足りる
          let bin = photons[pi * VTX_SLOTS + 2u].w;
          if (bin >= 0.0) {
            seen = seen + 1u;
            if (pcg(u32(m) * 2654435761u + seed) % seen == 0u) {
              creditBin = bin;
            }
          }
        }
      }
    }
  }
  // 集光で見つかった光子数の分布。SPPM の分散を直接動かしているのはここ。
  // 平均だけでなくばらつきを見たいので 2 乗和も取る
  if ((seed & 63u) == 0u) {
    let mi = u32(m);
    atomicAdd(&grid[statOff() + 2u], 1u);
    atomicAdd(&grid[statOff() + 3u], mi);
    atomicAdd(&grid[statOff() + 4u], mi * mi);
    if (m > 1.0 && powSum > 0.0) {
      let mu = powSum / m;
      let v = max(powSq / m - mu * mu, 0.0);
      // 変動係数を 100 倍の固定小数で貯める (WGSL に f32 の atomic がない)
      atomicAdd(&grid[statOff() + 5u], 1u);
      atomicAdd(&grid[statOff() + 6u], u32(min(sqrt(v) / mu, 40.0) * 100.0));
    }
  }

  // 記録は間引いてよい。ヒストグラムは何フレームも貯め続けるので、
  // 1 フレームで密に集める必要はない
  if (creditBin >= 0.0 && f32(seed >> 22u) < CREDIT_RATE * 1024.0) {
    atomicAdd(&grid[histOff() + u32(creditBin)], 1u);
  }
  *found = m;
  return sum;
}

/// SPPM のカメラ側。最初にギャザーできる面まで辿り、直接光は NEE、
/// 間接光はそこで光子を集めて求める
fn traceSppm(primary: Ray, radius: f32, flux: ptr<function, vec3f>, found: ptr<function, f32>,
  aov: ptr<function, Aov>) -> vec3f {
  var ray = primary;
  var throughput = vec3f(1.0);
  var radiance = vec3f(0.0);
  var bsdfPdf = -1.0;
  let useNee = U.nee != 0u && lightSelectCount() > 0u;
  let useEnvNee = U.nee != 0u && envIsActive();

  for (var depth = 0u; depth < U.maxBounces; depth = depth + 1u) {
    var hit: Hit;
    if (!hitScene(ray, 1e-3, 1e30, &hit)) {
      var w = 1.0;
      if (useEnvNee && bsdfPdf > 0.0) {
        if (U.mis != 0u) {
          w = misWeight(bsdfPdf, envPdf(ray.dir) / f32(lightSelectCount()));
        } else {
          w = 0.0;
        }
      }
      return radiance + throughput * envColor(ray.dir) * w;
    }

    if (depth == 0u) {
      (*aov).normal = hit.normal;
      (*aov).albedo = hit.mat.albedo;
      (*aov).dist = hit.t;
    }
    (*aov).bounces = f32(depth + 1u);

    var we = 1.0;
    if (useNee && bsdfPdf > 0.0 && isEmissive(hit.mat)) {
      if (U.mis != 0u) {
        let cosLight = abs(dot(hit.normal, ray.dir));
        we = misWeight(bsdfPdf,
          neePdfW(hit.lightQuad, ray.origin, hit.t, cosLight, hit.lightArea));
      } else {
        we = 0.0;
      }
    }
    radiance = radiance + throughput * hit.mat.emission * we;

    if (gatherableMat(hit.mat)) {
      if (useNee) {
        var mw = 1.0;
        var mdir = vec3f(0.0);
        radiance = radiance + throughput
          * sampleDirectLight(hit, ray.dir, vec2f(rand(), rand()), &mw, &mdir, false, 0.0, 0.0);

        // MIS のもう一方の戦略。ここで経路を打ち切ってしまうと BSDF
        // サンプリング側の寄与が丸ごと消えるので、直接光のぶんだけ
        // 1 本伸ばして拾う。間接光は光子が受け持つので、光源に当たった
        // 場合だけ加算する
        var att: vec3f;
        var sc: Ray;
        if (scatter(ray, hit, vec2f(rand(), rand()), &att, &sc, true)) {
          let pv = bsdfPdfFor(hit, ray.dir, sc.dir);
          if (pv > 0.0) {
            var h2: Hit;
            if (hitScene(sc, 1e-3, 1e30, &h2)) {
              if (isEmissive(h2.mat)) {
                let cosL = max(abs(dot(h2.normal, sc.dir)), 1e-6);
                let pL = neePdfW(h2.lightQuad, sc.origin, h2.t, cosL, h2.lightArea);
                radiance = radiance + throughput * att * h2.mat.emission
                  * select(0.0, misWeight(pv, pL), U.mis != 0u);
              }
            } else if (useEnvNee) {
              radiance = radiance + throughput * att * envColor(sc.dir)
                * select(0.0, misWeight(pv, envPdf(sc.dir) / f32(lightSelectCount())), U.mis != 0u);
            } else {
              radiance = radiance + throughput * att * envColor(sc.dir);
            }
          }
        }
      }
      *flux = throughput * gatherPhotons(hit, ray.dir, radius, found);
      return radiance;
    }

    // 集光できない面は通り抜けて次を探す。ただし pdf を評価できる面
    // (光沢の GGX など) では NEE は有効なので行う。誘電体だけは影レイが
    // 必ずガラス自身に遮られるので飛ばさない。
    // NEE を行った場合は次の頂点の放射を MIS で削る必要があるので、
    // bsdfPdf を残す。行わなかった場合は負のままにして重み 1 で受け持つ
    let neeHere = useNee && isDiffuseLike(hit.mat) && hit.mat.kind != MAT_DIELECTRIC;
    if (neeHere) {
      var mw2 = 1.0;
      var md2 = vec3f(0.0);
      radiance = radiance + throughput
        * sampleDirectLight(hit, ray.dir, vec2f(rand(), rand()), &mw2, &md2, false, 0.0, 0.0);
    }

    var attenuation: vec3f;
    var scattered: Ray;
    if (!scatter(ray, hit, vec2f(rand(), rand()), &attenuation, &scattered, true)) {
      return radiance;
    }
    if (neeHere) {
      let pv = bsdfPdfFor(hit, ray.dir, scattered.dir);
      bsdfPdf = select(-1.0, max(pv, 1e-8), pv > 0.0);
    } else {
      bsdfPdf = -1.0;
    }
    throughput = throughput * attenuation;
    ray = scattered;
  }
  return radiance;
}

// ------------------------------------------------- VCM: 頂点どうしの接続
/// 保存した光源側の頂点を Hit の形に戻す。BSDF を評価するのに要る
fn lightVertexHit(base: u32) -> Hit {
  let n = photons[base + 3u];
  let a = photons[base + 4u];
  var m: Material;
  m.albedo = a.xyz;
  m.roughness = a.w;
  m.emission = vec3f(0.0);
  m.ior = 1.5;
  m.kind = u32(n.w / 65536.0);
  var h: Hit;
  h.t = 0.0;
  h.p = photons[base + 0u].xyz;
  h.normal = n.xyz;
  h.frontFace = true;
  h.mat = m;
  h.lightArea = 0.0;
  return h;
}

/// この枠に今フレームの頂点が入っているか。使われなかった枠には
/// 前フレームの内容が残るので、フレーム番号を一緒に入れて見分ける
fn lightVertexAlive(base: u32) -> bool {
  // 環境マップから撒いた光子は dVCM / dVC を計算していないので接続に使えない。
  // 放出方向のビンが負のものがそれ (面光源からのものは必ず 0 以上)
  if (photons[base + 2u].w < 0.0) {
    return false;
  }
  let v = photons[base + 3u].w;
  if (v <= 0.0) {
    return false;
  }
  let kind = floor(v / 65536.0);
  return u32(v - kind * 65536.0) == (U.frameIndex % 65536u);
}

/// merging を MIS に統合するか。正しく動くようになったが、接続のみの構成の
/// 15 倍遅い (267 ms 対 3893 ms) ので既定は false。等時間では接続のみが勝つ。
///
///   cornell  PT 比 +0.72%   (修正前 +2.30%)
///   ajar     SPPM 比 +2.80% (約 300 spp 時点。収束不足の可能性あり)
///
/// merging を MIS に統合するか。正しく動くようになったが、接続のみの構成の
/// 15 倍遅い (267 ms 対 3893 ms) ので既定は false。等時間では接続のみが勝つ。
///
///   cornell  PT 比 +0.72%   (修正前 +2.30%)
///   ajar     SPPM 比 +2.80% (約 300 spp 時点。収束不足の可能性あり)
const VCM_MERGE: bool = false;

/// VCM の接続は既定 off。効きは大きいが、まだ過剰計上が残っている。
///
/// 効き (4096 spp、上位 1% 除外の relMSE、素のパストレース比):
///   indirect 5.4x / cornell 2.2x / enclosed と ajar は横ばい
///
/// 残っている偏り (8192 spp での全体平均、PT 比):
///   cornell +0.36% / enclosed +0.67% / maze +2.90% / veach -0.32% / indirect -0.04%
///
/// cornell はガラス球の集光が落ちる赤い壁 (x = 0, y 2.2〜3.3, z 0.2〜1.3) に
/// 局所的に +3.4% 出る。そこは PT が 1024 / 4096 / 16384 spp で
/// 0.04052 / 0.04055 / 0.04058 と完全に安定しているので、PT の未収束では
/// なく VCM 側の過剰計上で間違いない。一方 maze の +2.90% は PT 自体が
/// まだ収束していないので、どちらが正しいか判断できない。
///
/// 切り分け済み (cornell の集光領域、PT の収束値は 0.04055):
///  - RIS の候補数を 1 / 4 / 8 と変えても偏りは動かない
///    (0.04190 / 0.04185 / 0.04188) → 接続先の選び方は無罪
///  - 光子数を 1 / 1/2 / 1/4 にしても動かない (+3.40% / +3.35% / +3.44%)
///    → 光源側の経路の本数による正規化も無罪
///  - スペキュラ散乱後の漸化式を SmallVCM に合わせても動かない
/// 接続の寄与だけを 0 にすると -14.50% になるので、接続は「埋めるべき量の
/// 約 1.23 倍」を出している。つまり残っているのは MIS の重みの式そのもので、
/// 3 戦略の重みの和が 1 になっていない。直すには重みの再導出が要る

/// 半径の縮み方。0 と 1 の間なら、反復とともに半径は 0 に、累積光子数は
/// 無限に増える。SPPM と VCM で同じ値を使う
const ALPHA: f32 = 0.7;

/// VCM の半径。SPPM の画素ごとの半径と違い、全体で 1 つの値を反復ごとに縮める
fn vcmRadius() -> f32 {
  let it = f32(U.samplesBefore) / f32(max(U.sppPerFrame, 1u)) + 1.0;
  return U.radius0 * pow(max(it, 1.0), (ALPHA - 1.0) * 0.5);
}

/// merging の戦略 1 個が connection 何個ぶんに相当するかの比。
/// 半径 r の円盤に光子が入る確率が pi r^2 * 本数 に比例することから来る
fn etaVcm() -> f32 {
  if (!VCM_MERGE) {
    // merging を戦略として使わないなら、他の戦略の重みからも外す必要がある
    return 0.0;
  }
  let r = vcmRadius();
  return PI * r * r * f32(max(U.photonCount, 1u));
}

/// 候補として引く光源側の頂点の数。影レイを撃たずに評価するので安い
/// 候補として引く光源側の頂点の数。影レイを撃たずに評価するので安い
const RIS_CANDIDATES: u32 = 8u;

/// 光子を撒くときに使っている放出方向の pdf (立体角について)。
/// VCM の MIS 重みでは「この方向を光源側からたどって作る確率」として要る。
/// photonMain の混合分布とここは必ず同じ式にすること
fn emissionPdfDir(nrm: vec3f, d: vec3f) -> f32 {
  let cosE = abs(dot(d, nrm));
  var pdf = 0.5 * cosE / PI;
  if (histTotal() > 0u) {
    pdf = (1.0 - GUIDE_MIX) * pdf + GUIDE_MIX * histPdf(dirToBin(d));
  }
  return pdf;
}

fn luminanceOf(c: vec3f) -> f32 {
  return dot(c, vec3f(0.2126, 0.7152, 0.0722));
}

/// この頂点とつないだときの寄与。遮蔽も見る。
///
/// 遮蔽を見ないと、扉の裏にある頂点が「壁に近いので幾何項が大きい」という
/// 理由で選ばれてしまい、そのあと影レイで消える。このシーンは遮蔽こそが
/// 本質なので、選ぶ段階で見ないと意味がない
fn connectContribution(hit: Hit, rayDir: vec3f, base: u32, checkVis: bool) -> vec3f {
  let lv = lightVertexHit(base);
  let d = lv.p - hit.p;
  let dist2 = dot(d, d);
  if (dist2 < 1e-8) {
    return vec3f(0.0);
  }
  let dir = d * inverseSqrt(dist2);
  let cosCam = dot(hit.normal, dir);
  let cosLight = dot(lv.normal, -dir);
  if (cosCam <= 1e-6 || cosLight <= 1e-6) {
    return vec3f(0.0);
  }
  // bsdfEval は f * cos を返すので、余弦で割って裸の BSDF に戻す
  if (checkVis) {
    let dist = sqrt(dist2);
    if (occluded(hit.p + hit.normal * 1e-4, dir, dist - 1e-3)) {
      return vec3f(0.0);
    }
  }
  let fCam = bsdfEval(hit, rayDir, dir) / cosCam;
  let fLight = bsdfEval(lv, photons[base + 1u].xyz, -dir) / cosLight;
  return fCam * fLight * (cosCam * cosLight / dist2) * photons[base + 2u].xyz;
}

/// 候補を選ぶための安い目安。読むスロットを 3 つに抑え、BSDF の評価も省く。
/// RIS の目安は何を使っても不偏性は崩れない (選んだ確率で割るため)。
/// 1 頂点 5 スロットを全部読むと、ランダムアクセスのキャッシュミスで
/// 候補 8 個ぶんが実行時間の 4 割になる
fn candidateScore(hit: Hit, rayDir: vec3f, base: u32) -> f32 {
  let d = photons[base + 0u].xyz - hit.p;
  let dist2 = dot(d, d);
  if (dist2 < 1e-8) {
    return 0.0;
  }
  let dir = d * inverseSqrt(dist2);
  let cosCam = dot(hit.normal, dir);
  let cosLight = dot(photons[base + 3u].xyz, -dir);
  if (cosCam <= 1e-6 || cosLight <= 1e-6) {
    return 0.0;
  }
  return luminanceOf(photons[base + 2u].xyz) * cosCam * cosLight / dist2;
}

/// カメラ側の頂点で、半径内にある光源側の頂点を同一視して集める (merging)。
/// 影レイが要らないので遮蔽に強い。半径ぶんのボケが乗るが、反復とともに
/// 半径が縮むので消える
fn mergeAtVertex(
  hit: Hit,
  rayDir: vec3f,
  dVCMc: f32,
  dVCc: f32,
  dVMc: f32,
) -> vec3f {
  let r = vcmRadius();
  let r2 = r * r;
  let invEta = 1.0 / etaVcm();
  var sum = vec3f(0.0);
  let c = gridCoord(hit.p);
  for (var dz = -1; dz <= 1; dz = dz + 1) {
    for (var dy = -1; dy <= 1; dy = dy + 1) {
      for (var dx = -1; dx <= 1; dx = dx + 1) {
        let cell = gridHash(c.x + dx, c.y + dy, c.z + dz);
        let cnt = min(atomicLoad(&grid[cell]), GRID_CAP);
        for (var k = 0u; k < cnt; k = k + 1u) {
          let pi = atomicLoad(&grid[U.gridCells + cell * GRID_CAP + k]);
          let base = pi * VTX_SLOTS;
          let d = photons[base + 0u].xyz - hit.p;
          if (dot(d, d) > r2) {
            continue;
          }
          // gatherPhotons と同じ「同じ面か」の判定。距離だけで拾うと
          // 半径の中にある別の面の光子まで集めてしまう
          if (dot(photons[base + 3u].xyz, hit.normal) < GATHER_COS) {
            continue;
          }
          if (abs(dot(d, hit.normal)) > GATHER_PLANE * vcmRadius()) {
            continue;
          }
          let wi = -photons[base + 1u].xyz;
          let cosI = dot(hit.normal, wi);
          if (cosI <= 1e-4) {
            continue;
          }
          // bsdfEval は f * cos を返すので、余弦で割って裸の BSDF に戻す
          let f = bsdfEval(hit, rayDir, wi) / cosI;
          let pF = bsdfPdfFor(hit, rayDir, wi);
          let pR = bsdfPdfFor(hit, -wi, -rayDir);
          let wLight = photons[base + 0u].w * invEta + photons[base + 5u].x * pF;
          let wCamera = dVCMc * invEta + dVMc * pR;
          let mis = 1.0 / (wLight + 1.0 + wCamera);
          sum = sum + f * photons[base + 2u].xyz * mis;
        }
      }
    }
  }
  return sum / (PI * r2 * f32(max(U.photonCount, 1u)));
}

/// カメラ側の頂点 1 個を、光源側の経路の頂点 1 個とつなぐ。
///
/// 一様に 1 個引くだけだと、狭い隙間の向こうにしか光源がないシーンでは
/// ほとんど当たらない。光源側の頂点の大半は扉の裏の小部屋にあって
/// 部屋からは見えないため。そこで候補を何個か引いて、影レイを撃つ前の
/// 安い評価で 1 個に絞る (RIS)。選んだ確率で割るので不偏のまま
fn connectToLightVertex(
  hit: Hit,
  rayDir: vec3f,
  dVCMc: f32,
  dVCc: f32,
  stat: ptr<function, vec3f>,
) -> vec3f {
  let slots = U.photonCount * MAX_DEPOSITS;
  if (slots == 0u) {
    return vec3f(0.0);
  }

  var bestBase = 0u;
  var bestP = 0.0;
  var sumP = 0.0;
  for (var k = 0u; k < RIS_CANDIDATES; k = k + 1u) {
    let base = min(u32(rand() * f32(slots)), slots - 1u) * VTX_SLOTS;
    if (!lightVertexAlive(base)) {
      continue;
    }
    (*stat).x = (*stat).x + 1.0 / f32(RIS_CANDIDATES);
    let phat = candidateScore(hit, rayDir, base);
    if (phat <= 0.0) {
      continue;
    }
    (*stat).y = (*stat).y + 1.0 / f32(RIS_CANDIDATES);
    sumP = sumP + phat;
    // 貯留サンプリング。寄与の大きさに比例して 1 個選ぶ
    if (rand() * sumP < phat) {
      bestBase = base;
      bestP = phat;
    }
  }
  if (bestP <= 0.0) {
    return vec3f(0.0);
  }

  let lv = lightVertexHit(bestBase);
  let d = lv.p - hit.p;
  let dist = length(d);
  let dir = d / dist;
  let cosCam = dot(hit.normal, dir);
  let cosLight = dot(lv.normal, -dir);
  // 選抜は遮蔽なしの安い評価で行い、勝った 1 個にだけ影レイを撃つ。
  // 候補すべてに撃つと影レイが 8 倍になり、そこが実行時間の 7 割を占める
  let contrib = connectContribution(hit, rayDir, bestBase, true);

  // MIS。同じ経路を作りうる他の戦略 (カメラ側をもう 1 段伸ばす、
  // 光源側をもう 1 段伸ばす) との重み付け。
  //
  // **既知の不具合**: この重みだけを VCM の式にしても足りない。
  // NEE と発光の重みは trace 側で misWeight() (2 戦略の power heuristic) の
  // ままになっていて、その 2 つだけで和がちょうど 1 になっている。そこへ
  // 接続戦略の正の重みを足すので、必ず 1 を超えて二重計上になる (+41%)。
  // 直すには NEE と発光の重みも VCM の形に置き換える必要がある。
  //
  //   発光に当たったとき: 1 / (1 + directPdfA * dVCMc + emissionPdfW * dVCc)
  //   NEE:                1 / (wLight + 1 + wCamera)
  //     wLight  = bsdfDirPdfW / directPdfW
  //     wCamera = emissionPdfW * cosToLight / (directPdfW * cosAtLight)
  //               * (dVCMc + dVCc * bsdfRevPdfW)
  //
  // emissionPdfW は光子を撒くときに使っている混合分布の pdf と同じもの
  let lIn = photons[bestBase + 1u].xyz;
  let dist2 = dist * dist;
  let camPdfW = samplingPdf(hit, rayDir, dir);
  let camRevPdfW = samplingPdf(hit, -dir, -rayDir);
  let lightPdfW = bsdfPdfFor(lv, lIn, -dir);
  let lightRevPdfW = bsdfPdfFor(lv, dir, -lIn);
  let eta = etaVcm();
  let wLight = camPdfW * cosLight / dist2
    * (eta + photons[bestBase + 0u].w + photons[bestBase + 1u].w * lightRevPdfW);
  let wCamera = lightPdfW * cosCam / dist2 * (eta + dVCMc + dVCc * camRevPdfW);
  let mis = 1.0 / (wLight + 1.0 + wCamera);

  // RIS の重み。これで「全部の枠を足したもの」の不偏推定になる
  let risW = (sumP / f32(RIS_CANDIDATES)) * f32(slots) / bestP;
  // 光子の本数で割る。power は光束ではなく BDPT の経路スループット α
  // そのものになっている (alpha_2 = alpha_1 * cosE / pdf = 放出時の power)
  // ので、変換は要らず、光源側の経路 N 本の平均を取るぶんだけ割ればよい
  let n = f32(max(U.photonCount, 1u));
  let out = contrib * mis * risW / n;
  (*stat).z = (*stat).z + luminanceOf(out);
  return out;
}

// -------------------------------------------------------------- 光子パス
/// グリッドの個数カウンタを 0 に戻す
@compute @workgroup_size(64, 1, 1)
fn clearGrid(@builtin(global_invocation_id) gid: vec3u) {
  // 累積をやり直すときは学習した分布も捨てる。カメラが動くと
  // 「役に立つ方向」が変わるので、前の視点の学習は使えない
  let reset = U.samplesBefore == 0u;
  if (reset && gid.x < HIST_BINS) {
    atomicStore(&grid[histOff() + gid.x], 0u);
  }
  if (gid.x < 7u) {
    atomicStore(&grid[statOff() + gid.x], 0u);
  }
  // ガイディングの CDF をボクセルごとに作り直す。
  //
  // 毎フレームやると高い。4096 ボクセル x 256 ビンを 1 スレッド 1 ボクセルで
  // 舐めるので、走るのは 64 ワークグループだけ。並列度が出ないまま
  // 100 万回の atomic を回すことになり、ガイドを有効にしたときの
  // 時間増 (実測 +25%) の大半がここだった。
  //
  // 累積サンプル数が 2 倍になったときだけ作り直す。学習量が倍にならない
  // うちは分布もたいして変わらないので、これで十分追従する
  // (Muller 2017 の反復ごとに予算を倍にしていく構成と同じ考え方)。
  // 作り直しの回数は spp に対して対数でしか増えない
  let sBefore = U.samplesBefore;
  let sAfter = sBefore + U.sppPerFrame;
  let rebuild = sBefore == 0u || firstLeadingBit(sAfter) != firstLeadingBit(sBefore);
  if (rebuild && gid.x < GUIDE_VOX) {
    let hb = guideOff() + gid.x * GUIDE_BINS;
    let cb = guideCdfOff() + gid.x * (GUIDE_BINS + 1u);
    var acc = 0u;
    for (var i = 0u; i < GUIDE_BINS; i = i + 1u) {
      atomicStore(&grid[cb + i], acc);
      acc = acc + atomicLoad(&grid[hb + i]);
    }
    atomicStore(&grid[cb + GUIDE_BINS], acc);
    // 溜まりすぎたら半分に減衰させる。形は保たれ、桁あふれだけ防げる
    if (acc > GUIDE_CAP) {
      for (var i = 0u; i < GUIDE_BINS; i = i + 1u) {
        atomicStore(&grid[hb + i], atomicLoad(&grid[hb + i]) / 2u);
      }
    }
  }
  if (gid.x == 0u) {
    if (reset) {
      for (var i = 0u; i <= HIST_BINS; i = i + 1u) {
        atomicStore(&grid[cdfOff() + i], 0u);
      }
    } else {
      // 頭からの累積和を作る。512 個なので 1 スレッドで舐めても問題ない
      var acc = 0u;
      for (var i = 0u; i < HIST_BINS; i = i + 1u) {
        atomicStore(&grid[cdfOff() + i], acc);
        acc = acc + atomicLoad(&grid[histOff() + i]);
      }
      atomicStore(&grid[cdfOff() + HIST_BINS], acc);
    }
  }
  if (gid.x >= U.gridCells) {
    return;
  }
  atomicStore(&grid[gid.x], 0u);
}

/// 光子を 1 個グリッドに入れる。あふれたセルは捨てる (その分だけ暗くなる)。
///
/// 捨てるぶんは戻ってこないので、これは暗い側の偏りになる。GRID_CAP を
/// 96 -> 256 に上げて測ると、12 シーンの relMSE の幾何平均は 1.008 倍に
/// しかならず、効くのは光子が密集する enclosed の 1.13 倍だけだった。
/// メモリを 170MB 増やす価値はないと判断してそのままにしている。
/// 不偏に直すなら、あふれたときに貯留サンプリングで入れ替えて、
/// 集める側で (実際の個数 / GRID_CAP) を掛ける必要がある
fn depositPhoton(index: u32, p: vec3f) {
  let c = gridCoord(p);
  let cell = gridHash(c.x, c.y, c.z);
  let slot = atomicAdd(&grid[cell], 1u);
  if (slot < GRID_CAP) {
    atomicStore(&grid[U.gridCells + cell * GRID_CAP + slot], index);
  }
  // 無駄になっている光子の割合を測る。全部数えると atomic が詰まるので間引く
  if ((index & 63u) == 0u) {
    atomicAdd(&grid[statOff()], 1u);
    if (atomicLoad(&grid[markOff() + cell]) != 0u) {
      atomicAdd(&grid[statOff() + 1u], 1u);
    }
  }
}

@compute @workgroup_size(64, 1, 1)
fn photonMain(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= U.photonCount || lightSelectCount() == 0u) {
    return;
  }
  rngState = pcg(gid.x * 15487469u + U.frameIndex * 2654435761u + 7u);
  pixelSeed = rngState;
  sampleIdx = U.frameIndex;

  // 面光源と環境マップから 1 つ選ぶ。NEE 側の選び方と揃えてある
  let ln = lightSelectCount();
  let pick = min(u32(rand() * f32(ln)), ln - 1u);
  var power: vec3f;
  var ray: Ray;
  /// この光子を放出した方向のビン。環境マップからの光子は学習の対象外なので -1
  var emitBin = -1.0;
  /// VCM の MIS 用の量。環境マップからの光子には未対応なので 0 のまま
  var dVCM = 0.0;
  var dVC = 0.0;
  var dVM = 0.0;

  if (pick >= U.lightCount) {
    // 環境マップ。CDF から方向を引き、外接球の外から中へ向けて撒く
    let smp = sampleEnvDir(vec2f(rand(), rand()));
    if (smp.w <= 0.0) {
      return;
    }
    let dir = -smp.xyz;
    // 進行方向に垂直な円板から始点を引く。
    //
    // 外接球を覆う円板から一様に引くと、地面が巨大なシーン (spheres は
    // 半径 1000 の地面球のせいで外接球が 1732) では注目領域に届く光子が
    // 7.5e-5 しかなく使い物にならない。そこでカメラが見ているあたりを
    // 覆う小さい円板との混合から引く。選んだ確率で割るので不偏のまま
    let basis = onb(dir);
    let planeC = U.sceneCenter - dir * U.sceneRadius;
    let focus = U.camPos + U.camW * U.focusDist;
    let fRel = focus - planeC;
    let fx = dot(fRel, basis[0]);
    let fy = dot(fRel, basis[1]);
    let rBig = U.sceneRadius;
    let rSmall = max(U.focusDist * 1.5, rBig * 1e-3);

    var px = 0.0;
    var py = 0.0;
    if (rand() < 0.5) {
      let rr = rBig * sqrt(rand());
      let aa = rand() * 2.0 * PI;
      px = rr * cos(aa);
      py = rr * sin(aa);
    } else {
      let rr = rSmall * sqrt(rand());
      let aa = rand() * 2.0 * PI;
      px = fx + rr * cos(aa);
      py = fy + rr * sin(aa);
    }

    // 面積についての混合 pdf。その点を覆っている円板のぶんだけ足す
    var pdfPos = 0.0;
    if (px * px + py * py <= rBig * rBig) {
      pdfPos = pdfPos + 0.5 / (PI * rBig * rBig);
    }
    let dx = px - fx;
    let dy = py - fy;
    if (dx * dx + dy * dy <= rSmall * rSmall) {
      pdfPos = pdfPos + 0.5 / (PI * rSmall * rSmall);
    }
    if (pdfPos <= 0.0) {
      return;
    }

    let origin = planeC + basis[0] * px + basis[1] * py;
    power = envColor(smp.xyz) * f32(ln) / (smp.w * pdfPos);
    ray = Ray(origin, dir);
  } else {
    // 面光源。面上の点をとり、方向は「学習した分布」と「余弦分布」の
    // 混合から引く。混ぜてあるので、学習が偏っても pdf が 0 の方向は
    // できない = 不偏性が保たれる
    let light = quads[indices[pick]];
    let n = cross(light.u, light.v);
    let area = length(n);
    let nrm = n / area;
    let origin = light.q + light.u * rand() + light.v * rand();

    let learned = histTotal() > 0u;
    var dir: vec3f;
    if (learned && rand() < GUIDE_MIX) {
      dir = binToDir(sampleHistBin(rand()), vec2f(rand(), rand()));
    } else {
      // 発光面は両面なので、どちら側に出すかを選ぶ
      var side = nrm;
      if (rand() < 0.5) {
        side = -side;
      }
      dir = normalize(onb(side) * cosineHemisphere(vec2f(rand(), rand())));
    }
    let cosE = abs(dot(dir, nrm));
    if (cosE <= 1e-6) {
      return;
    }
    // 余弦側の pdf は表裏の選択を含めて 0.5 * cos / PI
    var pdf = 0.5 * cosE / PI;
    if (learned) {
      pdf = (1.0 - GUIDE_MIX) * pdf + GUIDE_MIX * histPdf(dirToBin(dir));
    }
    if (pdf <= 0.0) {
      return;
    }
    emitBin = f32(dirToBin(dir));
    // VCM の MIS 用の量。dVCM は「この頂点を NEE で作る確率 / 光源側から
    // たどって作る確率」、dVC は接続戦略の重みを組み立てるための累積。
    // どちらも pdf の比なので単位は無次元
    dVCM = 1.0 / pdf;
    dVC = cosE * area * f32(ln) / pdf;
    dVM = dVC / etaVcm();
    // 出力 = 放射輝度 * 面積 * cos * (選択確率の逆数) / pdf。
    // 撒いた総数で割るのは present 側なので、ここでは割らない
    power = light.mat.emission * area * f32(ln) * cosE / pdf;
    ray = Ray(origin + dir * 1e-4, dir);
  }

  // 格納先はスレッドごとに固定枠を取る。単一アドレスへの atomicAdd で
  // 割り当てると 26 万回が直列化してしまう
  var localSlot = 0u;
  var bounces = 0u;
  /// 放射してからの相対的な減衰。ロシアンルーレットはこれで決める。
  /// power は「放射照度 x 面積」の絶対値なので、そのまま生存確率にすると
  /// 常に 1 を超えて打ち切りが一度も効かない (maze なら 7000 x 面積)
  var relThr = 1.0;
  for (var depth = 0u; depth < U.maxBounces; depth = depth + 1u) {
    var hit: Hit;
    if (!hitScene(ray, 1e-3, 1e30, &hit)) {
      return;
    }
    let kind = hit.mat.kind;
    let gatherable = gatherableMat(hit.mat);

    // 面積についての pdf に直す補正。距離の 2 乗で割り、入射余弦で割る
    let cosIn = abs(dot(hit.normal, ray.dir));
    if (cosIn > 1e-6) {
      dVCM = dVCM * hit.t * hit.t / cosIn;
      dVC = dVC / cosIn;
      dVM = dVM / cosIn;
    }

    // SPPM では直接光はカメラ側の NEE が担当するので、1 回以上散乱した後だけ
    // 堆積させる。VCM の接続では逆で、光源が最初に当たった面の頂点こそ要る。
    // そこへつなぐ経路は NEE では作れないので、外すと丸ごと欠ける
    let minBounce = select(1u, 0u, U.vcm != 0u);
    if (gatherable && bounces >= minBounce && localSlot < MAX_DEPOSITS) {
      let slot = gid.x * MAX_DEPOSITS + localSlot;
      localSlot = localSlot + 1u;
      photons[slot * VTX_SLOTS + 0u] = vec4f(hit.p, dVCM);
      photons[slot * VTX_SLOTS + 1u] = vec4f(ray.dir, dVC);
      photons[slot * VTX_SLOTS + 2u] = vec4f(power, emitBin);
      // 材質の種類とフレーム番号を 1 つの f32 に詰める。使われなかった枠に
      // 前フレームの内容が残るので、これで今フレームのものだけを拾う
      photons[slot * VTX_SLOTS + 3u] =
        vec4f(hit.normal, f32(kind) * 65536.0 + f32(U.frameIndex % 65536u));
      photons[slot * VTX_SLOTS + 4u] = vec4f(hit.mat.albedo, hit.mat.roughness);
      photons[slot * VTX_SLOTS + 5u] = vec4f(dVM, 0.0, 0.0, 0.0);
      // 使われなかった枠は grid に載らないので、古い内容が参照されることはない
      depositPhoton(slot, hit.p);
    }

    var attenuation: vec3f;
    var scattered: Ray;
    // 光子は仕事率を運ぶので、屈折での eta^2 は掛けない
    if (!scatter(ray, hit, vec2f(rand(), rand()), &attenuation, &scattered, false)) {
      return;
    }

    // VCM の漸化式。pF は今サンプルした方向の pdf、pR は逆向きに
    // たどったときの pdf。接続戦略では光源側の経路を逆にたどるので、
    // 逆向きの pdf が要る
    let pF = bsdfPdfFor(hit, ray.dir, scattered.dir);
    if (pF > 0.0) {
      let pR = bsdfPdfFor(hit, -scattered.dir, -ray.dir);
      let cosOut = abs(dot(hit.normal, scattered.dir));
      let eta = etaVcm();
      dVC = (cosOut / pF) * (dVC * pR + dVCM + eta);
      dVM = (cosOut / pF) * (dVM * pR + dVCM / eta + 1.0);
      dVCM = 1.0 / pF;
    } else {
      // デルタ的な散乱。方向を選ぶ確率が定義できないので dVCM は 0 にするが、
      // dVC / dVM は「順方向と逆方向の pdf の比が 1」なので cos を掛けて
      // そのまま持ち越す (Georgiev らの VCM / SmallVCM と同じ扱い)。
      //
      // ここには else が無く、屈折前の値をそのまま持ち越していた。
      // 誘電体の透過は dielectricPdf が 0 を返すので、ガラスを抜けた光子は
      // 壊れた MIS 量を持ったまま堆積していた
      let cosOut = abs(dot(hit.normal, scattered.dir));
      dVC = dVC * cosOut;
      dVM = dVM * cosOut;
      dVCM = 0.0;
    }

    power = power * attenuation;
    relThr = relThr * max(attenuation.r, max(attenuation.g, attenuation.b));
    ray = scattered;
    bounces = bounces + 1u;

    // ロシアンルーレット
    if (PHOTON_RR && U.vcm == 0u && depth >= PHOTON_RR_START) {
      let q = min(relThr, 1.0);
      if (rand() > q) {
        return;
      }
      power = power / max(q, 1e-4);
      relThr = relThr / max(q, 1e-4);
    }
  }
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= U.width || gid.y >= U.height) {
    return;
  }
  let pixel = gid.y * U.width + gid.x;
  // 固定 seed では累積サンプル数から種を作るので、同条件なら毎回同じ絵になる
  let seedBase = select(U.frameIndex, U.samplesBefore, U.fixedSeed != 0u);
  // 画素とフレームを線形に混ぜてから 1 回ハッシュすると、
  // pixel * 9781 と seedBase * 6271 がぶつかる組み合わせで別の画素・
  // 別のフレームが同じ種を引く。入れ子にして潰す
  rngState = pcg(pixel + pcg(seedBase * 6271u + U.salt * 0x9e3779b9u + 1u));
  pixelSeed = pcg(pixel * 26699u + U.salt * 0x85ebca6bu + 1u);

  // 適応サンプリング。既に十分収束した画素は今フレームは撃たない。
  // 空いた時間はフレームレートとして返ってくるので、荒れている画素の
  // サンプル数が伸びる。
  //
  // **これは偏る**。打ち切りの判定に、その画素自身のサンプルから作った
  // 分散を使っているため、たまたま明るい経路をまだ引いていない画素が
  // 「収束した」と誤判定されて撃たれなくなり、そのまま暗いところで
  // 止まる (打ち切り則の偏り)。実測でも 8192 spp の平均輝度が
  // cornell -0.70% / glass -0.57% / indirect -0.16% と暗い側にずれ、
  // 上位 1% 除外の relMSE は spp を 4 倍にしても glass で 0.98 倍にしか
  // 減らない (無効時は 1.49 倍)。等時間でも 4096 spp で 0.41 〜 0.86 倍と
  // 負けるので既定は off
  var thisSpp = U.sppPerFrame;
  let oAcc = pixel * 4u;
  if (U.adaptivePixels != 0u && U.sppm == 0u && U.debugMode == 0u
    && U.samplesBefore > 0u) {
    let acc = histWrite[oAcc];
    let n = acc.w;
    if (n >= 64.0) {
      let mean = luminanceOf(acc.rgb) / n;
      // 2 乗和から分散を出し、平均に対する相対標準誤差で判定する
      let m2 = histWrite[oAcc + 2u].x / n;
      let varL = max(m2 - mean * mean, 0.0);
      let relErr = sqrt(varL / n) / max(mean, 1e-4);
      if (relErr < ADAPTIVE_TOL) {
        thisSpp = 0u;
      }
    }
  }

  // この画素がこれまでに出している平均輝度。firefly の閾値と、
  // ADRRS の「目指している値」の両方に使う
  var pixelMean = 0.0;
  var fireflyThr = 0.0;
  if (U.samplesBefore > 0u && U.debugMode == 0u) {
    let hc = select(histRead[oAcc], histWrite[oAcc], U.sppm != 0u);
    if (hc.w > 0.0) {
      pixelMean = luminanceOf(hc.rgb) / hc.w;
      if (FIREFLY_K > 0.0) {
        fireflyThr = pixelMean * FIREFLY_K * sqrt(hc.w);
      }
    }
  }

  var sum = vec3f(0.0);
  var sumSq = 0.0;
  var firstHit = vec4f(0.0);
  for (var s = 0u; s < thisSpp; s = s + 1u) {
    // 累積サンプル番号で低食い違い列を引く
    sampleIdx = U.samplesBefore + s;
    let jitter = sample2d(0u, true);
    let px = (f32(gid.x) + jitter.x) / f32(U.width) * 2.0 - 1.0;
    let py = 1.0 - (f32(gid.y) + jitter.y) / f32(U.height) * 2.0;
    var fh = vec4f(0.0);
    var aov = Aov(vec3f(0.0), vec3f(0.0), 0.0, 0.0, 0.0, 0.0, vec3f(0.0));
    let radiance = trace(makeRay(px, py, sample2d(1u, true)), pixelMean, &fh, &aov);
    if (U.debugMode == 0u) {
      if (accumulable(radiance)) {
        var r = radiance;
        let l = luminanceOf(r);
        if (fireflyThr > 0.0 && l > fireflyThr) {
          // 色は保ったまま輝度だけ閾値へ落とす
          r = r * (fireflyThr / l);
        }
        sum = sum + r;
        let lc = luminanceOf(r);
        sumSq = sumSq + lc * lc;
      }
    } else {
      sum = sum + aovColor(aov);
    }
    if (s == 0u) {
      firstHit = fh;
    }
  }

  let o = pixel * 4u;
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
        // 1 画素あたり 4 要素。ここが 2 のままだと別の画素を読んでしまう
        let pp = (u32(sy) * U.width + u32(sx)) * 4u;
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
    let hc = select(histRead[o], histWrite[o], U.sppm != 0u);
    prevSum = hc.rgb;
    prevCount = hc.w;
  }

  histWrite[o] = vec4f(prevSum + sum, prevCount + f32(thisSpp));
  if (U.sppm == 0u) {
    // 適応サンプリングの判定に使う 2 乗和。SPPM ではこの枠を使うので触らない
    let prevSq = select(0.0, histWrite[o + 2u].x, U.samplesBefore > 0u);
    histWrite[o + 2u] = vec4f(prevSq + sumSq, 0.0, 0.0, 0.0);
  }
  histWrite[o + 1u] = firstHit;

  // デノイザの手がかり。SPPM が有効なら sppmMain が書き直すが、
  // SPPM off + denoise on のときは main だけが通るのでここでも書いておく。
  // x (SPPM の半径) は自分の管轄外なので既存の値を壊さないよう読んでから書く
  if (U.denoise != 0u) {
    let pxC = (f32(gid.x) + 0.5) / f32(U.width) * 2.0 - 1.0;
    let pyC = 1.0 - (f32(gid.y) + 0.5) / f32(U.height) * 2.0;
    let g = guideFor(pxC, pyC);
    histWrite[o + 3u] = vec4f(histWrite[o + 3u].x, g.x, g.y, g.z);
  } else {
    histWrite[o + 3u] = vec4f(histWrite[o + 3u].x, 0.0, 0.0, 0.0);
  }
}
