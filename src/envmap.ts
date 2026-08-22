/**
 * lat-long の環境マップと、その重要度サンプリング用の 2 次元 CDF。
 *
 * HDRI ファイルを持ってくる代わりに、太陽つきの空を手続き的に生成する。
 * 「小さくて桁違いに明るい光源が環境マップに含まれる」という、重要度
 * サンプリングが必要になる性質はそのまま再現できる。
 */

/** 輝度。CDF の重みと pdf の評価で同じ式を使うこと */
function luminance(r: number, g: number, b: number): number {
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

/** pdf が 0 になる方向を作らないための下限 */
const MIN_WEIGHT = 1e-6;

export interface EnvMap {
  width: number;
  height: number;
  /**
   * GPU に載せる 1 本の配列。バッファ数の上限に余裕がないので
   * [テクセル rgba][周辺 CDF][条件付き CDF] を連結して持つ。
   */
  data: Float32Array;
  /** pdf_omega = max(luminance, MIN_WEIGHT) * pdfScale */
  pdfScale: number;
}

/** 太陽と空。dir は正規化済み、y が上 */
function skyRadiance(
  dx: number,
  dy: number,
  dz: number,
  sun: [number, number, number],
): [number, number, number] {
  if (dy < 0) {
    // 地面はごく暗く、水平線に向けて少し明るくする
    const t = Math.min(1, -dy * 4);
    const g = 0.16 * (1 - t) + 0.055 * t;
    return [g * 1.05, g, g * 0.92];
  }

  // 天頂ほど青く、水平線ほど白っぽく
  const t = Math.pow(dy, 0.42);
  const r = 0.72 * (1 - t) + 0.19 * t;
  const g = 0.79 * (1 - t) + 0.35 * t;
  const b = 0.92 * (1 - t) + 0.72 * t;

  const cosAngle = dx * sun[0] + dy * sun[1] + dz * sun[2];
  // 太陽のまわりのぼんやりした光暈
  const glow = Math.pow(Math.max(0, cosAngle), 220) * 1.6;

  // 太陽本体 (半径 2.5 度)
  const SUN_COS = Math.cos((2.5 * Math.PI) / 180);
  const disk = cosAngle > SUN_COS ? 1 : 0;
  const sunR = disk * 820;

  return [
    r + glow * 1.0 + sunR * 1.0,
    g + glow * 0.93 + sunR * 0.94,
    b + glow * 0.8 + sunR * 0.82,
  ];
}

export function buildSunSkyEnv(width = 512, height = 256): EnvMap {
  const elevation = (22 * Math.PI) / 180;
  const azimuth = (-35 * Math.PI) / 180;
  const sun: [number, number, number] = [
    Math.cos(elevation) * Math.cos(azimuth),
    Math.sin(elevation),
    Math.cos(elevation) * Math.sin(azimuth),
  ];

  const texelCount = width * height;
  const marginalOffset = texelCount * 4;
  const condOffset = marginalOffset + height + 1;
  const data = new Float32Array(condOffset + height * (width + 1));

  // 1. テクセルの色を埋める
  for (let v = 0; v < height; v++) {
    const theta = ((v + 0.5) / height) * Math.PI;
    const sinTheta = Math.sin(theta);
    const cosTheta = Math.cos(theta);
    for (let u = 0; u < width; u++) {
      const phi = ((u + 0.5) / width) * Math.PI * 2;
      const dx = sinTheta * Math.cos(phi);
      const dz = sinTheta * Math.sin(phi);
      const [r, g, b] = skyRadiance(dx, cosTheta, dz, sun);
      const o = (v * width + u) * 4;
      data[o] = r;
      data[o + 1] = g;
      data[o + 2] = b;
    }
  }

  // 2. サンプリングの重みを alpha に入れる。
  //
  // 色はバイリニア補間で引くのに pdf を最近傍テクセルから作ると、太陽の
  // ような小さく極端に明るい領域の縁で「暗いテクセルの pdf を使いながら
  // 補間で明るい値を拾う」ことになり、比が数百倍に跳ねて白い斑点になる。
  // 近傍 3x3 の最大を重みにすれば、pdf が補間値を下回らなくなる。
  // pdf は正であればどんな形でも不偏なので、これで偏りは出ない
  const lum = (u: number, v: number) => {
    const uu = ((u % width) + width) % width;
    const vv = Math.min(Math.max(v, 0), height - 1);
    const o = (vv * width + uu) * 4;
    return luminance(data[o], data[o + 1], data[o + 2]);
  };
  for (let v = 0; v < height; v++) {
    for (let u = 0; u < width; u++) {
      let m = 0;
      for (let dv = -1; dv <= 1; dv++) {
        for (let du = -1; du <= 1; du++) m = Math.max(m, lum(u + du, v + dv));
      }
      data[(v * width + u) * 4 + 3] = Math.max(m, MIN_WEIGHT);
    }
  }

  // 3. 行ごとに条件付き CDF を積み、同時に行の総和を集める
  const rowSums = new Float32Array(height);
  for (let v = 0; v < height; v++) {
    const sinTheta = Math.sin(((v + 0.5) / height) * Math.PI);
    const base = condOffset + v * (width + 1);
    let acc = 0;
    data[base] = 0;
    for (let u = 0; u < width; u++) {
      // sin(theta) は lat-long のテクセルが張る立体角の補正
      acc += data[(v * width + u) * 4 + 3] * sinTheta;
      data[base + u + 1] = acc;
    }
    rowSums[v] = acc;
    if (acc > 0) {
      for (let u = 1; u <= width; u++) data[base + u] /= acc;
    }
  }

  let total = 0;
  data[marginalOffset] = 0;
  for (let v = 0; v < height; v++) {
    total += rowSums[v];
    data[marginalOffset + v + 1] = total;
  }
  if (total > 0) {
    for (let v = 1; v <= height; v++) data[marginalOffset + v] /= total;
  }

  // 立体角に直すと sin(theta) が約分され、pdf は輝度に比例した定数倍になる
  return {
    width,
    height,
    data,
    pdfScale: (width * height) / (2 * Math.PI * Math.PI * total),
  };
}
