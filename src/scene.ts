export type Vec3 = [number, number, number];

export const MATERIAL = {
  lambert: 0,
  /** GGX マイクロファセット (導体) */
  ggx: 1,
  dielectric: 2,
  emissive: 3,
} as const;

export type MaterialKind = (typeof MATERIAL)[keyof typeof MATERIAL];

export interface Material {
  kind: MaterialKind;
  /** lambert の反射率 / ggx の垂直入射反射率 F0 / dielectric の減衰色 */
  albedo: Vec3;
  /** ggx の粗さ (知覚的)。0 に近いほど鏡面 */
  roughness?: number;
  /** dielectric の屈折率 */
  ior?: number;
  /** emissive の放射輝度 */
  emission?: Vec3;
}

export const lambert = (albedo: Vec3): Material => ({
  kind: MATERIAL.lambert,
  albedo,
});
/** GGX 導体。roughness が小さいほど鏡面に近い */
export const ggx = (albedo: Vec3, roughness: number): Material => ({
  kind: MATERIAL.ggx,
  albedo,
  roughness,
});
/**
 * 誘電体。roughness を上げるとすりガラスになる。
 * albedo は「1 単位距離あたりの透過色」として距離依存の吸収に使う
 */
export const dielectric = (
  ior = 1.5,
  roughness = 0,
  transmittance: Vec3 = [1, 1, 1],
): Material => ({
  kind: MATERIAL.dielectric,
  albedo: transmittance,
  roughness,
  ior,
});
export const emissive = (emission: Vec3): Material => ({
  kind: MATERIAL.emissive,
  albedo: [0, 0, 0],
  emission,
});

export interface Sphere {
  center: Vec3;
  radius: number;
  material: Material;
}

/** 三角形。法線は頂点法線を重心座標で補間する */
export interface Triangle {
  v0: Vec3;
  v1: Vec3;
  v2: Vec3;
  n0: Vec3;
  n1: Vec3;
  n2: Vec3;
  material: Material;
}

/** 角 q と 2 辺 u, v が張る平行四辺形 */
export interface Quad {
  q: Vec3;
  u: Vec3;
  v: Vec3;
  material: Material;
}

/** 背景の扱い。WGSL 側の ENV_* と一致させること */
export const ENV = {
  /** 空のグラデーション */
  sky: 0,
  /** 真っ黒 (閉じた部屋を光源だけで照らす) */
  black: 1,
  /** 太陽つきの空を焼いた lat-long マップ。重要度サンプリングの対象になる */
  hdri: 2,
} as const;

export type EnvKind = (typeof ENV)[keyof typeof ENV];

/** シーンごとに決まるカメラの初期値 */
export interface CameraPreset {
  target: Vec3;
  distance: number;
  yaw: number;
  pitch: number;
  fovDeg: number;
  aperture: number;
}

/** 箱で区切った一様媒質 */
export interface Fog {
  min: Vec3;
  max: Vec3;
  /** 散乱係数 */
  sigmaS: number;
  /** 吸収係数 */
  sigmaA: number;
  /** Henyey-Greenstein の非対称パラメータ。正で前方散乱 */
  g: number;
}

export interface Scene {
  spheres: Sphere[];
  quads: Quad[];
  triangles: Triangle[];
  env: EnvKind;
  camera: CameraPreset;
  fog?: Fog;
}

// ---------------------------------------------------------------- geometry
/** 軸並行な直方体を 6 枚の quad に展開する */
function box(min: Vec3, max: Vec3, material: Material): Quad[] {
  const dx: Vec3 = [max[0] - min[0], 0, 0];
  const dy: Vec3 = [0, max[1] - min[1], 0];
  const dz: Vec3 = [0, 0, max[2] - min[2]];
  const neg = (v: Vec3): Vec3 => [-v[0], -v[1], -v[2]];
  return [
    { q: [min[0], min[1], max[2]], u: dx, v: dy, material }, // front
    { q: [max[0], min[1], max[2]], u: neg(dz), v: dy, material }, // right
    { q: [max[0], min[1], min[2]], u: neg(dx), v: dy, material }, // back
    { q: [min[0], min[1], min[2]], u: dz, v: dy, material }, // left
    { q: [min[0], max[1], max[2]], u: dx, v: neg(dz), material }, // top
    { q: [min[0], min[1], min[2]], u: dx, v: dz, material }, // bottom
  ];
}

/** quad 群を Y 軸まわりに回してから平行移動する */
function place(quads: Quad[], yawDeg: number, offset: Vec3): Quad[] {
  const a = (yawDeg * Math.PI) / 180;
  const c = Math.cos(a);
  const s = Math.sin(a);
  const rot = (v: Vec3): Vec3 => [c * v[0] + s * v[2], v[1], -s * v[0] + c * v[2]];
  return quads.map(({ q, u, v, material }) => {
    const r = rot(q);
    return {
      q: [r[0] + offset[0], r[1] + offset[1], r[2] + offset[2]] as Vec3,
      u: rot(u),
      v: rot(v),
      material,
    };
  });
}

// ---------------------------------------------------------------- scenes
/** 決定的な擬似乱数 (シーンを毎回同じにするため) */
function makeRng(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

/** Ray Tracing in One Weekend 風の定番シーン */
function buildSpheresScene(): Scene {
  const spheres: Sphere[] = [
    // 地面 (巨大球)
    { center: [0, -1000, 0], radius: 1000, material: lambert([0.5, 0.5, 0.5]) },
    { center: [-2.6, 1, 0], radius: 1, material: dielectric(1.5) },
    { center: [0, 1, 0], radius: 1, material: lambert([0.55, 0.25, 0.18]) },
    { center: [2.6, 1, 0], radius: 1, material: ggx([0.7, 0.6, 0.5], 0.06) },
  ];

  // 周りに小球を散らす
  const rand = makeRng(20240819);
  for (let a = -5; a < 5; a++) {
    for (let b = -4; b < 4; b++) {
      const center: Vec3 = [a * 1.1 + 0.7 * rand(), 0.2, b * 1.1 + 0.7 * rand()];
      // 大球と重なる位置は避ける
      const tooClose = spheres
        .slice(1)
        .some(
          (s) =>
            Math.hypot(center[0] - s.center[0], center[2] - s.center[2]) <
            s.radius + 0.45,
        );
      if (tooClose) continue;

      const pick = rand();
      if (pick < 0.75) {
        spheres.push({
          center,
          radius: 0.2,
          material: lambert([rand() * rand(), rand() * rand(), rand() * rand()]),
        });
      } else if (pick < 0.93) {
        spheres.push({
          center,
          radius: 0.2,
          material: ggx(
            [0.5 + 0.5 * rand(), 0.5 + 0.5 * rand(), 0.5 + 0.5 * rand()],
            0.1 + 0.4 * rand(),
          ),
        });
      } else {
        spheres.push({ center, radius: 0.2, material: dielectric(1.5) });
      }
    }
  }

  return {
    spheres,
    quads: [],
    triangles: [],
    env: ENV.hdri,
    camera: {
      target: [0, 1, 0],
      distance: 12,
      yaw: Math.PI * 0.5,
      pitch: 0.14,
      fovDeg: 32,
      aperture: 0.08,
    },
  };
}

/**
 * Cornell box。オリジナルは一辺 555 だが、カメラの距離やレンズ半径を
 * 他のシーンと揃えたいので 1/100 スケール (一辺 5.55) で組む。
 */
function buildCornellScene(): Scene {
  const S = 5.55;
  const white = lambert([0.73, 0.73, 0.73]);
  const red = lambert([0.65, 0.05, 0.05]);
  const green = lambert([0.12, 0.45, 0.15]);

  const quads: Quad[] = [
    // 左右の壁
    { q: [S, 0, 0], u: [0, S, 0], v: [0, 0, S], material: green },
    { q: [0, 0, 0], u: [0, S, 0], v: [0, 0, S], material: red },
    // 床・天井・奥
    { q: [0, 0, 0], u: [S, 0, 0], v: [0, 0, S], material: white },
    { q: [S, S, S], u: [-S, 0, 0], v: [0, 0, -S], material: white },
    { q: [0, 0, S], u: [S, 0, 0], v: [0, S, 0], material: white },
    // 天井の面光源 (天井のわずかに下)
    {
      q: [3.43, S - 0.01, 3.32],
      u: [-1.3, 0, 0],
      v: [0, 0, -1.05],
      material: emissive([15, 15, 15]),
    },
    // 背の高い箱と低い箱
    ...place(box([0, 0, 0], [1.65, 3.3, 1.65], white), 15, [2.65, 0, 2.95]),
    ...place(box([0, 0, 0], [1.65, 1.65, 1.65], white), -18, [1.3, 0, 0.65]),
  ];

  return {
    spheres: [
      // 低い箱の上のガラス玉 (おまけ)
      { center: [2.1, 2.3, 1.45], radius: 0.65, material: dielectric(1.5) },
    ],
    quads,
    triangles: [],
    env: ENV.black,
    camera: {
      target: [S / 2, S / 2, S / 2],
      distance: 10.8,
      yaw: -Math.PI * 0.5,
      pitch: 0,
      fovDeg: 40,
      aperture: 0,
    },
  };
}


/**
 * 半開きの扉のコーネルボックス (ajar door)。
 * 奥の壁に戸口を切り、その裏の小部屋にだけ光源を置く。扉はほんの少しだけ
 * 開いていて、部屋の中はその細い隙間から漏れた光だけで照らされる。
 * 部屋の中の面から光源への影レイ (NEE) はほぼ全部が扉に遮られ、BSDF
 * サンプリングも隙間を偶然通る確率が低いのでパストレーシングは猛烈に
 * ノイズが乗る。逆に SPPM は光源から撒いた光子が隙間を通って部屋に
 * 入れるので有利になるはず。この対比を見るためのシーン
 */
function buildAjarDoorScene(): Scene {
  const S = 5.55;
  const white = lambert([0.73, 0.73, 0.73]);
  const red = lambert([0.65, 0.05, 0.05]);
  const green = lambert([0.12, 0.45, 0.15]);

  // 画像の左右について: このシーンのカメラ設定では +x が画面の左に来る
  // (緑の壁が画面左)。戸口は画面の右に開けたいので、蝶番を x が大きい側に
  // 置き、扉は x=0 の方へ伸ばす
  const hingeX = 4.35;
  const lintelY = 4.2;
  /** 扉の幅。戸口 (4.35) よりわずかに狭いだけにして、隙間を細く保つ */
  const doorW = 4.28;

  // 扉の裏の小部屋の奥行き
  const backDepth = 1.8;
  const backZ = S + backDepth;

  const quads: Quad[] = [
    // 左右の壁 (主室)
    { q: [S, 0, 0], u: [0, S, 0], v: [0, 0, S], material: green },
    { q: [0, 0, 0], u: [0, S, 0], v: [0, 0, S], material: red },
    // 床・天井 (主室。天井には面光源を置かない。光源は扉の裏だけ)
    { q: [0, 0, 0], u: [S, 0, 0], v: [0, 0, S], material: white },
    { q: [S, S, S], u: [-S, 0, 0], v: [0, 0, -S], material: white },

    // 奥の壁を戸口の分だけ 2 枚の quad に割る
    // (a) 蝶番の柱になる全高部分
    { q: [hingeX, 0, S], u: [S - hingeX, 0, 0], v: [0, S, 0], material: white },
    // (b) 戸口の上、鴨居の部分
    {
      q: [0, lintelY, S],
      u: [hingeX, 0, 0],
      v: [0, S - lintelY, 0],
      material: white,
    },

    // 扉本体。厚さ 0.08 の板を戸口の端 (x=hingeX) を蝶番にしてわずかに回す。
    // 幅を戸口とほぼ同じにしてあるので、閉じていれば隙間はごくわずか。
    // 8 度ほど開くと蝶番から遠い自由端 (x=0 側) だけ隙間が広がる、という
    // くさび形の開き方になる。扉は小部屋の側 (+z) へ開く
    ...place(box([-doorW, 0, 0], [0, lintelY, 0.08], white), 8, [hingeX, 0, S]),

    // 扉の裏の小部屋。主室と同じ x, y の範囲でそのまま奥へ延長し、
    // 光が漏れないようきっちり閉じる
    { q: [S, 0, S], u: [0, S, 0], v: [0, 0, backDepth], material: white }, // 右壁
    { q: [0, 0, S], u: [0, S, 0], v: [0, 0, backDepth], material: white }, // 左壁
    { q: [0, 0, S], u: [S, 0, 0], v: [0, 0, backDepth], material: white }, // 床
    { q: [0, S, S], u: [S, 0, 0], v: [0, 0, backDepth], material: white }, // 天井
    { q: [0, 0, backZ], u: [S, 0, 0], v: [0, S, 0], material: white }, // 一番奥の壁

    // 光源。小部屋のいちばん奥、扉の自由端に近い側に置く。
    // カメラから隙間越しに光源本体が直接見えない位置にすること
    {
      q: [0.35, 1.0, S + 1.7],
      u: [1.6, 0, 0],
      v: [0, 2.5, 0],
      material: emissive([55, 46, 32]),
    },

    // 背の高い箱と低い箱 (cornell と同じ配置)
    ...place(box([0, 0, 0], [1.65, 3.3, 1.65], white), 15, [2.65, 0, 2.95]),
    ...place(box([0, 0, 0], [1.65, 1.65, 1.65], white), -18, [1.3, 0, 0.65]),
  ];

  return {
    spheres: [],
    quads,
    triangles: [],
    env: ENV.black,
    camera: {
      target: [S / 2, S / 2, S / 2],
      distance: 10.8,
      yaw: -Math.PI * 0.5,
      pitch: 0,
      fovDeg: 40,
      aperture: 0,
    },
  };
}


/**
 * Veach の MIS テストシーン。
 * 鋭さの違う 4 枚の光沢プレートと、大きさの違う 4 個の面光源を並べる。
 * 「鋭いローブ x 大きい光源」は光源サンプリングが苦手、
 * 「広いローブ x 小さい光源」は BSDF サンプリングが苦手なので、
 * MIS の有無で結果が大きく変わる。
 */
function buildVeachScene(): Scene {
  const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const norm = (v: Vec3): Vec3 => {
    const l = Math.hypot(v[0], v[1], v[2]) || 1;
    return [v[0] / l, v[1] / l, v[2] / l];
  };

  const eye: Vec3 = [0, 3.5, -13];
  const lightRow: Vec3 = [0, 6.4, 2.6];

  /** center を中心に、eye と lightRow を鏡面反射で結ぶ向きへ傾けた板 */
  const plate = (center: Vec3, width: number, depth: number, material: Material): Quad => {
    const a = norm(sub(eye, center));
    const b = norm(sub(lightRow, center));
    const n = norm([a[0] + b[0], a[1] + b[1], a[2] + b[2]]);
    const u: Vec3 = [width, 0, 0];
    const vd = norm([0, n[2], -n[1]]);
    const v: Vec3 = [vd[0] * depth, vd[1] * depth, vd[2] * depth];
    return {
      q: [
        center[0] - u[0] / 2 - v[0] / 2,
        center[1] - u[1] / 2 - v[1] / 2,
        center[2] - u[2] / 2 - v[2] / 2,
      ],
      u,
      v,
      material,
    };
  };

  /** 下向きの正方形光源。放射輝度は面積で割って総パワーを揃える */
  const lamp = (x: number, size: number, power: number): Quad => ({
    q: [x - size / 2, lightRow[1], lightRow[2] - size / 2],
    u: [size, 0, 0],
    v: [0, 0, size],
    material: emissive([
      power / (size * size),
      power / (size * size),
      power / (size * size),
    ]),
  });

  // GGX は Phong より裾が長いので、見た目が揃うよう粗さは低めに取る
  const plateSpec: { center: Vec3; roughness: number }[] = [
    { center: [0, 0.4, -3.2], roughness: 0.34 },
    { center: [0, 1.5, -0.7], roughness: 0.2 },
    { center: [0, 2.6, 1.8], roughness: 0.11 },
    { center: [0, 3.7, 4.3], roughness: 0.055 },
  ];

  const quads: Quad[] = [
    // 床 (暗めの拡散面)
    { q: [-14, -1.6, -9], u: [28, 0, 0], v: [0, 0, 26], material: lambert([0.16, 0.16, 0.18]) },
    ...plateSpec.map((p) => plate(p.center, 11, 1.35, ggx([0.95, 0.95, 0.95], p.roughness))),
    lamp(4.9, 0.15, 9),
    lamp(1.7, 0.45, 9),
    lamp(-1.7, 1.1, 9),
    lamp(-5.0, 2.4, 9),
  ];

  return {
    spheres: [],
    quads,
    triangles: [],
    env: ENV.black,
    camera: {
      target: [0, 2.2, 0],
      distance: 13.06,
      yaw: -Math.PI * 0.5,
      pitch: 0.0997,
      fovDeg: 46,
      aperture: 0,
    },
  };
}


/** three.js の TorusKnotGeometry と同じ曲線 */
function knotPoint(u: number, p: number, q: number, radius: number): Vec3 {
  const cu = Math.cos(u);
  const su = Math.sin(u);
  const quOverP = (q / p) * u;
  const cs = Math.cos(quOverP);
  return [
    radius * (2 + cs) * 0.5 * cu,
    radius * (2 + cs) * su * 0.5,
    radius * Math.sin(quOverP) * 0.5,
  ];
}

/** トーラス結び目を三角形メッシュにする。頂点法線つき */
function torusKnot(
  segments: number,
  sides: number,
  radius: number,
  tube: number,
  center: Vec3,
  material: Material,
): Triangle[] {
  const p = 2;
  const q = 3;
  const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const add = (a: Vec3, b: Vec3): Vec3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
  const cross = (a: Vec3, b: Vec3): Vec3 => [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
  const norm = (v: Vec3): Vec3 => {
    const l = Math.hypot(v[0], v[1], v[2]) || 1;
    return [v[0] / l, v[1] / l, v[2] / l];
  };

  // 曲線に沿った枠を作り、円を掃引する
  const pos: Vec3[][] = [];
  const nrm: Vec3[][] = [];
  for (let i = 0; i <= segments; i++) {
    const u = (i / segments) * p * Math.PI * 2;
    const p1 = knotPoint(u, p, q, radius);
    const p2 = knotPoint(u + 0.01, p, q, radius);
    const t = sub(p2, p1);
    let n = add(p2, p1);
    const b = norm(cross(t, n));
    n = norm(cross(b, t));

    const ringPos: Vec3[] = [];
    const ringNrm: Vec3[] = [];
    for (let j = 0; j <= sides; j++) {
      const v = (j / sides) * Math.PI * 2;
      const cx = -tube * Math.cos(v);
      const cy = tube * Math.sin(v);
      const vert: Vec3 = [
        p1[0] + cx * n[0] + cy * b[0] + center[0],
        p1[1] + cx * n[1] + cy * b[1] + center[1],
        p1[2] + cx * n[2] + cy * b[2] + center[2],
      ];
      ringPos.push(vert);
      ringNrm.push(norm([vert[0] - p1[0] - center[0], vert[1] - p1[1] - center[1], vert[2] - p1[2] - center[2]]));
    }
    pos.push(ringPos);
    nrm.push(ringNrm);
  }

  const tris: Triangle[] = [];
  for (let i = 0; i < segments; i++) {
    for (let j = 0; j < sides; j++) {
      const a = { p: pos[i][j], n: nrm[i][j] };
      const bb = { p: pos[i + 1][j], n: nrm[i + 1][j] };
      const c = { p: pos[i + 1][j + 1], n: nrm[i + 1][j + 1] };
      const d = { p: pos[i][j + 1], n: nrm[i][j + 1] };
      tris.push({ v0: a.p, v1: bb.p, v2: c.p, n0: a.n, n1: bb.n, n2: c.n, material });
      tris.push({ v0: a.p, v1: c.p, v2: d.p, n0: a.n, n1: c.n, n2: d.n, material });
    }
  }
  return tris;
}

/** 三角形メッシュと BVH の動作確認用シーン */
function buildMeshScene(): Scene {
  const triangles = torusKnot(220, 22, 2.6, 0.62, [0, 1.9, 0], ggx([0.95, 0.78, 0.42], 0.16));
  return {
    spheres: [],
    quads: [
      // 床
      { q: [-12, 0, -12], u: [24, 0, 0], v: [0, 0, 24], material: lambert([0.45, 0.45, 0.48]) },
      // 天井の面光源
      {
        q: [-2.2, 8.5, -2.2],
        u: [4.4, 0, 0],
        v: [0, 0, 4.4],
        material: emissive([9, 8.6, 8]),
      },
    ],
    triangles,
    env: ENV.hdri,
    camera: {
      target: [0, 1.9, 0],
      distance: 17,
      yaw: Math.PI * 0.62,
      pitch: 0.26,
      fovDeg: 34,
      aperture: 0,
    },
  };
}


/** すりガラスと色ガラスの確認用。粗さを 0 から段階的に上げた球を並べる */
function buildGlassScene(): Scene {
  const stripe = (i: number, color: Vec3): Quad => ({
    q: [-6.6 + i * 2.2, 0, 3.2],
    u: [2.2, 0, 0],
    v: [0, 3.6, 0],
    material: lambert(color),
  });

  const balls: { x: number; mat: Material }[] = [
    { x: -4.75, mat: dielectric(1.5, 0) },
    { x: -2.85, mat: dielectric(1.5, 0.12) },
    { x: -0.95, mat: dielectric(1.5, 0.25) },
    { x: 0.95, mat: dielectric(1.5, 0.45) },
    // 色ガラス (距離依存の吸収)
    { x: 2.85, mat: dielectric(1.5, 0, [0.62, 0.34, 0.12]) },
    { x: 4.75, mat: dielectric(1.5, 0.08, [0.28, 0.72, 0.45]) },
  ];

  return {
    spheres: balls.map((b) => ({
      center: [b.x, 0.85, 0] as Vec3,
      radius: 0.85,
      material: b.mat,
    })),
    quads: [
      { q: [-14, 0, -12], u: [28, 0, 0], v: [0, 0, 24], material: lambert([0.5, 0.5, 0.53]) },
      // 屈折のぼけ具合が分かるよう、後ろに色の帯を置く
      ...[
        [0.85, 0.2, 0.2], [0.9, 0.6, 0.15], [0.85, 0.85, 0.2],
        [0.2, 0.7, 0.3], [0.2, 0.5, 0.85], [0.55, 0.3, 0.8],
      ].map((c, i) => stripe(i, c as Vec3)),
    ],
    triangles: [],
    env: ENV.hdri,
    camera: {
      target: [0, 0.95, 0.6],
      distance: 12.5,
      yaw: -Math.PI * 0.5,
      pitch: 0.1,
      fovDeg: 40,
      aperture: 0,
    },
  };
}


/** 参加媒質の確認用。窓の入った暗い部屋に霧を満たし、光芒を出す */
function buildShaftScene(): Scene {
  const wall = lambert([0.55, 0.53, 0.5]);
  const W = 8;
  const H = 5;
  const D = 11;

  // +x の壁を縦の板に分け、隙間から陽が差し込むようにする
  const slats: Quad[] = [];
  for (let i = 0; i < 7; i++) {
    const z0 = i * 1.62;
    slats.push({
      q: [W, 0, z0],
      u: [0, H, 0],
      v: [0, 0, 1.05],
      material: wall,
    });
  }

  return {
    spheres: [
      { center: [4.8, 0.9, 8.5], radius: 0.9, material: ggx([0.95, 0.82, 0.5], 0.14) },
    ],
    quads: [
      { q: [0, 0, 0], u: [W, 0, 0], v: [0, 0, D], material: lambert([0.42, 0.4, 0.38]) },
      { q: [0, H, 0], u: [W, 0, 0], v: [0, 0, D], material: wall },
      { q: [0, 0, 0], u: [0, H, 0], v: [0, 0, D], material: wall },
      { q: [0, 0, D], u: [W, 0, 0], v: [0, H, 0], material: wall },
      { q: [0, 0, 0], u: [W, 0, 0], v: [0, H, 0], material: wall },
      ...slats,
    ],
    triangles: [],
    env: ENV.hdri,
    fog: { min: [0, 0, 0], max: [W, H, D], sigmaS: 0.09, sigmaA: 0.006, g: 0.62 },
    camera: {
      target: [5.4, 2.2, 6.0],
      distance: 4.4,
      yaw: Math.PI * 0.97,
      pitch: 0.06,
      fovDeg: 60,
      aperture: 0,
    },
  };
}


/**
 * 間接照明が支配的なシーン。単方向パストレーシングの弱点を見るためのもの。
 *
 * 部屋を隔壁で 2 つに分け、細い隙間だけでつないである。光源は奥側で上を
 * 向いているので、手前側を照らす光は「天井で跳ね返り、隙間を通ってくる」
 * 経路しかない。手前の面から光源への影レイはほぼ必ず遮られるので、NEE が
 * ほとんど効かない。
 */
function buildIndirectScene(): Scene {
  const X = 11;
  const Y = 5;
  const Z = 8;
  const warm = lambert([0.62, 0.57, 0.5]);
  const cool = lambert([0.46, 0.5, 0.58]);
  const floor = lambert([0.5, 0.48, 0.45]);

  // 隔壁。z 方向に細い隙間 (扉) を残す
  // 隙間を狭めるほど NEE が効かなくなり、単方向パストレーシングには厳しくなる。
  // 幅 1.0 / 0.4 / 0.15 で必要な spp は cornell 比 2.4 / 7.1 / 17.4 倍だった
  const gap0 = 3.8;
  const gap1 = 4.2;
  const bx = 6.0;

  return {
    spheres: [
      { center: [2.6, 0.85, 3.2], radius: 0.85, material: lambert([0.72, 0.7, 0.66]) },
      { center: [4.3, 0.6, 5.4], radius: 0.6, material: ggx([0.95, 0.9, 0.8], 0.2) },
    ],
    quads: [
      { q: [0, 0, 0], u: [X, 0, 0], v: [0, 0, Z], material: floor },
      { q: [0, Y, 0], u: [X, 0, 0], v: [0, 0, Z], material: warm },
      { q: [0, 0, 0], u: [0, Y, 0], v: [0, 0, Z], material: cool },
      { q: [X, 0, 0], u: [0, Y, 0], v: [0, 0, Z], material: warm },
      { q: [0, 0, 0], u: [X, 0, 0], v: [0, Y, 0], material: cool },
      { q: [0, 0, Z], u: [X, 0, 0], v: [0, Y, 0], material: cool },
      // 隔壁 (隙間の手前と奥)
      { q: [bx, 0, 0], u: [0, Y, 0], v: [0, 0, gap0], material: warm },
      { q: [bx, 0, gap1], u: [0, Y, 0], v: [0, 0, Z - gap1], material: warm },
      // 奥側の光源。上を向いているので直接は手前を照らさない
      {
        q: [8.6, 0.35, 2.6],
        u: [0, 0, 2.4],
        v: [1.9, 0, 0],
        material: emissive([175, 162, 136]),
      },
    ],
    triangles: [],
    env: ENV.black,
    camera: {
      target: [4.0, 1.7, 3.0],
      distance: 3.6,
      yaw: Math.PI * 0.85,
      pitch: 0.12,
      fovDeg: 70,
      aperture: 0,
    },
  };
}


/**
 * BDPT の必要性を見るための、意図的に過酷なシーン。
 *
 * 光源を箱で囲い、細い隙間だけ残してある。部屋のどの面から光源へ影レイを
 * 飛ばしても、隙間を通る細い立体角に入らない限り遮られる。つまり NEE が
 * ほぼ全域で無効になり、カメラ側の経路が偶然その隙間に入るのを待つしかない。
 */
function buildEnclosedScene(): Scene {
  const R = 8;
  const H = 5;
  const wall = lambert([0.6, 0.58, 0.55]);
  // 箱の内側は明るくしておく。中で何度も跳ねてから隙間を抜ける
  const inner = lambert([0.86, 0.85, 0.82]);

  // 光源を囲う箱
  const x0 = 3.0;
  const x1 = 5.0;
  const y0 = 2.0;
  const y1 = 3.6;
  const z0 = 3.0;
  const z1 = 5.0;
  // -x 面に残す隙間 (幅 0.3)
  const s0 = 3.85;
  const s1 = 4.15;

  return {
    spheres: [
      { center: [1.9, 0.75, 2.4], radius: 0.75, material: lambert([0.7, 0.68, 0.64]) },
      { center: [6.3, 0.7, 6.0], radius: 0.7, material: ggx([0.95, 0.9, 0.82], 0.22) },
    ],
    quads: [
      { q: [0, 0, 0], u: [R, 0, 0], v: [0, 0, R], material: lambert([0.52, 0.5, 0.47]) },
      { q: [0, H, 0], u: [R, 0, 0], v: [0, 0, R], material: wall },
      { q: [0, 0, 0], u: [0, H, 0], v: [0, 0, R], material: wall },
      { q: [R, 0, 0], u: [0, H, 0], v: [0, 0, R], material: wall },
      { q: [0, 0, 0], u: [R, 0, 0], v: [0, H, 0], material: wall },
      { q: [0, 0, R], u: [R, 0, 0], v: [0, H, 0], material: wall },

      // 囲いの箱。-x 面だけ隙間を空ける
      { q: [x1, y0, z0], u: [0, y1 - y0, 0], v: [0, 0, z1 - z0], material: inner },
      { q: [x0, y0, z0], u: [0, y1 - y0, 0], v: [0, 0, s0 - z0], material: inner },
      { q: [x0, y0, s1], u: [0, y1 - y0, 0], v: [0, 0, z1 - s1], material: inner },
      { q: [x0, y1, z0], u: [x1 - x0, 0, 0], v: [0, 0, z1 - z0], material: inner },
      { q: [x0, y0, z0], u: [x1 - x0, 0, 0], v: [0, 0, z1 - z0], material: inner },
      { q: [x0, y0, z1], u: [x1 - x0, 0, 0], v: [0, y1 - y0, 0], material: inner },
      { q: [x0, y0, z0], u: [x1 - x0, 0, 0], v: [0, y1 - y0, 0], material: inner },

      // 隙間の正面に目隠しを置き、部屋から光源への直線をほぼ塞ぐ。
      // 光は隙間 -> 目隠しの裏 -> 縁を回り込む、という経路でしか部屋に出られない
      { q: [2.55, 1.6, 3.35], u: [0, 2.4, 0], v: [0, 0, 1.3], material: inner },

      // 囲いの中の光源。閉じている +x 面を向いている
      {
        q: [3.35, 2.3, 3.4],
        u: [0, 1.0, 0],
        v: [0, 0, 1.2],
        material: emissive([260, 245, 210]),
      },
    ],
    triangles: [],
    env: ENV.black,
    camera: {
      target: [4.5, 2.0, 4.2],
      distance: 3.4,
      yaw: Math.PI * 0.8,
      pitch: 0.1,
      fovDeg: 66,
      aperture: 0,
    },
  };
}


/**
 * 蛇腹の板の枚数。
 * 光源強度を固定して測ると、届く光は 1 枚 -> 2 枚で約 1/13、
 * 2 枚 -> 3 枚で約 1/925 に落ちる。相対ノイズは 1.2% / 47.3% / 420.7%。
 * 3 枚では cornell と同品質にするのに約 1600 倍の spp が要り、実質描けない。
 * 同梱するのは 2 枚。厳しいが絵は見える。
 */
const MAZE_BAFFLES = 2;

/**
 * 光源を蛇腹の板の奥に置いたシーン。
 *
 * 板は左右交互に、隣どうしが必ず重なるように置いてある。そのため光源から
 * 部屋まで一直線に通る経路が存在せず、板 1 枚につき最低 1 回の反射が
 * 強制される。難易度が枚数に対してどう伸びるかを測るためのもの。
 */
function buildMazeScene(): Scene {
  const X = 10;
  const Y = 5;
  const Z = 14;
  const wall = lambert([0.62, 0.6, 0.56]);
  const panel = lambert([0.72, 0.71, 0.68]);

  // 偶数枚目は左端から x=7 まで、奇数枚目は x=3 から右端まで。
  // x = 3..7 が必ず重なるので、直線では抜けられない
  const z0 = 6.0;
  const step = 1.3;
  const baffles: Quad[] = [];
  for (let i = 0; i < MAZE_BAFFLES; i++) {
    const even = i % 2 === 0;
    const xa = even ? 0 : 3;
    const xb = even ? 7 : X;
    baffles.push({
      q: [xa, 0, z0 + i * step],
      u: [xb - xa, 0, 0],
      v: [0, Y, 0],
      material: panel,
    });
  }

  const lightZ = z0 + MAZE_BAFFLES * step + 1.6;

  return {
    spheres: [
      { center: [2.6, 0.9, 2.6], radius: 0.9, material: lambert([0.72, 0.7, 0.66]) },
      { center: [6.6, 0.55, 1.8], radius: 0.55, material: ggx([0.95, 0.9, 0.82], 0.2) },
    ],
    quads: [
      { q: [0, 0, 0], u: [X, 0, 0], v: [0, 0, Z], material: lambert([0.5, 0.48, 0.45]) },
      { q: [0, Y, 0], u: [X, 0, 0], v: [0, 0, Z], material: wall },
      { q: [0, 0, 0], u: [0, Y, 0], v: [0, 0, Z], material: lambert([0.45, 0.5, 0.55]) },
      { q: [X, 0, 0], u: [0, Y, 0], v: [0, 0, Z], material: lambert([0.58, 0.45, 0.4]) },
      { q: [0, 0, 0], u: [X, 0, 0], v: [0, Y, 0], material: wall },
      { q: [0, 0, Z], u: [X, 0, 0], v: [0, Y, 0], material: wall },
      ...baffles,
      // 一番奥の光源
      {
        q: [4.2, 1.5, lightZ],
        u: [1.6, 0, 0],
        v: [0, 1.4, 0],
        material: emissive([7000, 6580, 5740]),
      },
    ],
    triangles: [],
    env: ENV.black,
    camera: {
      target: [5.0, 2.0, 4.2],
      distance: 3.7,
      yaw: Math.PI * 1.5,
      pitch: 0.13,
      fovDeg: 72,
      aperture: 0,
    },
  };
}


/**
 * 水面の高さ。
 * dynamic を立てると低周波のうねりと高周波のさざ波を足して荒くする。
 * 傾きの変化が大きくなるぶん、集光のコントラストが上がる。
 */
function waveHeight(x: number, z: number, amp: number, dynamic = false): number {
  const base =
    Math.sin(6.1 * x + 1.3) * Math.cos(5.3 * z) +
    0.6 * Math.sin(9.7 * z + 2.1) * Math.cos(8.3 * x) +
    0.35 * Math.sin(15.1 * (x * 0.7 + z * 0.7) + 0.7);
  if (!dynamic) {
    return amp * base;
  }
  // 大きなうねりを主役にする。高周波を足しすぎると集光が細かい格子に
  // 割れてしまい、力強い筋にならない
  return (
    amp *
    (1.55 * Math.sin(2.6 * x - 1.8 * z + 0.4) +
      1.25 * Math.sin(3.6 * z + 1.9) * Math.cos(3.1 * x) +
      0.8 * Math.sin(6.1 * x + 1.3) * Math.cos(5.3 * z) +
      0.42 * Math.sin(9.7 * z + 2.1) * Math.cos(8.3 * x) +
      0.16 * Math.sin(15.1 * (x * 0.7 + z * 0.7) + 0.7))
  );
}

/** 波打つ水面をハイトフィールドのメッシュにする。法線は差分から求める */
function waterSurface(
  x0: number,
  z0: number,
  size: number,
  level: number,
  amp: number,
  n: number,
  material: Material,
  dynamic = false,
): Triangle[] {
  const norm = (v: Vec3): Vec3 => {
    const l = Math.hypot(v[0], v[1], v[2]) || 1;
    return [v[0] / l, v[1] / l, v[2] / l];
  };
  const E = 1e-3;
  const pos: Vec3[][] = [];
  const nrm: Vec3[][] = [];
  for (let i = 0; i <= n; i++) {
    const px: Vec3[] = [];
    const nx: Vec3[] = [];
    for (let j = 0; j <= n; j++) {
      const x = x0 + (i / n) * size;
      const z = z0 + (j / n) * size;
      px.push([x, level + waveHeight(x, z, amp, dynamic), z]);
      // 上向きの法線 (-dh/dx, 1, -dh/dz)
      const hx =
        (waveHeight(x + E, z, amp, dynamic) - waveHeight(x - E, z, amp, dynamic)) / (2 * E);
      const hz =
        (waveHeight(x, z + E, amp, dynamic) - waveHeight(x, z - E, amp, dynamic)) / (2 * E);
      nx.push(norm([-hx, 1, -hz]));
    }
    pos.push(px);
    nrm.push(nx);
  }

  const tris: Triangle[] = [];
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const a = { p: pos[i][j], n: nrm[i][j] };
      const b = { p: pos[i + 1][j], n: nrm[i + 1][j] };
      const c = { p: pos[i + 1][j + 1], n: nrm[i + 1][j + 1] };
      const d = { p: pos[i][j + 1], n: nrm[i][j + 1] };
      tris.push({ v0: a.p, v1: b.p, v2: c.p, n0: a.n, n1: b.n, n2: c.n, material });
      tris.push({ v0: a.p, v1: c.p, v2: d.p, n0: a.n, n1: c.n, n2: d.n, material });
    }
  }
  return tris;
}

/**
 * Cornell box に水を張ったシーン。天井の光源が波打つ水面で屈折し、
 * プールの底に集光模様を作る。単方向パストレーシングでは届かない
 * 経路なので、SPPM の効きどころがそのまま見える。
 */
function buildWaterScene(): Scene {
  const S = 5.55;
  const white = lambert([0.76, 0.75, 0.72]);
  const red = lambert([0.65, 0.05, 0.05]);
  const green = lambert([0.12, 0.45, 0.15]);
  // わずかに青緑がかった水。1 単位距離あたりの透過色
  const water = dielectric(1.33, 0, [0.72, 0.9, 0.86]);

  return {
    spheres: [],
    quads: [
      { q: [S, 0, 0], u: [0, S, 0], v: [0, 0, S], material: green },
      { q: [0, 0, 0], u: [0, S, 0], v: [0, 0, S], material: red },
      { q: [0, 0, 0], u: [S, 0, 0], v: [0, 0, S], material: white },
      { q: [S, S, S], u: [-S, 0, 0], v: [0, 0, -S], material: white },
      { q: [0, 0, S], u: [S, 0, 0], v: [0, S, 0], material: white },
      // 集光を鋭くしたいので光源は小さめ
      {
        q: [3.15, S - 0.01, 3.1],
        u: [-0.75, 0, 0],
        v: [0, 0, -0.65],
        material: emissive([95, 90, 78]),
      },
    ],
    // 波打つ縁が見えないよう、水面は壁の外まで伸ばす
    triangles: waterSurface(-0.2, -0.2, S + 0.4, 1.15, 0.032, 210, water),
    env: ENV.black,
    camera: {
      target: [S / 2, 1.5, S / 2],
      distance: 9.6,
      yaw: -Math.PI * 0.5,
      pitch: 0.3,
      fovDeg: 42,
      aperture: 0,
    },
  };
}


/**
 * 部屋ごと水没した Cornell box。水面は天井のすぐ下にあり、箱も床も水中。
 * 天井の光源が波打つ水面で屈折して、床だけでなく左右の壁や箱にまで
 * 集光が回る。水面を下から見ると臨界角の外は全反射して鏡のようになる。
 */
function buildSubmergedScene(): Scene {
  const S = 5.55;
  const white = lambert([0.76, 0.75, 0.72]);
  const red = lambert([0.65, 0.05, 0.05]);
  const green = lambert([0.12, 0.45, 0.15]);
  const water = dielectric(1.33, 0, [0.86, 0.95, 0.93]);

  return {
    spheres: [],
    quads: [
      { q: [S, 0, 0], u: [0, S, 0], v: [0, 0, S], material: green },
      { q: [0, 0, 0], u: [0, S, 0], v: [0, 0, S], material: red },
      { q: [0, 0, 0], u: [S, 0, 0], v: [0, 0, S], material: white },
      { q: [S, S, S], u: [-S, 0, 0], v: [0, 0, -S], material: white },
      { q: [0, 0, S], u: [S, 0, 0], v: [0, S, 0], material: white },
      // 水面より上にある光源。小さめにして集光を鋭くする
      {
        q: [2.87, S - 0.01, 2.87],
        u: [-0.19, 0, 0],
        v: [0, 0, -0.16],
        material: emissive([2700, 2560, 2200]),
      },
      // 沈んでいる 2 つの箱
      ...place(box([0, 0, 0], [1.65, 3.3, 1.65], white), 15, [2.65, 0, 2.95]),
      ...place(box([0, 0, 0], [1.65, 1.65, 1.65], white), -18, [1.3, 0, 0.65]),
    ],
    // 水面は天井のすぐ下。壁の外まで伸ばして縁を隠す
    triangles: waterSurface(-0.25, -0.25, S + 0.5, 4.3, 0.085, 260, water, true),
    env: ENV.black,
    camera: {
      target: [S / 2, 2.4, S / 2],
      distance: 10.2,
      yaw: -Math.PI * 0.5,
      pitch: 0.07,
      fovDeg: 44,
      aperture: 0,
    },
  };
}

export interface SceneEntry {
  id: string;
  name: string;
  build: () => Scene;
}

export const SCENES: SceneEntry[] = [
  { id: "spheres", name: "spheres (RTIOW)", build: buildSpheresScene },
  { id: "cornell", name: "cornell box", build: buildCornellScene },
  { id: "veach", name: "veach MIS test", build: buildVeachScene },
  { id: "mesh", name: "torus knot (mesh)", build: buildMeshScene },
  { id: "glass", name: "rough / colored glass", build: buildGlassScene },
  { id: "shaft", name: "light shafts (fog)", build: buildShaftScene },
  { id: "indirect", name: "indirect only (hard)", build: buildIndirectScene },
  { id: "enclosed", name: "enclosed light (brutal)", build: buildEnclosedScene },
  { id: "maze", name: "baffle maze (BDPT test)", build: buildMazeScene },
  { id: "ajar", name: "ajar door (SPPM test)", build: buildAjarDoorScene },
  { id: "water", name: "water caustics", build: buildWaterScene },
  { id: "submerged", name: "submerged cornell", build: buildSubmergedScene },
];

export const DEFAULT_SCENE_ID = SCENES[0].id;

export function buildSceneById(id: string): Scene {
  const entry = SCENES.find((s) => s.id === id) ?? SCENES[0];
  return entry.build();
}

// ---------------------------------------------------------------- packing
/** WGSL 側の struct と一致させること (単位: バイト) */
export const MATERIAL_STRIDE = 48;
export const SPHERE_STRIDE = 64;
export const QUAD_STRIDE = 96;
export const TRIANGLE_STRIDE = 144;

/** o は float 単位のオフセット */
function writeMaterial(
  f32: Float32Array,
  u32: Uint32Array,
  o: number,
  m: Material,
) {
  f32[o + 0] = m.albedo[0];
  f32[o + 1] = m.albedo[1];
  f32[o + 2] = m.albedo[2];
  f32[o + 3] = m.roughness ?? 0;
  const e = m.emission ?? [0, 0, 0];
  f32[o + 4] = e[0];
  f32[o + 5] = e[1];
  f32[o + 6] = e[2];
  f32[o + 7] = m.ior ?? 1.5;
  u32[o + 8] = m.kind;
}

/** storage buffer にそのまま書ける形へパックする */
export function packSpheres(spheres: Sphere[]): ArrayBuffer {
  const buffer = new ArrayBuffer(Math.max(1, spheres.length) * SPHERE_STRIDE);
  const f32 = new Float32Array(buffer);
  const u32 = new Uint32Array(buffer);
  spheres.forEach((s, i) => {
    const o = (i * SPHERE_STRIDE) / 4;
    f32[o + 0] = s.center[0];
    f32[o + 1] = s.center[1];
    f32[o + 2] = s.center[2];
    f32[o + 3] = s.radius;
    writeMaterial(f32, u32, o + 4, s.material);
  });
  return buffer;
}

/**
 * NEE で直接サンプルする面光源 (emissive な quad) のインデックス列。
 * 球の面光源には対応していない。
 */
export function packLights(quads: Quad[]): { data: ArrayBuffer; count: number } {
  const indices = quads.flatMap((q, i) =>
    q.material.kind === MATERIAL.emissive ? [i] : [],
  );
  const data = new ArrayBuffer(Math.max(1, indices.length) * 4);
  new Uint32Array(data).set(indices);
  return { data, count: indices.length };
}

export function packTriangles(tris: Triangle[]): ArrayBuffer {
  const buffer = new ArrayBuffer(Math.max(1, tris.length) * TRIANGLE_STRIDE);
  const f32 = new Float32Array(buffer);
  const u32 = new Uint32Array(buffer);
  tris.forEach((t, i) => {
    const o = (i * TRIANGLE_STRIDE) / 4;
    f32.set(t.v0, o + 0);
    // 辺は原点を v0 に取った差分で持つ (Moller-Trumbore がそのまま使える)
    f32.set([t.v1[0] - t.v0[0], t.v1[1] - t.v0[1], t.v1[2] - t.v0[2]], o + 4);
    f32.set([t.v2[0] - t.v0[0], t.v2[1] - t.v0[1], t.v2[2] - t.v0[2]], o + 8);
    f32.set(t.n0, o + 12);
    f32.set(t.n1, o + 16);
    f32.set(t.n2, o + 20);
    writeMaterial(f32, u32, o + 24, t.material);
  });
  return buffer;
}

export function packQuads(quads: Quad[]): ArrayBuffer {
  const buffer = new ArrayBuffer(Math.max(1, quads.length) * QUAD_STRIDE);
  const f32 = new Float32Array(buffer);
  const u32 = new Uint32Array(buffer);
  quads.forEach((quad, i) => {
    const o = (i * QUAD_STRIDE) / 4;
    f32.set(quad.q, o + 0);
    f32.set(quad.u, o + 4);
    f32.set(quad.v, o + 8);
    writeMaterial(f32, u32, o + 12, quad.material);
  });
  return buffer;
}
