export type Vec3 = [number, number, number];

export const MATERIAL = {
  lambert: 0,
  metal: 1,
  dielectric: 2,
  emissive: 3,
} as const;

export type MaterialKind = (typeof MATERIAL)[keyof typeof MATERIAL];

export interface Material {
  kind: MaterialKind;
  /** lambert / metal のアルベド。dielectric では減衰色として使う */
  albedo: Vec3;
  /** metal のざらつき (0 で鏡面) */
  fuzz?: number;
  /** dielectric の屈折率 */
  ior?: number;
  /** emissive の放射輝度 */
  emission?: Vec3;
}

export const lambert = (albedo: Vec3): Material => ({
  kind: MATERIAL.lambert,
  albedo,
});
export const metal = (albedo: Vec3, fuzz = 0): Material => ({
  kind: MATERIAL.metal,
  albedo,
  fuzz,
});
export const dielectric = (ior = 1.5): Material => ({
  kind: MATERIAL.dielectric,
  albedo: [1, 1, 1],
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

export interface Scene {
  spheres: Sphere[];
  quads: Quad[];
  env: EnvKind;
  camera: CameraPreset;
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
    { center: [2.6, 1, 0], radius: 1, material: metal([0.7, 0.6, 0.5], 0.02) },
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
          material: metal(
            [0.5 + 0.5 * rand(), 0.5 + 0.5 * rand(), 0.5 + 0.5 * rand()],
            0.3 * rand(),
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
    env: ENV.sky,
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

export interface SceneEntry {
  id: string;
  name: string;
  build: () => Scene;
}

export const SCENES: SceneEntry[] = [
  { id: "spheres", name: "spheres (RTIOW)", build: buildSpheresScene },
  { id: "cornell", name: "cornell box", build: buildCornellScene },
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
  f32[o + 3] = m.fuzz ?? 0;
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
