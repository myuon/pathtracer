export const MATERIAL = {
  lambert: 0,
  metal: 1,
  dielectric: 2,
} as const;

export type MaterialType = (typeof MATERIAL)[keyof typeof MATERIAL];

export interface Sphere {
  center: [number, number, number];
  radius: number;
  material: MaterialType;
  /** lambert / metal のアルベド。dielectric では減衰色として使う */
  albedo: [number, number, number];
  /** metal のざらつき (0 で鏡面) */
  fuzz?: number;
  /** dielectric の屈折率 */
  ior?: number;
}

/** Sphere 1 個あたりのバイト数 (WGSL 側の struct Sphere と一致させること) */
export const SPHERE_STRIDE = 48;

/** 決定的な擬似乱数 (シーンを毎回同じにするため) */
function makeRng(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

/** Ray Tracing in One Weekend 風の定番シーン */
export function buildScene(): Sphere[] {
  const spheres: Sphere[] = [
    // 地面 (巨大球)
    {
      center: [0, -1000, 0],
      radius: 1000,
      material: MATERIAL.lambert,
      albedo: [0.5, 0.5, 0.5],
    },
    {
      center: [-2.6, 1, 0],
      radius: 1,
      material: MATERIAL.dielectric,
      albedo: [1, 1, 1],
      ior: 1.5,
    },
    {
      center: [0, 1, 0],
      radius: 1,
      material: MATERIAL.lambert,
      albedo: [0.55, 0.25, 0.18],
    },
    {
      center: [2.6, 1, 0],
      radius: 1,
      material: MATERIAL.metal,
      albedo: [0.7, 0.6, 0.5],
      fuzz: 0.02,
    },
  ];

  // 周りに小球を散らす
  const rand = makeRng(20240819);
  for (let a = -5; a < 5; a++) {
    for (let b = -4; b < 4; b++) {
      const center: [number, number, number] = [
        a * 1.1 + 0.7 * rand(),
        0.2,
        b * 1.1 + 0.7 * rand(),
      ];
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
          material: MATERIAL.lambert,
          albedo: [rand() * rand(), rand() * rand(), rand() * rand()],
        });
      } else if (pick < 0.93) {
        spheres.push({
          center,
          radius: 0.2,
          material: MATERIAL.metal,
          albedo: [0.5 + 0.5 * rand(), 0.5 + 0.5 * rand(), 0.5 + 0.5 * rand()],
          fuzz: 0.3 * rand(),
        });
      } else {
        spheres.push({
          center,
          radius: 0.2,
          material: MATERIAL.dielectric,
          albedo: [1, 1, 1],
          ior: 1.5,
        });
      }
    }
  }
  return spheres;
}

/** storage buffer にそのまま書ける形へパックする */
export function packSpheres(spheres: Sphere[]): ArrayBuffer {
  const buffer = new ArrayBuffer(spheres.length * SPHERE_STRIDE);
  const f32 = new Float32Array(buffer);
  const u32 = new Uint32Array(buffer);
  spheres.forEach((s, i) => {
    const o = (i * SPHERE_STRIDE) / 4;
    f32[o + 0] = s.center[0];
    f32[o + 1] = s.center[1];
    f32[o + 2] = s.center[2];
    f32[o + 3] = s.radius;
    f32[o + 4] = s.albedo[0];
    f32[o + 5] = s.albedo[1];
    f32[o + 6] = s.albedo[2];
    f32[o + 7] = s.fuzz ?? 0;
    u32[o + 8] = s.material;
    f32[o + 9] = s.ior ?? 1.5;
  });
  return buffer;
}
