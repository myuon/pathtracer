import type { Quad, Sphere, Triangle, Vec3 } from "./scene";

/** WGSL 側の struct BvhNode と一致させること */
export const BVH_NODE_STRIDE = 32;

/** 葉に入れるプリミティブ数の上限 */
const LEAF_SIZE = 4;

/** プリミティブ参照の型タグ。上位 2 ビットに入れる。WGSL 側と一致させること */
const TYPE_SPHERE = 0;
const TYPE_QUAD = 1;
const TYPE_TRIANGLE = 2;

const encode = (type: number, index: number) => ((type << 30) | index) >>> 0;

interface Ref {
  /** (type << 30) | index */
  code: number;
  min: Vec3;
  max: Vec3;
  centroid: Vec3;
}

interface Node {
  min: Vec3;
  max: Vec3;
  /** 葉なら refs の開始位置、内部ノードなら右の子のノード番号 */
  leftFirst: number;
  /** 0 なら内部ノード */
  count: number;
}

function boundsOf(points: Vec3[]): [Vec3, Vec3] {
  const min: Vec3 = [Infinity, Infinity, Infinity];
  const max: Vec3 = [-Infinity, -Infinity, -Infinity];
  for (const p of points) {
    for (let i = 0; i < 3; i++) {
      min[i] = Math.min(min[i], p[i]);
      max[i] = Math.max(max[i], p[i]);
    }
  }
  // 軸に平行な板は厚みが 0 になり、レイとの交差判定が不安定になるので少し膨らませる
  const EPS = 1e-4;
  for (let i = 0; i < 3; i++) {
    if (max[i] - min[i] < EPS) {
      min[i] -= EPS;
      max[i] += EPS;
    }
  }
  return [min, max];
}

function makeRef(code: number, min: Vec3, max: Vec3): Ref {
  return {
    code,
    min,
    max,
    centroid: [
      (min[0] + max[0]) * 0.5,
      (min[1] + max[1]) * 0.5,
      (min[2] + max[2]) * 0.5,
    ],
  };
}

function collectRefs(spheres: Sphere[], quads: Quad[], triangles: Triangle[]): Ref[] {
  const refs: Ref[] = [];
  spheres.forEach((s, i) => {
    const r = Math.abs(s.radius);
    const [min, max] = boundsOf([
      [s.center[0] - r, s.center[1] - r, s.center[2] - r],
      [s.center[0] + r, s.center[1] + r, s.center[2] + r],
    ]);
    refs.push(makeRef(encode(TYPE_SPHERE, i), min, max));
  });
  quads.forEach((q, i) => {
    const [min, max] = boundsOf([
      q.q,
      [q.q[0] + q.u[0], q.q[1] + q.u[1], q.q[2] + q.u[2]],
      [q.q[0] + q.v[0], q.q[1] + q.v[1], q.q[2] + q.v[2]],
      [
        q.q[0] + q.u[0] + q.v[0],
        q.q[1] + q.u[1] + q.v[1],
        q.q[2] + q.u[2] + q.v[2],
      ],
    ]);
    refs.push(makeRef(encode(TYPE_QUAD, i), min, max));
  });
  triangles.forEach((t, i) => {
    const [min, max] = boundsOf([t.v0, t.v1, t.v2]);
    refs.push(makeRef(encode(TYPE_TRIANGLE, i), min, max));
  });
  return refs;
}

function unionBounds(refs: Ref[], from: number, to: number): [Vec3, Vec3] {
  const min: Vec3 = [Infinity, Infinity, Infinity];
  const max: Vec3 = [-Infinity, -Infinity, -Infinity];
  for (let i = from; i < to; i++) {
    for (let a = 0; a < 3; a++) {
      min[a] = Math.min(min[a], refs[i].min[a]);
      max[a] = Math.max(max[a], refs[i].max[a]);
    }
  }
  return [min, max];
}

/**
 * 中央値分割の BVH。プリミティブ数が少ないおもちゃなので SAH までは行わない。
 * refs は再帰の過程で並べ替えられ、葉はその連続範囲を指す。
 */
function build(refs: Ref[], from: number, to: number, nodes: Node[]): number {
  const [min, max] = unionBounds(refs, from, to);
  const self = nodes.length;
  nodes.push({ min, max, leftFirst: from, count: to - from });

  if (to - from <= LEAF_SIZE) {
    return self;
  }

  // 一番広がっている軸の中央値で二分する
  const extent: Vec3 = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
  let axis = 0;
  if (extent[1] > extent[axis]) axis = 1;
  if (extent[2] > extent[axis]) axis = 2;

  const slice = refs.slice(from, to).sort((a, b) => a.centroid[axis] - b.centroid[axis]);
  for (let i = 0; i < slice.length; i++) refs[from + i] = slice[i];

  const mid = (from + to) >> 1;
  nodes[self].count = 0;
  build(refs, from, mid, nodes);
  // 左の子は自分の直後に置かれるので、右の子だけ番号を控える
  nodes[self].leftFirst = build(refs, mid, to, nodes);
  return self;
}

export interface BvhData {
  nodes: ArrayBuffer;
  refs: ArrayBuffer;
  nodeCount: number;
  /** シーン全体の AABB。SPPM の初期半径やセルの大きさを決めるのに使う */
  bounds: { min: Vec3; max: Vec3 };
}

export function buildBvh(
  spheres: Sphere[],
  quads: Quad[],
  triangles: Triangle[],
): BvhData {
  const refs = collectRefs(spheres, quads, triangles);
  const nodes: Node[] = [];
  if (refs.length > 0) {
    build(refs, 0, refs.length, nodes);
  } else {
    // 空のシーンでも 1 ノードは置く (WebGPU は 0 バイトのバッファを作れない)
    nodes.push({ min: [0, 0, 0], max: [0, 0, 0], leftFirst: 0, count: 0 });
  }

  const nodeBuf = new ArrayBuffer(nodes.length * BVH_NODE_STRIDE);
  const f32 = new Float32Array(nodeBuf);
  const u32 = new Uint32Array(nodeBuf);
  nodes.forEach((n, i) => {
    const o = (i * BVH_NODE_STRIDE) / 4;
    f32.set(n.min, o);
    u32[o + 3] = n.leftFirst;
    f32.set(n.max, o + 4);
    u32[o + 7] = n.count;
  });

  const refBuf = new ArrayBuffer(Math.max(1, refs.length) * 4);
  new Uint32Array(refBuf).set(refs.map((r) => r.code >>> 0));

  return {
    nodes: nodeBuf,
    refs: refBuf,
    nodeCount: nodes.length,
    bounds: { min: nodes[0].min, max: nodes[0].max },
  };
}
