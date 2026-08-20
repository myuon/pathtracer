import pathtraceWgsl from "./shaders/pathtrace.wgsl?raw";
import presentWgsl from "./shaders/present.wgsl?raw";
import { buildBvh } from "./bvh";
import { buildSunSkyEnv } from "./envmap";
import { ENV, packLights, packQuads, packSpheres, packTriangles, type Scene } from "./scene";

export interface GpuContext {
  device: GPUDevice;
  context: GPUCanvasContext;
  format: GPUTextureFormat;
  canvas: HTMLCanvasElement;
}

export class WebGpuUnsupportedError extends Error {}

export async function initGpu(canvas: HTMLCanvasElement): Promise<GpuContext> {
  if (!navigator.gpu) {
    throw new WebGpuUnsupportedError("navigator.gpu が見つかりません");
  }
  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: "high-performance",
  });
  if (!adapter) {
    throw new WebGpuUnsupportedError("GPUAdapter を取得できませんでした");
  }
  // 既定の上限 (128MB) では高解像度ディスプレイで履歴バッファが収まらない。
  // アダプタが許す範囲で引き上げておき、駄目なら既定のまま続行する
  const cap = 1024 * 1024 * 1024;
  const requiredLimits = {
    maxStorageBufferBindingSize: Math.min(
      adapter.limits.maxStorageBufferBindingSize,
      cap,
    ),
    maxBufferSize: Math.min(adapter.limits.maxBufferSize, cap),
    // SPPM で光子とグリッドを足したので既定の 8 本では足りない
    maxStorageBuffersPerShaderStage: Math.min(
      adapter.limits.maxStorageBuffersPerShaderStage,
      12,
    ),
  };
  const device = await adapter
    .requestDevice({ requiredLimits })
    .catch(() => adapter.requestDevice());
  device.lost.then((info) => {
    console.error("WebGPU device lost:", info.reason, info.message);
  });

  const context = canvas.getContext("webgpu");
  if (!context) {
    throw new WebGpuUnsupportedError("webgpu コンテキストを取得できませんでした");
  }
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: "opaque" });

  return { device, context, format, canvas };
}

/** WGSL 側の struct Uniforms と一致させること */
const UNIFORM_SIZE = 304;

/**
 * 履歴バッファの 1 画素あたりのバイト数。
 * WGSL 側の vec4f 2 個 (色とサンプル数 / 1 次交差の世界座標) と一致させること。
 * ここと maxPixels() がずれると、上限判定をすり抜けて bind group の作成が落ちる。
 */
const HIST_BYTES_PER_PIXEL = 64;
const WORKGROUP = 8;

/** 1 反復あたりに撒く光子の数 */
const PHOTON_COUNT = 1 << 17;
/** ハッシュグリッドのセル数 */
const GRID_CELLS = 1 << 18;
/** 1 セルに入る光子の上限。WGSL 側の GRID_CAP と一致させること */
const GRID_CAP = 48;

export interface FrameParams {
  camPos: [number, number, number];
  camU: [number, number, number];
  camV: [number, number, number];
  camW: [number, number, number];
  tanHalfFov: number;
  focusDist: number;
  lensRadius: number;
  width: number;
  height: number;
  frameIndex: number;
  sppPerFrame: number;
  maxBounces: number;
  /** このフレームより前に積んだサンプル数。0 なら accum を上書きする */
  samplesBefore: number;
  /** 面光源を直接サンプルする (next event estimation) */
  nee: boolean;
  /** 光源サンプリングと BSDF サンプリングを power heuristic で合成する */
  mis: boolean;
  /** スクランブル済み Sobol (0,2) 列を使う */
  qmc: boolean;
  /** 環境マップを光源としてサンプルする */
  envIs: boolean;
  /** 累積を前フレームから再投影して引き継ぐ */
  reproject: boolean;
  /** 0 なら通常描画、それ以外は中間量を疑似カラーで出す */
  debugMode: number;
  /** 乱数の種を累積サンプル数から作り、同条件を再現可能にする */
  fixedSeed: boolean;
  /** 参加媒質を有効にする */
  fog: boolean;
  /** SPPM を使う */
  sppm: boolean;
}

export class Renderer {
  private readonly device: GPUDevice;
  private readonly context: GPUCanvasContext;
  private readonly uniformBuffer: GPUBuffer;
  private readonly uniformData = new ArrayBuffer(UNIFORM_SIZE);
  private readonly uniformF32 = new Float32Array(this.uniformData);
  private readonly uniformU32 = new Uint32Array(this.uniformData);
  private readonly computePipeline: GPUComputePipeline;
  private readonly sppmPipeline: GPUComputePipeline;
  private readonly photonPipeline: GPUComputePipeline;
  private readonly clearGridPipeline: GPUComputePipeline;
  private readonly computeLayout: GPUBindGroupLayout;
  private readonly photonBuffer: GPUBuffer;
  private readonly gridBuffer: GPUBuffer;
  private radius0 = 0.05;
  private sceneCenter: [number, number, number] = [0, 0, 0];
  private sceneRadius = 1;
  private camDistance = 1;
  private photonsEmitted = 0;
  private readonly presentPipeline: GPURenderPipeline;

  /** ping-pong する履歴バッファ。1 画素あたり vec4f 2 個 */
  private histBuffers: [GPUBuffer, GPUBuffer] | null = null;
  private accumPixels = 0;
  private computeBindGroups: GPUBindGroup[] = [];
  private presentBindGroups: GPUBindGroup[] = [];
  private parity = 0;
  /** 再投影に使う 1 フレーム前のカメラ */
  private prevCam: FrameParams | null = null;

  private sphereBuffer: GPUBuffer | null = null;
  private quadBuffer: GPUBuffer | null = null;
  private indexBuffer: GPUBuffer | null = null;
  private bvhBuffer: GPUBuffer | null = null;
  private triBuffer: GPUBuffer | null = null;
  private envBuffer: GPUBuffer | null = null;
  private envWidth = 1;
  private envHeight = 1;
  private envPdfScale = 0;
  private fog: Scene["fog"] = undefined;
  private bvhNodeCount = 0;
  private sphereCount = 0;
  private quadCount = 0;
  private lightCount = 0;
  private env = 0;

  constructor(gpu: GpuContext, scene: Scene) {
    this.device = gpu.device;
    this.context = gpu.context;

    this.uniformBuffer = this.device.createBuffer({
      size: UNIFORM_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    // 光子 1 個あたり vec4f 3 個
    this.photonBuffer = this.device.createBuffer({
      // 1 本の光子が複数回堆積するので枠は 2 倍取る
      size: PHOTON_COUNT * 2 * 3 * 16,
      usage: GPUBufferUsage.STORAGE,
      label: "photons",
    });
    // [セルごとの個数][セルごとの光子インデックス][書き込んだ光子の総数]
    this.gridBuffer = this.device.createBuffer({
      size: (GRID_CELLS * (1 + GRID_CAP) + 1) * 4,
      usage: GPUBufferUsage.STORAGE,
      label: "grid",
    });

    const traceModule = this.device.createShaderModule({
      code: pathtraceWgsl,
      label: "pathtrace",
    });

    // エントリポイントごとに使う binding が違うので、layout: "auto" ではなく
    // 明示的なレイアウトを共有する
    const st = (type: GPUBufferBindingType): GPUBindGroupLayoutEntry["buffer"] => ({ type });
    this.computeLayout = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: st("uniform") },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: st("storage") },
        ...[2, 3, 4, 5, 6, 7, 8].map((b) => ({
          binding: b,
          visibility: GPUShaderStage.COMPUTE,
          buffer: st("read-only-storage"),
        })),
        { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: st("storage") },
        { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: st("storage") },
      ],
    });
    const pipelineLayout = this.device.createPipelineLayout({
      bindGroupLayouts: [this.computeLayout],
    });
    const mk = (entryPoint: string) =>
      this.device.createComputePipeline({
        layout: pipelineLayout,
        compute: { module: traceModule, entryPoint },
      });
    this.computePipeline = mk("main");
    this.sppmPipeline = mk("sppmMain");
    this.photonPipeline = mk("photonMain");
    this.clearGridPipeline = mk("clearGrid");

    const presentModule = this.device.createShaderModule({
      code: presentWgsl,
      label: "present",
    });
    this.presentPipeline = this.device.createRenderPipeline({
      layout: "auto",
      vertex: { module: presentModule, entryPoint: "vsMain" },
      fragment: {
        module: presentModule,
        entryPoint: "fsMain",
        targets: [{ format: gpu.format }],
      },
      primitive: { topology: "triangle-list" },
    });

    this.setScene(scene);
  }

  /** ジオメトリを丸ごと差し替える。バッファのサイズが変わるので作り直す */
  setScene(scene: Scene) {
    this.fog = scene.fog;
    this.sphereCount = scene.spheres.length;
    this.quadCount = scene.quads.length;
    this.env = scene.env;

    this.sphereBuffer?.destroy();
    this.sphereBuffer = this.uploadGeometry(packSpheres(scene.spheres), "spheres");
    this.quadBuffer?.destroy();
    this.quadBuffer = this.uploadGeometry(packQuads(scene.quads), "quads");

    const lights = packLights(scene.quads);
    this.lightCount = lights.count;

    this.triBuffer?.destroy();
    this.triBuffer = this.uploadGeometry(packTriangles(scene.triangles), "triangles");

    // 環境マップは使うシーンだけ焼く。それ以外はダミーを 1 個置く
    this.envBuffer?.destroy();
    if (scene.env === ENV.hdri) {
      const env = buildSunSkyEnv();
      this.envWidth = env.width;
      this.envHeight = env.height;
      this.envPdfScale = env.pdfScale;
      this.envBuffer = this.uploadGeometry(env.data.buffer as ArrayBuffer, "env");
    } else {
      this.envWidth = 1;
      this.envHeight = 1;
      this.envPdfScale = 0;
      this.envBuffer = this.uploadGeometry(new Float32Array(16).buffer, "env");
    }

    const bvh = buildBvh(scene.spheres, scene.quads, scene.triangles);
    this.bvhNodeCount = bvh.nodeCount;
    // シーンの対角長の 1% を光子ギャザーの初期半径にする
    const d = bvh.bounds.max.map((v, i) => v - bvh.bounds.min[i]);
    this.radius0 = Math.max(1e-3, Math.hypot(d[0], d[1], d[2]) * 0.012);
    // 環境マップから光子を撒くときの外接球
    this.sceneCenter = bvh.bounds.min.map((v, i) => (v + bvh.bounds.max[i]) / 2) as [
      number,
      number,
      number,
    ];
    this.sceneRadius = Math.max(1e-3, Math.hypot(d[0], d[1], d[2]) / 2);
    this.camDistance = scene.camera.distance;
    this.bvhBuffer?.destroy();
    this.bvhBuffer = this.uploadGeometry(bvh.nodes, "bvh");

    // storage buffer の本数を節約するため、面光源の索引と BVH の参照を 1 本にまとめる
    const lightIdx = new Uint32Array(lights.data);
    const refIdx = new Uint32Array(bvh.refs);
    const merged = new Uint32Array(this.lightCount + refIdx.length);
    merged.set(lightIdx.subarray(0, this.lightCount), 0);
    merged.set(refIdx, this.lightCount);
    this.indexBuffer?.destroy();
    this.indexBuffer = this.uploadGeometry(merged.buffer as ArrayBuffer, "indices");

    this.rebuildBindGroups();
  }

  /** pack 済みのジオメトリを storage buffer に載せる (空のシーンでも 1 要素分は確保される) */
  private uploadGeometry(packed: ArrayBuffer, label: string): GPUBuffer {
    const buffer = this.device.createBuffer({
      size: packed.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      label,
    });
    this.device.queue.writeBuffer(buffer, 0, packed);
    return buffer;
  }

  private rebuildBindGroups() {
    if (!this.histBuffers) return;
    // ping-pong の 2 通りぶん作っておき、フレームごとに入れ替える
    this.computeBindGroups = [0, 1].map((i) =>
      this.device.createBindGroup({
        layout: this.computeLayout,
        entries: [
          { binding: 0, resource: { buffer: this.uniformBuffer } },
          { binding: 1, resource: { buffer: this.histBuffers![i] } },
          { binding: 2, resource: { buffer: this.sphereBuffer! } },
          { binding: 3, resource: { buffer: this.quadBuffer! } },
          { binding: 4, resource: { buffer: this.indexBuffer! } },
          { binding: 5, resource: { buffer: this.bvhBuffer! } },
          { binding: 6, resource: { buffer: this.histBuffers![1 - i] } },
          { binding: 7, resource: { buffer: this.triBuffer! } },
          { binding: 8, resource: { buffer: this.envBuffer! } },
          { binding: 9, resource: { buffer: this.photonBuffer } },
          { binding: 10, resource: { buffer: this.gridBuffer } },
        ],
      }),
    );
    this.presentBindGroups = [0, 1].map((i) =>
      this.device.createBindGroup({
        layout: this.presentPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.uniformBuffer } },
          { binding: 1, resource: { buffer: this.histBuffers![i] } },
        ],
      }),
    );
  }

  /** 1 storage buffer に収まる最大ピクセル数 */
  maxPixels(): number {
    return Math.floor(
      this.device.limits.maxStorageBufferBindingSize / HIST_BYTES_PER_PIXEL,
    );
  }

  private ensureAccum(pixels: number) {
    if (this.histBuffers && this.accumPixels >= pixels) return;
    this.histBuffers?.forEach((b) => b.destroy());
    this.accumPixels = pixels;
    const make = (label: string) =>
      this.device.createBuffer({
        size: pixels * HIST_BYTES_PER_PIXEL,
        usage: GPUBufferUsage.STORAGE,
        label,
      });
    this.histBuffers = [make("hist0"), make("hist1")];
    this.rebuildBindGroups();
  }

  private writeUniforms(p: FrameParams) {
    const f = this.uniformF32;
    const u = this.uniformU32;
    f.set(p.camPos, 0);
    f[3] = p.lensRadius;
    f.set(p.camU, 4);
    f[7] = p.focusDist;
    f.set(p.camV, 8);
    f[11] = p.tanHalfFov;
    f.set(p.camW, 12);
    f[15] = p.width / p.height;
    u[16] = p.width;
    u[17] = p.height;
    u[18] = p.frameIndex;
    u[19] = p.sppPerFrame;
    u[20] = p.maxBounces;
    u[21] = this.sphereCount;
    u[22] = p.samplesBefore;
    u[23] = p.samplesBefore + p.sppPerFrame;
    u[24] = this.quadCount;
    u[25] = this.env;
    u[26] = this.lightCount;
    u[27] = p.nee ? 1 : 0;
    u[28] = p.mis ? 1 : 0;
    u[29] = p.qmc ? 1 : 0;
    u[30] = this.bvhNodeCount;
    u[31] = p.envIs ? 1 : 0;
    u[32] = this.envWidth;
    u[33] = this.envHeight;
    f[34] = this.envPdfScale;

    // 再投影用の 1 フレーム前のカメラ。初回は自分自身を入れておく
    const q = this.prevCam ?? p;
    f.set(q.camPos, 36);
    f[39] = q.tanHalfFov;
    f.set(q.camU, 40);
    f[43] = q.width / q.height;
    f.set(q.camV, 44);
    u[47] = p.reproject && this.prevCam ? 1 : 0;
    f.set(q.camW, 48);
    u[52] = p.debugMode;
    u[53] = p.fixedSeed ? 1 : 0;
    const fog = this.fog;
    if (fog && p.fog) {
      f.set(fog.min, 56);
      f[59] = fog.sigmaS;
      f.set(fog.max, 60);
      f[63] = fog.sigmaA;
      f[64] = fog.g;
      u[65] = 1;
    } else {
      u[65] = 0;
    }
    u[66] = p.sppm ? 1 : 0;
    u[67] = PHOTON_COUNT;
    u[68] = GRID_CELLS;
    f[69] = this.radius0;
    f[70] = this.radius0;
    f[71] = this.photonsEmitted;
    f.set(this.sceneCenter, 72);
    f[75] = this.sceneRadius;
    this.device.queue.writeBuffer(this.uniformBuffer, 0, this.uniformData);
  }

  render(p: FrameParams) {
    this.ensureAccum(p.width * p.height);
    // 光子は面光源と環境マップから撒く。環境マップは NEE が直接光を
    // 受け持っている前提なので、env importance sampling が要る。
    //
    // 環境マップの光子はシーンの外接球の外から撒くので、巨大なプリミティブが
    // あると円板が広がりすぎて、見えている範囲にほとんど届かなくなる
    // (spheres シーンは半径 1000 の地面球のせいで外接球が 1732 になり、
    //  有効なのは 7.5e-5 しかない)。そういうシーンは黙って暗く出るより
    // パストレースに落とすほうがましなので、見込みで切り分ける
    const envPhotons =
      this.envWidth > 1 && p.envIs && this.sceneRadius < this.camDistance * 20;
    const sppm = p.sppm && (this.lightCount > 0 || envPhotons);
    const q: FrameParams = sppm === p.sppm ? p : { ...p, sppm };
    // SPPM は画素ごとの状態を持ち越す必要があるので ping-pong を止める
    const write = sppm ? 0 : this.parity;
    if (sppm && p.samplesBefore === 0) this.photonsEmitted = 0;
    this.writeUniforms(q);

    const encoder = this.device.createCommandEncoder();
    const compute = encoder.beginComputePass();
    compute.setBindGroup(0, this.computeBindGroups[write]);

    if (sppm) {
      // グリッドを空にする -> 光子を撒く -> カメラ側で集める
      compute.setPipeline(this.clearGridPipeline);
      compute.dispatchWorkgroups(Math.ceil(GRID_CELLS / 64));
      compute.setPipeline(this.photonPipeline);
      compute.dispatchWorkgroups(Math.ceil(PHOTON_COUNT / 64));
      compute.setPipeline(this.sppmPipeline);
    } else {
      compute.setPipeline(this.computePipeline);
    }
    compute.dispatchWorkgroups(
      Math.ceil(p.width / WORKGROUP),
      Math.ceil(p.height / WORKGROUP),
    );
    compute.end();

    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: this.context.getCurrentTexture().createView(),
          loadOp: "clear",
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          storeOp: "store",
        },
      ],
    });
    pass.setPipeline(this.presentPipeline);
    pass.setBindGroup(0, this.presentBindGroups[write]);
    pass.draw(3);
    pass.end();

    this.device.queue.submit([encoder.finish()]);
    if (sppm) this.photonsEmitted += PHOTON_COUNT;
    this.prevCam = p;
    this.parity = sppm ? 0 : 1 - write;
  }
}
