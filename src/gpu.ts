import pathtraceWgsl from "./shaders/pathtrace.wgsl?raw";
import presentWgsl from "./shaders/present.wgsl?raw";
import { packQuads, packSpheres, type Scene } from "./scene";

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
  const device = await adapter.requestDevice();
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
const UNIFORM_SIZE = 112;
const WORKGROUP = 8;

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
}

export class Renderer {
  private readonly device: GPUDevice;
  private readonly context: GPUCanvasContext;
  private readonly uniformBuffer: GPUBuffer;
  private readonly uniformData = new ArrayBuffer(UNIFORM_SIZE);
  private readonly uniformF32 = new Float32Array(this.uniformData);
  private readonly uniformU32 = new Uint32Array(this.uniformData);
  private readonly computePipeline: GPUComputePipeline;
  private readonly presentPipeline: GPURenderPipeline;

  private accumBuffer: GPUBuffer | null = null;
  private accumPixels = 0;
  private computeBindGroup: GPUBindGroup | null = null;
  private presentBindGroup: GPUBindGroup | null = null;

  private sphereBuffer: GPUBuffer | null = null;
  private quadBuffer: GPUBuffer | null = null;
  private sphereCount = 0;
  private quadCount = 0;
  private env = 0;

  constructor(gpu: GpuContext, scene: Scene) {
    this.device = gpu.device;
    this.context = gpu.context;

    this.uniformBuffer = this.device.createBuffer({
      size: UNIFORM_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const traceModule = this.device.createShaderModule({
      code: pathtraceWgsl,
      label: "pathtrace",
    });
    this.computePipeline = this.device.createComputePipeline({
      layout: "auto",
      compute: { module: traceModule, entryPoint: "main" },
    });

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
    this.sphereCount = scene.spheres.length;
    this.quadCount = scene.quads.length;
    this.env = scene.env;

    this.sphereBuffer?.destroy();
    this.sphereBuffer = this.uploadGeometry(packSpheres(scene.spheres), "spheres");
    this.quadBuffer?.destroy();
    this.quadBuffer = this.uploadGeometry(packQuads(scene.quads), "quads");

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
    if (!this.accumBuffer) return;
    this.computeBindGroup = this.device.createBindGroup({
      layout: this.computePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: this.accumBuffer } },
        { binding: 2, resource: { buffer: this.sphereBuffer! } },
        { binding: 3, resource: { buffer: this.quadBuffer! } },
      ],
    });
    this.presentBindGroup = this.device.createBindGroup({
      layout: this.presentPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: this.accumBuffer } },
      ],
    });
  }

  /** 1 storage buffer に収まる最大ピクセル数 */
  maxPixels(): number {
    return Math.floor(this.device.limits.maxStorageBufferBindingSize / 16);
  }

  private ensureAccum(pixels: number) {
    if (this.accumBuffer && this.accumPixels >= pixels) return;
    this.accumBuffer?.destroy();
    this.accumPixels = pixels;
    this.accumBuffer = this.device.createBuffer({
      size: pixels * 16,
      usage: GPUBufferUsage.STORAGE,
      label: "accum",
    });
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
    this.device.queue.writeBuffer(this.uniformBuffer, 0, this.uniformData);
  }

  render(p: FrameParams) {
    this.ensureAccum(p.width * p.height);
    this.writeUniforms(p);

    const encoder = this.device.createCommandEncoder();

    const compute = encoder.beginComputePass();
    compute.setPipeline(this.computePipeline);
    compute.setBindGroup(0, this.computeBindGroup!);
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
    pass.setBindGroup(0, this.presentBindGroup!);
    pass.draw(3);
    pass.end();

    this.device.queue.submit([encoder.finish()]);
  }
}
