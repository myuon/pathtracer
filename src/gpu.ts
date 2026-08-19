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
