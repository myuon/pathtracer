import { initGpu, WebGpuUnsupportedError } from "./gpu";

async function main() {
  const canvas = document.getElementById("canvas") as HTMLCanvasElement;
  const gpu = await initGpu(canvas);

  const resize = () => {
    const dpr = Math.min(window.devicePixelRatio, 2);
    canvas.width = Math.max(1, Math.floor(canvas.clientWidth * dpr));
    canvas.height = Math.max(1, Math.floor(canvas.clientHeight * dpr));
  };
  resize();
  window.addEventListener("resize", resize);

  const frame = () => {
    const encoder = gpu.device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: gpu.context.getCurrentTexture().createView(),
          clearValue: { r: 0.05, g: 0.06, b: 0.08, a: 1 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });
    pass.end();
    gpu.device.queue.submit([encoder.finish()]);
    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
}

main().catch((err) => {
  console.error(err);
  if (err instanceof WebGpuUnsupportedError) {
    document.getElementById("unsupported")?.classList.add("show");
  }
});
