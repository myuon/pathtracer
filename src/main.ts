import { attachCameraControls, OrbitCamera } from "./camera";
import { initGpu, Renderer, WebGpuUnsupportedError } from "./gpu";
import { buildSceneById, DEFAULT_SCENE_ID, SCENES } from "./scene";
import { createUi } from "./ui";

/** 操作をやめてから収束モードに戻るまでの待ち時間 (ms) */
const SETTLE_MS = 150;

async function main() {
  const canvas = document.getElementById("canvas") as HTMLCanvasElement;
  const gpu = await initGpu(canvas);

  const initialScene = buildSceneById(DEFAULT_SCENE_ID);
  const renderer = new Renderer(gpu, initialScene);

  const camera = new OrbitCamera();
  camera.applyPreset(initialScene.camera);

  let samples = 0;
  let frameIndex = 0;
  const reset = () => {
    samples = 0;
  };

  const ui = createUi({
    camera,
    scenes: SCENES,
    initialSceneId: DEFAULT_SCENE_ID,
    onSceneChange: (id) => {
      const scene = buildSceneById(id);
      renderer.setScene(scene);
      camera.applyPreset(scene.camera);
      ui.syncCamera();
      reset();
    },
    onReset: reset,
  });

  let interacting = false;
  let settleAt = 0;
  attachCameraControls(canvas, camera, {
    onInteractingChange: (next) => {
      interacting = next;
      if (!next) settleAt = performance.now() + SETTLE_MS;
    },
  });

  let canvasW = 0;
  let canvasH = 0;
  const resizeCanvas = () => {
    const dpr = Math.min(window.devicePixelRatio, 2);
    const w = Math.max(1, Math.floor(canvas.clientWidth * dpr));
    const h = Math.max(1, Math.floor(canvas.clientHeight * dpr));
    if (w === canvasW && h === canvasH) return;
    canvasW = w;
    canvasH = h;
    canvas.width = w;
    canvas.height = h;
    reset();
  };
  resizeCanvas();
  window.addEventListener("resize", resizeCanvas);

  // 1 dispatch が重くなりすぎないように accum buffer の上限からピクセル数を抑える
  const maxPixels = renderer.maxPixels();

  let lastWidth = 0;
  let lastHeight = 0;
  let lastFast = false;
  let fpsEma = 0;
  let lastTime = performance.now();

  const frame = () => {
    requestAnimationFrame(frame);
    resizeCanvas();

    const now = performance.now();
    const dt = now - lastTime;
    lastTime = now;
    fpsEma = fpsEma === 0 ? 1000 / dt : fpsEma * 0.9 + (1000 / dt) * 0.1;

    const settling = !interacting && now < settleAt;
    const fast = interacting || settling;

    const scale = fast ? ui.settings.interactiveScale : ui.settings.resolutionScale;
    let width = Math.max(1, Math.round(canvasW * scale));
    let height = Math.max(1, Math.round(canvasH * scale));
    if (width * height > maxPixels) {
      const k = Math.sqrt(maxPixels / (width * height));
      width = Math.max(1, Math.floor(width * k));
      height = Math.max(1, Math.floor(height * k));
    }

    // カメラ・解像度・品質モードが変わったら累積をやり直す
    if (camera.dirty || width !== lastWidth || height !== lastHeight || fast !== lastFast) {
      camera.dirty = false;
      lastWidth = width;
      lastHeight = height;
      lastFast = fast;
      samples = 0;
    }

    const spp = fast ? 1 : ui.settings.sppPerFrame;
    const maxBounces = fast ? ui.settings.interactiveBounces : ui.settings.maxBounces;
    const b = camera.basis();

    renderer.render({
      camPos: b.position,
      camU: b.u,
      camV: b.v,
      camW: b.w,
      tanHalfFov: b.tanHalfFov,
      focusDist: b.focusDist,
      lensRadius: b.lensRadius,
      width,
      height,
      frameIndex,
      sppPerFrame: spp,
      maxBounces,
      samplesBefore: samples,
      nee: ui.settings.nee,
      mis: ui.settings.mis,
    });

    samples += spp;
    frameIndex++;

    ui.setStatus(
      `${width}x${height} / ${samples} spp / ${fpsEma.toFixed(0)} fps` +
        (fast ? " / interactive" : ""),
    );
  };

  requestAnimationFrame(frame);
}

main().catch((err) => {
  console.error(err);
  if (err instanceof WebGpuUnsupportedError) {
    document.getElementById("unsupported")?.classList.add("show");
  }
});
