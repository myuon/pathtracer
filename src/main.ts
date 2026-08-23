import { runBench } from "./bench";
import { attachCameraControls, OrbitCamera } from "./camera";
import { initGpu, Renderer, WebGpuUnsupportedError } from "./gpu";
import { buildSceneById, DEFAULT_SCENE_ID, SCENES } from "./scene";
import { createUi } from "./ui";

/** 操作をやめてから収束モードに戻るまでの待ち時間 (ms) */
const SETTLE_MS = 150;

async function main() {
  const canvas = document.getElementById("canvas") as HTMLCanvasElement;

  // 計測モード。UI も rAF ループも通さず、決めた予算ぶん回して結果を置く
  if (new URLSearchParams(location.search).has("bench")) {
    await runBench(canvas);
    return;
  }

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
      // 操作を始めたら自動で再開する。止まったまま動かないと分かりにくい
      if (next) ui.setPaused(false);
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

  // 目標とする GPU 時間 (ms)。操作中は応答性、収束中は処理量を優先する。
  // 収束中に細かく刻むと 1 フレームの固定費ばかり払うことになって逆に遅い
  const TARGET_FAST = 16;
  const TARGET_CONVERGED = 100;
  /** 前フレームの GPU がまだ終わっていないか。終わるまで次を投入しない */
  let gpuBusy = false;
  /** 直近の GPU フレーム時間 (ms)。操作中と収束中で別に持つ */
  let gpuMsFast = 0;
  let gpuMsSlow = 0;
  /** 操作中の解像度スケールの自動調整 (ユーザー設定に掛ける) */
  let autoResScale = 1;
  /** 自動調整した spp/frame */
  let autoSpp = 1;
  /**
   * 光子数の倍率。SPPM は spp に依存しない固定費 (光子を撒く処理) が
   * 支配的なので、spp を削っても効かない。こちらが本命の調整先
   */
  let photonScale = 1;

  const frame = () => {
    requestAnimationFrame(frame);
    // 投入しすぎるとキューが積み上がって操作に反応しなくなる。
    // 弱いマシンほどここで効く
    if (gpuBusy) {
      return;
    }
    resizeCanvas();

    const now = performance.now();
    const dt = now - lastTime;
    lastTime = now;
    fpsEma = fpsEma === 0 ? 1000 / dt : fpsEma * 0.9 + (1000 / dt) * 0.1;

    const settling = !interacting && now < settleAt;
    const fast = interacting || settling;

    // 操作中が重ければ解像度を落とす。収束中の解像度には触らない
    if (ui.settings.adaptive && fast && gpuMsFast > 0) {
      if (gpuMsFast > TARGET_FAST * 1.3) {
        autoResScale = Math.max(0.25, autoResScale * 0.8);
      } else if (gpuMsFast < TARGET_FAST * 0.5) {
        autoResScale = Math.min(1, autoResScale * 1.1);
      }
    } else if (!ui.settings.adaptive) {
      autoResScale = 1;
    }
    const scale = fast
      ? ui.settings.interactiveScale * autoResScale
      : ui.settings.resolutionScale;
    let width = Math.max(1, Math.round(canvasW * scale));
    let height = Math.max(1, Math.round(canvasH * scale));
    if (width * height > maxPixels) {
      const k = Math.sqrt(maxPixels / (width * height));
      width = Math.max(1, Math.floor(width * k));
      height = Math.max(1, Math.floor(height * k));
    }

    // 解像度や品質モードが変わるとバッファの意味が変わるので、素直に捨てる
    const layoutChanged = width !== lastWidth || height !== lastHeight || fast !== lastFast;
    let reproject = false;
    if (layoutChanged) {
      lastWidth = width;
      lastHeight = height;
      lastFast = fast;
      samples = 0;
    } else if (camera.dirty) {
      // 操作中だけ累積を新しい視点へ投影して引き継ぐ。
      // 収束モードは毎回きれいにやり直すので、最終的な絵に再投影は混ざらない
      if (fast && ui.settings.reproject) {
        reproject = true;
      } else {
        samples = 0;
      }
    }
    camera.dirty = false;

    // 収束中は目標が緩い。1 フレームが長くなりすぎない範囲で処理量を稼ぐ
    if (ui.settings.adaptive && gpuMsSlow > 0) {
      // 削る順番が肝心。SPPM では光子を撒く処理が spp に依存しない固定費
      // として支配的なので、spp から削ると時間が減らないまま収束だけ遅くなる
      const photonsFirst = ui.settings.sppm;
      if (gpuMsSlow > TARGET_CONVERGED * 1.2) {
        if (photonsFirst && photonScale > 1 / 16) {
          photonScale = Math.max(1 / 16, photonScale / 2);
        } else if (autoSpp > 1) {
          autoSpp = Math.max(1, autoSpp >> 1);
        } else {
          photonScale = Math.max(1 / 16, photonScale / 2);
        }
      } else if (gpuMsSlow < TARGET_CONVERGED * 0.85) {
        // 減らしても時間が変わらない場合に下がったまま戻れなくならないよう、
        // 戻す側の閾値は緩めに取る
        if (autoSpp < ui.settings.sppPerFrame) {
          autoSpp = autoSpp + 1;
        } else {
          photonScale = Math.min(1, photonScale * 2);
        }
      }
    } else {
      autoSpp = ui.settings.sppPerFrame;
      photonScale = 1;
    }
    const spp = fast ? 1 : autoSpp;
    const maxBounces = fast ? ui.settings.interactiveBounces : ui.settings.maxBounces;
    const b = camera.basis();

    const taken = renderer.render({
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
      qmc: ui.settings.qmc,
      envIs: ui.settings.envIs,
      reproject,
      debugMode: ui.settings.debugMode,
      fixedSeed: ui.settings.fixedSeed,
      fog: ui.settings.fog,
      // 操作中は SPPM を切る。光子パスは解像度に依存しない固定費なので
      // 低解像度プレビューには重すぎるうえ、カメラが動いている間は
      // 画素ごとの統計 (半径や累積光子数) が意味を持たない
      sppm: ui.settings.sppm && !fast,
      vcm: ui.settings.vcm && !fast,
      guide: ui.settings.guide,
      ears: ui.settings.ears,
      salt: 0,
      denoise: ui.settings.denoise,
      photonScale,
      paused: ui.settings.paused,
    });

    // GPU が終わるまで次を投入しない。ついでに 1 spp あたりの時間を測る
    gpuBusy = true;
    const submitted = performance.now();
    void gpu.device.queue.onSubmittedWorkDone().then(() => {
      const ms = performance.now() - submitted;
      if (fast) {
        gpuMsFast = gpuMsFast === 0 ? ms : gpuMsFast * 0.7 + ms * 0.3;
      } else {
        gpuMsSlow = gpuMsSlow === 0 ? ms : gpuMsSlow * 0.7 + ms * 0.3;
      }
      gpuBusy = false;
    });

    samples += taken;
    if (!ui.settings.paused) {
      frameIndex++;
    }

    ui.setStatus(
      (ui.settings.paused ? "[停止中] " : "") +
        `${width}x${height} / ${samples} spp / ${fpsEma.toFixed(0)} fps` +
        // 実際に積んだ数を出す。SPPM は spp/frame の設定に関係なく 1
        (fast ? " / interactive" : ` / ${taken} spp per frame`) +
        ` / ${(fast ? gpuMsFast : gpuMsSlow).toFixed(0)} ms` +
        (photonScale < 1 ? ` / photons 1/${Math.round(1 / photonScale)}` : ""),
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
