import { OrbitCamera } from "./camera";
import { initGpu, Renderer } from "./gpu";
import { buildSceneById } from "./scene";

/**
 * 計測モード。`?bench=1&...` で入る。
 *
 * UI も入力も rAF の可変ロジックも通さず、決められた設定で決められた
 * 予算 (spp か ms) ぶんだけ回して、累積バッファを HDR のまま返す。
 * 1 回の計測につきページを 1 枚使い切る前提なので、学習した分布 (ガイド
 * 格子や光子の放出ヒストグラム) が前の計測から漏れることはない
 */

export interface BenchResult {
  scene: string;
  width: number;
  height: number;
  /** 実際に積んだ spp */
  spp: number;
  /** 実際に回したフレーム数 */
  frames: number;
  /** 実測時間 (ms)。ウォームアップの 1 フレームは含まない */
  ms: number;
  /** Float32Array (画素あたり RGB) を base64 にしたもの */
  hdr: string;
}

function flag(q: URLSearchParams, key: string, dflt: boolean): boolean {
  const v = q.get(key);
  if (v === null) return dflt;
  return v !== "0" && v !== "false";
}

function num(q: URLSearchParams, key: string, dflt: number): number {
  const v = q.get(key);
  if (v === null) return dflt;
  const n = Number(v);
  return Number.isFinite(n) ? n : dflt;
}

function toBase64(buf: ArrayBufferLike): string {
  const bytes = new Uint8Array(buf);
  const chunk = 0x8000;
  let s = "";
  for (let i = 0; i < bytes.length; i += chunk) {
    s += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }
  return btoa(s);
}

export async function runBench(canvas: HTMLCanvasElement): Promise<void> {
  const q = new URLSearchParams(location.search);
  const sceneId = q.get("scene") ?? "cornell";
  const width = Math.max(1, Math.round(num(q, "w", 320)));
  const height = Math.max(1, Math.round(num(q, "h", 240)));

  // present パスは canvas のサイズで描くので、計測解像度に合わせておく
  canvas.width = width;
  canvas.height = height;
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;

  const gpu = await initGpu(canvas);
  const scene = buildSceneById(sceneId);
  const renderer = new Renderer(gpu, scene);
  const camera = new OrbitCamera();
  camera.applyPreset(scene.camera);
  const b = camera.basis();

  const sppPerFrame = Math.max(1, Math.round(num(q, "sppf", 1)));
  // 予算。spp を指定すればその spp まで、ms を指定すればその時間まで
  const targetSpp = q.has("spp") ? Math.round(num(q, "spp", 0)) : 0;
  const targetMs = q.has("ms") ? num(q, "ms", 0) : 0;
  if (targetSpp <= 0 && targetMs <= 0) throw new Error("spp か ms のどちらかが要る");

  const base = {
    camPos: b.position,
    camU: b.u,
    camV: b.v,
    camW: b.w,
    tanHalfFov: b.tanHalfFov,
    focusDist: b.focusDist,
    lensRadius: flag(q, "dof", true) ? b.lensRadius : 0,
    width,
    height,
    maxBounces: Math.round(num(q, "bounces", 8)),
    nee: flag(q, "nee", true),
    mis: flag(q, "mis", true),
    qmc: flag(q, "qmc", true),
    envIs: flag(q, "envIs", true),
    reproject: false,
    debugMode: 0,
    // 種を累積サンプル数から作る。同条件なら毎回同じ絵になる。
    // seed=0 にすると種の作り方が変わり、同じ推定量の別の実現が得られる。
    // 「その差が本物か、乱数のばらつきか」を切り分けるのに使う
    fixedSeed: flag(q, "seed", true),
    fog: flag(q, "fog", true),
    sppm: flag(q, "sppm", false),
    vcm: flag(q, "vcm", false),
    guide: flag(q, "guide", false),
    adaptivePixels: flag(q, "adaptive", false),
    // 参照画像と検証画像で別の値にして、同じ点列を共有しないようにする
    salt: Math.round(num(q, "salt", 0)),
    denoise: flag(q, "denoise", false),
    photonScale: num(q, "photonScale", 1),
    paused: false,
  };

  // ウォームアップ。パイプラインの構築とシェーダのコンパイルを計測から外す。
  // この後 samplesBefore を 0 に戻すので、累積も光子の状態もやり直しになる
  renderer.render({ ...base, frameIndex: 0, sppPerFrame: 1, samplesBefore: 0 });
  await gpu.device.queue.onSubmittedWorkDone();

  let samples = 0;
  let frames = 0;
  const t0 = performance.now();
  for (;;) {
    renderer.render({
      ...base,
      frameIndex: frames,
      sppPerFrame,
      samplesBefore: samples,
    });
    await gpu.device.queue.onSubmittedWorkDone();
    samples += sppPerFrame;
    frames += 1;
    const elapsed = performance.now() - t0;
    if (targetSpp > 0 && samples >= targetSpp) break;
    if (targetMs > 0 && elapsed >= targetMs) break;
  }
  const ms = performance.now() - t0;

  const hdr = await renderer.readHdr(width, height);
  const result: BenchResult = {
    scene: sceneId,
    width,
    height,
    spp: samples,
    frames,
    ms,
    hdr: toBase64(hdr.buffer),
  };
  (window as unknown as { __bench?: BenchResult }).__bench = result;
}
