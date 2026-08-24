import { OrbitCamera } from "./camera";
import { initGpu, PhotonStats, Renderer } from "./gpu";
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
  /** present を通した sRGB 8bit (RGBA)。present=1 のときだけ入る */
  ldr?: string;
  /** 最後のフレームの光子統計。SPPM / VCM のときだけ意味がある */
  stats?: PhotonStats;
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
  // before= を渡すと、まずそのシーンを少し描いてから本命へ切り替える。
  // setScene の後始末 (学習した分布の破棄など) が効いているかの検証用
  const before = q.get("before");
  const scene = buildSceneById(sceneId);
  const renderer = new Renderer(gpu, before ? buildSceneById(before) : scene);
  const camera = new OrbitCamera();
  camera.applyPreset(scene.camera);
  const b = camera.basis();

  const sppPerFrame = Math.max(1, Math.round(num(q, "sppf", 1)));
  // 表示された絵 (トーンマップ + デノイザ通し) も返すか
  const wantLdr = flag(q, "present", false);
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
    debugMode: Math.round(num(q, "debug", 0)),
    // 種を累積サンプル数から作る。同条件なら毎回同じ絵になる。
    // seed=0 にすると種の作り方が変わり、同じ推定量の別の実現が得られる。
    // 「その差が本物か、乱数のばらつきか」を切り分けるのに使う
    fixedSeed: flag(q, "seed", true),
    fog: flag(q, "fog", true),
    sppm: flag(q, "sppm", false),
    vcm: flag(q, "vcm", false),
    guide: flag(q, "guide", false),
    ears: flag(q, "ears", false),
    wavefront: flag(q, "wavefront", false),
    // 参照画像と検証画像で別の値にして、同じ点列を共有しないようにする
    salt: Math.round(num(q, "salt", 0)),
    denoise: flag(q, "denoise", false),
    photonScale: num(q, "photonScale", 1),
    photons: q.has("photons") ? num(q, "photons", 0) : undefined,
    paused: false,
  };

  if (before) {
    // 前のシーンを一通り描いて、学習した分布や光子グリッドを埋めておく
    const b = new OrbitCamera();
    b.applyPreset(buildSceneById(before).camera);
    const bb = b.basis();
    for (let i = 0; i < 24; i++) {
      renderer.render({
        ...base,
        camPos: bb.position, camU: bb.u, camV: bb.v, camW: bb.w,
        tanHalfFov: bb.tanHalfFov, focusDist: bb.focusDist, lensRadius: bb.lensRadius,
        frameIndex: i, sppPerFrame: 1, samplesBefore: i,
      });
    }
    await gpu.device.queue.onSubmittedWorkDone();
    renderer.setScene(scene);
  }

  // ウォームアップ。パイプラインの構築とシェーダのコンパイルを計測から外す。
  // この後 samplesBefore を 0 に戻すので、累積も光子の状態もやり直しになる
  renderer.render({ ...base, frameIndex: 0, sppPerFrame: 1, samplesBefore: 0 });
  await gpu.device.queue.onSubmittedWorkDone();

  let samples = 0;
  let frames = 0;
  const t0 = performance.now();
  for (;;) {
    const taken = renderer.render({
      ...base,
      frameIndex: frames,
      sppPerFrame,
      samplesBefore: samples,
      // 最後の 1 フレームだけ、表示された絵も写す
      capture: wantLdr,
    });
    await gpu.device.queue.onSubmittedWorkDone();
    // SPPM は spp/frame に関係なく 1 フレーム 1 サンプルなので、
    // 実際に積んだ数で数える
    samples += taken;
    frames += 1;
    const elapsed = performance.now() - t0;
    if (targetSpp > 0 && samples >= targetSpp) break;
    if (targetMs > 0 && elapsed >= targetMs) break;
  }
  const ms = performance.now() - t0;

  const stats = await renderer.readStats();
  const hdr = await renderer.readHdr(width, height);
  const ldr = wantLdr ? await renderer.readPresented(width, height) : undefined;
  const result: BenchResult = {
    scene: sceneId,
    width,
    height,
    spp: samples,
    frames,
    ms,
    hdr: toBase64(hdr.buffer),
    ldr: ldr ? toBase64(ldr.buffer) : undefined,
    stats,
  };
  (window as unknown as { __bench?: BenchResult }).__bench = result;
}
