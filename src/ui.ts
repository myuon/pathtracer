import type { OrbitCamera } from "./camera";
import type { SceneEntry } from "./scene";

/** 収束モード・操作モードの描画設定 */
export interface RenderSettings {
  /** 収束モードでの最大バウンス数 */
  maxBounces: number;
  /** 収束モードでの 1 フレームあたりのサンプル数 */
  sppPerFrame: number;
  /** 収束モードの解像度スケール (0..1) */
  resolutionScale: number;
  /** 操作中の解像度スケール (0..1) */
  interactiveScale: number;
  /** 操作中の最大バウンス数 */
  interactiveBounces: number;
  /** 面光源を直接サンプルする (next event estimation) */
  nee: boolean;
  /** 光源サンプリングと BSDF サンプリングを power heuristic で合成する */
  mis: boolean;
  /** スクランブル済み Sobol (0,2) 列を使う */
  qmc: boolean;
  /** 環境マップを光源としてサンプルする */
  envIs: boolean;
  /** 操作中に累積を再投影して引き継ぐ */
  reproject: boolean;
  /** 0 なら通常描画、それ以外は中間量を疑似カラーで出す */
  debugMode: number;
  /** 乱数の種を累積サンプル数から作り、同条件を再現可能にする */
  fixedSeed: boolean;
  /** 参加媒質を有効にする */
  fog: boolean;
  /** SPPM を使う */
  sppm: boolean;
  vcm: boolean;
  guide: boolean;
  adaptivePixels: boolean;
  /// ロシアンルーレットの生存確率を、この先期待される放射輝度から決める
  ears: boolean;
  /** アルベド/法線ガイド付き a-trous デノイザをかける */
  denoise: boolean;
  /** 1 フレームの GPU 時間が目標に収まるよう spp を自動調整する */
  adaptive: boolean;
  /** 計算を止める。表示は保つ */
  paused: boolean;
}

export interface UiOptions {
  camera: OrbitCamera;
  /** セレクタに並べるシーン */
  scenes: SceneEntry[];
  /** 最初に選択されているシーンの id */
  initialSceneId: string;
  /** シーンが選び直されたとき */
  onSceneChange: (id: string) => void;
  /** 描画設定やカメラが変わり、累積をリセットすべきとき */
  onReset: () => void;
}

export interface UiHandle {
  /** 一時停止を外から切り替える (操作を始めたら自動で再開するため) */
  setPaused: (paused: boolean) => void;
  settings: RenderSettings;
  /** 画面下部のステータス行を更新する */
  setStatus: (text: string) => void;
  /** カメラ側を書き換えたあと、スライダーの表示を追従させる */
  syncCamera: () => void;
}

/** デバッグ表示。WGSL 側の aovColor と一致させること */
const DEBUG_MODES = [
  { id: "0", name: "off (通常描画)" },
  { id: "1", name: "normal" },
  { id: "2", name: "albedo" },
  { id: "3", name: "distance" },
  { id: "4", name: "bsdf pdf" },
  { id: "5", name: "MIS weight" },
  { id: "6", name: "bounces" },
  { id: "7", name: "VCM connect" },
  { id: "8", name: "photon waste" },
  { id: "9", name: "direct only" },
  { id: "10", name: "indirect only" },
];

/** 解像度スケール系スライダーで選べる離散値 */
const RESOLUTION_SCALES = [0.25, 0.33, 0.5, 0.75, 1];

/** 離散値リストの中から値に最も近いインデックスを探す */
function closestIndex(values: number[], value: number): number {
  let bestIndex = 0;
  let bestDiff = Infinity;
  for (let i = 0; i < values.length; i++) {
    const diff = Math.abs(values[i] - value);
    if (diff < bestDiff) {
      bestDiff = diff;
      bestIndex = i;
    }
  }
  return bestIndex;
}

/** スライダー行を作るための共通オプション */
interface SliderRowOptions {
  label: string;
  min: number;
  max: number;
  step: number;
  value: number;
  /** スライダーの生値からラベル表示用の文字列を作る */
  format: (value: number) => string;
  /** スライダーが動いたときに呼ばれる (raw value を渡す) */
  onInput: (value: number) => void;
}

interface SliderRowHandle {
  row: HTMLDivElement;
  /** 値表示を更新する (外部から値を変えた場合に使う) */
  setValue: (value: number) => void;
}

/** ラベル・現在値表示・スライダーからなる 1 行を作る */
function createSliderRow(options: SliderRowOptions): SliderRowHandle {
  const row = document.createElement("div");
  row.className = "pt-row";

  const labelLine = document.createElement("div");
  labelLine.className = "pt-row-label";

  const labelText = document.createElement("span");
  labelText.textContent = options.label;

  const valueText = document.createElement("span");
  valueText.className = "pt-row-value";
  valueText.textContent = options.format(options.value);

  labelLine.appendChild(labelText);
  labelLine.appendChild(valueText);

  const slider = document.createElement("input");
  slider.type = "range";
  slider.min = String(options.min);
  slider.max = String(options.max);
  slider.step = String(options.step);
  slider.value = String(options.value);
  slider.className = "pt-slider";

  slider.addEventListener("input", () => {
    const value = Number(slider.value);
    valueText.textContent = options.format(value);
    options.onInput(value);
  });

  row.appendChild(labelLine);
  row.appendChild(slider);

  return {
    row,
    setValue: (value: number) => {
      slider.value = String(value);
      valueText.textContent = options.format(value);
    },
  };
}

/** ラベルとドロップダウンからなる 1 行を作る */
function createSelectRow(options: {
  label: string;
  items: { id: string; name: string }[];
  value: string;
  onChange: (id: string) => void;
}): HTMLDivElement {
  const row = document.createElement("div");
  row.className = "pt-row";

  const labelLine = document.createElement("div");
  labelLine.className = "pt-row-label";
  const labelText = document.createElement("span");
  labelText.textContent = options.label;
  labelLine.appendChild(labelText);

  const select = document.createElement("select");
  select.className = "pt-select";
  for (const item of options.items) {
    const opt = document.createElement("option");
    opt.value = item.id;
    opt.textContent = item.name;
    select.appendChild(opt);
  }
  select.value = options.value;
  select.addEventListener("change", () => options.onChange(select.value));

  row.appendChild(labelLine);
  row.appendChild(select);
  return row;
}

/** ラベルとチェックボックスからなる 1 行を作る */
function createToggleRow(options: {
  label: string;
  value: boolean;
  onChange: (value: boolean) => void;
}): HTMLDivElement {
  const row = document.createElement("div");
  row.className = "pt-row";

  const labelLine = document.createElement("label");
  labelLine.className = "pt-row-label pt-toggle-row";
  const labelText = document.createElement("span");
  labelText.textContent = options.label;

  const check = document.createElement("input");
  check.type = "checkbox";
  check.className = "pt-check";
  check.checked = options.value;
  check.addEventListener("change", () => options.onChange(check.checked));

  labelLine.appendChild(labelText);
  labelLine.appendChild(check);
  row.appendChild(labelLine);
  return row;
}

/** スタイルを 1 回だけ document.head に注入する */
function injectStyles(): void {
  const STYLE_ID = "pt-ui-style";
  if (document.getElementById(STYLE_ID)) return;

  const style = document.createElement("style");
  style.id = STYLE_ID;
  style.textContent = `
.pt-panel {
  position: fixed;
  top: 12px;
  left: 12px;
  width: 220px;
  max-width: calc(100vw - 24px);
  background: rgba(20, 22, 26, 0.82);
  border: 1px solid rgba(255, 255, 255, 0.12);
  border-radius: 8px;
  backdrop-filter: blur(8px);
  -webkit-backdrop-filter: blur(8px);
  padding: 8px 10px;
  font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
  font-size: 11px;
  color: #e6e6e6;
  user-select: none;
  z-index: 10;
}

.pt-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  font-size: 12px;
  font-weight: 600;
  letter-spacing: 0.02em;
  cursor: default;
}

.pt-toggle {
  background: transparent;
  border: 1px solid rgba(255, 255, 255, 0.2);
  border-radius: 4px;
  color: inherit;
  font-family: inherit;
  font-size: 11px;
  line-height: 1;
  width: 18px;
  height: 18px;
  cursor: pointer;
}

.pt-toggle:hover {
  background: rgba(255, 255, 255, 0.08);
}

.pt-body {
  margin-top: 6px;
}

.pt-row {
  margin-top: 6px;
}

.pt-row-label {
  display: flex;
  align-items: center;
  justify-content: space-between;
  opacity: 0.85;
}

.pt-row-value {
  opacity: 0.6;
}

.pt-slider {
  width: 100%;
  margin-top: 2px;
}

.pt-select {
  width: 100%;
  margin-top: 2px;
  background: rgba(255, 255, 255, 0.08);
  border: 1px solid rgba(255, 255, 255, 0.18);
  border-radius: 4px;
  color: inherit;
  font-family: inherit;
  font-size: 11px;
  padding: 3px 4px;
  cursor: pointer;
}

.pt-select option {
  background: #14161a;
  color: #e6e6e6;
}

.pt-toggle-row {
  cursor: pointer;
}

.pt-check {
  margin: 0;
  cursor: pointer;
}

.pt-reset {
  margin-top: 8px;
  width: 100%;
  background: rgba(255, 255, 255, 0.08);
  border: 1px solid rgba(255, 255, 255, 0.18);
  border-radius: 4px;
  color: inherit;
  font-family: inherit;
  font-size: 11px;
  padding: 4px 0;
  cursor: pointer;
}

.pt-reset:hover {
  background: rgba(255, 255, 255, 0.16);
}

.pt-status {
  margin-top: 6px;
  padding-top: 6px;
  border-top: 1px solid rgba(255, 255, 255, 0.1);
  color: rgba(230, 230, 230, 0.55);
  white-space: normal;
  word-break: break-word;
  line-height: 1.4;
}
`;
  document.head.appendChild(style);
}

/** フローティングのコントロールパネルを作り、body に追加する */
export function createUi(options: UiOptions): UiHandle {
  injectStyles();

  const settings: RenderSettings = {
    maxBounces: 12,
    sppPerFrame: 2,
    resolutionScale: 1,
    interactiveScale: 0.33,
    interactiveBounces: 3,
    nee: true,
    mis: true,
    qmc: true,
    envIs: true,
    reproject: true,
    debugMode: 0,
    fixedSeed: false,
    fog: true,
    sppm: true,
    denoise: false,
    adaptive: true,
    paused: false,
    vcm: false,
    // 既定は off。等 spp / 等時間のどちらで測っても差し引きマイナスだった。
    // bench/run.mjs で 320x240 / 12 バウンス / 1024 spp、効率 (relMSE x 秒) の
    // 対 PT 倍率は 0.86x (水面の集光だけ 1.56x で、そこだけは効く)。
    // 学習の式は直したが、一様な 16^3 ボクセル x 256 方向という分解能では
    // 「狭い隙間の向こうに光源がある」を表現しきれていない。
    // なお SPPM が有効なときは sppmMain が走るので、この設定は効かない
    guide: false,
    adaptivePixels: false,
    // 既定は off。差し引きはほぼ互角 (12 シーンの効率の幾何平均 0.98x、
    // relMSE は 1.18x) だが、間接光が奥にあるシーンにだけ大きく効く。
    // bench/run.mjs の 1024 spp で relMSE / 効率が
    // indirect 1.87x / 1.33x、enclosed 1.77x / 1.34x、
    // maze 1.51x / 1.10x、ajar 1.42x / 0.94x。
    // 逆に簡単なシーンでは経路が伸びるぶん遅くなるだけなので既定では入れない。
    // なお SPPM が有効なときは sppmMain が走るのでこの設定は効かない
    ears: false,
  };

  const panel = document.createElement("div");
  panel.className = "pt-panel";

  // ヘッダー (タイトル + 折りたたみトグル)
  const header = document.createElement("div");
  header.className = "pt-header";

  const title = document.createElement("span");
  title.textContent = "path tracer";

  const toggleButton = document.createElement("button");
  toggleButton.type = "button";
  toggleButton.className = "pt-toggle";
  toggleButton.textContent = "−";

  header.appendChild(title);
  header.appendChild(toggleButton);

  const body = document.createElement("div");
  body.className = "pt-body";

  let expanded = true;
  toggleButton.addEventListener("click", () => {
    expanded = !expanded;
    body.style.display = expanded ? "" : "none";
    toggleButton.textContent = expanded ? "−" : "+";
  });

  const notifyChange = () => {
    options.onReset();
  };

  const sceneRow = createSelectRow({
    label: "scene",
    items: options.scenes,
    value: options.initialSceneId,
    onChange: (id) => {
      options.onSceneChange(id);
    },
  });

  const maxBouncesRow = createSliderRow({
    label: "max bounces",
    min: 1,
    max: 32,
    step: 1,
    value: settings.maxBounces,
    format: (v) => String(v),
    onInput: (v) => {
      settings.maxBounces = v;
      notifyChange();
    },
  });

  const sppRow = createSliderRow({
    label: "spp / frame",
    min: 1,
    max: 16,
    step: 1,
    value: settings.sppPerFrame,
    format: (v) => String(v),
    onInput: (v) => {
      settings.sppPerFrame = v;
      notifyChange();
    },
  });

  const resolutionRow = createSliderRow({
    label: "resolution",
    min: 0,
    max: RESOLUTION_SCALES.length - 1,
    step: 1,
    value: closestIndex(RESOLUTION_SCALES, settings.resolutionScale),
    format: (index) => `${Math.round(RESOLUTION_SCALES[index] * 100)}%`,
    onInput: (index) => {
      settings.resolutionScale = RESOLUTION_SCALES[index];
      notifyChange();
    },
  });

  const fovRow = createSliderRow({
    label: "fov",
    min: 15,
    max: 90,
    step: 1,
    value: options.camera.fovDeg,
    format: (v) => `${Math.round(v)}°`,
    onInput: (v) => {
      options.camera.fovDeg = v;
      options.camera.dirty = true;
      notifyChange();
    },
  });

  const apertureRow = createSliderRow({
    label: "aperture",
    min: 0,
    max: 0.5,
    step: 0.01,
    value: options.camera.aperture,
    format: (v) => v.toFixed(2),
    onInput: (v) => {
      options.camera.aperture = v;
      options.camera.dirty = true;
      notifyChange();
    },
  });

  const interactiveResRow = createSliderRow({
    label: "interactive res",
    min: 0,
    max: RESOLUTION_SCALES.length - 1,
    step: 1,
    value: closestIndex(RESOLUTION_SCALES, settings.interactiveScale),
    format: (index) => `${Math.round(RESOLUTION_SCALES[index] * 100)}%`,
    onInput: (index) => {
      settings.interactiveScale = RESOLUTION_SCALES[index];
      notifyChange();
    },
  });

  const interactiveBouncesRow = createSliderRow({
    label: "interactive bounces",
    min: 1,
    max: 8,
    step: 1,
    value: settings.interactiveBounces,
    format: (v) => String(v),
    onInput: (v) => {
      settings.interactiveBounces = v;
      notifyChange();
    },
  });

  const neeRow = createToggleRow({
    label: "next event estimation",
    value: settings.nee,
    onChange: (v) => {
      settings.nee = v;
      notifyChange();
    },
  });

  const misRow = createToggleRow({
    label: "MIS (power heuristic)",
    value: settings.mis,
    onChange: (v) => {
      settings.mis = v;
      notifyChange();
    },
  });

  const qmcRow = createToggleRow({
    label: "low-discrepancy sampling",
    value: settings.qmc,
    onChange: (v) => {
      settings.qmc = v;
      notifyChange();
    },
  });

  const envIsRow = createToggleRow({
    label: "env importance sampling",
    value: settings.envIs,
    onChange: (v) => {
      settings.envIs = v;
      notifyChange();
    },
  });

  const reprojectRow = createToggleRow({
    label: "temporal reprojection",
    value: settings.reproject,
    onChange: (v) => {
      settings.reproject = v;
      notifyChange();
    },
  });

  const debugRow = createSelectRow({
    label: "debug view",
    items: DEBUG_MODES,
    value: String(settings.debugMode),
    onChange: (id) => {
      settings.debugMode = Number(id);
      notifyChange();
    },
  });

  const fogRow = createToggleRow({
    label: "participating media",
    value: settings.fog,
    onChange: (v) => {
      settings.fog = v;
      notifyChange();
    },
  });

  const sppmRow = createToggleRow({
    label: "SPPM (photon mapping)",
    value: settings.sppm,
    onChange: (v) => {
      settings.sppm = v;
      notifyChange();
    },
  });

  const denoiseRow = createToggleRow({
    label: "denoise (a-trous)",
    value: settings.denoise,
    onChange: (v) => {
      settings.denoise = v;
      notifyChange();
    },
  });

  const guideRow = createToggleRow({
    label: "path guiding",
    value: settings.guide,
    onChange: (v) => {
      settings.guide = v;
      notifyChange();
    },
  });

  const earsRow = createToggleRow({
    label: "adaptive RR (ADRRS)",
    value: settings.ears,
    onChange: (v) => {
      settings.ears = v;
      notifyChange();
    },
  });

  const adaptivePixelsRow = createToggleRow({
    label: "adaptive sampling",
    value: settings.adaptivePixels,
    onChange: (v) => {
      settings.adaptivePixels = v;
      notifyChange();
    },
  });

  const vcmRow = createToggleRow({
    label: "VCM (vertex connection)",
    value: settings.vcm,
    onChange: (v) => {
      settings.vcm = v;
      notifyChange();
    },
  });

  const pauseRow = createToggleRow({
    label: "pause (space)",
    value: settings.paused,
    onChange: (v) => {
      settings.paused = v;
    },
  });

  const adaptiveRow = createToggleRow({
    label: "adaptive spp",
    value: settings.adaptive,
    onChange: (v) => {
      settings.adaptive = v;
      notifyChange();
    },
  });

  const fixedSeedRow = createToggleRow({
    label: "fixed seed",
    value: settings.fixedSeed,
    onChange: (v) => {
      settings.fixedSeed = v;
      notifyChange();
    },
  });

  const resetButton = document.createElement("button");
  resetButton.type = "button";
  resetButton.className = "pt-reset";
  resetButton.textContent = "reset";
  resetButton.addEventListener("click", () => {
    options.onReset();
  });

  body.appendChild(sceneRow);
  body.appendChild(maxBouncesRow.row);
  body.appendChild(sppRow.row);
  body.appendChild(resolutionRow.row);
  body.appendChild(fovRow.row);
  body.appendChild(apertureRow.row);
  body.appendChild(interactiveResRow.row);
  body.appendChild(interactiveBouncesRow.row);
  body.appendChild(neeRow);
  body.appendChild(misRow);
  body.appendChild(qmcRow);
  body.appendChild(envIsRow);
  body.appendChild(reprojectRow);
  body.appendChild(fogRow);
  body.appendChild(sppmRow);
  body.appendChild(denoiseRow);
  body.appendChild(vcmRow);
  body.appendChild(guideRow);
  body.appendChild(earsRow);
  body.appendChild(adaptivePixelsRow);
  body.appendChild(pauseRow);
  body.appendChild(adaptiveRow);
  body.appendChild(fixedSeedRow);
  body.appendChild(debugRow);
  body.appendChild(resetButton);

  const status = document.createElement("div");
  status.className = "pt-status";

  panel.appendChild(header);
  panel.appendChild(body);
  panel.appendChild(status);

  document.body.appendChild(panel);

  // 空白キーでも切り替えられるようにする
  const pauseCheck = pauseRow.querySelector("input") as HTMLInputElement;
  const setPaused = (v: boolean) => {
    if (settings.paused === v) return;
    settings.paused = v;
    pauseCheck.checked = v;
  };
  window.addEventListener("keydown", (e) => {
    if (e.code !== "Space" || e.repeat) return;
    e.preventDefault();
    setPaused(!settings.paused);
  });

  return {
    settings,
    setPaused,
    setStatus: (text: string) => {
      status.textContent = text;
    },
    syncCamera: () => {
      fovRow.setValue(options.camera.fovDeg);
      apertureRow.setValue(options.camera.aperture);
    },
  };
}
