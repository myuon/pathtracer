# pathtracer

ブラウザで動く WebGPU パストレーサーのおもちゃ。

**デモ: https://myuon.github.io/pathtracer/**

WebGPU 対応ブラウザ (Chrome / Edge 113+, Safari 26+) が必要です。非対応の環境では画面にエラーが出ます。

- compute shader (WGSL) で 1 パス、accumulation buffer にプログレッシブ蓄積
- プリミティブは球・quad (平行四辺形)・三角形メッシュ / Lambert・GGX マイクロファセット (導体と誘電体)・面光源
- 誘電体は粗さを持てる (すりガラス) ほか、距離依存の吸収で色ガラスになる
- 箱で区切った一様な参加媒質 (霧)。距離サンプリングと Henyey-Greenstein 位相関数で光芒が出る
- SPPM (確率的プログレッシブ photon mapping、既定で有効)。集光や、光源が奥まったシーンで単方向パストレーシングより桁違いに速い。光子は面光源と環境マップの両方から撒く。巨大なプリミティブがあって外接球が広がりすぎるシーンでは自動的にパストレースへ落ちる
- 交差判定は BVH (中央値分割)。9,680 三角形のシーンでも対話的に動く
- シーンは UI から切り替え。RTIOW の定番シーン、Cornell box、Veach の MIS テストシーン、トーラス結び目メッシュ、すりガラス/色ガラス、霧の光芒、水面の集光、水没した部屋、間接照明だけの部屋、囲われた光源を同梱
- NEE (next event estimation) で拡散面から面光源を直接サンプル。Cornell box では同じノイズに落ちるまでの spp が 1/30 ほどになる
- MIS (power heuristic) で光源サンプリングと BSDF サンプリングを合成。光源と受光面が近接平行な箇所のスパイクが消える
- 画素ジッタ・レンズ・1 次バウンスに画素ごとスクランブルした Sobol (0,2) 列を使用。被写界深度のあるシーンでノイズが 1.3 倍ほど減る
- 極端に明るいサンプル (firefly) は画素の走行平均から作った閾値まで押し下げる。閾値をサンプル数の平方根に比例させてあるので収束先はずれない。12 シーンの relMSE が 1.5 倍改善
- orbit カメラ (左ドラッグ=回転 / 右ドラッグ=パン / ホイール=ドリー) + defocus blur
- 太陽つきの空を焼いた環境マップを 2 次元 CDF で重要度サンプリング。同 spp でノイズが 3.7 倍減る
- 操作中は低解像度・低バウンスで追従し、累積を再投影して引き継ぐ。手を止めるとフル品質で蓄積し直す
- 1 フレームの GPU 時間を実測して仕事量を自動調整する。非力な GPU でも操作の応答性が保たれる
- スペースキーまたは pause トグルで計算を止められる。絵はそのまま残り、再開すると続きから積む
- デバッグ用に法線・アルベド・深度・BSDF の pdf・MIS 重み・バウンス数を疑似カラー表示できる。固定 seed にすると同条件が完全に再現される

## 計測

`bench/run.mjs` で等時間 / 等 spp のノイズを自動で測れる。参照画像との
RMSE / relMSE / ACES-RMSE を出すので、「◯◯を入れたら何倍良くなったか」を
目視ではなく数字で決められる。使い方は [bench/README.md](bench/README.md)。

```sh
node bench/run.mjs ref --spp=32768 --sppf=32 --bounces=12   # 参照画像を焼く
node bench/run.mjs run --scenes=cornell --ms=10000 \
  --config "base:guide=0" --config "guide:guide=1"          # 等時間で比べる
```

## 動かす

```sh
pnpm install
pnpm dev
```

## 構成

| ファイル | 役割 |
| --- | --- |
| `src/main.ts` | エントリ、rAF ループ、品質モードの切り替え |
| `src/gpu.ts` | device 初期化、パイプラインとバッファの構築 |
| `src/camera.ts` | orbit カメラと入力 |
| `src/scene.ts` | シーン定義 (ジオメトリ・マテリアル・カメラ初期値) と storage buffer へのパック |
| `src/ui.ts` | 素の DOM のコントロールパネル |
| `src/shaders/pathtrace.wgsl` | パストレース本体 |
| `src/shaders/present.wgsl` | トーンマップ + フルスクリーン三角形 blit |
| `src/bench.ts` | 計測モード (`?bench=1`)。決めた予算だけ回して HDR を返す |
| `bench/run.mjs` | 計測ドライバ。Chrome を回して参照画像との誤差を出す |
