# pathtracer

ブラウザで動く WebGPU パストレーサーのおもちゃ。

**デモ: https://myuon.github.io/pathtracer/**

WebGPU 対応ブラウザ (Chrome / Edge 113+, Safari 26+) が必要です。非対応の環境では画面にエラーが出ます。

- compute shader (WGSL) で 1 パス、accumulation buffer にプログレッシブ蓄積
- プリミティブは球・quad (平行四辺形)・三角形メッシュ / Lambert・GGX マイクロファセット (導体)・Dielectric・面光源
- 交差判定は BVH (中央値分割)。9,680 三角形のシーンでも対話的に動く
- シーンは UI から切り替え。RTIOW の定番シーン、Cornell box、Veach の MIS テストシーン、トーラス結び目メッシュを同梱
- NEE (next event estimation) で拡散面から面光源を直接サンプル。Cornell box では同じノイズに落ちるまでの spp が 1/30 ほどになる
- MIS (power heuristic) で光源サンプリングと BSDF サンプリングを合成。光源と受光面が近接平行な箇所のスパイクが消える
- 画素ジッタ・レンズ・1 次バウンスに画素ごとスクランブルした Sobol (0,2) 列を使用。被写界深度のあるシーンでノイズが 1.3 倍ほど減る
- orbit カメラ (左ドラッグ=回転 / 右ドラッグ=パン / ホイール=ドリー) + defocus blur
- 操作中は低解像度・低バウンスで追従し、手を止めると数フレームでフル品質の蓄積に戻る

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
