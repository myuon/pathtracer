# pathtracer

ブラウザで動く WebGPU パストレーサーのおもちゃ。

- compute shader (WGSL) で 1 パス、accumulation buffer にプログレッシブ蓄積
- 球のみ / Lambert・Metal・Dielectric / 環境光は空のグラデーション
- orbit カメラ (左ドラッグ=回転 / 右ドラッグ=パン / ホイール=ドリー) + defocus blur
- 操作中は低解像度・低バウンスで追従し、手を止めると数フレームでフル品質の蓄積に戻る

## 動かす

```sh
pnpm install
pnpm dev
```

WebGPU 対応ブラウザ (Chrome / Safari の新しめのバージョン) が必要です。

## 構成

| ファイル | 役割 |
| --- | --- |
| `src/main.ts` | エントリ、rAF ループ、品質モードの切り替え |
| `src/gpu.ts` | device 初期化、パイプラインとバッファの構築 |
| `src/camera.ts` | orbit カメラと入力 |
| `src/scene.ts` | 球とマテリアルの定義、storage buffer へのパック |
| `src/ui.ts` | 素の DOM のコントロールパネル |
| `src/shaders/pathtrace.wgsl` | パストレース本体 |
| `src/shaders/present.wgsl` | トーンマップ + フルスクリーン三角形 blit |
