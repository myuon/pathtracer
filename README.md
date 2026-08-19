# pathtracer

ブラウザで動く WebGPU パストレーサーのおもちゃ。

**デモ: https://myuon.github.io/pathtracer/**

WebGPU 対応ブラウザ (Chrome / Edge 113+, Safari 26+) が必要です。非対応の環境では画面にエラーが出ます。

- compute shader (WGSL) で 1 パス、accumulation buffer にプログレッシブ蓄積
- プリミティブは球と quad (平行四辺形) / Lambert・Metal・Dielectric・面光源
- シーンは UI から切り替え。RTIOW の定番シーンと Cornell box を同梱
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
