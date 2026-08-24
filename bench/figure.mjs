// イシューや README に貼る比較画像を撮る。
//
//   node bench/figure.mjs --scene=maze --w=860 --h=540 --spp=145 \
//     --a "path tracing|単方向|sppm=0" \
//     --b "SPPM|photon mapping|sppm=1" \
//     --caption "baffle maze (板 2 枚) / 860x540 / どちらも 145 spp / max bounces 12" \
//     --out docs/sppm-maze.webp
//
// bench/run.mjs の PNG 出力は参照画像がある解像度でしか使えないので、
// 図を撮るためだけの経路を分けてある。?bench=1&present=1 で表示と同じ
// トーンマップを通した絵を読み戻し、そのまま並べる
import { readFileSync, writeFileSync, mkdtempSync, rmSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";
import { chromium } from "playwright-core";
import { createServer } from "vite";
import { toPng } from "./png.mjs";

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));
const CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";

const { values } = parseArgs({
  options: {
    scene: { type: "string" },
    w: { type: "string", default: "860" },
    h: { type: "string", default: "540" },
    spp: { type: "string" },
    // spp の代わりに壁時計時間で揃える (ms)
    ms: { type: "string" },
    bounces: { type: "string", default: "12" },
    // "見出し|右肩の小見出し|設定" を並べる。設定は run.mjs と同じ key=value,...
    a: { type: "string", multiple: true },
    caption: { type: "string", default: "" },
    out: { type: "string" },
    // パネルの表示幅 (px)。原寸より小さくすると縮小して貼る
    cw: { type: "string", default: "860" },
    // 2 段目に拡大を並べる "x,y,w,h" (元画像の画素座標)
    zoom: { type: "string" },
    zoomLabel: { type: "string", default: "拡大" },
  },
});

const W = Number(values.w), H = Number(values.h);
const BASE = "nee=1,mis=1,qmc=1,envIs=1,sppm=0,vcm=0,guide=0,ears=0,denoise=0";

/**
 * 画素ごとに 3x3 平均との差を取り、その二乗平均平方根をノイズ量とする。
 * 平均や明るさの違いには反応せず、粒状感だけを拾う。
 * あわせて、周りより飛び抜けて明るい画素 (firefly) の数も数える
 */
function noiseOf(rgba, w, h) {
  const lum = new Float64Array(w * h);
  for (let i = 0; i < w * h; i++) {
    lum[i] = 0.2126 * rgba[i * 4] + 0.7152 * rgba[i * 4 + 1] + 0.0722 * rgba[i * 4 + 2];
  }
  let ss = 0, n = 0, fly = 0, mean = 0;
  for (let i = 0; i < w * h; i++) mean += lum[i] / (w * h);
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      let sum = 0;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) sum += lum[(y + dy) * w + (x + dx)];
      }
      const d = lum[y * w + x] - sum / 9;
      ss += d * d;
      n += 1;
      // 8bit で 24 階調ぶん周りより明るい画素は、点として見える
      if (d > 24) fly += 1;
    }
  }
  return { sigma: Math.sqrt(ss / n), fly, mean };
}

/** RGBA から矩形を切り出す */
function crop(rgba, w, [x, y, cw, ch]) {
  const out = Buffer.alloc(cw * ch * 4);
  for (let j = 0; j < ch; j++) {
    rgba.copy(out, j * cw * 4, ((y + j) * w + x) * 4, ((y + j) * w + x + cw) * 4);
  }
  return out;
}

async function shoot(browser, origin, config) {
  const q = new URLSearchParams({
    bench: "1", present: "1", salt: "0",
    scene: values.scene, w: String(W), h: String(H),
    sppf: "1", bounces: values.bounces,
  });
  if (values.ms) q.set("ms", values.ms); else q.set("spp", values.spp ?? "128");
  for (const kv of `${BASE},${config}`.split(",").filter(Boolean)) {
    const [k, v] = kv.split("=");
    q.set(k, v ?? "1");
  }
  const page = await browser.newPage({ viewport: { width: W + 40, height: H + 40 } });
  const logs = [];
  page.on("console", (m) => logs.push(`[${m.type()}] ${m.text()}`));
  page.on("pageerror", (e) => logs.push(`[pageerror] ${e.message}`));
  try {
    await page.goto(`${origin}/?${q}`, { waitUntil: "load" });
    await page.waitForFunction(() => window.__bench !== undefined, null, {
      timeout: 30 * 60 * 1000,
    });
    const r = await page.evaluate(() => window.__bench);
    const bad = logs.filter(
      (l) => (l.startsWith("[error]") || l.startsWith("[pageerror]") ||
        l.includes("Error while parsing WGSL")) && !l.includes("Failed to load resource"),
    );
    if (bad.length) throw new Error(`ページがエラーを出した:\n${bad.slice(0, 3).join("\n")}`);
    return { ldr: Buffer.from(r.ldr, "base64"), spp: r.spp, ms: r.ms };
  } finally {
    await page.close();
  }
}

const panels = (values.a ?? []).map((s) => {
  const [label, sub, config] = s.split("|");
  return { label, sub: sub ?? "", config: config ?? "" };
});
if (!panels.length || !values.out) throw new Error("--a と --out が要る");

const server = await createServer({ root: ROOT, server: { port: 0 }, logLevel: "warn" });
await server.listen();
const origin = server.resolvedUrls.local[0].replace(/\/$/, "");
const browser = await chromium.launch({
  executablePath: CHROME, headless: true, args: ["--enable-unsafe-webgpu"],
});
const zoom = values.zoom ? values.zoom.split(",").map(Number) : null;
const dir = mkdtempSync(join(tmpdir(), "fig-"));
try {
  for (const [i, p] of panels.entries()) {
    const r = await shoot(browser, origin, p.config);
    p.png = join(dir, `panel${i}.png`);
    writeFileSync(p.png, toPng(r.ldr, W, H));
    p.stat = noiseOf(r.ldr, W, H);
    if (zoom) {
      const c = crop(r.ldr, W, zoom);
      p.zoomPng = join(dir, `panel${i}-zoom.png`);
      writeFileSync(p.zoomPng, toPng(c, zoom[2], zoom[3]));
      p.zoomStat = noiseOf(c, zoom[2], zoom[3]);
    }
    console.log(
      `${p.label.padEnd(16)} ${String(r.spp).padStart(6)} spp ` +
        `${(r.ms / 1000).toFixed(1).padStart(6)}s  ノイズ ${p.stat.sigma.toFixed(2)}  ` +
        `firefly ${p.stat.fly}  平均輝度 ${p.stat.mean.toFixed(2)}` +
        (p.zoomStat ? `  [拡大部 ノイズ ${p.zoomStat.sigma.toFixed(2)} ` +
          `firefly ${p.zoomStat.fly}]` : ""),
    );
  }

  const CW = Number(values.cw);
  const b64 = (p) => "data:image/png;base64," + readFileSync(p).toString("base64");
  const cells = panels.map((p) => `
    <div class="cell">
      <div class="head"><b>${p.label}</b><span>${p.sub}</span></div>
      <img src="${b64(p.png)}">
    </div>`).join("");
  const zooms = zoom ? panels.map((p) => `
    <div class="cell"><img src="${b64(p.zoomPng)}"></div>`).join("") : "";
  const html = `<!doctype html><meta charset="utf-8"><style>
  body{margin:0;background:#14161a;color:#e8e8e8;
    font:14px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace}
  .wrap{display:inline-block;padding:12px}
  .cols{display:flex;gap:10px}
  .cell{width:${CW}px}
  .head{display:flex;justify-content:space-between;align-items:baseline;padding:2px 2px 8px}
  .head b{font-size:19px}
  .head span{opacity:.55;font-size:13px}
  .cell img{width:${CW}px;display:block;border-radius:3px}
  .note{opacity:.7;font-size:13px;padding:10px 2px 2px;white-space:pre-line}
  .zlab{opacity:.55;font-size:13px;padding:8px 2px 4px}
</style><div class="wrap">
  <div class="cols">${cells}</div>
  ${zoom ? `<div class="zlab">${values.zoomLabel}</div><div class="cols">${zooms}</div>` : ""}
  ${values.caption ? `<div class="note">${values.caption}</div>` : ""}
</div>`;

  const page = await browser.newPage({
    viewport: { width: CW * panels.length + 40, height: 400 },
    deviceScaleFactor: 1,
  });
  await page.setContent(html);
  await page.waitForTimeout(500);
  const shot = join(dir, "figure.png");
  writeFileSync(shot, await page.locator(".wrap").screenshot());
  execFileSync("cwebp", ["-q", "88", "-quiet", shot, "-o", resolve(ROOT, values.out)]);
  console.log("wrote", values.out);
} finally {
  await browser.close();
  await server.close();
  rmSync(dir, { recursive: true, force: true });
}
