// 撮った絵をラベル付きで横に並べる。画像ライブラリを足したくないので
// HTML を組んでブラウザに描かせ、それを撮る
import { chromium } from "playwright-core";
import { readFileSync, writeFileSync } from "node:fs";
import { parseArgs } from "node:util";

const { values } = parseArgs({ options: {
  a: { type: "string" }, b: { type: "string" },
  labelA: { type: "string", default: "変更前" }, labelB: { type: "string", default: "変更後" },
  scenes: { type: "string" }, out: { type: "string" },
  cw: { type: "string", default: "620" }, note: { type: "string", default: "" },
}});
const CW = Number(values.cw);
const scenes = values.scenes.split(",");
const b64 = (p) => "data:image/png;base64," + readFileSync(p).toString("base64");
// shot.mjs が書いた到達 spp があれば絵に添える
const sppOf = (dir) => {
  try { return JSON.parse(readFileSync(`${dir}/spp.json`, "utf8")); } catch { return {}; }
};
const sa = sppOf(values.a), sb = sppOf(values.b);

const rows = scenes.map((s) => `
  <div class="row">
    <div class="name">${s}</div>
    <div class="spp left">${sa[s] ?? ""}</div>
    <div class="spp right">${sb[s] ?? ""}</div>
    <img src="${b64(`${values.a}/${s}.png`)}">
    <img src="${b64(`${values.b}/${s}.png`)}">
  </div>`).join("");

const html = `<!doctype html><meta charset="utf-8"><style>
  body{margin:0;background:#14161a;color:#e8e8e8;
    font:14px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace}
  .wrap{width:${CW * 2 + 24}px;padding:12px}
  .head{display:flex;gap:8px;margin:2px 0 8px}
  .head div{width:${CW}px;text-align:center;font-size:15px;padding:5px 0;
    background:#242830;border-radius:4px}
  .row{position:relative;display:flex;gap:8px;margin-bottom:8px}
  .row img{width:${CW}px;display:block;border-radius:3px}
  .name{position:absolute;left:8px;top:6px;z-index:2;background:rgba(0,0,0,.62);
    padding:2px 8px;border-radius:3px;font-size:13px}
  .spp{position:absolute;bottom:6px;z-index:2;background:rgba(0,0,0,.62);
    padding:2px 8px;border-radius:3px;font-size:13px}
  .spp.left{left:8px}
  .spp.right{left:calc(50% + 8px)}
  .note{opacity:.68;font-size:12px;padding:2px 2px 6px}
</style><div class="wrap">
  <div class="head"><div>${values.labelA}</div><div>${values.labelB}</div></div>
  ${rows}
  ${values.note ? `<div class="note">${values.note}</div>` : ""}
</div>`;

const browser = await chromium.launch({
  executablePath: "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  headless: true,
});
const page = await browser.newPage({ viewport: { width: CW * 2 + 24, height: 800 } });
await page.setContent(html);
await page.waitForTimeout(400);
writeFileSync(values.out, await page.locator(".wrap").screenshot());
await browser.close();
console.log("wrote", values.out);
