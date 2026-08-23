// アプリ本体を実際に動かす煙テスト。bench は main.ts の rAF ループを
// 丸ごと迂回しているので、こちらでしか出ない不具合がある
import { chromium } from "playwright-core";
import { createServer } from "vite";
import { dirname } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));
const server = await createServer({ root: ROOT, server: { port: 0 }, logLevel: "warn" });
await server.listen();
const origin = server.resolvedUrls.local[0].replace(/\/$/, "");
const browser = await chromium.launch({
  executablePath: process.env.CHROME_PATH ??
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  headless: true,
  args: ["--enable-unsafe-webgpu"],
});
const page = await browser.newPage({ viewport: { width: 900, height: 650 } });
const bad = [];
page.on("console", (m) => {
  const t = m.text();
  if (m.type() === "error" || t.includes("Error while parsing WGSL") ||
      t.includes("is invalid due to a previous error")) {
    if (!t.includes("Failed to load resource")) bad.push(`[${m.type()}] ${t}`);
  }
});
page.on("pageerror", (e) => bad.push(`[pageerror] ${e.message}`));

await page.goto(origin, { waitUntil: "load" });
await page.waitForTimeout(2500);

const status = async () => page.evaluate(() => document.body.innerText.match(/\d+x\d+ \/ \d+ spp[^\n]*/)?.[0] ?? "");
console.log("起動後      :", await status());

// カメラを動かす (対話モード + 再投影を通す)
const c = await page.$("#canvas");
const box = await c.boundingBox();
await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
await page.mouse.down();
for (let i = 0; i < 12; i++) {
  await page.mouse.move(box.x + box.width / 2 + i * 9, box.y + box.height / 2 + i * 4);
  await page.waitForTimeout(45);
}
await page.mouse.up();
await page.waitForTimeout(1200);
console.log("カメラ操作後:", await status());

// シーンを一通り切り替える
const opts = await page.$$eval("select option", (o) => o.map((x) => x.value));
const scenes = opts.filter((v) => /^[a-z]+$/.test(v));
for (const s of scenes.slice(0, 12)) {
  await page.selectOption("select", s).catch(() => {});
  await page.waitForTimeout(500);
}
console.log("全シーン切替後:", await status());

// 一時停止と再開
await page.keyboard.press("Space");
await page.waitForTimeout(400);
const paused = await status();
await page.keyboard.press("Space");
await page.waitForTimeout(600);
console.log("一時停止     :", paused);
console.log("再開後      :", await status());

await browser.close();
await server.close();
if (bad.length) { console.log("\n=== エラー ===\n" + bad.slice(0, 10).join("\n")); process.exit(1); }
console.log("\nエラーなし");
