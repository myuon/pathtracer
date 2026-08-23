// アプリを実際に動かして、同じ壁時計時間だけ回した絵を撮る。
// main には bench が無いので、こちらで前後を揃える
import { chromium } from "playwright-core";
import { createServer } from "vite";
import { writeFileSync } from "node:fs";
import { parseArgs } from "node:util";

const { values } = parseArgs({ options: {
  root: { type: "string" }, out: { type: "string" },
  scenes: { type: "string" }, ms: { type: "string", default: "10000" },
  w: { type: "string", default: "700" }, h: { type: "string", default: "480" },
}});
const W = Number(values.w), H = Number(values.h), MS = Number(values.ms);

const server = await createServer({ root: values.root, server: { port: 0 }, logLevel: "warn" });
await server.listen();
const origin = server.resolvedUrls.local[0].replace(/\/$/, "");
const browser = await chromium.launch({
  executablePath: "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  headless: true, args: ["--enable-unsafe-webgpu"],
});
const page = await browser.newPage({ viewport: { width: W, height: H }, deviceScaleFactor: 1 });
page.on("pageerror", (e) => console.error("[pageerror]", e.message));
await page.goto(origin, { waitUntil: "load" });
await page.waitForTimeout(1500);

for (const scene of values.scenes.split(",")) {
  // シーンを選ぶ (options に scene id を持つ select を探す)
  await page.evaluate((s) => {
    for (const sel of document.querySelectorAll("select")) {
      if ([...sel.options].some((o) => o.value === s)) {
        sel.value = s;
        sel.dispatchEvent(new Event("change", { bubbles: true }));
        return;
      }
    }
  }, scene);
  await page.waitForTimeout(MS);
  // パネルを隠してから撮る
  await page.evaluate(() => {
    for (const el of document.querySelectorAll("body > div")) {
      if (el.id !== "unsupported") el.style.visibility = "hidden";
    }
  });
  const buf = await page.locator("#canvas").screenshot();
  writeFileSync(`${values.out}/${scene}.png`, buf);
  await page.evaluate(() => {
    for (const el of document.querySelectorAll("body > div")) el.style.visibility = "";
  });
  const status = await page.evaluate(() => document.body.innerText.match(/\d+ spp/)?.[0] ?? "");
  console.log(`  ${scene.padEnd(10)} ${status}`);
}
await browser.close();
await server.close();
