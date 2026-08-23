#!/usr/bin/env node
// 等時間 / 等 spp のノイズを自動で測る。
//
//   node bench/run.mjs ref --scenes=cornell,glass --spp=4096
//   node bench/run.mjs run --scenes=cornell --ms=20000 \
//        --config "base:guide=0" --config "guide:guide=1"
//
// 1 回の計測につきページを 1 枚使い切る。学習した分布 (ガイド格子や
// 光子の放出ヒストグラム) が前の計測から漏れないようにするため
import { mkdirSync, readFileSync, writeFileSync, existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";
import { chromium } from "playwright-core";
import { createServer } from "vite";

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));
const REF_DIR = join(ROOT, "bench", "ref");

const CHROME =
  process.env.CHROME_PATH ??
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";

/** 参照画像を作るときの設定。基本は無バイアスなパストレース */
const REF_CONFIG = "nee=1,mis=1,qmc=1,envIs=1,sppm=0,vcm=0,guide=0,denoise=0";

/**
 * パストレースが収束しないシーンだけ、別の推定量で焼く。
 *
 * submerged は小さな光源が波打つ水面の向こうにあり、PT はその経路を
 * ほとんど引けない (平均輝度が 256 spp で 0.276、32768 spp でも 0.339 と
 * 伸び続ける)。SPPM は 1024 spp で 0.45075、4096 spp で 0.45118 と止まり、
 * しかも集光半径を 1/4 にしても 0.45107 で動かない = 残っている偏りは
 * 無視できる。こちらを参照にする
 */
const REF_OVERRIDE = {
  // SPPM は 1024 spp で止まるので、予算も PT ほど要らない
  submerged: {
    config: "nee=1,mis=1,qmc=1,envIs=1,sppm=1,vcm=0,guide=0,denoise=0",
    spp: "4096",
    sppf: "1",
  },
  // 半開きの扉の隙間を通す経路を PT はほとんど引けず、131072 spp でも
  // 平均輝度が伸び続ける (0.10730 -> 0.11189 -> 0.11361 -> 0.11482)。
  // VCM は同じ予算でずっと速く収束する (1024 -> 0.11156、16384 -> 0.11399、
  // 増分が +1.77% -> +0.24% -> +0.17% と落ちる) ので、そちらで焼く
  ajar: {
    config: "nee=1,mis=1,qmc=1,envIs=1,sppm=0,vcm=1,guide=0,denoise=0",
    spp: "32768",
  },
};

const DEFAULT_SCENES = [
  "spheres", "cornell", "veach", "mesh", "glass", "shaft",
  "indirect", "enclosed", "maze", "ajar", "water", "submerged",
];

const { values, positionals } = parseArgs({
  allowPositionals: true,
  options: {
    scenes: { type: "string" },
    config: { type: "string", multiple: true },
    spp: { type: "string" },
    ms: { type: "string" },
    w: { type: "string", default: "320" },
    h: { type: "string", default: "240" },
    bounces: { type: "string", default: "12" },
    sppf: { type: "string", default: "1" },
    headed: { type: "boolean", default: false },
    json: { type: "string" },
    dump: { type: "boolean", default: false },
    present: { type: "boolean", default: false },
    repeat: { type: "string", default: "1" },
  },
});

const mode = positionals[0] ?? "run";
const W = Number(values.w);
const H = Number(values.h);
const scenes = (values.scenes ?? DEFAULT_SCENES.join(",")).split(",").filter(Boolean);

// 参照画像はカメラ・解像度・バウンス数に固有なので、取り違えないよう
// ファイル名に全部入れる
function refPath(scene) {
  return join(REF_DIR, `${scene}_${W}x${H}_b${values.bounces}.f32`);
}

/** "name:k=v,k=v" を分解する。名前を省いたら設定文字列をそのまま名前にする */
function parseConfig(s) {
  const i = s.indexOf(":");
  if (i < 0) return { name: s || "default", params: s };
  return { name: s.slice(0, i), params: s.slice(i + 1) };
}

function buildUrl(origin, scene, params) {
  const q = new URLSearchParams();
  q.set("bench", "1");
  q.set("scene", scene);
  q.set("w", String(W));
  q.set("h", String(H));
  q.set("bounces", values.bounces);
  q.set("sppf", values.sppf);
  // 参照画像と検証画像で乱数の塩を変える。同じにすると、検証側の N 点が
  // 参照側の点列の先頭 N 点とそのまま一致してしまい、両者の誤差が
  // 打ち消し合って実際より良く見える (実測で 4%ほど下駄を履いていた)
  q.set("salt", mode === "ref" ? "0" : "1");
  // 表示された絵 (トーンマップ + デノイザ通し) も返させる
  if (values.present && mode !== "ref") q.set("present", "1");
  for (const kv of params.split(",")) {
    if (!kv) continue;
    const [k, v] = kv.split("=");
    q.set(k, v ?? "1");
  }
  return `${origin}/?${q.toString()}`;
}

async function measure(browser, origin, scene, params, budget) {
  const q = { ...budget };
  const url = buildUrl(origin, scene, params) +
    (q.spp ? `&spp=${q.spp}` : "") + (q.ms ? `&ms=${q.ms}` : "");
  const page = await browser.newPage({ viewport: { width: W + 40, height: H + 40 } });
  const logs = [];
  page.on("console", (m) => logs.push(`[${m.type()}] ${m.text()}`));
  page.on("pageerror", (e) => logs.push(`[pageerror] ${e.message}`));
  try {
    await page.goto(url, { waitUntil: "load" });
    const timeout = (q.ms ? Number(q.ms) : 0) * 4 + 15 * 60 * 1000;
    await page.waitForFunction(() => window.__bench !== undefined, null, { timeout });
    const r = await page.evaluate(() => window.__bench);
    // シェーダのコンパイルに失敗しても WebGPU は非同期にエラーを出すだけで
    // 走り続けてしまい、真っ黒な絵が「結果」として返ってくる。
    // 数値だけ見ていると気づけないので、ここで落とす
    // WGSL のコンパイル失敗は console.warning にしか出ず、しかも WebGPU は
    // そのまま走り続けるので、真っ黒な絵が「結果」として返ってくる。
    // 数値だけ見ていると気づけないのでここで落とす
    const errs = logs.filter(
      (l) =>
        (l.startsWith("[error]") ||
          l.startsWith("[pageerror]") ||
          l.includes("Error while parsing WGSL") ||
          l.includes("is invalid due to a previous error")) &&
        // favicon の 404 のような無害なものは無視する
        !l.includes("Failed to load resource"),
    );
    if (errs.length) throw new Error(`ページがエラーを出した:\n${errs.slice(0, 3).join("\n")}`);
    return {
      ...r,
      hdr: Buffer.from(r.hdr, "base64"),
      ldr: r.ldr ? Buffer.from(r.ldr, "base64") : undefined,
    };
  } catch (e) {
    throw new Error(`${scene} [${params}] の計測に失敗:\n${e.message}\n${logs.join("\n")}`);
  } finally {
    await page.close();
  }
}

function toF32(buf) {
  return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
}

function aces(x) {
  const v = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14);
  return Math.pow(Math.min(Math.max(v, 0), 1), 1 / 2.2);
}

/** 参照との誤差。値が壊れている画素は数だけ報告して除外する */
function metrics(img, ref) {
  let se = 0, rel = 0, ase = 0, n = 0, bad = 0;
  // 画素ごとの相対二乗誤差。裾に強く引きずられる relmse を補うため、
  // 上位 1% を落としたものと中央値も出す
  const per = [];
  const px = img.length / 3;
  for (let i = 0; i < px; i++) {
    let pe = 0, pn = 0;
    for (let j = 0; j < 3; j++) {
      const a = img[i * 3 + j], b = ref[i * 3 + j];
      if (!Number.isFinite(a) || !Number.isFinite(b)) { bad++; continue; }
      const d = a - b;
      se += d * d;
      const r = (d * d) / (b * b + 0.01);
      rel += r;
      pe += r; pn++;
      const da = aces(a) - aces(b);
      ase += da * da;
      n++;
    }
    if (pn) per.push(pe / pn);
  }
  per.sort((a, b) => a - b);
  const keep = Math.max(1, Math.floor(per.length * 0.99));
  let trimSum = 0;
  for (let i = 0; i < keep; i++) trimSum += per[i];
  return {
    rmse: Math.sqrt(se / n),
    relmse: rel / n,
    // 上位 1% の画素を落とした relMSE。光沢面のシルエットに出る十数個の
    // firefly が relMSE の半分以上を占めることがあり、素の relMSE では
    // サンプラの良し悪しがまったく見えない
    trimmed: trimSum / keep,
    /// 画素ごとの相対二乗誤差の中央値。典型的な画素がどれだけ合っているか
    medRel: per[Math.floor(per.length * 0.5)],
    acesRmse: Math.sqrt(ase / n),
    bad,
  };
}

async function withBrowser(fn) {
  const server = await createServer({ root: ROOT, server: { port: 0 }, logLevel: "warn" });
  await server.listen();
  const origin = server.resolvedUrls.local[0].replace(/\/$/, "");
  const browser = await chromium.launch({
    executablePath: CHROME,
    headless: !values.headed,
    args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan"],
  });
  try {
    return await fn(browser, origin);
  } finally {
    await browser.close();
    await server.close();
  }
}

function meanOf(buf) {
  const f = toF32(buf);
  let s = 0;
  for (let i = 0; i < f.length; i++) s += f[i];
  return s / f.length;
}

/** 独立な実行を平均する。裾の重いシーンでは参照そのものが firefly を拾うので */
function averageHdr(bufs) {
  const n = bufs.length;
  if (n === 1) return bufs[0];
  const acc = new Float64Array(toF32(bufs[0]).length);
  for (const b of bufs) { const f = toF32(b); for (let i = 0; i < f.length; i++) acc[i] += f[i]; }
  const out = new Float32Array(acc.length);
  for (let i = 0; i < acc.length; i++) out[i] = acc[i] / n;
  return Buffer.from(out.buffer);
}

async function cmdRef() {
  const spp = values.spp ?? "4096";
  const repeat = Math.max(1, Number(values.repeat));
  mkdirSync(REF_DIR, { recursive: true });
  await withBrowser(async (browser, origin) => {
    for (const scene of scenes) {
      const t = Date.now();
      // 検証側 (salt=1) と重ならない塩で、独立な実行を repeat 回まわして平均する
      const runs = [];
      for (let i = 0; i < repeat; i++) {
        const ov = REF_OVERRIDE[scene];
        const cfg = ov ? ov.config : REF_CONFIG;
        const budget = { spp: ov?.spp ?? spp };
        const extra = ov?.sppf ? `,sppf=${ov.sppf}` : "";
        runs.push(await measure(browser, origin, scene, `${cfg}${extra},salt=${1000 + i}`, budget));
      }
      const r = { ...runs[0], hdr: averageHdr(runs.map((x) => x.hdr)) };
      writeFileSync(refPath(scene), r.hdr);
      let note = "";
      // 収束の確認。予算の 1/4 と比べて平均輝度がまだ動いているなら、
      // その参照画像は収束しておらず、誤差の基準に使えない。
      // submerged は 32768 spp でもまだ +13%/4倍 で伸びていて、
      // 「SPPM が正しいのに参照が間違っている」状態になっていた
      const ov2 = REF_OVERRIDE[scene];
      const quarter = await measure(
        browser, origin, scene,
        (ov2 ? ov2.config : REF_CONFIG) + (ov2?.sppf ? `,sppf=${ov2.sppf}` : ""),
        { spp: String(Math.max(1, Math.round(Number(ov2?.spp ?? spp) / 4))) },
      );
      const drift = meanOf(r.hdr) / meanOf(quarter.hdr) - 1;
      if (Math.abs(drift) > 0.02) {
        note = `  <- 未収束 (1/4 予算から ${(drift * 100).toFixed(1)}% 動いている)`;
      }
      console.log(
        `ref ${scene.padEnd(10)} ${r.spp} spp / ${(r.ms / 1000).toFixed(1)}s ` +
          `(壁時計 ${((Date.now() - t) / 1000).toFixed(1)}s)${note}`,
      );
    }
  });
}

async function cmdRun() {
  const configs = (values.config ?? ["default:"]).map(parseConfig);
  const budget = values.ms ? { ms: values.ms } : { spp: values.spp ?? "256" };
  const rows = [];
  await withBrowser(async (browser, origin) => {
    for (const scene of scenes) {
      if (!existsSync(refPath(scene))) {
        console.log(`skip ${scene}: 参照画像がない (先に ref を作る)`);
        continue;
      }
      const ref = toF32(readFileSync(refPath(scene)));
      for (const c of configs) {
        const r = await measure(browser, origin, scene, c.params, budget);
        const img = toF32(r.hdr);
        if (img.length !== ref.length) throw new Error("参照画像とサイズが違う");
        const m = metrics(img, ref);
        // 表示に出る絵での誤差。デノイザは present の中で効くので、
        // 累積バッファを読むだけでは効果が測れない
        if (r.ldr) {
          let se = 0;
          const px = ref.length / 3;
          for (let i = 0; i < px; i++) {
            for (let j = 0; j < 3; j++) {
              const want = Math.round(aces(ref[i * 3 + j]) * 255);
              const got = r.ldr[i * 4 + j];
              se += (want - got) * (want - got);
            }
          }
          m.ldrRmse = Math.sqrt(se / (px * 3));
        }
        // 効率 = relMSE x 秒。等 spp で測っても等時間の優劣が出る指標。
        // 「10 秒でどこまで行くか」を直接測るより、フレーム数の刻みに
        // 影響されないぶん再現性が高い
        const eff = m.relmse * (r.ms / 1000);
        rows.push({ scene, config: c.name, spp: r.spp, ms: r.ms, eff, ...m });
        // 調査用に HDR をそのまま落とす。どの画素が壊れているかを見るため
        if (values.dump) {
          mkdirSync(join(ROOT, "bench", "out"), { recursive: true });
          writeFileSync(join(ROOT, "bench", "out", `${scene}_${c.name}.f32`), r.hdr);
        }
        const last = rows[rows.length - 1];
        console.log(
          `${scene.padEnd(10)} ${c.name.padEnd(12)} ${String(last.spp).padStart(6)} spp ` +
            `${(last.ms / 1000).toFixed(1).padStart(6)}s  ` +
            `relmse ${last.relmse.toExponential(3)}  trim ${last.trimmed.toExponential(3)}  ` +
            `med ${last.medRel.toExponential(3)}  aces ${last.acesRmse.toExponential(3)}  ` +
            `eff ${last.eff.toExponential(3)}` +
            (last.ldrRmse !== undefined ? `  ldr ${last.ldrRmse.toFixed(3)}` : "") +
            (last.bad ? `  (壊れた成分 ${last.bad})` : ""),
        );
      }
    }
  });

  // 先頭の設定を基準にした倍率。1 より大きいほど良い
  const names = configs.map((c) => c.name);
  for (const key of ["relmse", "trimmed", "medRel", "eff"]) {
    console.log(
      `\n=== 基準 (${names[0]}) に対する ${
        { eff: "効率 (relMSE x 秒)", relmse: "relMSE", trimmed: "relMSE (上位 1% 除く)", medRel: "相対誤差の中央値" }[key]
      } の倍率 ===`,
    );
    console.log(["scene".padEnd(10), ...names.map((n) => n.padStart(12))].join(""));
    const all = [];
    for (const scene of scenes) {
      const rs = rows.filter((r) => r.scene === scene);
      if (!rs.length) continue;
      const b = rs[0][key];
      console.log(
        [scene.padEnd(10), ...names.map((n) => {
          const r = rs.find((x) => x.config === n);
          if (r) all.push({ n, v: b / r[key] });
          return (r ? (b / r[key]).toFixed(3) + "x" : "-").padStart(12);
        })].join(""),
      );
    }
    // 相乗平均。1 シーンの外れ値に引きずられないよう積で平均する
    console.log(
      ["幾何平均".padEnd(8), ...names.map((n) => {
        const vs = all.filter((x) => x.n === n).map((x) => x.v);
        const g = Math.exp(vs.reduce((a, v) => a + Math.log(v), 0) / vs.length);
        return (vs.length ? g.toFixed(3) + "x" : "-").padStart(12);
      })].join(""),
    );
  }
  if (values.json) writeFileSync(values.json, JSON.stringify(rows, null, 2));
}

if (mode === "ref") await cmdRef();
else if (mode === "run") await cmdRun();
else {
  console.error(`不明なモード: ${mode} (ref か run)`);
  process.exit(1);
}
