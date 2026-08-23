// WGSL と TS で手で揃えている定数がずれていないかを確かめる。
// ずれても何も言わずに壊れる (バッファの大きさや索引の計算が狂う) ので、
// 定数を触ったら必ず走らせること。pnpm bench:consts
import fs from "node:fs";
const w = fs.readFileSync("src/shaders/pathtrace.wgsl", "utf8");
const t = fs.readFileSync("src/gpu.ts", "utf8");
const num = (s) => {
  if (s === null || s === undefined) return null;
  const c = s.replace(/u\b/g, "").trim();
  try { return Function(`"use strict";return (${c})`)(); } catch { return c; }
};
const wg = (n) => { const m = w.match(new RegExp(`^const ${n}\\s*:\\s*\\w+\\s*=\\s*([^;]+);`, "m")); return m ? m[1] : null; };
const ts = (n) => { const m = t.match(new RegExp(`^const ${n}\\s*=\\s*([^;]+);`, "m")); return m ? m[1] : null; };

const names = ["GRID_CAP", "MAX_DEPOSITS", "HIST_BINS", "GUIDE_VOX", "GUIDE_BINS", "VTX_SLOTS"];
var ng = 0;
console.log("WGSL と TS で一致させる必要がある定数:");
for (const n of names) {
  const a = num(wg(n)), b = num(ts(n));
  if (a === null || b === null) { console.log(`  ${n.padEnd(13)} WGSL=${a} TS=${b}  (片方にしか無い)`); continue; }
  const ok = a === b;
  if (!ok) ng++;
  console.log(`  ${n.padEnd(13)} WGSL=${String(a).padStart(8)}  TS=${String(b).padStart(8)}  ${ok ? "一致" : "★不一致★"}`);
}
// GRID_CELLS は WGSL 側では uniform 経由なので TS だけ
console.log(`  GRID_CELLS    TS=${num(ts("GRID_CELLS"))} (WGSL は uniform 経由)`);

// Uniforms struct のサイズが UNIFORM_SIZE に収まるか (std140 相当の切り上げ)
const st = w.slice(w.indexOf("struct Uniforms"), w.indexOf("};", w.indexOf("struct Uniforms")));
const fields = [...st.matchAll(/^\s+(\w+)\s*:\s*(u32|f32|vec3f)\s*,/gm)].map((m) => m[2]);
let off = 0;
for (const ty of fields) { if (ty === "vec3f") { off = Math.ceil(off / 4) * 4; off += 3; } else off += 1; }
const need = Math.ceil(off / 4) * 4 * 4;
const us = num(ts("UNIFORM_SIZE"));
console.log(`\nUniforms: フィールド ${fields.length} 個 / 必要 ${need} バイト / UNIFORM_SIZE ${us}  ${need <= us ? "収まっている" : "★溢れている★"}`);
if (need > us) ng++;
// struct Uniforms の各フィールドが何番目のスロットに来るかを出し、
// gpu.ts の writeUniforms が同じ添字へ書いているかを突き合わせる。
// フィールドを 1 つ消して TS 側の添字を詰め忘れると、以降の値が全部
// 1 つずれて読まれる。絵は出るので気づきにくい
{
  const names = [];
  let o = 0;
  for (const m of st.matchAll(/^\s+(\w+)\s*:\s*(u32|f32|vec3f)\s*,/gm)) {
    if (m[2] === "vec3f") { o = Math.ceil(o / 4) * 4; names[o] = m[1]; o += 3; }
    else { names[o] = m[1]; o += 1; }
  }
  const body = t.slice(t.indexOf("writeUniforms("), t.indexOf("render(p: FrameParams)"));
  let mism = 0;
  for (const m of body.matchAll(/^\s*[uf]\[(\d+)\]\s*=\s*(?:p|this)\.(\w+)/gm)) {
    const idx = Number(m[1]), prop = m[2];
    const want = names[idx];
    // 名前は違うが意味は同じもの (計算して入れているなど)
    const alias = {
      aspect: "width", samplesAfter: "samplesBefore",
      photonCount: "photonsThisFrame", cellSize: "radius0",
    };
    if (alias[want] === prop) continue;
    if (want && want.toLowerCase() !== prop.toLowerCase() &&
        !want.toLowerCase().includes(prop.toLowerCase()) &&
        !prop.toLowerCase().includes(want.toLowerCase())) {
      console.log(`  ★ u[${idx}] に ${prop} を書いているが、WGSL の同じ位置は ${want}`);
      mism++;
    }
  }
  console.log(`\nuniform の添字: ${mism ? "★" + mism + " 件ずれている★" : "ずれなし"}`);
  if (mism) ng++;
}

const hb = num(ts("HIST_BYTES_PER_PIXEL"));
console.log(`HIST_BYTES_PER_PIXEL ${hb} (WGSL は vec4f 4 個 = 64 バイト)  ${hb === 64 ? "一致" : "★不一致★"}`);
if (hb !== 64) ng++;
console.log(ng ? `\n★ ${ng} 件の不一致` : "\nすべて一致");
process.exit(ng ? 1 : 0);
