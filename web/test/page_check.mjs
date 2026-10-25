/*
 * page_check.mjs — headless-Chromium acceptance run for the page (run
 * from /tmp/ortvendor where playwright is installed; Node resolves the
 * package from there):
 *
 *   node page_check.mjs
 *
 * Exercises what the IAB webview could not (it cannot cross-origin
 * isolate): real COOP/COEP-isolated Chromium, the full play loop
 * (click e2e4 → net replies through the ORT bridge), and the
 * walkthrough — a real shard pulled straight from Hugging Face through
 * the panel's own loaders (the page has no local-file inputs).
 * Screenshots land in web/test/shots/.
 */
import { mkdirSync } from "node:fs";
import { createRequire } from "node:module";
import path from "node:path";
import { fileURLToPath } from "node:url";
// playwright lives in a scratch install outside the repo (the page itself
// has no dependencies) — bind it explicitly by location.
const { chromium } = createRequire("/tmp/ortvendor/package.json")("playwright");

const ROOT = path.resolve(fileURLToPath(new URL("../../", import.meta.url)));
const BASE = "http://127.0.0.1:8468";
const SHOTS = `${ROOT}/web/test/shots`;
mkdirSync(SHOTS, { recursive: true });

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1280, height: 860 } });
page.on("pageerror", (e) => console.log("PAGE ERROR:", e.message));
page.on("console", (m) => console.log(`CONSOLE[${m.type()}]:`, m.text().slice(0, 300)));

await page.goto(`${BASE}/web/`);
// single-threaded engine: no SharedArrayBuffer, no isolation requirement
await page.waitForFunction(
  () => document.getElementById("status").textContent.includes("your move"),
  { timeout: 15000 });
console.log("play mode ready:", await page.locator("#status").textContent());

// load the real net through the panel's HF flow (list → pick → load).
// No .onnx pushed to the models repo yet → skip and run the dummy eval.
await page.locator("#netList").click();
let netLoaded = false;
try {
  await page.waitForFunction(
    () => document.querySelectorAll("#netSel option").length > 0,
    { timeout: 20000 });
  await page.selectOption("#netSel", { index: 0 });
  await page.locator("#netLoad").click();
  await page.waitForFunction(() => {
    const t = document.getElementById("netName").textContent;
    return t.includes("(") && !t.startsWith("loading");
  }, { timeout: 120000 });
  netLoaded = true;
} catch (e) {
  console.log("NET LOAD SKIPPED:",
    await page.locator("#netName").textContent());
}
console.log("net:", await page.locator("#netName").textContent(),
            netLoaded ? "(loaded from HF)" : "(dummy evaluator)");

// 1. e4 by clicking the board cells
await page.locator('.cell[data-sq="e2"]').click();
await page.locator('.cell[data-sq="e4"]').click();
try {
  await page.waitForFunction(
    () => document.querySelectorAll("#moves li").length >= 2, { timeout: 90000 });
} catch (e) {
  console.log("MOVES WAIT FAILED. status:", await page.locator("#status").textContent());
  console.log("moves html:", await page.locator("#moves").innerHTML());
  console.log("pieces:", await page.locator("#grid .piece").count(),
              "fen:", await page.locator("#fen").textContent());
  console.log("dbg:", JSON.stringify(await page.evaluate("window.__dbg()")));
  throw e;
}
const afterE4 = await page.locator("#moves").textContent();
console.log("after 1.e4, engine replied:", afterE4.trim().slice(0, 40));
await page.screenshot({ path: `${SHOTS}/play.png` });

// walkthrough: a real v3 shard from the HF datasets repo
await page.locator("#tabWalk").click();
await page.locator("#shardList").click();
await page.waitForFunction(
  () => document.querySelectorAll("#shardSel option").length > 0,
  { timeout: 60000 });
console.log("shards listed:", await page.locator("#shardSel option").count());
await page.selectOption("#shardSel", { index: 0 });
await page.waitForFunction(
  () => +document.querySelectorAll("#wtGameSel option").length > 100,
  { timeout: 120000 });
console.log("shard:",
  await page.locator("#shardInfo").textContent(),
  "| games listed:", await page.locator("#wtGameSel option").count());
// pick a game, step to ply 12
await page.selectOption("#wtGameSel", { index: 8 });
await page.locator("#wtSlider").evaluate((el) => {
  el.value = "12";
  el.dispatchEvent(new Event("input", { bubbles: true }));
});
await page.waitForFunction(
  () => document.querySelector("#status").textContent.includes("ply 12/"),
  { timeout: 15000 });
console.log("walkthrough:", await page.locator("#status").textContent());
const arrows = await page.locator("#arrows line").count();
const evalv = await page.locator("#evalbar").getAttribute("data-v");
console.log(`ply-12 overlay: ${arrows} arrows, eval ${evalv}`);
if (arrows < 2) throw new Error("v3 π arrows missing — replay/zip broken?");
await page.screenshot({ path: `${SHOTS}/walkthrough.png` });

await browser.close();
console.log("PAGE CHECK OK");
