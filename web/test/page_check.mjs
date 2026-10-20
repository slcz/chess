/*
 * page_check.mjs — headless-Chromium acceptance run for the page (run
 * from /tmp/ortvendor where playwright is installed; Node resolves the
 * package from there):
 *
 *   node page_check.mjs
 *
 * Exercises what the IAB webview could not (it cannot cross-origin
 * isolate): real COOP/COEP-isolated Chromium, the full play loop
 * (click e2e4 → net replies through the ORT bridge), the walkthrough
 * (a real g14 shard opened through the file input), and takes
 * screenshots to web/test/shots/.
 */
import { mkdirSync } from "node:fs";
import { createRequire } from "node:module";
// playwright lives in a scratch install outside the repo (the page itself
// has no dependencies) — bind it explicitly by location.
const { chromium } = createRequire("/tmp/ortvendor/package.json")("playwright");

const BASE = "http://127.0.0.1:8468";
const ROOT = "/Users/sileizhang/dev/chess";
const SHOTS = `${ROOT}/web/test/shots`;
mkdirSync(SHOTS, { recursive: true });

// Feed a local file through an <input type=file> without a native picker.
async function setInputFile(page, selector, path) {
  await page.evaluate(async ({ selector, path }) => {
    const bytes = new Uint8Array(await (await fetch(path)).arrayBuffer());
    const name = path.split("/").pop();
    const dt = new DataTransfer();
    dt.items.add(new File([bytes], name));
    const input = document.querySelector(selector);
    input.files = dt.files;
    input.dispatchEvent(new Event("change", { bubbles: true }));
  }, { selector, path: path.replace(ROOT, "") });
}

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

// load the real net through the panel's file input
await setInputFile(page, "#netFile", `${ROOT}/web/models/gen14.onnx`);
await page.waitForFunction(
  () => !document.getElementById("netName").textContent.includes("loading"),
  { timeout: 60000 });
console.log("net:", await page.locator("#netName").textContent());

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

// walkthrough: open the real g14 shard
await page.locator("#tabWalk").click();
await setInputFile(page, "#shardFile", `${ROOT}/data/hf/g14-shard-0.spbin`);
await page.waitForFunction(
  () => +document.querySelectorAll("#wtGameSel option").length > 100,
  { timeout: 30000 });
console.log("shard games listed:", await page.locator("#wtGameSel option").count());
// pick a decent-length game, step to ply 12
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
await page.screenshot({ path: `${SHOTS}/walkthrough.png` });

await browser.close();
console.log("PAGE CHECK OK");
