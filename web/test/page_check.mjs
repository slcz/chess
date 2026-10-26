/*
 * page_check.mjs — headless-Chromium acceptance run for the page (run
 * from /tmp/ortvendor where playwright is installed; Node resolves the
 * package from there):
 *
 *   node page_check.mjs
 *
 * Exercises what the IAB webview could not (it cannot cross-origin
 * isolate): real COOP/COEP-isolated Chromium and the full play loop
 * (click e2e4 → net replies through the ORT bridge; the net itself
 * loads from Hugging Face through the panel's own loader).
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

// load the real net through the panel's HF flow (auto-listed on boot
// from the pre-filled repo → pick → load). No .onnx in the repo → skip.

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

await browser.close();
console.log("PAGE CHECK OK");
