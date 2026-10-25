#!/usr/bin/env node
/*
 * eval_hang.mjs — driver for the minimal evaluator-hang probe (see
 * eval_hang.html). Runs the synthetic-rows loop in headless Chromium and
 * reports whether any session.run wedges. No engine, no MCTS.
 *
 *   node web/test/eval_hang.mjs [timeoutSeconds]
 */
import { createRequire } from "node:module";
const { chromium } = createRequire("/tmp/ortvendor/package.json")("playwright");

const browser = await chromium.launch();
const page = await browser.newPage();
page.on("pageerror", (e) => console.log("PAGE ERROR:", e.message));
page.on("console", (m) => console.log(`[${m.type()}]`, m.text().slice(0, 160)));

await page.goto("http://127.0.0.1:8468/web/test/eval_hang.html");
await page.waitForFunction(() => window.__probe, { timeout: 15000 });

const t0 = Date.now();
const result = await Promise.race([
  page.evaluate(() => window.__probe(null, null, 40))
    .catch((e) => "EVALUATE FAILED: " + e.message.slice(0, 80)),
  new Promise((r) => setTimeout(() => r("TIMEOUT"),
    (+process.argv[2] || 180) * 1000)),
]);
if (result === "TIMEOUT")
  console.log(`RESULT: HANG — no completion in ${((Date.now() - t0) / 1000) | 0}s ` +
    `(main thread unreachable: page context is blocked too)`);
else
  console.log("RESULT:", JSON.stringify(result));
await browser.close();
