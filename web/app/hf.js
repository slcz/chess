/*
 * hf.js — Hugging Face Hub access for the page and the engine worker.
 *
 * The GUI's only artifact source: generation checkpoints (.onnx — converted
 * offline by scripts/export_onnx.py and pushed to a models repo) and
 * self-play shards (.spbin in a datasets repo). No local files, no server:
 * both huggingface.co and its CDN send permissive CORS headers, so plain
 * fetch from any origin works.
 *
 *   listFiles("model", "user/repo", "")            → [{path, size, type}]
 *   resolveUrl("dataset", "user/repo", "a/b.spbin") → the download URL
 *   fetchWithProgress(url, cb)                     → ArrayBuffer
 *
 * listing is recursive=true and filtered by the caller; fetchWithProgress
 * streams so a 30 MB net shows a real progress bar (content-length comes
 * from the resolve redirect's final response).
 */
"use strict";

const HF = "https://huggingface.co";

/** Download URL for one file (follows the CDN redirect transparently). */
export function resolveUrl(repoType, repo, path) {
  const kind = repoType === "dataset" ? "datasets/" : "";
  return `${HF}/${kind}${repo}/resolve/main/${path}`;
}

/** Accept a bare repo id ("owner/name"), an id with an embedded path
 * ("owner/name/loop1"), or a pasted huggingface.co page/download URL
 * ("https://huggingface.co/owner/name/tree/main/loop1/", ".../resolve/
 * main/loop1/gen1.pt"). Returns {repo, path} — path is "" when the input
 * carries none (callers then fall back to their own prefix field). */
export function parseRepo(raw) {
  let s = (raw || "").trim();
  s = s.replace(/^https?:\/\//, "")
    .replace(/^(www\.)?(huggingface\.co|hf\.co)\//, "")
    .replace(/^datasets\//, "")
    .replace(/\/+$/, "");
  const m = s.match(/\/(tree|resolve|blob)\/([^/]+)\/?(.*)$/);
  if (m) return { repo: s.slice(0, m.index), path: m[3] };
  const seg = s.split("/").filter(Boolean);
  return { repo: seg.slice(0, 2).join("/"), path: seg.slice(2).join("/") };
}

/** Recursive tree listing under `prefix` ("" = whole repo). Raises with
 * the Hub's error message on 4xx (typo'd repo id, private repo). No
 * trailing slash before the query — the slashed form 302-redirects
 * without CORS headers, which browsers refuse. */
export async function listFiles(repoType, repo, prefix) {
  const kind = repoType === "model" ? "models" : "datasets";
  const base = prefix ? "/" + prefix.replace(/\/+$/, "") : "";
  const url = `${HF}/api/${kind}/${repo}/tree/main${base}?recursive=true`;
  const r = await fetch(url);
  if (!r.ok) {
    let detail = "";
    try { detail = (await r.json()).error || ""; } catch { /* html body */ }
    throw new Error(`HF list failed (${r.status}): ${detail || url}`);
  }
  return r.json();                      // [{type, path, size, ...}]
}

/** GET a URL as bytes, reporting (loaded, total) as chunks arrive. */
export async function fetchWithProgress(url, onProgress) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`fetch failed (${r.status}): ${url}`);
  const total = +r.headers.get("content-length") || 0;
  const chunks = [];
  let loaded = 0;
  if (r.body && onProgress) {
    const reader = r.body.getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      loaded += value.length;
      onProgress(loaded, total);
    }
  } else {
    chunks.push(new Uint8Array(await r.arrayBuffer()));
    loaded = chunks[0].length;
  }
  const out = new Uint8Array(loaded);
  for (let i = 0, o = 0; i < chunks.length; i++) {
    out.set(chunks[i], o);
    o += chunks[i].length;
  }
  return out.buffer;
}
