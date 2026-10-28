/*
 * board.js — board rendering + hit detection, adapted from the retired
 * chess_ai/static/board.html (CSS board, Unicode pieces, dot/ring move
 * markers, promotion via callback). Adds the pieces the old page never
 * had: an SVG arrow overlay for policy/visit distributions and an eval
 * bar. Rendering is pure: state in, DOM out — all chess truth arrives
 * from the engine worker.
 *
 * Squares are "e2"-style names; index 0 = a1 … 63 = h8.
 */
const TEXT = "\uFE0E";   // force text presentation, not emoji, on iOS
const GLYPH = {
  P: "\u265F" + TEXT, N: "\u265E" + TEXT, B: "\u265D" + TEXT,
  R: "\u265C" + TEXT, Q: "\u265B" + TEXT, K: "\u265A" + TEXT,
};
const FILES = "abcdefgh";
const CELL = 60;

/** Unicode glyph for a piece. Both colors use the solid glyph; CSS fills
 * it white for White. The text-presentation selector keeps iOS from
 * treating it as an emoji (which ignores the CSS color). */
export function pieceGlyph(ch) {
  return GLYPH[ch.toUpperCase()];
}

export class Board {
  constructor(root) {
    root.innerHTML = "";
    root.style.position = "relative";
    const grid = document.createElement("div");
    grid.id = "grid";
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.id = "arrows";
    root.append(svg, grid);
    this.cells = {};
    this.grid = grid;
    this.svg = svg;
    this.clickCb = null;
    for (let row = 0; row < 8; row++) {
      for (let col = 0; col < 8; col++) {
        const rank = 7 - row, file = col;
        const name = FILES[file] + (rank + 1);
        const d = document.createElement("div");
        d.className = "cell " + (((rank + file) % 2 === 0) ? "dark" : "light");
        d.dataset.sq = name;
        d.addEventListener("click", () => this.clickCb && this.clickCb(name));
        grid.appendChild(d);
        this.cells[name] = d;
      }
    }
  }

  /** state: {fen, lastMove?, check?, sel?, legal?[]} — the parts of the
   * engine state the board can show on its own. */
  render(state) {
    const placement = state.fen.split(" ")[0];
    const targets = state.lastMove
      ? [state.lastMove.slice(0, 2), state.lastMove.slice(2, 4)] : [];
    for (const name in this.cells) {
      const d = this.cells[name];
      d.classList.remove("hl", "sel", "check", "wp", "bp", "dotsrc");
      d.querySelectorAll(".dot,.ring,.piece").forEach((e) => e.remove());
    }
    let row = 0, file = 0;
    for (const ch of placement) {
      if (ch === "/") { row++; file = 0; continue; }
      if (ch >= "1" && ch <= "8") { file += +ch; continue; }
      const name = FILES[file] + (8 - row);
      const white = ch === ch.toUpperCase();
      const span = document.createElement("span");
      span.className = "piece";
      span.textContent = pieceGlyph(ch);
      this.cells[name].classList.add(white ? "wp" : "bp");
      this.cells[name].appendChild(span);
      file++;
    }
    if (targets.length)
      for (const t of targets) this.cells[t].classList.add("hl");
    if (state.check) {
      const side = state.side === "w" ? "K" : "k";
      const flat = placement.replace(/[0-8]/g, (m) => ".".repeat(+m))
        .replace(/\//g, "");
      const k = flat.indexOf(side);
      if (k >= 0)
        this.cells[FILES[k % 8] + (8 - Math.floor(k / 8))].classList.add("check");
    }
    if (state.sel) {
      this.cells[state.sel].classList.add("sel");
      for (const m of state.legal || []) {
        if (m.slice(0, 2) !== state.sel) continue;
        const to = m.slice(2, 4);
        const marker = document.createElement("div");
        marker.className =
          this.cells[to].querySelector(".piece") ? "ring" : "dot";
        this.cells[to].appendChild(marker);
      }
    }
    this.arrows([]);
  }

  /** arrows: [{from, to, weight (0..1), kind: "pi"|"best"|"played"}] —
   * drawn on the SVG overlay, thickest first so thin ones stay visible. */
  arrows(list) {
    this.svg.innerHTML = "";
    const n = 8 * CELL;
    this.svg.setAttribute("viewBox", `0 0 ${n} ${n}`);
    list.sort((a, b) => b.weight - a.weight);
    for (const a of list.slice(0, 10)) {
      const [f, t] = [sqCenter(a.from), sqCenter(a.to)];
      const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
      const color = a.kind === "pi" ? "#3d6fb5"
        : a.kind === "best" ? "#1e9e50" : "#b53d3d";
      const w = 3 + 9 * a.weight, o = 0.3 + 0.6 * a.weight;
      g.innerHTML =
        `<line x1="${f.x}" y1="${f.y}" x2="${t.x}" y2="${t.y}" ` +
        `stroke="${color}" stroke-width="${w}" stroke-opacity="${o}" ` +
        `stroke-linecap="round" marker-end="url(#ah)" />`;
      this.svg.appendChild(g);
    }
    if (list.length) {
      const defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
      defs.innerHTML =
        `<marker id="ah" viewBox="0 0 10 10" refX="6" refY="5" ` +
        `markerWidth="4.5" markerHeight="4.5" orient="auto-start-reverse">` +
        `<path d="M 0 0 L 10 5 L 0 10 z" fill="#fff" fill-opacity="0.85"/></marker>`;
      this.svg.prepend(defs);
    }
  }
}

/** Eval bar: value v is WHITE's view in [-1, 1]; the white share is the
 * --w custom property (the CSS gradient) and data-v shows the number. */
export function renderEvalBar(el, v) {
  if (v === null || v === undefined) {
    el.style.removeProperty("--w");
    el.dataset.v = "";
    return;
  }
  const pct = Math.round(100 * Math.min(1, Math.max(0, (v + 1) / 2)));
  el.style.setProperty("--w", pct + "%");
  el.dataset.v = v.toFixed(2);
}

const sqCenter = (name) => ({
  x: (FILES.indexOf(name[0]) + 0.5) * CELL,
  y: (7 - (+name[1] - 1) + 0.5) * CELL,
});
