/* CamTrol++ project page. Every number below is copied from the paper tables. */

// ---------------------------------------------------------------- data
const ARCS = [10, 12, 15, 20, 30, 60];
const STEP_SERIES = [
  { key: "full", name: "SVD, full CamTrol++", color: "--series-1",
    eae: [2.22, 2.49, 2.86, 3.03, 3.80, 6.56], zmr: [0.065, 0.064, 0.058, 0.068, 0.073, 0.234], time: [332, 277, 222, 166, 111, 56] },
  { key: "mech", name: "SVD, reprojection only", color: "--series-2",
    eae: [2.26, 2.41, 2.87, 2.98, 3.74, 6.81], zmr: [0.062, 0.048, 0.048, 0.054, 0.067, 0.238], time: [293, 244, 195, 146, 97, 49] },
  { key: "ltx", name: "LTX-Video, reprojection only", color: "--series-3",
    eae: [3.18, 3.09, 2.63, 2.44, 2.31, 3.25], zmr: [0.341, 0.292, 0.196, 0.140, 0.080, 0.024], time: [299, 249, 199, 149, 99, 49] },
];
const STEP_METRICS = {
  eae: { label: "EAE (×10⁻³), lower is better", fmt: v => v.toFixed(2),
    note: "Geometric error grows steadily with the arc per pass for SVD, and the single pass is the worst case. LTX-Video covers a 60° orbit in one pass with half the error of SVD, and reaches its lowest error at 20° to 30°, so the best bound depends on the backbone." },
  zmr: { label: "fraction of static pixels (ZMR), lower is better", fmt: v => v.toFixed(3),
    note: "With SVD a single pass leaves about a quarter of the pixels static, against 5 to 7% for multi-pass settings. LTX-Video does the opposite at very short arcs, so it holds more of the image still when passes become too small." },
  time: { label: "seconds per scene on one L40S GPU", fmt: v => v.toFixed(0) + " s",
    note: "Run time grows with the number of passes. 15° balances geometric accuracy and cost." },
};

// Orbit comparison and 3DGS (Table 2)
const ORBIT_COLS = [
  { key: "lpips", label: "LPIPS-next", dir: -1, fmt: 3 },
  { key: "eae", label: "EAE ×10⁻³", dir: -1, fmt: 2 },
  { key: "sampson", label: "Sampson ×10⁻⁴", dir: -1, fmt: 0 },
  { key: "fpa", label: "FPA", dir: 1, fmt: 3 },
  { key: "zmr", label: "ZMR", dir: -1, fmt: 3 },
  { key: "time", label: "Time (s)", dir: -1, fmt: 0 },
  { key: "colmap", label: "COLMAP", dir: 1, fmt: "frac", group: true },
  { key: "psnr", label: "PSNR", dir: 1, fmt: 2, group: true },
  { key: "ssim", label: "SSIM", dir: 1, fmt: 3, group: true },
  { key: "lpips3d", label: "LPIPS", dir: -1, fmt: 3, group: true },
];
const ORBIT_ROWS = [
  { name: "WAVE", passes: 1, lpips: 0.187, eae: 5.41, sampson: 25478, fpa: 0.577, zmr: 0.047, time: 24, colmap: 0, psnr: 18.58, ssim: 0.541, lpips3d: 0.237 },
  { name: "NVS-Solver", passes: 1, lpips: 0.109, eae: 4.27, sampson: 15714, fpa: 0.769, zmr: 0.199, time: 595, colmap: 14, psnr: 17.85, ssim: 0.551, lpips3d: 0.223 },
  { name: "CamTrol", passes: 1, lpips: 0.140, eae: 5.80, sampson: 27518, fpa: 0.714, zmr: 0.243, time: 218, colmap: 7, psnr: 16.31, ssim: 0.413, lpips3d: 0.343 },
  { name: "CamTrol, 15° passes", passes: 4, lpips: 0.134, eae: 2.94, sampson: 5374, fpa: 0.833, zmr: 0.049, time: 760, colmap: 20, psnr: 18.41, ssim: 0.600, lpips3d: 0.211 },
  { name: "CamTrol++, single pass", passes: 1, lpips: 0.137, eae: 6.56, sampson: 44472, fpa: 0.715, zmr: 0.234, time: 56, colmap: 8, psnr: 16.02, ssim: 0.393, lpips3d: 0.355 },
  { name: "CamTrol++", passes: 4, lpips: 0.124, eae: 2.86, sampson: 4869, fpa: 0.829, zmr: 0.058, time: 222, colmap: 20, psnr: 19.25, ssim: 0.626, lpips3d: 0.185, ours: true },
];
const SPLIT = [
  { name: "CamTrol++ (SVD)", single: 16.02, split: 19.25, cs: 8, cm: 20 },
  { name: "CamTrol", single: 16.31, split: 18.41, cs: 7, cm: 20 },
  { name: "CamTrol++ with LTX-Video", single: 19.23, split: 20.93, cs: 17, cm: 19 },
];
const RE10K = {
  head: ["Method", "Passes", "PSNR ↑", "SSIM ↑", "LPIPS ↓", "Time (s) ↓"],
  rows: [
    ["Input frame repeated", "n/a", "13.60", "0.352", "0.357", "n/a"],
    ["WAVE", "1", "15.76", "0.473", "0.254", "11"],
    ["CamTrol", "1", "17.63", "0.576", "0.179", "141"],
    ["NVS-Solver", "1", "18.16", "0.624", "0.164", "595"],
    ["CamTrol++, fixed 6 passes", "6", "16.36", "0.494", "0.258", "336"],
    ["CamTrol++", "1 to 6", "17.64", "0.568", "0.179", "74"],
  ],
  best: { 2: 3, 3: 3, 4: 3, 5: 1 },
  ours: 5,
};
const ABL = {
  head: ["Configuration", "LPIPS-next", "EAE", "Sampson", "Chroma drift"],
  rows: [
    ["Small-step reprojection", "0.125", "2.87", "5107", "2.63"],
    ["+ Epipolar attention", "0.122", "2.81", "4678", "2.49"],
    ["+ LAB at boundaries only", "0.122", "2.80", "4537", "1.08"],
    ["+ LAB at boundaries<br>and inpainted frames (CamTrol++)", "0.124", "2.86", "4869", "1.04"],
  ],
  ours: 3,
};
const DEPTH = [
  ["No corruption", 2.86, 6.56], ["Collapse halfway to a plane", 2.68, 5.68], ["Collapse to a quarter", 2.80, 5.50],
  ["Fully flat", 2.69, 7.91], ["Scale ×0.5 (too near)", 4.18, 12.39], ["Scale ×2 (too far)", 2.24, 3.83],
  ["Edge blur 8 px", 2.73, 6.15], ["Edge blur 24 px", 2.68, 6.06],
];
const PIPELINE = [
  "<b>Depth.</b> The first frame of the pass is lifted to a 3D point cloud with monocular depth from ZoeDepth. The point cloud is rebuilt at every pass boundary, so errors in depth do not accumulate along the path.",
  "<b>Registration-free warp.</b> Points are splatted into every pose of the pass with a 3×3 kernel, and small gaps are filled from the nearest valid pixel. Removing the per-frame point cloud registration of CamTrol makes warping 6.1× faster.",
  "<b>Inpaint.</b> Regions that were not visible in the source view are filled by Stable Diffusion inpainting and color matched to the input image.",
  "<b>Refine.</b> The warped clip is noised to an intermediate level and denoised by a frozen Stable Video Diffusion model. Epipolar attention in the decoder ties each pixel to the epipolar line of the commanded camera.",
  "<b>Anchor color.</b> The chroma of the last frame is matched to the input image in CIELAB. Every pass therefore restarts from the source colors and drift does not build up across passes.",
  "<b>Next pass.</b> The anchored frame becomes the input of the next pass. Consecutive passes share their boundary frame, so K passes of 14 frames give 13K + 1 frames.",
];
const VIDEO_SCENES = ["Outdoor trajectory", "Indoor trajectory", "Complex camera motion", "Scene stylisation"];
const VIDEO_METHODS = [["wave", "WAVE"], ["camtrol", "CamTrol"], ["ours", "CamTrol++"]];

// ---------------------------------------------------------------- helpers
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const svgEl = (tag, attrs = {}) => {
  const e = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  return e;
};
const tooltip = $("#tooltip");
function showTip(html, x, y) {
  tooltip.innerHTML = html;
  tooltip.style.opacity = 1;
  const w = tooltip.offsetWidth;
  tooltip.style.left = Math.min(window.innerWidth - w - 8, Math.max(8, x + 12)) + "px";
  tooltip.style.top = (y - 40) + "px";
}
const hideTip = () => (tooltip.style.opacity = 0);

// ---------------------------------------------------------------- theme
(function theme() {
  let saved = null;
  try { saved = localStorage.getItem("theme"); } catch (e) {}
  if (saved) document.documentElement.dataset.theme = saved;
  $("#themeBtn").addEventListener("click", () => {
    const dark = document.documentElement.dataset.theme
      ? document.documentElement.dataset.theme === "dark"
      : matchMedia("(prefers-color-scheme: dark)").matches;
    document.documentElement.dataset.theme = dark ? "light" : "dark";
    try { localStorage.setItem("theme", document.documentElement.dataset.theme); } catch (e) {}
    drawStepChart(); drawDepthChart(); drawSplit(); window.redrawCalc();
  });
})();

// ---------------------------------------------------------------- sprite scrubbers
function setSprite(el, scene) {
  const n = +(el.dataset.n || 14);
  el.style.backgroundImage = `url(assets/sprites/${scene}_${el.dataset.seq}.jpg)`;
  el.style.backgroundSize = `${n * 100}% 100%`;
}
function showFrame(scope, i) {
  $$(".sprite", scope).forEach(el => {
    const n = +(el.dataset.n || 14);
    el.style.backgroundPosition = `${(i / (n - 1)) * 100}% 0`;
  });
}

const players = [];
$$(".scrub").forEach(scrub => {
  const scope = scrub.closest(".card");
  const range = $("input", scrub), btn = $(".play", scrub), out = $(".angle", scrub);
  const n = +(scrub.dataset.n || 14);
  const labels = scrub.dataset.labels ? scrub.dataset.labels.split(",") : null;
  const player = { scope, timer: null, dir: 1, auto: scrub.dataset.group === "hero" || scrub.dataset.group === "orbit" };
  const set = i => {
    range.value = i;
    out.textContent = labels ? labels[i] : `${(60 * i / (n - 1)).toFixed(1)}°`;
    showFrame(scope, i);
  };
  player.play = () => {
    if (player.timer) return;
    btn.textContent = "❚❚";
    player.timer = setInterval(() => {
      let i = +range.value + player.dir;
      if (i >= n - 1 || i <= 0) player.dir *= -1;
      if (i <= 0 && player.onLoop) player.onLoop();
      set(Math.max(0, Math.min(n - 1, i)));
    }, labels ? 900 : 180);
  };
  player.stop = () => { clearInterval(player.timer); player.timer = null; btn.textContent = "▶"; };
  btn.addEventListener("click", () => (player.timer ? player.stop() : player.play()));
  range.addEventListener("input", () => { player.stop(); set(+range.value); });
  set(0);
  players.push(player);
});

// Play automatically only while visible.
const io = new IntersectionObserver(entries => entries.forEach(e => {
  const p = players.find(p => p.scope === e.target);
  if (!p || !p.auto) return;
  e.isIntersecting ? p.play() : p.stop();
}), { threshold: 0.35 });
players.forEach(p => io.observe(p.scope));

$$(".sprite[data-scene]").forEach(el => setSprite(el, el.dataset.scene));
function bindTabs(tabsSel, gridSel) {
  const tabs = $(tabsSel), grid = $(gridSel);
  const apply = scene => $$(".sprite", grid).forEach(el => setSprite(el, scene));
  apply($("button.active", tabs).dataset.scene);
  tabs.addEventListener("click", e => {
    const b = e.target.closest("button");
    if (!b) return;
    $$("button", tabs).forEach(x => x.classList.toggle("active", x === b));
    apply(b.dataset.scene);
  });
}
bindTabs("#orbitTabs", "#orbitGrid");
bindTabs("#stepTabs", "#stepGrid");
bindTabs("#gsTabs", "#gsView");

// ---------------------------------------------------------------- pipeline steps
(function pipeline() {
  const text = $("#stepText");
  const show = i => {
    $$(".step").forEach(s => s.classList.toggle("active", +s.dataset.i === i));
    text.innerHTML = PIPELINE[i];
  };
  $("#pipeline").addEventListener("click", e => { const s = e.target.closest(".step"); if (s) show(+s.dataset.i); });
  show(0);
})();

// ---------------------------------------------------------------- pass calculator
(function calculator() {
  const theta = $("#theta"), limit = $("#limit"), svg = $("#arcSvg");
  const draw = () => {
    const T = +theta.value, L = +limit.value;
    const K = Math.max(1, Math.ceil(T / L - 1e-9));
    $("#thetaOut").textContent = T + "°";
    $("#limitOut").textContent = L + "°";
    $("#kOut").textContent = K;
    $("#fOut").textContent = 13 * K + 1;
    $("#arcOut").textContent = (T / K).toFixed(1) + "°";

    svg.innerHTML = "";
    const cx = 160, cy = 162, r = 138, a0 = -90 - T / 2;
    const pt = (deg, rad) => [cx + rad * Math.cos(deg * Math.PI / 180), cy + rad * Math.sin(deg * Math.PI / 180)];
    const arc = (s, e, rad) => { const [x1, y1] = pt(s, rad), [x2, y2] = pt(e, rad); return `M${x1},${y1} A${rad},${rad} 0 ${e - s > 180 ? 1 : 0} 1 ${x2},${y2}`; };
    svg.append(svgEl("path", { d: arc(-150, -30, r), fill: "none", stroke: css("--line"), "stroke-width": 6, "stroke-linecap": "round" }));
    for (let k = 0; k < K; k++) {
      const s = a0 + k * T / K, e = s + T / K;
      svg.append(svgEl("path", { d: arc(s, e, r), fill: "none", "stroke-width": 6, "stroke-linecap": "round",
        stroke: k % 2 ? css("--series-1") : css("--series-2") }));
    }
    for (let k = 0; k <= K; k++) {
      const [x, y] = pt(a0 + k * T / K, r);
      svg.append(svgEl("circle", { cx: x, cy: y, r: K > 16 ? 2.5 : 4.5, fill: css("--card"), stroke: css("--ink"), "stroke-width": 2 }));
    }
    const label = (txt, x, y, size, weight, fill) => {
      const t = svgEl("text", { x, y, "text-anchor": "middle", "font-size": size, "font-weight": weight, fill });
      t.textContent = txt; svg.append(t);
    };
    const [ix, iy] = pt(a0, r + 18);
    label("input", ix, iy, 11, 500, css("--ink-3"));
    svg.append(svgEl("rect", { x: cx - 22, y: cy - 30, width: 44, height: 30, rx: 6, fill: css("--bg-alt"), stroke: css("--line") }));
    label("scene", cx, cy - 11, 11, 500, css("--ink-3"));
    label(`${K} pass${K > 1 ? "es" : ""} of ${(T / K).toFixed(1)}°`, cx, cy + 24, 15, 700, css("--ink"));
  };
  theta.addEventListener("input", draw);
  limit.addEventListener("input", draw);
  draw();
  window.redrawCalc = draw;
})();

// ---------------------------------------------------------------- step-size line chart
let stepMetric = "eae";
function drawStepChart() {
  const box = $("#stepChart"), m = STEP_METRICS[stepMetric];
  const W = 760, H = 320, pad = { l: 56, r: 24, t: 34, b: 44 };
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": `Step size against ${m.label}` });
  const all = STEP_SERIES.flatMap(s => s[stepMetric]);
  const raw = Math.max(...all) / 4, mag = 10 ** Math.floor(Math.log10(raw));
  const tick = [1, 2, 2.5, 5, 10].map(f => f * mag).find(t => t >= raw);
  const ymax = tick * 4;
  const x = i => pad.l + i * (W - pad.l - pad.r) / (ARCS.length - 1);
  const y = v => H - pad.b - v / ymax * (H - pad.t - pad.b);

  for (let g = 0; g <= 4; g++) {
    const v = ymax * g / 4;
    svg.append(svgEl("line", { x1: pad.l, x2: W - pad.r, y1: y(v), y2: y(v), class: "grid-line" }));
    const t = svgEl("text", { x: pad.l - 8, y: y(v) + 4, "text-anchor": "end" });
    t.textContent = +v.toFixed(3);
    svg.append(t);
  }
  ARCS.forEach((a, i) => {
    const t = svgEl("text", { x: x(i), y: H - pad.b + 20, "text-anchor": "middle" });
    t.textContent = a === 60 ? "60° (1 pass)" : a + "°";
    svg.append(t);
  });
  const xt = svgEl("text", { x: (W + pad.l) / 2, y: H - 4, "text-anchor": "middle" });
  xt.textContent = "arc per pass";
  svg.append(xt);
  const yt = svgEl("text", { x: pad.l - 8, y: 14 });
  yt.textContent = m.label;
  svg.append(yt);

  const cross = svgEl("line", { y1: pad.t, y2: H - pad.b, stroke: css("--ink-3"), "stroke-dasharray": "3 3", opacity: 0 });
  svg.append(cross);
  STEP_SERIES.forEach(s => {
    const col = css(s.color);
    svg.append(svgEl("path", { d: s[stepMetric].map((v, i) => `${i ? "L" : "M"}${x(i)},${y(v)}`).join(""), fill: "none", stroke: col, "stroke-width": 2 }));
    s[stepMetric].forEach((v, i) => svg.append(svgEl("circle", { cx: x(i), cy: y(v), r: 4.5, fill: col, stroke: css("--card"), "stroke-width": 2 })));
  });
  // invisible hit columns, one per arc
  ARCS.forEach((a, i) => {
    const w = (W - pad.l - pad.r) / (ARCS.length - 1);
    const hit = svgEl("rect", { x: x(i) - w / 2, y: pad.t, width: w, height: H - pad.t - pad.b, fill: "transparent" });
    hit.addEventListener("mousemove", ev => {
      cross.setAttribute("x1", x(i)); cross.setAttribute("x2", x(i)); cross.setAttribute("opacity", 1);
      showTip(`<b>${a}° per pass</b><br>` + STEP_SERIES.map(s => `${s.name}: ${m.fmt(s[stepMetric][i])}`).join("<br>"), ev.clientX, ev.clientY);
    });
    hit.addEventListener("mouseleave", () => { cross.setAttribute("opacity", 0); hideTip(); });
    svg.append(hit);
  });
  box.replaceChildren(svg);
  $("#stepNote").textContent = m.note;
  $("#stepLegend").innerHTML = STEP_SERIES.map(s => `<span><i style="background:${css(s.color)}"></i>${s.name}</span>`).join("");
}
$("#metricSeg").addEventListener("click", e => {
  const b = e.target.closest("button");
  if (!b) return;
  $$("#metricSeg button").forEach(x => x.classList.toggle("active", x === b));
  stepMetric = b.dataset.m;
  drawStepChart();
});
drawStepChart();

// ---------------------------------------------------------------- orbit table and bars
let sortKey = "eae";
function fmtCell(v, c) { return c.fmt === "frac" ? `${v}/20` : v.toFixed(c.fmt); }
function drawOrbit() {
  const best = {};
  ORBIT_COLS.forEach(c => { const vals = ORBIT_ROWS.map(r => r[c.key]); best[c.key] = c.dir < 0 ? Math.min(...vals) : Math.max(...vals); });
  const t = $("#orbitTable");
  t.innerHTML = `<thead><tr class="group"><th></th><th></th><th colspan="6">Generated sequence</th><th colspan="4">3DGS, held-out views</th></tr>
    <tr><th>Method</th><th>Passes</th>${ORBIT_COLS.map(c => `<th data-k="${c.key}" class="${c.key === sortKey ? "sorted" : ""}">${c.label} ${c.dir < 0 ? "↓" : "↑"}</th>`).join("")}</tr></thead>
    <tbody>${ORBIT_ROWS.map(r => `<tr class="${r.ours ? "ours" : ""}"><td>${r.name}</td><td>${r.passes}</td>${ORBIT_COLS.map(c =>
      `<td class="${r[c.key] === best[c.key] ? "best" : ""}">${fmtCell(r[c.key], c)}</td>`).join("")}</tr>`).join("")}</tbody>`;
  $$("th[data-k]", t).forEach(th => th.addEventListener("click", () => { sortKey = th.dataset.k; drawOrbit(); }));

  const c = ORBIT_COLS.find(c => c.key === sortKey);
  const rows = [...ORBIT_ROWS].sort((a, b) => c.dir * (b[c.key] - a[c.key]));
  const max = Math.max(...ORBIT_ROWS.map(r => r[c.key])) || 1;
  $("#orbitBars").innerHTML = `<div class="bar-title">Ranked by ${c.label}, ${c.dir < 0 ? "lower" : "higher"} is better</div>` +
    rows.map(r => `<div class="bar-row ${r.ours ? "ours" : ""}" data-tip="${r.name}: ${fmtCell(r[c.key], c)}"><span>${r.name}</span>
      <div class="track"><div class="fill" style="width:${Math.max(1.5, r[c.key] / max * 100)}%"></div></div><span class="v">${fmtCell(r[c.key], c)}</span></div>`).join("");
}
drawOrbit();

function staticTable(sel, spec) {
  $(sel).innerHTML = `<thead><tr>${spec.head.map(h => `<th>${h}</th>`).join("")}</tr></thead><tbody>${spec.rows.map((r, i) =>
    `<tr class="${i === spec.ours ? "ours" : ""}">${r.map((v, j) => `<td class="${spec.best && spec.best[j] === i ? "best" : ""}">${v}</td>`).join("")}</tr>`).join("")}</tbody>`;
}
staticTable("#re10kTable", RE10K);
staticTable("#ablTable", ABL);

// ---------------------------------------------------------------- splitting gains
function drawSplit() {
  const max = 22;
  $("#splitList").innerHTML = SPLIT.map(s => `<div class="split-item"><b>${s.name}</b>
    <div class="split-row"><span>one pass</span><div class="track"><div class="fill" style="width:${s.single / max * 100}%;background:${css("--series-3")}"></div></div><span>${s.single.toFixed(2)} dB · ${s.cs}/20</span></div>
    <div class="split-row"><span>15° passes</span><div class="track"><div class="fill" style="width:${s.split / max * 100}%;background:${css("--series-1")}"></div></div><span><b>${s.split.toFixed(2)} dB</b> · ${s.cm}/20</span></div></div>`).join("");
}
drawSplit();

// ---------------------------------------------------------------- depth failures (grouped bars)
function drawDepthChart() {
  const W = 480, rowH = 30, pad = { l: 180, r: 40, t: 6 };
  const H = pad.t + DEPTH.length * rowH + 4;
  const max = 13;
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "EAE under depth failures" });
  const xs = v => v / max * (W - pad.l - pad.r);
  const cols = [css("--series-1"), css("--series-3")];
  DEPTH.forEach(([name, a, b], i) => {
    const y0 = pad.t + i * rowH;
    const lab = svgEl("text", { x: pad.l - 10, y: y0 + 17, "text-anchor": "end", class: "lab" });
    lab.textContent = name;
    svg.append(lab);
    [a, b].forEach((v, k) => {
      const r = svgEl("rect", { x: pad.l, y: y0 + 4 + k * 11, width: xs(v), height: 9, rx: 3, fill: cols[k] });
      r.addEventListener("mousemove", ev => showTip(`${name}<br>${k ? "single pass" : "15° passes"}: ${v.toFixed(2)}`, ev.clientX, ev.clientY));
      r.addEventListener("mouseleave", hideTip);
      svg.append(r);
    });
    const t = svgEl("text", { x: pad.l + xs(b) + 6, y: y0 + 22 });
    t.textContent = b.toFixed(2);
    svg.append(t);
  });
  $("#depthChart").replaceChildren(svg);
  $("#depthLegend").innerHTML = `<span><i class="box" style="background:${cols[0]}"></i>15° passes</span><span><i class="box" style="background:${cols[1]}"></i>single 60° pass</span>`;
}
drawDepthChart();

// bar-row hover tooltips
document.addEventListener("mousemove", e => {
  const row = e.target.closest && e.target.closest(".bar-row");
  if (row) showTip(row.dataset.tip, e.clientX, e.clientY);
});
document.addEventListener("mouseout", e => { if (e.target.closest && e.target.closest(".bar-row")) hideTip(); });

// ---------------------------------------------------------------- videos
(function videos() {
  const tabs = $("#videoTabs"), rows = $("#videoRows");
  tabs.innerHTML = VIDEO_SCENES.map((s, i) => `<button class="${i ? "" : "active"}" data-i="${i + 1}">${s}</button>`).join("");
  const vio = new IntersectionObserver(es => es.forEach(e => { e.isIntersecting ? e.target.play().catch(() => {}) : e.target.pause(); }), { threshold: 0.25 });
  const show = scene => {
    rows.innerHTML = [1, 2, 3, 4].map(r => `<div class="video-row">${VIDEO_METHODS.map(([k, label]) =>
      `<figure class="${k === "ours" ? "ours" : ""}"><video muted loop playsinline preload="metadata" src="assets/videos/scene${scene}_row${r}_${k}.mp4"></video><figcaption>${k === "ours" ? `<b>${label}</b>` : label}</figcaption></figure>`).join("")}</div>`).join("");
    $$("video", rows).forEach(v => vio.observe(v));
    // keep the three videos of a row in step
    $$(".video-row", rows).forEach(row => {
      const vids = $$("video", row);
      vids[2].addEventListener("play", () => vids.forEach(v => { v.currentTime = 0; v.play().catch(() => {}); }), { once: true });
    });
  };
  tabs.addEventListener("click", e => {
    const b = e.target.closest("button");
    if (!b) return;
    $$("button", tabs).forEach(x => x.classList.toggle("active", x === b));
    show(b.dataset.i);
  });
  show(1);
})();

// ---------------------------------------------------------------- BibTeX copy
$("#copyBib").addEventListener("click", async () => {
  try { await navigator.clipboard.writeText($("#bibText").textContent); $("#copyBib").textContent = "Copied"; }
  catch (e) { $("#copyBib").textContent = "Select and copy"; }
  setTimeout(() => ($("#copyBib").textContent = "Copy"), 1500);
});

// redraw color-dependent graphics when the OS theme changes
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { drawStepChart(); drawDepthChart(); drawSplit(); window.redrawCalc(); });

// Figures from the paper, numbered as in the current version
const PAPER_FIGS = [
  ["1", "Large camera motion", "fig1_teaser.jpg", "CamTrol++ against CamTrol and WAVE on a ±60° rotation. The baselines lose coherence and flicker, while CamTrol++ splits the motion into small passes and stays stable, with 1.4 to 3.7× lower flicker. Times are for this 56-frame example (matched timings in Table 2)."],
  ["4", "Static frames and ablation", "fig4_static_ablation.jpg", "<b>(a)</b> Frames 30 to 42 of a 56-frame sequence. WAVE (red boxes) stops moving, while CamTrol++ (blue boxes) keeps following the camera path. <b>(b)</b> Visual ablation. LAB anchoring removes the color shift (top row) and epipolar attention keeps structures aligned along epipolar lines (bottom row)."],
  ["5", "Orbit on MegaScenes", "fig5_orbit.jpg", "Three scenes along the same 60° orbit, shown at 23°, 42° and 60°. CamTrol++ turns around each monument and keeps it sharp and in place. CamTrol adds structures that are not in the scene, NVS-Solver barely moves and inserts objects, and WAVE moves the camera but redraws the monuments."],
  ["6", "3D reconstruction and color", "fig6_3dgs_lab.jpg", "<b>(a)</b> Held-out views of 3D Gaussian Splatting fitted to CamTrol++ frames. 15° passes give a clean reconstruction, one 60° pass gives streaks and floaters. <b>(b)</b> Last frame of two RealEstate10K clips without and with LAB matching at pass boundaries. Matching keeps the colors close to the input."],
  ["7", "Pass size on two backbones", "fig7_stepsize.jpg", "The same 60° orbit as four 15° passes or one 60° pass, with SVD and with LTX-Video (reprojection only). With 15° passes both backbones turn around the monument. With one pass SVD freezes and the facade melts, and LTX-Video fills new regions with noisy structures."],
  ["8", "Effect of step size", "fig8_step_size.jpg", "Every arc size follows the same 60° orbit and is scored on the same 14 poses of 20 scenes. Geometric error grows steadily with the arc, and only the single 60° pass freezes large parts of the image (high ZMR). LPIPS-next stays nearly flat. The yellow band marks the 15° default."],
  ["9", "Long horizon and depth", "fig9_longhorizon_depth.jpg", "<b>(a)</b> LPIPS-next, ZMR and edge density over an 84-frame extension on 20 scenes. Consistency and motion stay stable through the standard 56 frames, and detail then slowly fades. Detail loss is the most common failure (13 of 20 scenes). <b>(b)</b> LPIPS-next under synthetic depth corruption on 20 scenes, along the earlier two-direction 56-frame path. It rises by about 16%, from 0.099 to 0.115, at severity 0.5 and stays well below the WAVE reference from the larger set of Table 1."],
  ["10", "Novel views on RealEstate10K", "figS1_re10k.jpg", "Input frame, real frame at a target pose and the view of each method for that pose. CamTrol++ matches the layout, colors and lighting of the real frame. CamTrol gives similar views on these short moves, while NVS-Solver and WAVE darken scenes and add clutter."],
  ["11", "Inpainted area against arc per pass", "figS2_inpainted_area.png", "Fraction of each warped frame handed to the inpainter, for the first pass of every setting of the step-size study, on 20 MegaScenes images with the same image, depth and poses. A 15° pass leaves 5.2% of each warped frame to the inpainter on average and 10.7% at its last frame, against 32.5% and 64.6% for one 60° pass. The area grows faster than the arc, which is why large passes lose geometry and freeze.", "560px"],
];
(function paperFigures() {
  const tabs = $("#figTabs"), img = $("#figImg"), cap = $("#figCap"), link = $("#figLink");
  tabs.innerHTML = PAPER_FIGS.map((f, i) => `<button class="${i ? "" : "active"}" data-i="${i}">Fig. ${f[0]}</button>`).join("");
  const show = i => {
    const [n, title, file, text, maxWidth] = PAPER_FIGS[i];
    img.style.maxWidth = maxWidth || "";
    img.style.margin = maxWidth ? "0 auto" : "";
    img.src = link.href = `assets/figures/${file}`;
    img.alt = `Figure ${n}, ${title}`;
    cap.innerHTML = `<b>Figure ${n}. ${title}.</b> ${text}`;
    $$("button", tabs).forEach(b => b.classList.toggle("active", +b.dataset.i === i));
  };
  tabs.addEventListener("click", e => { const b = e.target.closest("button"); if (b) show(+b.dataset.i); });
  show(0);
})();

// Hero scenes. The viewer moves to the next scene after each full sweep of the
// orbit, until the visitor picks a scene themselves.
(function heroScenes() {
  const tabs = $("#heroTabs"), grid = $("#heroGrid");
  const player = players.find(p => p.scope.classList.contains("hero-viewer"));
  const buttons = $$("button", tabs);
  const preload = scene => ["ours15", "ours60"].forEach(seq => { new Image().src = `assets/sprites/${scene}_${seq}.jpg`; });
  const show = b => {
    buttons.forEach(x => x.classList.toggle("active", x === b));
    $$(".sprite", grid).forEach(el => setSprite(el, b.dataset.scene));
    const next = buttons[(buttons.indexOf(b) + 1) % buttons.length];
    preload(next.dataset.scene);
  };
  player.onLoop = () => show(buttons[(buttons.indexOf($("button.active", tabs)) + 1) % buttons.length]);
  tabs.addEventListener("click", e => {
    const b = e.target.closest("button");
    if (!b) return;
    player.onLoop = null;
    show(b);
  });
  preload(buttons[1].dataset.scene);
})();
