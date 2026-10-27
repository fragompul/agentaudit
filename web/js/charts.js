// Chart renderers. Rows and bars are HTML (they reflow on phones); only the line chart is SVG.
const pct = (v, d = 0) => `${(v * 100).toFixed(d)}%`;
const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

function intervalRows(host, rows, { lo, hi, ticks, fmt }) {
  const pos = (v) => `${(((v - lo) / (hi - lo)) * 100).toFixed(2)}%`;
  const grid = ticks.map((t) => `<i class="tick" style="left:${pos(t)}"></i>`).join("");
  host.innerHTML =
    rows
      .map(
        (r) => `
    <div class="ir">
      <div class="ir-l" title="${esc(r.label)}">${esc(r.label)}</div>
      <div class="ir-t">${grid}<i class="ir-bar" style="left:${pos(r.lo)};width:${(((r.hi - r.lo) / (hi - lo)) * 100).toFixed(2)}%;background:${r.color}"></i><b class="ir-dot" style="left:${pos(r.point)};background:${r.color}"></b></div>
      <div class="ir-r">${r.right}</div>
    </div>`,
      )
      .join("") +
    `<div class="ir ir-axis"><div></div><div class="ir-t">${ticks.map((t) => `<span style="left:${pos(t)}">${fmt(t)}</span>`).join("")}</div><div></div></div>`;
}

function sweRank(el, d) {
  const rows = d.swebench.top.map((t, i) => ({
    label: t.name,
    lo: t.rank_lo,
    hi: t.rank_hi,
    point: i + 1,
    color: `var(--s${(i % 6) + 1})`,
    right: `<span class="mono">${pct(t.score, 1)}</span>`,
  }));
  const top = Math.ceil(Math.max(...rows.map((r) => r.hi)) / 5) * 5;
  intervalRows(el.querySelector(".plot"), rows, {
    lo: 0.5,
    hi: top + 0.5,
    ticks: [1, ...Array.from({ length: top / 5 }, (_, i) => (i + 1) * 5)],
    fmt: (t) => `#${t}`,
  });
  const t = d.swebench.top;
  el.insertAdjacentHTML(
    "beforeend",
    `<p class="note">The first three submissions sit within ${((t[0].score - t[2].score) * 100).toFixed(1)} points of each other; the chance of being truly best is about ${pct(t[0].p_best)}, ${pct(t[1].p_best)} and ${pct(t[2].p_best)}.</p>`,
  );
}

function sweGap(el, d) {
  const lab = (g) => (g.gap_hi >= 1 ? `over ${Math.round(g.gap_lo * 100)} pts` : `${Math.round(g.gap_lo * 100)} to ${Math.round(g.gap_hi * 100)} pts`);
  el.querySelector(".plot").innerHTML =
    d.swebench.by_gap
      .map(
        (g) => `
    <div class="br">
      <div class="br-l">${lab(g)}<small>${g.pairs.toLocaleString("en")} pairs</small></div>
      <div class="br-t">
        <div class="bar raw" style="width:${g.frac_distinguishable_raw * 88}%"><span>${pct(g.frac_distinguishable_raw)}</span></div>
        <div class="bar holm" style="width:${Math.max(g.frac_distinguishable_holm * 88, 0.4)}%"><span>${pct(g.frac_distinguishable_holm)}</span></div>
      </div>
    </div>`,
      )
      .join("") +
    `<div class="legend"><span><i style="background:var(--ink-2);opacity:.45"></i>raw p &lt; 0.05</span><span><i style="background:var(--accent)"></i>after Holm adjustment</span></div>`;
}

function peeking(el, d) {
  const r = d.peeking.find((x) => x.true_diff === 0);
  const rows = [
    ["Peek at every step", r.peek, "var(--accent)"],
    ["Fixed sample size", r.fixed, "var(--ink-2)"],
    ["Anytime-valid sequential", r.seq, "var(--blue)"],
  ];
  const scale = 80 / 0.4; // 0.4 of false positives fills 80% of the track
  const nom = `<i class="nom" style="left:${0.05 * scale}%"></i>`;
  el.querySelector(".plot").innerHTML = rows
    .map(
      ([l, v, c]) =>
        `<div class="br"><div class="br-l">${l}</div><div class="br-t peek">${nom}<div class="bar" style="width:${Math.max(v * scale, 0.5)}%;background:${c}"><span style="left:calc(${Math.max(1, (0.05 * scale) / Math.max(v * scale, 0.5)) * 100}% + 8px)">${pct(v, 1)}</span></div></div></div>`,
    )
    .join("");
  el.insertAdjacentHTML("beforeend", `<p class="note">The dashed line is the 5% error rate everyone assumes they have.</p>`);
}

function variance(el, d) {
  const s = d.local.shares;
  const parts = [
    ["Agent skill", s.model, "var(--s2)"],
    ["Task difficulty", s.task, "var(--s5)"],
    ["Agent × task", s.model_task, "var(--s4)"],
    ["Run noise", s.run, "var(--accent)"],
  ];
  el.querySelector(".plot").innerHTML = `<div class="stack" role="img" aria-label="${parts.map(([l, v]) => `${l} ${pct(v)}`).join(", ")}">${parts
    .map(([, v, c]) => `<div style="flex:${v};background:${c}"></div>`)
    .join("")}</div>
    <div class="legend">${parts.map(([l, v, c]) => `<span><i style="background:${c}"></i>${l} <b>${pct(v)}</b></span>`).join("")}</div>`;
  el.insertAdjacentHTML("beforeend", `<p class="note">Only about a quarter is the thing a leaderboard claims to measure.</p>`);
}

function local(el, d) {
  const rows = d.local.models.map((m, i) => ({
    label: m.model,
    lo: m.lo,
    hi: m.hi,
    point: m.score,
    color: `var(--s${(i % 6) + 1})`,
    right: `<span class="mono">${pct(m.score, 1)}</span>`,
  }));
  intervalRows(el.querySelector(".plot"), rows, { lo: 0, hi: 0.8, ticks: [0, 0.2, 0.4, 0.6, 0.8], fmt: (t) => pct(t) });
}

function coverage(el, d) {
  const all = d.coverage.rows;
  const tasks = [...new Set(all.map((r) => r.tasks))];
  const chips = el.querySelector(".chips");
  const plot = el.querySelector(".plot");
  const W = 520, H = 270, L = 44, R = 84, T = 18, B = 40, PAD = 28;
  const Y = (v) => T + (1 - (v - 0.5) / 0.5) * (H - T - B);
  const series = [
    ["pooled", "All runs pooled (Wilson)", "var(--accent)"],
    ["two_stage", "Two-stage bootstrap", "var(--blue)"],
  ];
  function draw(n) {
    const rows = all.filter((r) => r.tasks === n).sort((a, b) => a.runs - b.runs);
    const X = (i) => L + PAD + (i / (rows.length - 1)) * (W - L - R - 2 * PAD);
    let g = [0.5, 0.6, 0.7, 0.8, 0.9, 1]
      .map((t) => `<line class="grid" x1="${L}" x2="${W - R}" y1="${Y(t)}" y2="${Y(t)}"/><text x="${L - 8}" y="${Y(t) + 4}" text-anchor="end">${pct(t)}</text>`)
      .join("");
    g += rows.map((r, i) => `<text x="${X(i)}" y="${H - 14}" text-anchor="middle">${r.runs} run${r.runs > 1 ? "s" : ""} per task</text>`).join("");
    const ny = Y(d.coverage.nominal);
    g += `<line x1="${L}" x2="${W - R + 4}" y1="${ny}" y2="${ny}" stroke="var(--ink)" stroke-dasharray="5 4"/><text x="${W - R + 8}" y="${ny + 4}" class="lbl">nominal 95%</text>`;
    for (const [k, , c] of series) {
      g += `<polyline points="${rows.map((r, i) => `${X(i)},${Y(r[k])}`).join(" ")}" fill="none" stroke="${c}" stroke-width="3" stroke-linejoin="round"/>`;
      g += rows
        .map((r, i) => {
          const above = r[k] >= (k === "pooled" ? r.two_stage : r.pooled);
          return `<circle cx="${X(i)}" cy="${Y(r[k])}" r="5" fill="${c}"/><text class="num" x="${X(i)}" y="${Y(r[k]) + (above ? -11 : 21)}" text-anchor="middle">${pct(r[k])}</text>`;
        })
        .join("");
    }
    plot.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Interval coverage by runs per task for ${n} tasks">${g}</svg>
      <div class="legend">${series.map(([, l, c]) => `<span><i style="background:${c}"></i>${l}</span>`).join("")}</div>`;
    chips.querySelectorAll(".chip").forEach((b) => b.setAttribute("aria-pressed", String(+b.dataset.n === n)));
  }
  chips.innerHTML = tasks.map((n) => `<button class="chip" type="button" data-n="${n}">${n} tasks</button>`).join("");
  chips.addEventListener("click", (e) => {
    const b = e.target.closest(".chip");
    if (b) draw(+b.dataset.n);
  });
  draw(20);
}

const RENDER = { "swe-rank": sweRank, "swe-gap": sweGap, peeking, variance, local, coverage };

export function renderCharts(data) {
  document.querySelectorAll("[data-chart]").forEach((el) => RENDER[el.dataset.chart]?.(el, data));
  const bind = { "swe-agents": data.swebench.n_agents, "swe-pairs": data.swebench.n_pairs.toLocaleString("en") };
  document.querySelectorAll("[data-bind]").forEach((n) => {
    n.textContent = bind[n.dataset.bind];
  });

  const c = data.coverage.rows.find((r) => r.tasks === 20 && r.runs === 10);
  const pk = data.peeking.find((x) => x.true_diff === 0);
  const sw = data.swebench;
  const stats = [
    [`${Math.round(sw.adjacent_holm * sw.adjacent_n)}<small> / ${sw.adjacent_n}</small>`, "adjacent places on the SWE-bench Verified leaderboard that are <b>statistically distinguishable</b> (Holm, α = 0.05)."],
    [pct(c.pooled), "actual coverage of a <b>“95%” interval</b> when ten runs per task are pooled into one binomial sample (20 tasks)."],
    [pct(pk.peek, 1), `false positives when you <b>peek</b> at an A/A test and stop at the first win. Sequential inference: ${pct(pk.seq, 1)}.`],
    [pct(data.local.shares.run), `of score variance on local models is <b>pure run noise</b>, more than task difficulty (${pct(data.local.shares.task)}).`],
  ];
  document.getElementById("stats").innerHTML = stats.map(([n, t]) => `<div class="stat"><div class="n">${n}</div><p>${t}</p></div>`).join("");
}
