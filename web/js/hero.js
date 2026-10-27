// A live leaderboard: six agents with known skill, re-evaluated once per press.
export const SKILLS = [0.55, 0.56, 0.57, 0.58, 0.59, 0.6];
const NAMES = ["Atlas", "Birch", "Cobalt", "Dune", "Ember", "Fjord"];
export const N_TASKS = 100;
const ROW = 54;
const LO = 0.4, HI = 0.8;

export function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const normal = (r) => Math.sqrt(-2 * Math.log(1 - r())) * Math.cos(2 * Math.PI * r());
const sig = (x) => 1 / (1 + Math.exp(-x));
const logit = (p) => Math.log(p / (1 - p));

// Task difficulty is fixed; each agent's intercept is solved so its mean success equals its skill.
export function buildWorld() {
  const r = rng(7);
  const d = Array.from({ length: N_TASKS }, () => normal(r) * 1.2);
  return SKILLS.map((theta) => {
    let lo = -6, hi = 6;
    for (let i = 0; i < 40; i++) {
      const mid = (lo + hi) / 2;
      const m = d.reduce((s, x) => s + sig(mid + x), 0) / N_TASKS;
      if (m < theta) lo = mid; else hi = mid;
    }
    return d.map((x) => sig(lo + x));
  });
}

function wilson(k, n, z = 1.96) {
  const p = k / n, z2 = z * z, den = 1 + z2 / n;
  const c = (p + z2 / (2 * n)) / den;
  const h = (z * Math.sqrt((p * (1 - p)) / n + z2 / (4 * n * n))) / den;
  return [c - h, c + h];
}

const pc = (v) => `${(((v - LO) / (HI - LO)) * 100).toFixed(2)}%`;

export function initHero() {
  const host = document.getElementById("board-rows");
  if (!host) return;
  const world = buildWorld();
  const wins = new Array(6).fill(0);
  let run = 0;
  const rows = SKILLS.map((theta, i) => {
    const row = document.createElement("div");
    row.className = "row";
    row.style.setProperty("--c", `var(--s${i + 1})`);
    row.innerHTML = `<span class="rk"></span><span class="nm"><i></i>${NAMES[i]}</span><span class="tr"></span><span class="sc"></span>`;
    const tr = row.querySelector(".tr");
    tr.innerHTML = '<i class="ln"></i><i class="ci"></i><i class="tk"></i><i class="dt"></i>';
    const ci = tr.querySelector(".ci");
    const truth = tr.querySelector(".tk");
    truth.style.left = pc(theta);
    const dot = tr.querySelector(".dt");
    host.appendChild(row);
    return { row, ci, truth, dot, rk: row.querySelector(".rk"), sc: row.querySelector(".sc") };
  });

  const ciBox = document.getElementById("show-ci");
  const trBox = document.getElementById("show-truth");
  const tally = document.getElementById("tally");

  function evaluate() {
    run += 1;
    const r = rng(1000 + run * 97);
    const res = world.map((p, i) => {
      let k = 0;
      for (let j = 0; j < N_TASKS; j++) if (r() < p[j]) k++;
      return { i, k, score: k / N_TASKS, ci: wilson(k, N_TASKS), tie: r() };
    });
    res.sort((a, b) => b.k - a.k || a.tie - b.tie);
    wins[res[0].i] += 1;
    res.forEach((o, pos) => {
      const u = rows[o.i];
      u.row.style.transform = `translateY(${pos * ROW}px)`;
      u.row.classList.toggle("first", pos === 0);
      u.rk.textContent = pos + 1;
      u.sc.textContent = `${(o.score * 100).toFixed(0)}%`;
      u.dot.style.left = pc(o.score);
      u.ci.style.left = pc(o.ci[0]);
      u.ci.style.width = `${((o.ci[1] - o.ci[0]) / (HI - LO)) * 100}%`;
    });
    const parts = wins.map((w, i) => [w, i]).filter(([w]) => w > 0).sort((a, b) => b[0] - a[0])
      .map(([w, i]) => `${NAMES[i]} <b>×${w}</b>`).join(" · ");
    const leaders = wins.filter((w) => w > 0).length;
    const truth = SKILLS.indexOf(Math.max(...SKILLS));
    tally.innerHTML = `Run <b>${run}</b>: <b>${NAMES[res[0].i]}</b> comes first with ${(res[0].score * 100).toFixed(0)}%.<br>Winners so far: ${parts}.${run >= 3 ? `<br><b>${NAMES[truth]}</b> is truly the best, yet led ${wins[truth]} of ${run} runs; <b>${leaders}</b> different agents have led.` : ""}`;
  }
  const sync = () => {
    rows.forEach((u) => { u.ci.style.opacity = ciBox.checked ? 0.9 : 0; u.truth.style.opacity = trBox.checked ? 1 : 0; });
  };
  rows.forEach((u) => { u.row.style.transition = "none"; });
  evaluate(); sync();
  requestAnimationFrame(() => requestAnimationFrame(() => rows.forEach((u) => { u.row.style.transition = ""; })));
  document.getElementById("rerun").addEventListener("click", evaluate);
  ciBox.addEventListener("change", sync);
  trBox.addEventListener("change", sync);
}
