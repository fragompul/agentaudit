import { initTheme } from "./theme.js";

const PYODIDE = "https://cdn.jsdelivr.net/pyodide/v0.26.4/full/pyodide.js";
const $ = (id) => document.getElementById(id);
let py = null;
let lastHtml = "";

function status(msg, state) {
  $("status").textContent = msg;
  const e = $("engine");
  e.classList.toggle("ready", state === "ready");
  e.classList.toggle("err", state === "err");
}

async function boot() {
  await new Promise((res, rej) => {
    const s = document.createElement("script");
    s.src = PYODIDE;
    s.onload = res;
    s.onerror = () => rej(new Error("could not load Pyodide"));
    document.head.appendChild(s);
  });
  status("Starting Python…");
  py = await loadPyodide();
  status("Loading numpy and scipy…");
  await py.loadPackage(["numpy", "scipy"]);
  status("Loading agentaudit…");
  const zip = await (await fetch("../py/agentaudit.zip")).arrayBuffer();
  py.unpackArchive(zip, "zip");
  py.runPython(await (await fetch("../bridge.py")).text());
  status("Python ready. Running locally.", "ready");
  refreshButton();
}

const refreshButton = () => {
  $("analyze").disabled = !py || !$("data").value.trim();
};

function describe(text) {
  const rows = text.split("\n").filter((l) => l.trim());
  const models = new Set(), tasks = new Set();
  let bad = 0;
  for (const l of rows) {
    try {
      const o = JSON.parse(l);
      models.add(o.model);
      tasks.add(o.task_id);
    } catch (e) {
      bad += 1;
    }
  }
  return { n: rows.length, models: models.size, tasks: tasks.size, bad };
}

function setData(text, label) {
  $("data").value = text;
  const d = describe(text);
  $("summary").innerHTML = d.n
    ? `<b>${label}</b>: ${d.n.toLocaleString("en")} attempts · ${d.models} agents · ${d.tasks} tasks${d.bad ? ` · <b>${d.bad} unreadable lines</b>` : ""}`
    : "";
  refreshButton();
}

async function loadExample(name) {
  const r = await fetch(`../examples/${name}.jsonl`);
  setData(await r.text(), name);
  document.querySelectorAll("[data-example]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.example === name)));
}

function renderPower() {
  if (!py) return;
  const g = (id) => parseFloat($(id).value);
  const out = JSON.parse(py.globals.get("run_power")(g("p-delta"), g("p-tv"), g("p-rv"), g("p-ct"), g("p-cr"), g("p-pw")));
  const maxA = Math.max(...out.rows.map((r) => r.total_attempts));
  const cheapest = out.rows.reduce((a, b) => (b.cost < a.cost ? b : a));
  let h = "<table><tr><th>runs / task</th><th>tasks needed</th><th>total attempts</th><th></th><th>cost</th></tr>";
  for (const r of out.rows) {
    h += `<tr class="${r === cheapest ? "best" : ""}"><td>${r.n_runs}</td><td>${r.n_tasks}</td><td>${r.total_attempts.toLocaleString("en")}</td><td style="width:26%"><i class="meter" style="width:${(r.total_attempts / maxA) * 100}%"></i></td><td>${r.cost.toFixed(0)}</td></tr>`;
  }
  const opt = out.optimal_runs >= 1e6 ? "unbounded (no task-level variance)" : out.optimal_runs;
  h += `</table><p class="power-note">Cost-optimal runs per task: <b>${opt}</b>. Smallest detectable difference with 100 tasks × 5 runs: <b>${(100 * out.mde_100x5).toFixed(1)} points</b>.</p>`;
  $("power-out").innerHTML = h;
}

const pct = (v) => `${(v * 100).toFixed(0)}%`;
function renderKpis(res) {
  const k = [[res.n_models, "agents"], [res.n_tasks, "tasks"], [res.n_runs, "runs per task"]];
  if (res.fraction_indistinguishable != null) k.push([pct(res.fraction_indistinguishable), "pairs indistinguishable"]);
  if (res.shares) k.push([pct(res.shares.run), "variance is run noise"]);
  $("kpis").innerHTML = k.map(([n, t]) => `<div class="kpi"><b>${n}</b><span>${t}</span></div>`).join("");
}

function fitFrame() {
  const f = $("report");
  try {
    f.style.height = `${f.contentDocument.documentElement.scrollHeight + 8}px`;
  } catch (e) {
    /* keep the default height */
  }
}

async function analyze() {
  const text = $("data").value.trim();
  if (!text) return status("Load or paste some trajectories first.", "err");
  status("Analyzing…");
  $("analyze").disabled = true;
  await new Promise((r) => setTimeout(r, 30));
  const res = JSON.parse(py.globals.get("run_analysis")(text, $("level").value, $("nboot").value));
  refreshButton();
  if (res.error) return status(`Error: ${res.error}`, "err");
  lastHtml = res.html;
  $("output").hidden = false;
  renderKpis(res);
  $("report").onload = () => {
    fitFrame();
    setTimeout(fitFrame, 400);
  };
  $("report").srcdoc = res.html.replace("<title>", '<link rel="stylesheet" href="../css/report-embed.css"><title>');
  if (res.diff_variance) {
    $("p-tv").value = res.diff_variance.task.toFixed(3);
    $("p-rv").value = res.diff_variance.run.toFixed(3);
  }
  renderPower();
  status("Python ready. Analysis complete.", "ready");
  $("output").scrollIntoView({ behavior: "smooth", block: "start" });
}

function download() {
  const url = URL.createObjectURL(new Blob([lastHtml], { type: "text/html" }));
  const a = Object.assign(document.createElement("a"), { href: url, download: "agentaudit-report.html" });
  a.click();
  URL.revokeObjectURL(url);
}

async function readFile(f) {
  if (f) setData(await f.text(), f.name);
}

initTheme();
const drop = $("drop");
$("analyze").addEventListener("click", analyze);
$("download").addEventListener("click", download);
$("data").addEventListener("input", () => setData($("data").value, "pasted data"));
$("file").addEventListener("change", (e) => readFile(e.target.files[0]));
drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("file").click(); } });
for (const ev of ["dragenter", "dragover"]) drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); });
for (const ev of ["dragleave", "drop"]) drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); });
drop.addEventListener("drop", (e) => readFile(e.dataTransfer.files[0]));
document.querySelectorAll("[data-example]").forEach((b) => b.addEventListener("click", () => loadExample(b.dataset.example)));
for (const id of ["p-delta", "p-tv", "p-rv", "p-ct", "p-cr", "p-pw"]) $(id).addEventListener("input", renderPower);
loadExample("taubench_retail");
boot().then(renderPower).catch((e) => status(`Failed to start: ${e.message}`, "err"));
