const PYODIDE = "https://cdn.jsdelivr.net/pyodide/v0.26.4/full/pyodide.js";
const $ = (id) => document.getElementById(id);
const status = (msg) => { $("status").textContent = msg; };
let py = null;

async function boot() {
  await new Promise((res, rej) => {
    const s = document.createElement("script");
    s.src = PYODIDE; s.onload = res; s.onerror = () => rej(new Error("could not load Pyodide"));
    document.head.appendChild(s);
  });
  py = await loadPyodide();
  status("Loading numpy and scipy…");
  await py.loadPackage(["numpy", "scipy"]);
  status("Loading agentaudit…");
  const zip = await (await fetch("py/agentaudit.zip")).arrayBuffer();
  py.unpackArchive(zip, "zip");
  const bridge = await (await fetch("bridge.py")).text();
  py.runPython(bridge);
  status("Ready.");
  $("analyze").disabled = false;
}

async function loadExample() {
  const name = $("example").value;
  const r = await fetch(`examples/${name}.jsonl`);
  $("data").value = await r.text();
  status(`Loaded ${name}.`);
}

function renderPower() {
  if (!py) return;
  const g = (id) => parseFloat($(id).value);
  const out = JSON.parse(py.globals.get("run_power")(g("p-delta"), g("p-tv"), g("p-rv"), g("p-ct"), g("p-cr"), g("p-pw")));
  let h = "<table><tr><th>runs per task</th><th>tasks needed</th><th>total attempts</th><th>cost</th></tr>";
  for (const r of out.rows) h += `<tr><td>${r.n_runs}</td><td>${r.n_tasks}</td><td>${r.total_attempts}</td><td>${r.cost.toFixed(0)}</td></tr>`;
  h += `</table><p class="mut">Cost-optimal runs per task: <b>${out.optimal_runs >= 1e6 ? "∞ (no task-level variance)" : out.optimal_runs}</b>. Minimum detectable difference with 100 tasks × 5 runs: <b>${(100 * out.mde_100x5).toFixed(1)} points</b>.</p>`;
  $("power-out").innerHTML = h;
}

async function analyze() {
  const text = $("data").value.trim();
  if (!text) { status("Paste or load some trajectories first."); return; }
  status("Analyzing…"); $("analyze").disabled = true;
  await new Promise((r) => setTimeout(r, 30));
  const res = JSON.parse(py.globals.get("run_analysis")(text, $("level").value, $("nboot").value));
  $("analyze").disabled = false;
  if (res.error) { status("Error: " + res.error); return; }
  $("output").hidden = false;
  $("report").srcdoc = res.html;
  if (res.diff_variance) {
    $("p-tv").value = res.diff_variance.task.toFixed(3);
    $("p-rv").value = res.diff_variance.run.toFixed(3);
  }
  renderPower();
  status(`${res.n_models} agents, ${res.n_tasks} tasks, ${res.n_runs} runs.`);
}

$("load-example").addEventListener("click", loadExample);
$("analyze").addEventListener("click", analyze);
$("file").addEventListener("change", async (e) => { const f = e.target.files[0]; if (f) { $("data").value = await f.text(); status(`Loaded ${f.name}.`); } });
for (const id of ["p-delta", "p-tv", "p-rv", "p-ct", "p-cr", "p-pw"]) $(id).addEventListener("input", renderPower);
boot().then(() => { loadExample().then(renderPower); }).catch((e) => status("Failed to start: " + e.message));
