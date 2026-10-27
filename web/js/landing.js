import { initTheme } from "./theme.js";
import { initHero } from "./hero.js";
import { renderCharts } from "./charts.js";

initTheme();
initHero();

fetch("data/results.json")
  .then((r) => r.json())
  .then(renderCharts)
  .catch(() => {
    document.querySelectorAll(".plot").forEach((p) => {
      p.textContent = "Chart data could not be loaded.";
    });
  });

document.querySelectorAll(".copy").forEach((b) =>
  b.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(b.dataset.copy);
      b.textContent = "Copied";
    } catch (e) {
      b.textContent = "Select and copy";
    }
    setTimeout(() => {
      b.textContent = "Copy";
    }, 1600);
  }),
);
