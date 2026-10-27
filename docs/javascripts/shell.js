// Tie the docs to the rest of the site: the logo returns to the home page and the header links to the other parts.
(() => {
  const inner = document.querySelector(".md-header__inner");
  if (!inner) return;
  const root = location.pathname.includes("/docs/") ? `${location.pathname.split("/docs/")[0]}/` : "/";
  const logo = inner.querySelector("a.md-logo");
  if (logo) {
    logo.href = root;
    logo.title = "AgentAudit home";
    logo.setAttribute("aria-label", "AgentAudit home");
  }
  const nav = document.createElement("nav");
  nav.className = "aa-links";
  nav.setAttribute("aria-label", "Site");
  nav.innerHTML = `<a href="${root}">Home</a><a href="${root}playground/">Playground</a><a href="${root}paper/agentaudit-paper.pdf">Paper</a>`;
  inner.insertBefore(nav, inner.querySelector(".md-search") || null);
})();
