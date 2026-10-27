"""Self-contained Markdown and HTML reports (inline SVG, no external assets or dependencies)."""

from __future__ import annotations

import html
from typing import Any

PALETTE = ["#e5432a", "#2740d9", "#1f7a4d", "#8a3f9e", "#c98a14", "#0e7c86", "#6b6459", "#b92d17"]


def _at(d: dict[Any, Any], k: int) -> Any:
    """Dictionary lookup tolerant to int keys turned into strings by a JSON round trip."""
    return d[k] if k in d else d[str(k)]


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def forest_svg(models: list[dict[str, Any]], width: int = 640) -> str:
    """Horizontal interval plot of benchmark scores (sorted best first)."""
    rows = sorted(models, key=lambda m: -m["score"])
    h = 28 * len(rows) + 40
    left = 170

    def x(v: float) -> float:
        return left + v * (width - left - 20)

    parts = [f'<svg viewBox="0 0 {width} {h}" role="img" aria-label="Scores with intervals">']
    for t in (0, 0.25, 0.5, 0.75, 1.0):
        parts.append(
            f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="10" y2="{h - 24}" stroke="#8884" />'
            f'<text x="{x(t):.1f}" y="{h - 8}" font-size="11" text-anchor="middle" fill="currentColor">{int(t * 100)}%</text>'
        )
    for i, m in enumerate(rows):
        y = 24 + 28 * i
        c = PALETTE[i % len(PALETTE)]
        parts.append(
            f'<text x="{left - 8}" y="{y + 4}" font-size="12" text-anchor="end" fill="currentColor">{html.escape(m["model"])}</text>'
            f'<line x1="{x(m["lo"]):.1f}" x2="{x(m["hi"]):.1f}" y1="{y}" y2="{y}" stroke="{c}" stroke-width="3" />'
            f'<circle cx="{x(m["score"]):.1f}" cy="{y}" r="5" fill="{c}" />'
            f'<text x="{x(m["hi"]) + 6:.1f}" y="{y + 4}" font-size="11" fill="currentColor">{_pct(m["score"])}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def rank_heatmap_svg(ranking: dict[str, Any], size: int = 36) -> str:
    models = ranking["models"]
    probs = ranking["rank_probs"]
    m = len(models)
    left = 150
    w, h = left + size * m + 10, size * m + 50
    parts = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Rank probabilities">']
    for j in range(m):
        parts.append(
            f'<text x="{left + size * j + size / 2}" y="16" font-size="11" text-anchor="middle" fill="currentColor">#{j + 1}</text>'
        )
    for i, name in enumerate(models):
        y = 24 + size * i
        parts.append(
            f'<text x="{left - 8}" y="{y + size / 2 + 4}" font-size="12" text-anchor="end" fill="currentColor">{html.escape(name)}</text>'
        )
        for j in range(m):
            p = probs[i][j]
            parts.append(
                f'<rect x="{left + size * j}" y="{y}" width="{size - 2}" height="{size - 2}" rx="3" '
                f'fill="#2a6fdb" fill-opacity="{0.06 + 0.94 * p:.3f}" />'
                f'<text x="{left + size * j + size / 2 - 1}" y="{y + size / 2 + 3}" font-size="10" text-anchor="middle" '
                f'fill="{"#fff" if p > 0.55 else "currentColor"}">{round(100 * p)}</text>'
            )
    parts.append("</svg>")
    return "".join(parts)


def variance_svg(shares: dict[str, float], width: int = 640) -> str:
    labels = {
        "model": "agent skill",
        "task": "task difficulty",
        "model_task": "agent x task",
        "run": "run noise",
    }
    parts = [f'<svg viewBox="0 0 {width} 110" role="img" aria-label="Variance shares">']
    x0 = 0.0
    for i, (k, lab) in enumerate(labels.items()):
        w = shares.get(k, 0.0) * (width - 4)
        c = PALETTE[i]
        parts.append(f'<rect x="{x0 + 2:.1f}" y="10" width="{w:.1f}" height="32" fill="{c}" />')
        parts.append(f'<rect x="{2 + i * 150}" y="64" width="12" height="12" fill="{c}" />')
        parts.append(
            f'<text x="{20 + i * 150}" y="75" font-size="12" fill="currentColor">{lab} {_pct(shares.get(k, 0.0))}</text>'
        )
        x0 += w
    parts.append("</svg>")
    return "".join(parts)


def to_markdown(a: dict[str, Any], title: str = "AgentAudit report") -> str:
    lvl = int(100 * a["level"])
    lines = [
        f"# {title}",
        "",
        f"{a['n_models']} agents, {a['n_tasks']} tasks, up to {a['n_runs']} runs per task "
        f"({'balanced' if a['balanced'] else 'unbalanced: restricted to ' + str(a['balanced_tasks']) + ' tasks x ' + str(a['balanced_runs']) + ' runs for rank/variance'}).",
        "",
        f"## Scores with {lvl}% task-bootstrap intervals",
        "",
        "| agent | score | interval | pass^k (k=max) | pass@k (k=max) |",
        "|---|---|---|---|---|",
    ]
    for m in sorted(a["models"], key=lambda m: -m["score"]):
        kmax = max(int(k) for k in m["pass_hat_k"])
        ph, pa = (
            m["pass_hat_k"][str(kmax) if str(kmax) in m["pass_hat_k"] else kmax],
            m["pass_at_k"][str(kmax) if str(kmax) in m["pass_at_k"] else kmax],
        )
        lines.append(
            f"| {m['model']} | {_pct(m['score'])} | [{_pct(m['lo'])}, {_pct(m['hi'])}] | "
            f"{_pct(ph[0])} (k={kmax}) | {_pct(pa[0])} (k={kmax}) |"
        )
    if "ranking" in a:
        lines += [
            "",
            "## Ranking uncertainty",
            "",
            "| agent | rank | rank interval | P(best) |",
            "|---|---|---|---|",
        ]
        for r in a["ranking"]["table"]:
            lines.append(
                f"| {r['model']} | {r['rank']:.1f} | [{r['rank_lo']:.0f}, {r['rank_hi']:.0f}] | {_pct(r['p_best'])} |"
            )
        lines += [
            "",
            f"Pairs statistically indistinguishable (Holm-adjusted sign-flip test, alpha=0.05): "
            f"**{_pct(a['fraction_indistinguishable'])}**. Split-half rank correlation "
            f"(Kendall tau): mean {a['split_half_tau']['mean']:.2f}.",
        ]
    if "variance" in a:
        s = a["variance"]["shares"]
        lines += [
            "",
            "## Variance decomposition",
            "",
            f"agent skill {_pct(s['model'])}, task difficulty {_pct(s['task'])}, agent x task "
            f"{_pct(s['model_task'])}, run noise {_pct(s['run'])}. Generalizability coefficient for "
            f"agent comparisons: {a['variance']['g_coefficient']:.2f}.",
        ]
    if "power" in a:
        p = a["power"]
        lines += [
            "",
            "## Power planning",
            "",
            f"Minimum detectable difference with the current design (80% power): **{_pct(p['mde_current'])}**. "
            f"To detect {_pct(p['target_delta'])} you need {p['tasks_needed_at_current_runs']} tasks at the current runs per task.",
            "",
            "| runs per task | tasks needed | total attempts |",
            "|---|---|---|",
        ]
        for r in p["plan"]:
            lines.append(f"| {r['n_runs']} | {r['n_tasks']} | {r['total_attempts']} |")
    return "\n".join(lines) + "\n"


CSS = (
    ":root{--bg:#f4efe6;--fg:#17140f;--mut:#5a5346;--line:#d9cfbc;--acc:#b92d17}"
    "@media(prefers-color-scheme:dark){:root{--bg:#12100d;--fg:#f1ebdf;--mut:#a79f90;--line:#332d25;--acc:#ff8a72}}"
    "body{background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,sans-serif;max-width:860px;margin:2rem auto;padding:0 16px}"
    "h1,h2,h3{font-family:Georgia,'Times New Roman',serif;font-weight:600;line-height:1.15;letter-spacing:-.01em}"
    "h1{font-size:2rem}h2{font-size:1.4rem;margin-top:2.2rem;border-top:2px solid var(--fg);padding-top:.8rem}"
    "table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:.92rem}"
    "th{font-size:.72rem;letter-spacing:.08em;text-transform:uppercase;color:var(--mut)}"
    "td,th{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left}"
    "svg{width:100%;height:auto}p.mut{color:var(--mut)}"
)


def to_html(a: dict[str, Any], title: str = "AgentAudit report") -> str:
    md_tables = to_markdown(a, title)
    body = [
        f"<h1>{html.escape(title)}</h1>",
        f"<p class='mut'>{html.escape(md_tables.splitlines()[2])}</p>",
    ]
    body += ["<h2>Scores</h2>", forest_svg(a["models"])]
    if "ranking" in a:
        body += ["<h2>Probability of each rank</h2>", rank_heatmap_svg(a["ranking"])]
    if "variance" in a:
        body += ["<h2>Where the variance comes from</h2>", variance_svg(a["variance"]["shares"])]
    body.append("<h2>Details</h2>")
    for block in md_tables.split("\n## ")[1:]:
        head, _, rest = block.partition("\n")
        body.append(f"<h3>{html.escape(head)}</h3>")
        body.append(_md_block(rest))
    return (
        f"<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{html.escape(title)}</title><style>{CSS}</style><body>{''.join(body)}</body></html>"
    )


def _md_block(text: str) -> str:
    out, rows = [], []
    for line in text.strip().splitlines() + [""]:
        if line.startswith("|"):
            rows.append([c.strip() for c in line.strip("|").split("|")])
            continue
        if rows:
            hdr, body = rows[0], rows[2:]
            out.append(
                "<table><tr>"
                + "".join(f"<th>{html.escape(c)}</th>" for c in hdr)
                + "</tr>"
                + "".join(
                    "<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in r) + "</tr>"
                    for r in body
                )
                + "</table>"
            )
            rows = []
        if line.strip():
            out.append(f"<p>{html.escape(line).replace('**', '')}</p>")
    return "".join(out)
