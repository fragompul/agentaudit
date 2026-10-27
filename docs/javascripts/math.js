// Load MathJax only on pages that actually contain equations.
if (document.querySelector(".arithmatex")) {
  window.MathJax = { tex: { inlineMath: [["\(", "\)"]], displayMath: [["\[", "\]"]] }, options: { ignoreHtmlClass: ".*", processHtmlClass: "arithmatex" } };
  const s = document.createElement("script");
  s.src = "https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-mml-chtml.js";
  s.async = true;
  document.head.appendChild(s);
}
