// Load MathJax (default TeX delimiters, which is what arithmatex emits) only on pages with equations.
if (document.querySelector(".arithmatex")) {
  const s = document.createElement("script");
  s.src = "https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-mml-chtml.js";
  s.async = true;
  document.head.appendChild(s);
}
