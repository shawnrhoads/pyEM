// MathJax v3 config for the pyEM docs.
// Renders LaTeX both in Markdown pages (via pymdownx.arithmatex `\( \)` / `\[ \]`)
// and in the rendered example notebooks (which use `$ $` / `$$ $$`).
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"], ["$", "$"]],
    displayMath: [["\\[", "\\]"], ["$$", "$$"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: "tex2jax_ignore",
    processHtmlClass: "tex2jax_process|arithmatex"
  }
};
