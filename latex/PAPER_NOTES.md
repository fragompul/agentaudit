# Paper notes

## Build

```bash
python scripts/make_paper_assets.py     # figures, tables and macros from results/ (no number is typed by hand)
cd latex && tectonic main.tex           # or: ./build.sh
```

Template: the **official TMLR style** (`tmlr.sty`, `tmlr.bst`, from https://github.com/JmlrOrg/tmlr-style-file),
loaded with the `preprint` option so the author is visible. For a TMLR submission change
`\usepackage[preprint]{tmlr}` to `\usepackage{tmlr}` (anonymous) and remove the author block, the e-mail and the
GitHub link from the text. To target a different venue (COLM, NeurIPS Datasets & Benchmarks) swap the style file and keep
all `sections/*.tex` unchanged.

## Candidate venues (dates checked on 2026-10-23; verify before relying on them)

| Venue | Fit | Deadline / status |
|---|---|---|
| **TMLR** | Evaluation methodology with simulation + empirical study; rewards rigour and honest negative results | Rolling submissions; double-blind; ~9 weeks to decision per the TMLR FAQ; main body over 12 pages may take longer; the unmodified TMLR style is mandatory. https://jmlr.org/tmlr/author-guide.html |
| **COLM 2027** | LLM agents and evaluation | Tentative paper deadline around 30-31 March 2027 according to third-party trackers; no official call found yet. Check https://colmweb.org |
| **NeurIPS 2027, Datasets & Benchmarks** | The benchmark + statistical protocol contribution | Deadline not found; check https://neurips.cc when the call opens (historically May) |
| ICLR 2027 | -- | Passed: abstract 18 Sep 2026, paper 25 Sep 2026 (https://iclr.cc/Conferences/2027/Dates) |

## Submitting to TMLR

1. Switch to the anonymous style (above); remove author, e-mail, repository URL and acknowledgements that reveal identity. Host code in an anonymous repository for review if you prefer, or state it will be released.
2. Check the page budget (main body, excluding references and appendices).
3. Submit through OpenReview (https://openreview.net/group?id=TMLR) with `main.pdf` and a short cover note. TMLR allows posting to arXiv before or during review, but the submission itself must not link to an identified version.
4. Be prepared to supply the code link and the artifacts after acceptance (the repository is already public).

## Posting to arXiv

1. `cd latex && tectonic main.tex --keep-intermediates` and check that `main.bbl` exists; arXiv needs the `.bbl` rather than running BibTeX.
2. Upload `main.tex`, `sections/`, `generated/` (macros, tables, prompt text), `figures/*.pdf`, `tmlr.sty`, `fancyhdr.sty`, `math_commands.tex`, `main.bbl`. arXiv compiles with TeX Live; if it complains, compile locally with `pdflatex`-compatible packages (the preamble only uses standard packages).
3. Category: cs.LG primary, cs.AI and stat.ME cross-lists are reasonable.
4. License: CC BY 4.0 is a common choice; the code is MIT.

## Honest checklist before sending (self "Reviewer 2")

* Are the real-model conclusions limited to small local models? (stated in the abstract and Limitations)
* Is the SWE-bench full-coverage assumption flagged? (Limitations, Protocol)
* Do non-significant results get described as "not distinguishable", never as "equal"? (Discussion)
* Are the numbers in the abstract produced by macros? (yes, `generated/macros.tex`)
