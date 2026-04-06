# Repository Guidelines

## Project Structure & Module Organization
`main.tex` is the REVTeX 4.2 manuscript entrypoint. Keep citations in `main.bib` and figure assets in `images/` with relative paths like `./images/arpack_loglog.png`. Root PDFs are reference papers, not source files.

## Paper Overview & Section Map
The paper explains a faster way to compute AKV spins for binary black hole simulations by replacing the dense LAPACK generalized eigensolve with an ARPACK-based iterative method. It motivates the shift from `dggev` to shift-invert ARPACK, discusses numerical choices such as the shift `\sigma`, and compares timing/scaling data.

Current manuscript flow:

- `Introduction`
- `AKV spin definition`
- `Current AKV algorithm`
- `Sparse AKV algorithm`
- `Future improvements`
- `Conclusions`

The `Sparse AKV algorithm` section currently contains the main technical subsections: shift-invert, a worked example, conversion to a standard eigenvalue problem, sigma selection, and timing data.

## Build, Test, and Development Commands
Use `latexmk` when available:

- `latexmk -pdf main.tex` builds the paper and updates bibliography output.
- `latexmk -c` removes auxiliary files such as `*.aux` and `*.log`.
- `pdflatex main.tex` runs a single compile pass.
- `bibtex main` refreshes bibliography data when compiling manually.

Without `latexmk`, run `pdflatex main.tex`, `bibtex main`, then `pdflatex main.tex` twice.

## Coding Style & Naming Conventions
Preserve the existing LaTeX style: aligned option blocks in the preamble, descriptive labels such as `sec:Sparse AKV algorithm`, and minimal custom macros. For bibliography entries, follow the existing lowercase, topic-first key style such as `lovelace_binary-black-hole_2008`.

## Testing Guidelines
There is no automated test suite. Validate by rebuilding the paper and checking for:

- unresolved citations or references,
- missing figures from `images/`,
- overfull boxes or formatting regressions,
- unintended bibliography changes.

Before marking an issue as solved, run a clean build and review `main.log` for warnings.

## Commit & Pull Request Guidelines
Recent history uses short subjects such as `main bib`, `added papers`, and `changed to revtex`. Keep commits small and scoped to one manuscript change. Do not commit ignored LaTeX artifacts such as `main.pdf`, `*.aux`, `*.bbl`, or `mainNotes.bib`.

## Agent-Specific Instructions
Do not open pull requests or commit things unless explicitly asked. Do not add new manuscript text, rewrite sections, or expand arguments unless the author requests it directly. Default assistance should be limited to grammatical fixes and LaTeX error correction. If asked for readability feedback, provide suggestions only and do not edit the manuscript unless the request explicitly says to make the changes.
