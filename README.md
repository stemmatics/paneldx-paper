# Entity-linkage failure in longitudinal health data

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22086703.svg)](https://doi.org/10.5281/zenodo.22086703)
[![Manuscript](https://img.shields.io/badge/manuscript-CMPB%20(under%20review)-blue)](tex/manuscript.tex)
[![Software](https://img.shields.io/badge/software-paneldx-informational)](https://github.com/stemmatics/paneldx)
[![Code license](https://img.shields.io/badge/code-Apache%202.0-green)](LICENSE)
[![Content license](https://img.shields.io/badge/text%20%26%20figures-CC%20BY%204.0-green)](LICENSE)

Analysis code, intermediate results and manuscript for a study of entity-linkage
failure in panel data.

A longitudinal dataset asserts that rows sharing an identifier describe the same
entity observed repeatedly over time. Lags, first differences, trajectories,
sequence models and entity-grouped cross-validation all depend on that assertion,
and almost nothing verifies it. Where it is false, no error is raised and the
numbers become meaningless while remaining plausible.

## Findings

| | |
|---|---|
| Corrupted entity keys rejected | 100% (48 of 48) |
| Columns explained by a permuted key | 0, in all 63 datasets |
| Specificity where evidence exists | 67.4%, and 92% on clinical panels |
| Case study: earlier model against carry-forward | 2.9 times worse |

The case study is a self-audit. An earlier pipeline of ours keyed physicians by
row position in a table that had been sorted by platform rank, so position *i*
named a different doctor each quarter. The defects proved mutually concealing:
under the broken key the naive baseline scored R² 0.191 and looked useless, so
there was no reason to compare a model against it. Under the recovered key it
scored 0.971 and beat the model outright.

## Reproducing

```bash
pip install -r requirements.txt
python analysis/find_panels.py          # discovers the 63 public panels
python analysis/evaluate.py             # specificity and sensitivity
python analysis/make_figures.py         # all three figures
python analysis/run_analysis.py --data /path/to/hospital_data.xlsx
```

| Script | Produces |
|---|---|
| `run_analysis.py` | `results/findings.json`, every case-study number |
| `find_panels.py` | `results/panels.json`, the 63 public panels |
| `evaluate.py` | `results/evaluation.json`, sensitivity and specificity |
| `make_figures.py` | `figures/`, all three manuscript figures |

The manuscript is `tex/manuscript.tex`. It compiles with pdfLaTeX, or upload the
folder to Overleaf and it builds in the browser with nothing to install.

Dependencies are pinned, including `paneldx==0.3.1`, so the reported numbers
remain reproducible as that package evolves. Do not upgrade them to reproduce
this manuscript.

The first three scripts run unaided; they retrieve public datasets at runtime.
`run_analysis.py` requires the physician panel, which is **not distributed here**
and never will be: it contains attributes inferred from photographs of
identifiable individuals, which are sensitive personal information under PIPL and
GDPR Article 9. It takes a path to a local copy.

## Layout

```
tex/               the paper, LaTeX source using Elsevier's elsarticle class
analysis/          four scripts, each regenerating one part of the results
results/           machine-readable outputs underlying every figure and table
figures/           vector PDF for typesetting, PNG for preview
```

## The method

The validation method is released separately as
[paneldx](https://github.com/stemmatics/paneldx), installable with
`pip install paneldx` and licensed Apache-2.0. This repository contains only the
study; the tool is independently useful and versioned on its own.

## Citation

See [CITATION.cff](CITATION.cff), or use GitHub's "Cite this repository" button.

| What | DOI |
|---|---|
| This archive, all versions | [10.5281/zenodo.22086703](https://doi.org/10.5281/zenodo.22086703) |
| This archive, v1.0.2 | [10.5281/zenodo.22087095](https://doi.org/10.5281/zenodo.22087095) |
| The paneldx software | [10.5281/zenodo.22086706](https://doi.org/10.5281/zenodo.22086706) |

Cite the manuscript for the study and the software for the method. Use the
all-versions DOI unless you are reproducing a specific release.

## License

Code under Apache-2.0, manuscript and figures under CC BY 4.0. See
[LICENSE](LICENSE). No data is distributed.
