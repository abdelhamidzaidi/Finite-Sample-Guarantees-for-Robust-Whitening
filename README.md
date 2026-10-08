# Code and results for "Finite-Sample Guarantees for Robust Whitening"

Python code and stored simulation results that reproduce every table and figure of the manuscript
*"Finite-Sample Guarantees for Robust Whitening: The Margin of a Positive Definite Combination of
Lagged Covariance Matrices"* (A. T. Zaidi). Appendix G of the manuscript lists every function of this
repository, its purpose and its link to the paper; Appendix F gives seeds, grids and the rules for
failed or capped runs.

## Contents
| File | Role |
|---|---|
| `core.py` | the three algorithms: MNP (Algorithms 1 and 4), PC (Algorithm 2), PDC (Algorithm 3) |
| `model.py` | BSS model of Section 2, population matrices (2), estimators of Section 4 |
| `sep.py`, `wilson.py` | joint diagonalization and Amari index; Wilson intervals |
| `table2_lemma3.py`, `gamma_s.py` | Table 2; source margins of Section 6.1 |
| `sec6_runs.py`, `exact_families.py`, `sec7_runs.py`, `robust_runs.py`, `remarks_pdc.py` | simulations of Sections 6 and 7 |
| `make_tables.py`, `make_figures.py` | Tables 3-7 and Figures 1-5 from the stored results |
| `run_all.py` (`run_all.sh`) | runs everything in order |
| `results.zip` | stored result files (.pkl) used in the manuscript, and `tables_output.txt` |
| `requirements.txt` | exact library versions (Python 3.12.3) |

## Quick start
    pip install -r requirements.txt
    python run_all.py --smoke          # five-minute installation test (tiny sample sizes)

## Reproduce the tables and figures from the stored results (a few minutes)
Unzip `results.zip` (on Windows: right-click > Extract All); this gives the folder `results`. Then:

    cd results
    python ../make_tables.py > tables_check.txt     # identical to tables_output.txt
    python ../make_figures.py                         # writes fig_*.pdf
    The results folder can also be passed as an argument, so the figures
       can be drawn from any folder:

       python make_figures.py path/to/results     # writes fig_*.pdf in the current folder
    cd ..
    python table2_lemma3.py                           # Table 2 (a few minutes)

## Rerun every simulation from scratch (about 1.5 hours on one core)
    python run_all.py              # Windows, macOS or Linux
    sh run_all.sh                  # alternative on macOS or Linux

All random numbers come from fixed seeds, so a complete rerun reproduces the stored results
exactly with the library versions of `requirements.txt`; only computing times vary.
A complete rerun from a clean copy reproduced all 141 stored result files value by value.

## Citation
See `CITATION.cff`. The archived version of this repository has a permanent DOI on Zenodo.

## License
MIT (see `LICENSE`).
