#!/bin/sh
# Reproduces every table and figure of the manuscript (about 1.5 h on one core).
# Python 3.12, NumPy 2.4.4, SciPy 1.17.1, Matplotlib 3.10.8 (see requirements.txt).
set -e
python table2_lemma3.py                                  # Table 2
python gamma_s.py                                        # source margins, Section 6.1
T=500,1000,2000,5000,10000,20000,50000,100000
python sec6_runs.py 10 $T 200                            # Section 6, K = 10
python sec6_runs.py 6  $T 200                            # Section 6, K = 6
for pk in "5 3 100" "5 5 100" "10 5 100" "10 10 100" "20 10 60" "20 20 50"; do
  python exact_families.py $pk                           # Table 4, Figure 2
done
python sec7_runs.py T    $T 500                          # Table 5, Figure 3
python sec7_runs.py snr  -5,0,5,10,20,40 500             # Figure 4 (top)
python sec7_runs.py cond 1,2,4,8,16 500                  # Figure 4 (bottom)
python robust_runs.py 0 100                              # Tables 6-7, Figure 5
python remarks_pdc.py                                    # Remark 6
python make_tables.py > tables.txt                       # Tables 3-7 and quoted numbers
python make_figures.py                                   # Figures 1-5
