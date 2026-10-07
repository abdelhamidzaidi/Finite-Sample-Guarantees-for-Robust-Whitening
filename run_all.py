"""
run_all.py -- reproduces every table and figure of the manuscript (about 1.5 hours).
Works on Windows, macOS and Linux:   python run_all.py
It runs the same commands as run_all.sh, one after the other, and stops at the first error.
Quick installation test (about 5 minutes, tiny sample sizes, results NOT those of the paper):
                                     python run_all.py --smoke
"""
import subprocess, sys

T = "500,1000,2000,5000,10000,20000,50000,100000"
steps = [
    ["table2_lemma3.py"],                                   # Table 2
    ["gamma_s.py"],                                         # source margins, Section 6.1
    ["sec6_runs.py", "10", T, "200"],                       # Section 6, K = 10
    ["sec6_runs.py", "6", T, "200"],                        # Section 6, K = 6
    ["exact_families.py", "5", "3", "100"],                 # Table 4, Figure 2
    ["exact_families.py", "5", "5", "100"],
    ["exact_families.py", "10", "5", "100"],
    ["exact_families.py", "10", "10", "100"],
    ["exact_families.py", "20", "10", "60"],
    ["exact_families.py", "20", "20", "50"],
    ["sec7_runs.py", "T", T, "500"],                        # Table 5, Figure 3
    ["sec7_runs.py", "snr", "-5,0,5,10,20,40", "500"],      # Figure 4 (top)
    ["sec7_runs.py", "cond", "1,2,4,8,16", "500"],          # Figure 4 (bottom)
    ["robust_runs.py", "0", "100"],                         # Tables 6-7, Figure 5
    ["remarks_pdc.py"],                                     # Remark 6
]
if "--smoke" in sys.argv:          # same pipeline, very few trials, only to test the installation
    T = "2000,100000"
    steps = [["gamma_s.py"],
             ["sec6_runs.py", "10", T, "5"], ["sec6_runs.py", "6", T, "5"],
             ["exact_families.py", "5", "3", "5"],
             ["sec7_runs.py", "T", "10000", "5"], ["sec7_runs.py", "snr", "5", "5"], ["sec7_runs.py", "cond", "4", "5"],
             ["robust_runs.py", "0", "2"]]
for step in steps:
    print(">>> python", " ".join(step), flush=True)
    subprocess.run([sys.executable] + step, check=True)     # sys.executable = this Python
if "--smoke" in sys.argv:
    print("Smoke test finished: the pipeline runs. For the paper's results run: python run_all.py")
    sys.exit()
with open("tables.txt", "w") as f:                          # Tables 3-7 and quoted numbers
    subprocess.run([sys.executable, "make_tables.py"], stdout=f, check=True)
subprocess.run([sys.executable, "make_figures.py"], check=True)   # Figures 1-5
print("Done: see tables.txt and the files fig_*.pdf")
