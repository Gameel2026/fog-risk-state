# fog-risk-state

Levodopa OFF-state detection from freeze-free walking in people with freezing of gait, using a single lower-back accelerometer.

This repository holds the analysis plan, the code, the results and the embedded implementation of the study.
The plan was committed before any medication-state result was computed. Later changes are recorded in dated files.

## Timeline of commits
| Commit | Content |
|---|---|
| 4f589e9 | Analysis plan (PLAN.md) and pre-plan data checks. No medication-state result computed. |
| a1ff8ec | Frozen gait gate (results/frozen_gate.json), window and feature code, full analysis code. Committed before any medication-state result. |
| e7010eb | Step 3 results, implementation corrections (ANALYSIS_NOTES.md) and Amendment 01, committed before the amendment analyses were run. |
| 2f3993c | Amendment 01 code, FoG distances and a dry run with shuffled labels, committed before the real run. |
| 1167613 | Amendment 01 results (single run). |
| 78565c5 | Embedded C implementation for a Cortex-M4F, emulator agreement and resource estimates. |

## Documents
- `PLAN.md` — pre-specified analysis plan (version 1).
- `ANALYSIS_NOTES.md` — how points left open in the plan were implemented, and later corrections.
- `AMENDMENT_01.md` — post hoc exploratory analyses (task matching, personal baseline, windows away from FoG).

## Code and order of execution
| Step | Script | Output |
|---|---|---|
| Pre-plan checks | `preanalysis_data_checks.py` | data checks only |
| 1 | `p1_build_windows.py` | freeze-free windows and features (`results/windows_<variant>.pkl`, `results/recordings.csv`) |
| 2 | `p2_freeze_gate.py` | label-free gait gate (`results/frozen_gate.json`) |
| 3 | `p3_analysis.py` | planned analyses (`results/results.json`, predictions, figures) |
| Amendment 01 | `p1b_fog_distance.py`, `p3_amend01.py` | `results/fog_distance_primary.csv`, `results/results_amend01.json` |
| Embedded | `p4_deploy_model.py`, `mcu/` | deployable model, C code for Cortex-M4F, emulator agreement (`mcu/mcu_results.txt`) |

Shared settings are in `p_config.py`. Set `DATA_ROOT` to a local copy of the data before running.
The Windows batch files (`run_*.bat`) run each step in an Anaconda environment.
The analysis was run with Python 3.8.5, NumPy 1.19.2, pandas 1.1.3 and SciPy 1.5.2.
The embedded code was built with clang 18 and ld.lld (`mcu/build.sh`) and run in the Unicorn emulator (`mcu/run_fog_mcu.py`).

## Data
The study uses the public FoG detection contest data (Salomon et al., Nat. Commun. 2024; Zenodo doi:10.5281/zenodo.12565292).
The raw data are not redistributed here. Files in `results/` contain derived features, predictions and summary results only.

## Licence
Code: MIT. Documents and results: CC BY 4.0.
