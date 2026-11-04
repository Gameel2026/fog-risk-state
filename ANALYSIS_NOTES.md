# Analysis notes — implementation details of PLAN.md (version 1)

Dated 3 November 2026. Written and committed before any medication-state result was computed.
These notes do not change the plan. These notes record how points left open in the plan are implemented in code. Later changes, if needed, will be recorded in dated amendments with their reasons.

## Step order
1. `p1_build_windows.py` builds freeze-free windows and features for all recordings. Medication labels are not used.
2. `p2_freeze_gate.py` fixes the gait-gate threshold on pooled windows without labels and writes `results/frozen_gate.json`.
3. `p3_analysis.py` runs every planned analysis and writes `results/results.json`, the single source of all reported numbers.

## Frozen gate (step 2 output)
- Rule: Otsu threshold on log 0.5–3 Hz band power of all primary windows pooled over both datasets, and step autocorrelation peak > 0.3.
- Threshold: 1.6852 (log power units). 7,720 of 21,842 primary windows pass.
- sha256 of `frozen_gate.json`: 1d5fe4cb01cbaf0375c99567748efac4375115d34c98bb000577bfbdecc6ea77

## Implementation choices
- H1 uses the pooled leave-one-patient-out configuration, in which each patient is predicted by a model trained on all other patients of both datasets. The same test on within-dataset predictions pooled across datasets is reported as a sensitivity result.
- H2a needs both datasets to pass. Its p value is the larger of the two. H2b likewise needs both directions to pass.
- H2b permutation: the trained source model is fixed and ON/OFF labels are swapped within each patient-visit of the target data.
- H4: the UPDRS-III OFF−ON difference is averaged over a patient's visits in `subjects.csv`.
- Patient weights are 1/(number of training windows of the patient), rescaled to mean 1 so that C = 1 keeps its usual meaning.
- Logistic regression is solved by Newton steps on the scikit-learn objective. Agreement with scikit-learn was checked to 1e-8.
- Missing feature values are replaced by the training-fold median.
- Variants with another sampling rate or window length (32 Hz, 16 Hz, 10 s) use the same label-free Otsu rule recomputed on their own windows, because band power depends on window length and rate. Margin variants use the frozen threshold.
- Random-split control: 5-fold recording-level split, repeated 5 times, within each dataset.
- Negative control: one fixed label shuffle within patient-visit (seed 43), plus the mean of the H2a permutation null.
- Order checks: Spearman correlation of recording probability with test number in tDCS-FoG within each patient-visit-state, and of window probability with time within each recording.
- Operating point: in each fold the threshold gives 80% specificity on the training recordings' fitted probabilities.
- Temporally separated FoG link (DeFOG): median OFF probability of freeze-free windows in the first half of scored time versus FoG percentage in the second half, across patients within each state.
- Window-count curve: the first k windows of each recording at prediction time. The model is unchanged.
- Gradient boosting (exploratory): scikit-learn HistGradientBoostingClassifier with default settings and the same patient weights.
- Task-matched slowness comparison: not run, because no task labels are available for most recordings. The single-feature baseline is reported.
- Seeds: 42 for bootstrap and controls, 1000–1999 for H2a permutations, 49 for H2b permutations.

## Corrections (3 November 2026, after step 3 and before Amendment 01)
- The statement above that task labels are not available for most recordings was incorrect.
  In tDCS-FoG the Test field gives the test type for every recording, with 3 the most challenging (official data description).
  In DeFOG the official tasks file gives task intervals for the 91 training recordings. The planned task-matched comparison is carried out under Amendment 01.
- The order check named "tdcsfog_test_order" in results.json concerns the test type (difficulty), not the order of recording.
- The order of OFF and ON recordings in tDCS-FoG is not documented in the data description.
- The figure code in p3_analysis.py was changed for compatibility with an older matplotlib version, and results are now saved before the figures are drawn. No analysis was changed.
- Step 3 was run with run_analysis_step3.bat, which activates the Anaconda environment. It runs the same command as run_step3.bat.

## Label-free gate check (4 November 2026, after review comments)
- `p2b_gate_loo_check.py` recomputes the Otsu threshold without each patient in turn and from each source cohort alone. It reads no medication label and computes no medication-state result.
- Leaving out one patient moved the threshold between 1.622 and 1.694 and changed the gate decision for 7 of 21,842 primary windows.
- A threshold from tDCS-FoG alone changed 17 of 11,238 DeFOG windows. A threshold from DeFOG alone changed 5 of 10,604 tDCS-FoG windows.
- The frozen gate and all reported results are unchanged.

## Exploratory 16-Hz embedded build (4 November 2026, after Amendment 02)
- PLAN.md allows variants at lower sampling rates. This build ports the 16-Hz variant to the same Cortex-M4F C code. It is exploratory and does not change the 64-Hz primary configuration.
- `p4b_deploy_model_fs16.py` refits the pooled deployment model on the 16-Hz windows with the label-free Otsu gate recomputed at 16 Hz (threshold -1.0998). It writes `results/deploy_model_fs16.json` and `mcu16/fog_params.h`.
- `mcu16/fog_state.c` is the 64-Hz C code with a 96-sample window. `mcu16/run_fog_mcu16.py` runs the compiled binary in the Unicorn emulator on all 16-Hz windows and compares it with the Python reference.
- Logs: `p4b_log.txt` and `mcu16_log.txt`. Outputs: `mcu16/mcu16_results.txt`, `mcu16/mcu16_agreement.csv` and `mcu16/mcu16_recording_scores.csv`.
- No medication-state test was added. The 16-Hz state results quoted in the manuscript are the planned exploratory variant already stored in `results/results.json`.

## Supplementary feature figure (4 November 2026)
- `supplementary/fig_s1_features.png` and `supplementary/feature_table.csv` are drawn by `supplementary/fig_s1_features.py` from `results/dz_features.csv` and the pooled coefficients in `results/results.json`. No new analysis was run.
