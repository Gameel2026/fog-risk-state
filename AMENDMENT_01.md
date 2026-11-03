# Amendment 01 — post hoc exploratory analyses

Dated 3 November 2026. Written and committed before any of the analyses below was run.
PLAN.md (version 1) and its confirmatory results are unchanged. H2a was not supported, and this amendment does not change that conclusion.
All analyses below are post hoc and exploratory. They are intended to be reported beside the original results in all cases.

## Reasons
PLAN §7 planned a task-matched comparison "where task labels allow it".
The implementation notes stated that task labels were not available for most recordings. That statement was incorrect.
In tDCS-FoG, the Test field gives one of three test types for every recording, with 3 the most challenging (official data description).
In DeFOG, the official tasks file gives task intervals for the training recordings.
The official protocol description states that the ON protocol includes a MiniBest test that is absent from OFF.

Inspection of counts suggested three possible influences on the reported results.
- Laboratory recordings were retained at different rates across Test types in OFF and ON (OFF 73/65/60%, ON 68/70/74% for Tests 1/2/3).
- At home, MiniBest and rest windows appear only in ON. In the training recordings they formed 19% of ON windows after the gait gate (225 of 1,178), plus 20 rest windows. Task proportions also differed between states.
- Recording-level AUC under leave-one-patient-out is biased by changes in training-fold prevalence.

Within-patient laboratory feature differences were also sizeable (dz_features.csv). This motivated a model trained on within-visit contrasts.

## What was examined before this amendment
Counts of recordings, windows, visits, Test types, time bins and tasks per medication state.
Gate pass rates per state. A label-free proxy of distance to excluded segments.
An intercept-only leave-one-patient-out AUC that uses no sensor feature (laboratory 0.25, home 0.43, pooled 0.28).
No medication-state performance of any analysis below was computed.

## Common definitions
- Data: primary windows after the frozen gate, as in the original analysis.
- Paired patient-visit (pv): a visit with retained recordings in both states.
- Recording score: median window logit.
- Laboratory strata: the Test type. There is one recording per pv, state and Test type.
- Home strata: the task name in the official tasks file.
  A window is assigned to a task only if it lies wholly within that task interval.
  MiniBest items and rest periods are excluded.
  A home stratum is used only if it has at least 2 windows in each state. Its score is the median window logit in that stratum and state.
  Home matched analyses use the DeFOG training recordings only, because only these have task intervals.
  If fewer than 20 patients would remain, cued and uncued versions of a task are merged, and the two hotspot tasks are merged.
- Matched difference for one pv: mean over usable strata of (OFF score − ON score). The patient value is the mean over that patient's pvs.
- Unmatched difference: OFF − ON of recording scores averaged within pv and state, as in the original analysis.
- Within-visit pairwise accuracy: the proportion of OFF–ON pairs within the same pv and stratum in which OFF scores higher (ties count 0.5).
  It is averaged per patient and then across patients.

## A. Task-matched comparison of the original model
The original within-dataset leave-one-patient-out window logits are regenerated with the unchanged code.
They must reproduce pred_tdcsfog.csv and pred_defog.csv to within 1e-6. Otherwise the analysis stops and the discrepancy is reported.
The matched difference is computed in each dataset.

As a descriptive decomposition, the home unmatched difference of the original model is also reported in four forms:
all recordings; training recordings only; training recordings without MiniBest and rest windows; and the matched difference.
The unmatched difference in the DeFOG test recordings is reported separately, with the note that MiniBest windows could not be removed there.

## B. Personal-baseline model
Only paired pvs are used. At home, only training recordings are used, and MiniBest and rest windows are removed before any other step.
Recordings then need at least 3 windows. Within each leave-one-patient-out fold:
1. Missing values are replaced by the training-fold median.
2. Each feature is centred within each pv on the mean of the recording medians. No medication label is used.
3. Features are standardised on the training fold.
4. L2 logistic regression with C = 1.0 is fitted on the same 15 features.
   Weights give each patient equal total weight, each visit equal weight within a patient, and each state equal weight within a visit.

The test patient is centred by the same rule. For a linear model, within-visit differences in logits do not depend on that centring.
B is evaluated only by within-patient statistics: the matched difference (main), the unmatched difference and the pairwise accuracy.

## C. Windows away from FoG (descriptive)
A per-window distance to the nearest annotated FoG sample is computed by a separate script from the raw data, without medication labels.
The original model's logits are restricted to windows at least 30 s from FoG, with 60 s as a sensitivity.
Each recording needs at least 3 such windows. At home, training recordings are used without MiniBest and rest windows.
The unmatched difference is reported if at least 15 paired patients (home) or 20 (laboratory) remain. Otherwise only counts are reported.

As a further descriptive contrast within recordings, windows ending less than 30 s before a FoG onset are compared with windows at least 60 s from FoG, within each state.
At home this comparison is made within the same task. It needs at least 2 windows of each kind per recording or task and at least 15 patients.

## D. Structural reference (disclosure)
For each recording-level leave-one-patient-out AUC reported, the intercept-only AUC and the permutation-null mean are given beside it.

## Tests and multiplicity
The family has four tests: A-lab, A-home, B-lab and B-home.
Each is a one-sided Wilcoxon signed-rank test (matched difference > 0), with Holm adjustment at familywise alpha 0.05.
All other quantities are reported as estimates with 95% bootstrap confidence intervals (1,000 patient resamples, seed 42). Any p values for them are nominal.

## Execution order
1. Commit this amendment.
2. Compute the FoG distances (label-free) and run a dry run with labels shuffled within pv (seed 77).
3. Commit the analysis code and the sha256 values of the distance file and the tasks file.
4. Run the analysis once.

## Reporting
All results are intended to be reported whatever their direction, beside the original results and labelled exploratory.
An effect will be described as observed under a given analysis only if its Holm-adjusted p is below 0.05.
If the home effect is smaller after task matching, this will be reported as a qualification of the original home result.
Coding errors found after the run that do not change this specification will be corrected and logged in the analysis notes.
