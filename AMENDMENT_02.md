# Amendment 02 — replication on unseen DeFOG visits

Dated 4 November 2026. Written and committed before any window, feature, score or outcome of the recordings below was computed.
PLAN.md (version 1), Amendment 01 and all reported results are unchanged.
This amendment adds one pre-specified replication test and a short fixed sequence of secondary tests. Other quantities are descriptive. Later changes, if needed, will be recorded in dated amendments with their reasons.

## Reasons
The contest data folder `train/notype` holds 46 DeFOG recordings that the original pipeline did not read.
`p1_build_windows.py` reads only the train, public and private folders of tDCS-FoG and DeFOG.
These recordings carry a FoG indicator (`Event`) without FoG type, together with the `Valid` and `Task` flags.
Their task intervals are in the official tasks file.
They allow the home within-patient result to be tested again on visits that were not used in any earlier analysis.

## What was examined before this amendment
- Metadata only (`defog_metadata.csv`, `tasks.csv`, `subjects.csv`).
  - 46 recordings from 25 subjects, all of them already in the DeFOG cohort.
  - 38 recordings form 19 subject-visits (all Visit 2) with one OFF and one ON recording each. None of these subject-visits was used before.
  - The other 8 recordings are ON recordings of subject-visits whose OFF recording was already analysed (six from Visit 1, two from Visit 2).
- Task durations per state from `tasks.csv`.
- The header, and the column means of one of the 8 excluded recordings (8228c6fdee). No medication information was used.
- No signal-derived feature, window, score or outcome of these recordings was computed.

## Recordings
- **Replication set:** the 38 recordings of the 19 unused subject-visits (Visit 2, one OFF and one ON each).
- **The 8 ON recordings of already-used subject-visits are not part of the replication set.** Their OFF partners were analysed before. They are reported descriptively only.

## Pre-processing (unchanged rules)
- Acceleration is already in g at 100 Hz. It is resampled to 64 Hz with the same code as `p1_build_windows.py`.
- Scored samples: `Valid` and `Task` both true. The flags are parsed as text equal to "true", ignoring case.
- FoG: `Event` > 0. Every annotated FoG sample is removed with a 2-s margin.
- Windows of 6 s with 50% overlap. The same 15 features (`p_features.py`).
- The frozen gait gate: log 0.5–3 Hz power above 1.6852 and step autocorrelation peak above 0.3. It is not recomputed.
- Recordings need at least 3 gated windows.

## Strata (as in Amendment 01 A)
- A window is assigned to a task only if it lies wholly within one task interval of `tasks.csv`.
- MiniBest items (`MB*`) and rest periods (`Rest*`) are excluded.
- Strata are exact task names, so cued versions are separate strata.
- A stratum is used only if it has at least 2 windows in each state. Its score is the median window logit in that stratum and state.
- Matched difference of a subject-visit: mean over usable strata of (OFF score − ON score). Each subject has one replication visit, so this is also the patient value.
- Fallback, applied only if fewer than 15 patients have a usable stratum under exact names: cued and uncued versions of a task are merged, and Hotspot1 is merged with Hotspot2.

## Models (no retraining choices, no tuning)
- **Primary model:** the original home leave-one-patient-out model of each subject.
  - It is trained on all original DeFOG windows of all other patients, exactly as in the original analysis.
  - It is applied unchanged to that subject's new visit.
  - The test subject therefore contributes no data to training, and the visit is new.
  - Reproduction check: the refitted fold models must reproduce `pred_defog.csv` for the subject's original recordings to within 1e-6. Otherwise the analysis stops and the discrepancy is reported.
- **Secondary model S1:** the laboratory-trained transfer model. It is trained on all tDCS-FoG windows as in the original analysis and applied unchanged.
- **Secondary model S2:** the Amendment 01 personal-baseline model.
  - It is trained, as in Amendment 01 B, on the DeFOG training recordings of all other patients, without MiniBest and rest windows.
  - The new visit is centred by the same label-free rule after removal of MiniBest and rest windows.
- **Descriptive only:** the pooled full-data model used for the Cortex-M4F port. It was trained on these subjects' earlier visits.

## Tests
- **Primary test:** one-sided Wilcoxon signed-rank test that the matched difference of the primary model is greater than zero. Alpha 0.05, using SciPy's default exact distribution for up to 25 patients.
  - It is run if at least 10 patients have a usable stratum. Otherwise only counts are reported.
- **Fixed sequence after the primary test:** S1 and then S2, each a one-sided Wilcoxon test of the matched difference at alpha 0.05.
  - A secondary test is interpreted as confirmatory only if the primary test and every earlier test in the sequence were significant.
  - Otherwise it is reported as exploratory with its nominal p value.
- **Descriptive, with 95% patient-bootstrap CIs (1,000 resamples, seed 42):**
  - number of patients with OFF > ON;
  - median difference and dz;
  - within-visit pairwise accuracy;
  - the unmatched difference after MiniBest and rest removal;
  - the deployment model;
  - the 8 excluded ON recordings paired with their earlier OFF recordings.

## Reporting
- All results are intended to be reported whatever their direction, beside the earlier results.
- The replication is described as a test on new visits of patients from the same cohort, not on an independent cohort.
- A replication will be described as observed only if the primary test is significant.
- Coding errors found after the run that do not change this specification will be corrected and logged in the analysis notes.

## Execution order
1. Commit this amendment.
2. Write `p1c_build_notype.py` and `p3_amend02.py`.
3. Run `p1c_build_notype.py`. It is label-free.
4. Run `p3_amend02.py --dryrun`. Labels are swapped within each subject-visit (seed 77), and the run reports counts only.
5. Commit the code and the sha256 of the new window file.
6. Run the analysis once.
