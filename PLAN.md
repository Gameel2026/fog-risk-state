# Analysis plan — Levodopa OFF-state detection from freeze-free walking in people with freezing of gait

Version 1. Committed before any medication-state result is computed.
Deviations, if any, will be recorded in dated amendments with their reasons.

## 1. Aim
We intend to test whether the levodopa OFF state, the state with the highest freezing-of-gait (FoG) risk,
can be recognised from walking without freezing, recorded by a single lower-back accelerometer in people with FoG.

## 2. Data
- Public FoG detection contest data (Salomon et al., Nat. Commun. 2024; Zenodo doi:10.5281/zenodo.12565292).
  We intend to use the training recordings and the public and private test recordings.
  FoG labels of the test recordings come from the official solution file. Samples marked "Ignored" are not scored.
  Test files that belong to patients also present in training are kept, because no model trained for the contest is evaluated here.
  - tDCS-FoG: laboratory FoG-provoking protocol, 128 Hz, acceleration in m/s².
  - DeFOG: home FoG-provoking protocol, 100 Hz, acceleration in g.
- Clinical data from subjects.csv: UPDRS-III ON and OFF, NFOG-Q.
- Patients recorded in one medication state only may be used for training but not for paired tests.

### Checks done before this plan (no medication-state result computed)
- In the metadata 44 tDCS-FoG patients and 55 DeFOG patients have ON and OFF recordings within the same visit.
- No patient ID appears in both datasets. No identical profile was found on age, sex, years since diagnosis, UPDRS-III ON and OFF and NFOG-Q.
- Units: the mean acceleration magnitude was 9.84 in tDCS-FoG (m/s²) and 1.00 in DeFOG (g).
- DeFOG: about 30% of samples carry both Valid and Task flags.
- In the training recordings alone, 6-s freeze-free windows (2-s margin) were plentiful in DeFOG (median 65–78 per recording) and scarce in tDCS-FoG (median 10, 62 recordings with fewer than 3).
  Restricted to training recordings, only 67 patients would keep a paired visit. The test recordings are therefore added to reach the full paired cohort.

## 3. Pre-processing (fixed)
- Convert to g and resample both datasets to 64 Hz (primary rate).
- DeFOG training recordings: keep samples with Valid and Task flags true. Test recordings: keep samples not marked "Ignored".
- Remove every annotated FoG episode (any type) with a margin of 2 s before onset and after end.
- Windows of 6 s with 50% overlap.
- Gait gate: a window is kept if its 0.5–3 Hz band power exceeds a fixed threshold and its step-frequency autocorrelation peak exceeds 0.3. The threshold values will be fixed on the pooled data without using medication labels and written to the repository before modelling.
- Recordings with fewer than 3 retained windows are excluded. The number of excluded recordings and patients will be reported.

## 4. Features (fixed list, orientation- and scale-robust where possible)
Per window, from the magnitude and the three axes:
- log band power 0.5–3 Hz and 3–8 Hz, and their ratio
- RMS of vertical, mediolateral and anteroposterior acceleration
- dominant step frequency and its spectral peak power
- harmonic ratios (vertical, anteroposterior, mediolateral)
- autocorrelation step regularity and stride regularity, and their ratio (symmetry)
- normalised jerk (RMS of derivative)
No feature encodes the number of windows, the fraction of retained time or the recording length.

## 5. Model (fixed, no tuning)
- L2-regularised logistic regression, C = 1.0, features standardised on training folds only.
- Training is weighted so that each patient contributes equally.
- Aggregation: median window probability per recording, then mean over a patient's recordings in each state within a visit, then mean over visits.
- Validation: leave-one-patient-out within each dataset. Cross-dataset: a model trained on all of one dataset is applied unchanged to the other. A third configuration pools both datasets under leave-one-patient-out.
- A single-feature baseline (0.5–3 Hz log band power) is computed with the same pipeline.

## 6. Hypotheses and testing order
Fixed-sequence testing at one-sided alpha = 0.05. The confirmatory chain stops at the first non-significant test, and later tests are then reported as exploratory.
- H1 (primary): the estimated OFF probability is higher in a patient's OFF recordings than in the same patient's ON recordings. Pooled over both datasets. One-sided Wilcoxon signed-rank on the per-patient difference. Per-dataset results are reported as replications.
- H2a: recording-level AUC under leave-one-patient-out exceeds 0.5 within each dataset. The p value comes from 1,000 permutations in which the ON and OFF labels are swapped at random within each patient-visit.
- H2b: the cross-dataset AUC exceeds 0.5 in each direction.
- H4: the per-patient OFF−ON difference in estimated probability correlates positively with the UPDRS-III OFF−ON difference (Spearman, one-sided).
Confidence intervals come from 1,000 bootstrap resamples of patients.

## 7. Control and robustness analyses (planned, reported in all cases)
- Split control: the same pipeline with a random recording-level split in which one patient can appear in training and testing, compared with leave-one-patient-out.
- Negative control: ON and OFF labels shuffled within each patient-visit before training. Performance is expected to fall to chance.
- Order: in DeFOG the OFF recording precedes the ON recording. Within each state we will test whether the estimated OFF probability trends with recording order or time.
- Slowness: the single-feature baseline above, and a task-matched comparison where task labels allow it.
- Residual FoG: margins of 0 s and 5 s, and recordings without any annotated FoG.
- An operating point: sensitivity at 80% specificity, with the threshold chosen on training folds.

## 8. Exploratory analyses (labelled as such)
- Within-patient concordance versus between-patient AUC.
- Sampling rate (64, 32, 16 Hz) and number of windows per recording (1, 2, 4, 8, all) versus performance, with estimated energy.
- Within-patient paired effect sizes (dz) of each feature in both datasets.
- Temporally separated FoG link in DeFOG: estimated OFF probability from earlier freeze-free walking versus FoG burden later in the same session, within each medication state.
- Other sensitivity: window 10 s; model without absolute-amplitude features; mean-logit aggregation.
- Embedded: C implementation of the feature extractor and the frozen model for a Cortex-M4F, decision agreement with Python in an ARM emulator, flash, RAM and instructions per window, and estimated energy from datasheet values.

## 9. Fallback
If H1 is not supported, within-patient paired tests of each listed feature (Benjamini–Hochberg across features) will be reported in both datasets as the main descriptive result.

## 10. Known limitations to be reported
- In DeFOG the OFF recording precedes the ON recording on the same day, so order effects cannot be fully separated from medication.
- tDCS-FoG recordings come from a stimulation trial. ON and OFF are paired within visit so that stimulation is common to both.
- Accelerometer only; turns cannot be removed reliably.
- Energy values are estimates and not hardware measurements.
