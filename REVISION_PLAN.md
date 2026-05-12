# Revision Plan: Geophysical Navigation Fixability Chapter

**Status**: Draft revision roadmap addressing editorial feedback
**Target**: Strengthen real-terrain generalization narrative and add operational rigor
**Effort Estimate**: 25-35 hours total work
**Timeline**: 2-3 weeks working 2-3 hours/day

---

## Executive Summary

The editorial review identifies a critical gap: the chapter achieves R²=0.947 on synthetic data but fails on real terrain (R²=-1.42), with systematic over-optimism. The revision reframes the work from "PCRB predictor" to "screening heuristic" and:

1. **Paper-level changes**: Restructure to lead with distribution-shift finding; add principled uncertainty and correction framework
2. **Code/analysis changes**: Implement ablations, quantile regression, and diagnostic-driven correction factors
3. **Scope clarification**: Document valid parameter ranges, known failure modes, and remediation roadmap

---

## Phase 1: Reframe and Restructure (Paper-Only, ~6 hours)

These tasks clarify the narrative without requiring new experiments.

### Task #1: Revise Abstract (Priority: CRITICAL, Effort: 1 hr)
**Why first**: Reader's first impression; sets tone for entire chapter
- Current framing hides real-terrain failure until Section 4.4
- **Action**: Lead with synthetic success (R²=0.947), immediately reveal real-terrain gap (R²=-1.42)
- **Key change**: "...achieving R² = 0.947 on synthetic data. Real-terrain validation reveals systematic over-optimism (R² = −1.42) due to sub-patch heterogeneity, non-power-law spectral structure, and normalization sensitivity. The model succeeds as a conservative screening heuristic for synthetic-like terrain."
- **Dependencies**: None
- **Output**: Revised abstract (1-2 paragraphs)

### Task #2: Restructure Section 4 Results (Priority: HIGH, Effort: 2 hrs)
**Why**: Real-terrain findings must not be buried on page 17-18
- Current order: synthetic PCRB → particle filter → regression on synthetic → real-terrain results
- **Action**: Move Table 3 (real-terrain validation) to immediately after synthetic regression model performance
- **New Section 4 structure**:
  - 4.1: Parameter Sweep Summary (synthetic only)
  - 4.2: Particle Filter Empirical Validation (synthetic only)
  - 4.3: Regression Model Performance
    - 4.3a: Synthetic test set (R²=0.947)
    - **4.3b: Real-terrain validation [MOVED FROM 4.4]** (R²=-1.42, Table 3)
  - 4.4: Real-terrain predictions and landscape maps (page 17 content shifted down)
- **Dependencies**: None (pure reorganization)
- **Output**: Revised Section 4 with updated page numbers in cross-references

### Task #3: Add "Limitations & Remediation Roadmap" (Priority: HIGH, Effort: 3 hrs)
**Why**: Diagnostic findings (Section 5.3) need concrete, testable paths forward
- Current: Three failure modes identified; solutions hand-waved
- **Action**: Create Section 5.9 with structure:
  - **Valid parameter ranges**: σ₀ ∈ {100, 500, 1000, 2000} m (note: σ₀=100 m uses 4-D model), T ≤ 3600 s, Δx ∈ {10...1000} m, star-pattern only, σ_meas = 0.1 field units
  - **Known failure modes** (quantified severity):
    - Sub-patch heterogeneity: 4-10× errors on patchy terrain (diagnose by computing local vs. global gradient RMS)
    - Non-power-law PSD: ~3-5× error on terrain with concentrated ridgeline features (diagnose via power-law goodness-of-fit)
    - Normalization sensitivity: extreme outliers compress weak features (diagnose via skewness and clipping effect)
  - **Remediation strategies** (each with testable hypothesis):
    1. Mixed synthetic-real training (expect 2× bias reduction)
    2. Local sliding-window features (expect improved correlation and ~3-5× better real-terrain R²)
    3. Patch heterogeneity metric (expect to capture Everest-Amazon variance)
    4. Uncertainty quantification (expectation: bounds contain 90% of actual errors)
- **Dependencies**: None (pure writing; informs later code tasks)
- **Output**: New subsection (~1.5 pages) clearly stating what future work should test

---

## Phase 2: Code-Level Analysis Tasks (~12 hours)

These require new experiments and retraining models. Run these to support revised narrative.

### Task #4: Ablation Study on Feature Categories (Priority: HIGH, Effort: 2 hrs)
**Why**: Show which feature groups drive R²=0.947; clarify if spectral features (I, D, L) justify complexity
- **Action**: Train XGBoost with feature groups dropped:
  - Baseline: all 9 features, R²_baseline = 0.947
  - Ablate 1: map features only → R²
  - Ablate 2: trajectory features only → R²
  - Ablate 3: spectral features (I, D, L) only → R²
  - Ablate 4: drop I, D, L; keep grad_rms → R² (critical: does spectral info help beyond gradient?)
- **Code changes**: ~50 lines (loop over feature subsets, retrain, log R²)
- **Dependencies**: Task #3 (provides context for why this matters)
- **Wall time**: ~50 min (training 5 models × ~10 min each)
- **Output**: 
  - Figure: ablation bar chart (ΔR² relative to baseline)
  - Table: feature groups and their ±ΔR² contribution
  - **Finding**: Likely shows grad_rms dominates; spectral features are secondary corrections
- **Use in paper**: Section 5.5 (Feature Dominance), replace/enhance current importance plot

### Task #5: Cross-Validation Bias Analysis (Priority: MEDIUM, Effort: 1.5 hrs)
**Why**: Random 80/20 split has training neighbors in parameter space; inflates apparent generalization
- **Action**: Two experiments:
  1. Random holdout: repeat 5 times with different seeds → report R² variance
  2. Configuration-stratified holdout: withhold entire levels of key parameters
     - Withhold all Δx=1000 m (coarse resolution)
     - Withhold all β=3.5 (very smooth terrain)
     - Withhold all σ₀=2000 m (high uncertainty)
  3. Compare R² across strategies
- **Code changes**: ~100 lines (stratified split utilities, rerun validation)
- **Dependencies**: None (uses existing trained model)
- **Wall time**: ~60 min (retraining 4 models)
- **Output**:
  - Table: "Generalization by split strategy" showing interpolation vs. extrapolation R²
  - **Finding**: Likely reveals 20-30% R² drop on stratified holdout (model doesn't extrapolate coarse resolution)
- **Use in paper**: Section 4.3a, revised model-performance text: "On interpolation tasks, R²=0.947; on extrapolation (e.g., unseen resolutions), R² is lower"

### Task #6: Fit Empirical Scaling Law (Priority: MEDIUM, Effort: 1.5 hrs)
**Why**: Eq. 12 is currently proposed but not validated; compare 2-feature power law to 9-feature ML model
- **Action**: Fit CEP ≈ C · σ₀^α / (g_rms · d_total)^γ
  - Log-transform and linear regression to extract (C, α, γ)
  - Report R² on synthetic test set
  - Compare scaling-law R² to XGBoost R²
  - Test both on real terrain; which generalizes better?
- **Code changes**: ~200 lines (power-law fitting, comparison analysis)
- **Dependencies**: None
- **Wall time**: ~45 min
- **Output**:
  - Figure: log-log plot of CEP vs. (g_rms · d_total) with fitted power law
  - Table: fitted coefficients (C, α, γ with ±95% CI)
  - **Finding**: Likely shows scaling law captures ~85-90% of XGBoost R² (simplicity vs. accuracy tradeoff)
- **Use in paper**: Section 5.5 (Feature Dominance), replace current hand-waved Eq. 12 with fitted coefficients and explicit comparison

### Task #7: Quantile Regression for Uncertainty (Priority: HIGH, Effort: 2.5 hrs)
**Why**: Operational use needs confidence bounds; point estimates are insufficient
- **Action**: Train quantile regression XGBoost (0.05, 0.50, 0.95 quantiles)
  - Retrain three models with quantile objectives
  - Evaluate on synthetic test set: plot median ± 90% band
  - Compute empirical coverage (% of actual CEP within band)
  - Apply to real-terrain scenarios; compare band widths across regions
- **Code changes**: ~300 lines (quantile trainer, visualization, coverage metrics)
- **Dependencies**: None (uses existing features)
- **Wall time**: ~90 min (training 3 quantile models)
- **Output**:
  - Figure: predicted vs. actual CEP with confidence band (replaces Figure 10)
  - Figure: coverage analysis (what % of test samples fall within 90% band?)
  - Table: mean interval width by CEP magnitude (e.g., CEP 1 m: ±0.5 m band; CEP 100 m: ±40 m band)
- **Use in paper**: Section 5.4 (Operational Screening), update protocol: "Use 0.95 quantile for conservative screening; 0.05 for optimistic bound"

### Task #8: Normalization Sensitivity & Heterogeneity Features (Priority: HIGH, Effort: 3 hrs)
**Why**: Two key failure modes identified in Section 5.3; quantify and propose fixes
- **Part A: Normalization sensitivity** (~90 min)
  - For 7 SRTM tiles, compute: skewness, outlier fraction, effective gradient RMS after clipping
  - Repredict with clipped normalizer vs. robust scaler
  - Compare to standard normalizer
  - **Finding**: Likely shows 30-50% error reduction on Amazon/Sahara with robust scaler
  
- **Part B: Heterogeneity metric** (~90 min)
  - Define: partition each patch into 4×4 sub-windows; compute gradient RMS variance
  - Add as 10th feature to model
  - Retrain XGBoost with 10 features
  - Test on real terrain: does heterogeneity help predict error?
  
- **Code changes**: ~400 lines (heterogeneity computation, alternative normalizers, retraining)
- **Dependencies**: Task #3 (context for why this matters)
- **Wall time**: ~120 min (feature eng. + retraining + validation)
- **Output**:
  - Figure: heterogeneity vs. prediction error on real terrain
  - Table: model performance (9-feature baseline vs. 10-feature with heterogeneity + normalizer choice)
  - **Finding**: Likely shows heterogeneity metric helps on patchy terrain; robust normalizer improves Amazon case
- **Use in paper**: 
  - Section 5.3 (Distribution Shift): add quantified diagnostics
  - New appendix section: "Proposed robustness improvements" with experimental results

### Task #9: Computational Complexity (Priority: LOW, Effort: 1 hr)
**Why**: Practitioners need to know deployment feasibility; currently vague "<1 second per tile"
- **Action**:
  - Add timing instrumentation to `analyze-results predict` pipeline
  - Benchmark on commodity hardware (laptop CPU)
  - Report: FFT time, feature extraction time, model inference time
  - Big-O complexity for each stage
- **Code changes**: ~150 lines (timing decorators, benchmarking harness)
- **Dependencies**: None
- **Wall time**: ~30 min (instrumentation + running benchmarks)
- **Output**:
  - Table: timing breakdown for 50×50 km tile at 30 m resolution
  - Equation: O(N² log N) for FFT, O(depth × n_trees) for inference
  - **Finding**: <1 s is achievable on CPU; scales linearly with tile size
- **Use in paper**: Section 3.5, new subsection "Computational Requirements"

---

## Phase 3: Principled Uncertainty Framework (~8 hours)

Most important for operational viability. Addresses the crude "×3.8 correction factor."

### Task #10: Expand Trajectory-Geometry Discussion (Priority: MEDIUM, Effort: 2 hrs)
**Why**: Star-pattern assumption is critical but currently buried on page 28
- **Action**:
  - Move from Section 5.7 to Section 3.2 (Trajectory Generation)
  - Show math: for straight-line heading ψ, effective gradient info ∝ g_rms(ψ); star covers all ψ uniformly
  - Quantify impact: if anisotropy r=2, straight-line mission predicted 2× better than reality
  - Propose future experiment (not executed): re-run PCRB sweep with straight-line trajectories
- **Code changes**: None (pure writing)
- **Dependencies**: None
- **Output**: Expanded trajectory discussion with equations and example from SRTM tiles
- **Use in paper**: Section 3.2 + Introduction (as stated limitation) + Section 5.7 (future work)

### Task #15: Principled Real-Terrain Correction Framework (Priority: CRITICAL, Effort: 4 hrs)
**Why**: Current "×3.8 correction" is ad-hoc and opaque; operationally dangerous
- **Action**: Develop diagnostic-driven correction factor model
  1. Compute per-region correction factors from Table 3: f_Everest ≈ 2.0×, f_Amazon ≈ 10.3×, etc.
  2. Identify terrain diagnostics: skewness, heterogeneity, power-law fit quality
  3. Train lightweight regression (linear or shallow tree) to predict f from diagnostics
  4. New pipeline: raw_CEP_pred → terrain_diagnostics → correction_factor → corrected_CEP
  
- **Code changes**: ~500 lines (diagnostic extraction, correction model training, pipeline integration)
- **Dependencies**: Task #8 (heterogeneity feature), conceptually depends on Task #3 (diagnostics)
- **Wall time**: ~150 min (feature eng. + model training + validation)
- **Output**:
  - Correction factor model artifact (scikit-learn pickle)
  - Figure: correction factor vs. terrain diagnostics (scatter with fit line)
  - Table 3 extended: per-region correction factors (2.0-10.3× range, not flat 3.8×)
  - Operational output: "Predicted CEP = 50 m; terrain diagnostics → correction factor 2.5; corrected CEP = 125 m [with uncertainty]"
- **Use in paper**: Completely rewrite Section 5.4 (Operational Screening) with principled framework

### Task #11: Expand Related Work (Priority: MEDIUM, Effort: 2 hrs)
**Why**: Current coverage is 1 page; misses modern approaches and positions contribution poorly
- **Action**: 
  - Dedicated "Related Work" section (or Section 2.5)
  - Cover: TERCOM/SITAN, particle filters, recent deep learning, PCRB bounds, info-theoretic mission planning, magnetic/gravity methods
  - Clearly state gap: "No prior work systematically maps (map spectral properties, trajectory parameters) → PCRB via ML"
- **Code changes**: None (literature review + writing)
- **Dependencies**: None
- **Output**: ~1.5 page section positioning this work
- **Use in paper**: Introduction → Related Work → Background flow

---

## Phase 4: Documentation & Polish (~4 hours)

### Task #12: Clarify Fixed-Q Validity Regime (Priority: MEDIUM, Effort: 1 hr)
**Why**: Section 5.8 is tacked on; reader left unsure if results are valid
- **Action**: Move to Section 2.1 (Problem Formulation); state validity condition upfront
  - "Fixed-Q model valid when σ₀ > σ_a t^(3/2); for our swept σ₀ ∈ {500, 1000, 2000} m and T ≤ 2400 s, condition holds"
  - σ₀ = 100 m cases use 4-D velocity-state model (footnote with reference)
  - Future work: extend to low-uncertainty regime
- **Code changes**: Audit PCRB code to confirm it switches to 4-D for σ₀ < 390 m if needed
- **Dependencies**: Task #3 (scope context)
- **Output**: Moved/expanded discussion in Section 2.1 + footnote

### Task #13: Improve Figures & Captions (Priority: LOW, Effort: 1.5 hrs)
**Why**: Some figures (esp. 17, 3) have dense captions; operational figures lack uncertainty
- **Action**:
  - Figure 17: add uncertainty visualization (desaturated color for low-confidence regions)
  - Figure 3: break dense caption into summary + findings + notation
  - Figure 10 (or replaced by quantile version): add confidence band and coverage metric
  - Regenerate all with Task #7 quantile regression results
- **Code changes**: ~200 lines (visualization utilities)
- **Dependencies**: Task #7 (quantile model), Task #15 (confidence framework)
- **Output**: Revised 4-5 figures with improved captions

### Task #14: Restructure Conclusion & Chapter Summary (Priority: LOW, Effort: 1.5 hrs)
**Why**: Sections 6-7 are redundant; no clear role for Chapter Summary
- **Action**: Reframe Section 7 as dissertation-level context
  - 7.1: Results summary (brief)
  - 7.2: Contribution to dissertation context (NEW: how this chapter connects to other chapters)
  - 7.3: Open problems and extensions (NEW: natural next steps)
  - 7.4: Practitioner takeaways (NEW: actionable summary)
- **Dependencies**: All earlier tasks (adds context from revised narrative)
- **Output**: Restructured Sections 6-7 with distinct value for each

### Task #16: Documentation & Deployment Guide (Priority: LOW, Effort: 1 hr)
**Why**: Code is now complex (multiple models, diagnostics); users need guidance
- **Action**:
  - Update docstrings for all new functions (ablation, quantile, heterogeneity, correction)
  - Create `DEPLOYMENT.md` with operational guide
  - Add example notebooks showing: (1) quick screening, (2) conservative planning, (3) detailed analysis
  - Update README with model evolution and performance summary
- **Code changes**: ~300 lines (docstrings, examples, guides)
- **Dependencies**: All code tasks (documents the completed work)
- **Output**: `DEPLOYMENT.md`, updated `README.md`, 3-5 example notebooks

---

## Execution Order & Dependencies

### Week 1 (Paper Restructuring + Key Analysis)
1. **Task #1** (1 hr) — Revise abstract; unblocks reading of later feedback
2. **Task #2** (2 hrs) — Restructure Section 4
3. **Task #3** (3 hrs) — Limitations & Remediation Roadmap
4. **Task #11** (2 hrs) — Expand related work
5. **Task #4** (2 hrs) — Ablation study [PARALLEL with above]
6. **Task #6** (1.5 hrs) — Fit scaling law [PARALLEL]

### Week 2 (Uncertainty Framework + Diagnostics)
7. **Task #7** (2.5 hrs) — Quantile regression [HIGH PRIORITY]
8. **Task #8** (3 hrs) — Normalization + heterogeneity [HIGH PRIORITY]
9. **Task #5** (1.5 hrs) — Cross-validation bias [PARALLEL]
10. **Task #15** (4 hrs) — Correction framework [CRITICAL, depends on #8]
11. **Task #9** (1 hr) — Computational complexity [PARALLEL]

### Week 3 (Polish & Documentation)
12. **Task #10** (2 hrs) — Trajectory geometry
13. **Task #12** (1 hr) — Fixed-Q clarity
14. **Task #13** (1.5 hrs) — Figures & captions [depends on #7, #15]
15. **Task #14** (1.5 hrs) — Conclusion restructure
16. **Task #16** (1 hr) — Documentation

---

## Critical Path (Minimum Viable Revision)

If time is limited, focus on these in order:
1. **Task #1** (abstract reframing) — 1 hr
2. **Task #2** (restructure Section 4) — 2 hrs
3. **Task #3** (Limitations & Remediation) — 3 hrs
4. **Task #7** (quantile regression) — 2.5 hrs
5. **Task #15** (correction framework) — 4 hrs
6. **Task #8** (heterogeneity diagnostics) — 3 hrs

**Total**: 15.5 hours (achievable in 1 week at 2-3 hrs/day)

This minimum covers:
- ✅ Front-load real-terrain failure in narrative
- ✅ Add uncertainty quantification
- ✅ Replace ad-hoc correction with principled framework
- ✅ Identify and quantify failure modes

---

## Success Criteria

After revision, chapter should satisfy:

1. **Narrative clarity**
   - Abstract leads with distribution shift (not buried on page 18)
   - Real-terrain results appear immediately after regression model performance
   - Limitations & Remediation roadmap gives actionable next steps

2. **Operational soundness**
   - Correction framework is diagnostic-driven, not ad-hoc
   - Uncertainty bounds quantify prediction confidence
   - Valid parameter ranges explicitly stated with caveats

3. **Analytical depth**
   - Feature ablation shows which features matter (addresses "why 9 features?")
   - Generalization bias quantified (interpolation vs. extrapolation)
   - Failure modes have quantified severity and diagnostic features

4. **Practitioner guidance**
   - Deployment guide shows how to use model operationally
   - Examples for quick screening, conservative planning, detailed analysis
   - Clear statement: "This is a screening heuristic, not a deployed predictor"

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Task #7 (quantile) takes longer than estimated | Pre-code XGBoost quantile wrapper; test on simple dataset first |
| Task #15 (correction framework) has insufficient real-terrain data to train secondary model | Use SRTM tiles only (7 regions × 90 scenarios = 630 data points; enough for lightweight regression) |
| New analysis reveals XGBoost model quality is worse than expected | Fallback: document findings honestly; frame as "screening heuristic requires validation" |
| Dependencies between tasks cause delays | Keep paper drafts in separate branches; merge after code results are in |
| Time pressure forces incomplete work | Prioritize critical path (Tasks 1, 2, 3, 7, 15, 8) over nice-to-haves (11, 13, 14) |

---

## Deliverables Checklist

- [ ] Revised abstract with upfront limitation statement
- [ ] Restructured Section 4 with real-terrain results earlier
- [ ] New Section 5.9: Limitations & Remediation Roadmap
- [ ] Ablation study results (Figure: feature importance by category)
- [ ] Cross-validation analysis (Table: interpolation vs. extrapolation R²)
- [ ] Empirical scaling law fitted (Eq. 12 with coefficients)
- [ ] Quantile regression model and uncertainty figure
- [ ] Heterogeneity feature and robustness analysis
- [ ] Diagnostic-driven correction framework model
- [ ] Expanded real-terrain Section 5.4 (Operational Screening)
- [ ] Expanded trajectory-geometry discussion
- [ ] Expanded related work section
- [ ] Improved figures with uncertainty visualization
- [ ] Restructured conclusion and chapter summary
- [ ] Deployment guide and example notebooks
- [ ] Updated code documentation
- [ ] All code integrated into `geo-fixability` package with tests

