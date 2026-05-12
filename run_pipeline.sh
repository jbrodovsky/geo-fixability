#!/usr/bin/env bash
# Full geo-fixability experiment pipeline.
#
# Steps:
#   1. Unified parameter sweep            → data/sweep/sweep_results.csv
#   2. PF experiment sweep                → data/experiments/pf_experiment_results.csv
#   3. Q-validation figure                → paper/fig_q_validation_*.{pdf,png}
#   4. PF results figures                 → data/analysis/pf/
#   5. Publication figures (plot-sweep)   → paper/figures/
#   6. Model training & analysis          → data/analysis/
#      (analyze, ablation, crossval, quantile, scaling-law)
#   7. Prediction plots                   → data/predictions/
#   8. Real-terrain validation            → data/validation/
#   9. Correction framework               → data/validation/
#
# Usage:
#   bash run_pipeline.sh                  # full run, default settings
#   bash run_pipeline.sh --workers 12     # pass extra flags to sweep runners
#
# Extra flags are forwarded only to the two sweep runners (steps 1 & 2).
# Edit the variables below to tune output directories and model settings.
set -euo pipefail
cd "$(dirname "$0")"

SWEEP_CSV="data/sweep/sweep_results.csv"
ANALYSIS_OUTDIR="data/analysis"
ANALYSIS_MODEL="${ANALYSIS_OUTDIR}/model.pkl"
PRED_OUTDIR="data/predictions"
VAL_OUTDIR="data/validation"

# ── Step 1: Unified parameter sweep ────────────────────────────────────────
echo ""
echo "=== [1/9] Unified parameter sweep ==="
uv run run-sweep --config sweep_config.yaml "$@"

# ── Step 2: PF experiment sweep ─────────────────────────────────────────────
echo ""
echo "=== [2/9] Particle Filter experiment ==="
uv run run-pf-experiment --config sweep_config.yaml "$@"

# ── Step 3: Q-validation figure ─────────────────────────────────────────────
echo ""
echo "=== [3/9] Q-validation plots ==="
uv run python src/geo_fixability/scripts/plot_q_validation.py

# ── Step 4: PF results figures ───────────────────────────────────────────────
echo ""
echo "=== [4/9] PF results figures ==="
uv run plot-pf-results

# ── Step 5: Publication figures ──────────────────────────────────────────────
echo ""
echo "=== [5/9] Publication figures ==="
uv run plot-sweep \
    --csv "${SWEEP_CSV}" \
    --outdir paper/figures

# ── Step 6a: Train model + EDA ──────────────────────────────────────────────
echo ""
echo "=== [6a/9] Model training (analyze) ==="
uv run analyze-results analyze \
    --csv "${SWEEP_CSV}" \
    --outdir "${ANALYSIS_OUTDIR}" \
    --model-type xgb

# ── Step 6b: Feature ablation ────────────────────────────────────────────────
echo ""
echo "=== [6b/9] Feature ablation ==="
uv run analyze-results ablation \
    --csv "${SWEEP_CSV}" \
    --outdir "${ANALYSIS_OUTDIR}"

# ── Step 6c: Cross-validation ────────────────────────────────────────────────
echo ""
echo "=== [6c/9] Cross-validation ==="
uv run analyze-results crossval \
    --csv "${SWEEP_CSV}" \
    --outdir "${ANALYSIS_OUTDIR}"

# ── Step 6d: Quantile regression ─────────────────────────────────────────────
echo ""
echo "=== [6d/9] Quantile regression ==="
uv run analyze-results quantile \
    --csv "${SWEEP_CSV}" \
    --outdir "${ANALYSIS_OUTDIR}"

# ── Step 6e: Scaling law ─────────────────────────────────────────────────────
echo ""
echo "=== [6e/9] Scaling law ==="
uv run analyze-results scaling-law \
    --csv "${SWEEP_CSV}" \
    --outdir "${ANALYSIS_OUTDIR}"

# ── Step 7: Prediction plots (all presets, with slices) ──────────────────────
echo ""
echo "=== [7/9] Prediction plots (all presets) ==="
uv run analyze-results predict-presets \
    --model "${ANALYSIS_MODEL}" \
    --outdir "${PRED_OUTDIR}" \
    --plot-slices

# ── Step 8: Real-terrain validation ──────────────────────────────────────────
echo ""
echo "=== [8/9] Real-terrain PCRB validation ==="
uv run validate-terrain \
    --model "${ANALYSIS_MODEL}" \
    --outdir "${VAL_OUTDIR}"

# ── Step 9: Correction framework ─────────────────────────────────────────────
echo ""
echo "=== [9/9] Diagnostic correction framework ==="
uv run validate-terrain correction \
    --model "${ANALYSIS_MODEL}" \
    --outdir "${VAL_OUTDIR}"

echo ""
echo "=== Pipeline complete ==="
echo "  Sweep CSV  : ${SWEEP_CSV}"
echo "  PF CSV     : data/experiments/pf_experiment_results.csv"
echo "  Q-val figs : paper/figures/q-val/"
echo "  PF figs    : data/analysis/pf/"
echo "  Pub figs   : paper/figures/"
echo "  Analysis   : ${ANALYSIS_OUTDIR}/"
echo "  Predictions: ${PRED_OUTDIR}/"
echo "  Validation : ${VAL_OUTDIR}/"

# ── Step 10: Update paper with new results ────────────────────────────────────
echo ""
echo "=== [10/10] Updating paper with new results ==="
PAPER_FILE="paper/draft.tex"
claude -p "The geo-fixability experiment pipeline has just finished. Please update or write the paper at paper/draft.tex with the new quantitative results. This paper is intended for submission to a top-tier IEEE journal or conference (probably IEEE TAES), but the exact venue is not yet decided. The paper should be structured like a typical research paper with sections such as Introduction, Background, Methodology, Results, Discussion, and Conclusion. The paper should clearly describe the experimental setup, the results obtained from the new outputs, and the implications of these results for the field of geophysical navigation and robotics. You may use the references in paper/references.bib to support your claims and provide context. Please ensure that all numerical results, tables, and quantitative claims in the paper are updated to reflect the new outputs from the pipeline.

New outputs to incorporate:
  - ${ANALYSIS_OUTDIR}/          model performance, ablation, cross-validation, quantile coverage, scaling law
  - ${VAL_OUTDIR}/               real-terrain validation and correction framework results
  - ${PRED_OUTDIR}/              prediction outputs
  - paper/figures/               updated figures from the sweep and PF experiments"
