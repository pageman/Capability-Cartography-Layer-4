# Minimal End-to-End Credibility Roadmap

This roadmap defines the smallest package that materially strengthens the Tao-facing claim in Capability Cartography Layer 4.

## Goal

Build one complete, reproducible case study that shows:

1. a pre-training forecast registered before optimization,
2. a real checkpointed training run,
3. coarse evaluation masking smoother internal or fine-grained progress,
4. mechanistic signal growth across checkpoints,
5. causal intervention via ablation and restoration,
6. one held-out related evaluation split.

## Files Added

- `benchmark/manifest.json`
  Frozen manifest for the minimal benchmark package and success criteria.
- `capability_cartography_layer4/case_study.py`
  Deterministic modular masking case study runner and artifact generator.
- `capability_cartography_layer4/demo_case_study.py`
  CLI entrypoint for generating the end-to-end case study artifacts.
- `tests/test_case_study.py`
  Regression checks for forecast registration, masking evidence, mechanistic growth, and causal validation.

## Artifacts Generated

The case study writes to `artifacts/minimal_case_study/`:

- `forecast_registration.json`
- `checkpoint_metrics.json`
- `causal_validation.json`
- `summary_report.md`

## Success Criteria

The minimal package counts as a success if:

1. the forecast is recorded before training and predicts the eventual curve class,
2. fine-grained error improves substantially before coarse accuracy sharply rises,
3. Fourier-signature and circuit-completeness metrics increase across checkpoints,
4. final ablation causes a large held-out degradation and restoration recovers it,
5. the held-out split shows the same qualitative step-like masking behavior.

## Next Layer After This Package

This package is intentionally narrow. After it lands, the next work should be:

1. replace the toy modular learner with a real model checkpoint pipeline,
2. add metric-ablation suites beyond one thresholded task,
3. replace heuristic circuit extraction with actual ACDC or activation patching,
4. add frozen benchmark splits and metadata baselines for out-of-sample forecasting.
