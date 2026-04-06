# Capability Cartography Layer 4: Narrow Predictive & Mechanistic Benchmark

**Capability Cartography Layer 4 (CCL4)** is currently best read as an evidence-backed narrow benchmark package. While Layer 3 provided causal explanations for past failures, this repo now focuses on two bounded directions:

1.  **Predictive (The "When")**: A regime-theoretic heuristic for forecasting trajectory classes *before* training on a frozen small benchmark.
2.  **Mechanistic (The "How")**: A narrow checkpointed case-study pipeline for tracking proxy circuit formation and testing simple causal interventions.

## The Layer Evolution

| Layer | Question | Verb | Key Addition |
|-------|----------|------|-------------|
| **1** | "What happened?" | **Measures** | Schemas, sweeps, surfaces, validation, falsifiable laws |
| **2** | "What kind of failure is this?" | **Classifies** | Failure atlas, visualization, agent briefs |
| **3** | "Why did it fail?" | **Explains** | Causal registry, middle-regime analysis, transfer diagnostics |
| **4** | **"How does it work & When will it scale?"** | **Forecasts & Maps** | **Regime-theoretic forecasting, circuit discovery, grokking analysis, optional quantum-analogy layer** |

## Current Scope

### 1. Regime-Theoretic Forecasting
Using the $(m,r,d,s^*)$ interface, CCL4 classifies the asymptotic regime of a task and predicts its scaling trajectory on a frozen small benchmark:
-   **Power-law**: Gradual accumulation in classical regimes ($r \gg 1$).
-   **Emergent**: Phase transitions in sparse, high-dimensional regimes ($d > m, s^* \ll d$) with high relational depth.
-   **Step-function**: Sudden onset at resource thresholds or huge compressibility gaps ($>20\times$).

### 2. Mechanistic Proxy Mapping
Instead of treating the model as a black box, CCL4 tracks simple mechanistic proxies in checkpointed toy runs:
-   **Fourier-style signals** in periodic modular tasks.
-   **Circuit-completeness proxies** based on sparse weight structure and held-out interventions.

### 3. Optional Quantum Analogy Layer
CCL4 includes an optional interpretive layer that compares some modular-arithmetic circuits to Fourier-style motifs also discussed in quantum algorithms like **Shor's Algorithm** and the **Quantum Fourier Transform (QFT)**. In the current repo this is analogy only, not a validated equivalence claim.

### 4. False vs Real Unpredictability
The repo now explicitly splits the Tao-Keating concern into two narrower categories:
- **False unpredictability**: the apparent jump is mostly created by the observation layer, such as thresholded or coarse scoring.
- **Real unpredictability**: the surprise remains after metric controls, weak precursor signals, or forecast-mechanism feedback.

In the current benchmark package, the periodic modular case is treated as mostly `false unpredictability`, while the sparse relational stress case is treated as a current instance of `real unpredictability`. This split is surfaced in [`artifacts/minimal_suite/suite_summary.json`](/Users/hifi/Capability-Cartography-Layer-4/artifacts/minimal_suite/suite_summary.json).

## What The Repo Demonstrates

### Narrative Arc: From Measurement to Mastery
The current repo demonstrates a narrow package:
- a registered pre-training forecast on frozen toy cases,
- a checkpointed modular case study with metric masking,
- simple held-out causal intervention,
- a two-family small-transformer evidence bundle with one periodic family and one smooth control family,
- baseline comparisons on a frozen small benchmark.
- a root [`verification.yaml`](/Users/hifi/Capability-Cartography-Layer-4/verification.yaml) that records scope, claim coverage, and failure modes for the current evidence artifacts.
- a frozen [`deliverable_manifest.json`](/Users/hifi/Capability-Cartography-Layer-4/artifacts/small_transformer_case/deliverable_manifest.json) for the canonical small-transformer artifact package.

It does **not** yet demonstrate a general solution to Tao’s unpredictability puzzle, frontier-model circuit discovery, or strong multi-domain generalization.

### Methodological Arc: The Dual-Engine Pipeline
The methodology currently implements a narrow dual-engine process:
1.  **Forecasting Engine (Pre-training)**: Utilizes $(m,r,d,s^*)$ and metadata heuristics to predict trajectory classes on a frozen small benchmark.
2.  **Mapping Engine (Post-training)**: Tracks checkpointed proxy signals and runs simple ablation/restoration checks on tiny benchmark models plus a two-family small-transformer bundle.
3.  **Observability Layer**: Writes artifact-level `claim_coverage`, `failure_modes`, and provenance manifests so the repo’s evidence boundary is machine-readable.

## Research Arc

The current research arc is narrower and better instrumented than earlier repo drafts:
- move from broad historical rhetoric to frozen narrow benchmark cases,
- separate `false unpredictability` from `real unpredictability`,
- connect pre-training forecasts to post-training mechanism checks,
- package the strongest current evidence as a canonical small-transformer artifact bundle,
- freeze scope and provenance in [`verification.yaml`](/Users/hifi/Capability-Cartography-Layer-4/verification.yaml) and [`deliverable_manifest.json`](/Users/hifi/Capability-Cartography-Layer-4/artifacts/small_transformer_case/deliverable_manifest.json).

The resulting posture is: evidence-backed narrow forecasting plus narrow mechanism discovery, with explicit failure modes and planned extensions rather than implied generality.

## Project Structure

-   `regime_forecaster.py`: Logic for pre-training trajectory prediction.
-   `circuit_discovery.py`: Methods for mapping capabilities to internal circuits.
-   `orchestration.py`: Unified pipeline for forecasting and verification.
-   `schemas.py`: Data structures for records, forecasts, and circuit definitions.
-   `demo.py`: Verification script across Scaling Laws, Transformers, and Modular Exponentiation.
-   `case_study.py`: Minimal end-to-end Tao-facing case study with registered forecast, checkpoint metrics, masking analysis, and causal validation.
-   `demo_case_study.py`: Entry point that generates the reproducible case-study artifacts in `artifacts/minimal_case_study/`.
-   `benchmark/manifest.json`: Frozen manifest for the minimal benchmark package and success criteria.
-   `benchmark/frozen_cases.json`: Frozen small benchmark cases for the minimum-viable-pass suite.
-   `benchmark/task_families.json`: Scaffold for moving from frozen tiny cases to held-out task-family prediction.
-   `benchmark/baselines.py`: Baselines for the frozen trajectory-class benchmark.
-   `benchmark/metric_ablation.py`: Metric-ablation and falsification-control suite for the minimal case study.
-   `benchmark/run_minimal_suite.py`: End-to-end runner for the small benchmark plus real tiny-model suite.
-   `CLAIMS.md`: Claim ledger separating replicated, preliminary, and hypothetical statements.
-   `verification.yaml`: Root observability artifact describing current evidence scope, claim coverage, and failure modes.
-   `PARAMETERS.md`: Operational definitions and uncertainty notes for `(m, r, d, s*)`.
-   `TRANSFER.md`: Narrow statement of what SplitUP/Schur-inspired transfer does and does not justify in this repo.
-   `ROADMAP.md`: File-by-file roadmap for moving the repo from framework sketch to evidence-backed package.
-   `small_transformer_case.py`: Implemented checkpointed two-family small-transformer benchmark with artifact outputs.
-   `checkpointed_attention_discovery.py`: Narrow discovery utilities for stable attention-route extraction and targeted-vs-random ablation summaries.

## Quick Start

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
python3 capability_cartography_layer4/demo.py
```

For the minimal end-to-end evidence package:

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
python3 capability_cartography_layer4/demo_case_study.py
python3 -m unittest discover -s tests -p 'test_case_study.py'
python3 benchmark/baselines.py
python3 benchmark/metric_ablation.py
python3 benchmark/run_minimal_suite.py
```

## References
-   Schur, F. et al. (2026). *Many Experiments, Few Repetitions, Unpaired Data, and Sparse Effects*. arXiv:2601.15254
-   Pajo, P. (2026). *Predicting AI Capability Trajectories Before Training: A Regime-Theoretic Framework*.
-   Pajo, P. (2026). *Mechanistic Causal Analysis for LLM Capability Cartography*.
-   **Capability Cartography Layer 3**: [GitHub](https://github.com/pageman/Capability-Cartography-Layer-3) | [DeepWiki](https://deepwiki.com/pageman/Capability-Cartography-Layer-3)
-   **Capability Cartography Layer 2**: [GitHub](https://github.com/pageman/Capability-Cartography-Layer-2) | [DeepWiki](https://deepwiki.com/pageman/Capability-Cartograpy-Layer-2)
-   **Capability Cartography Layer 1**: [GitHub](https://github.com/pageman/Capability-Cartography-Layer) | [DeepWiki](https://deepwiki.com/pageman/Capability-Cartography-Layer)

## Citation

```bibtex
@misc{capability-cartography-layer-4-2026,
  author    = {Paul "The Pageman" Pajo, pageman@gmail.com},
  title     = {Capability-Cartography-Layer-4: Predictive and Mechanistic Framework},
  year      = {2026},
  publisher = {GitHub},
  journal   = {GitHub Repository},
  howpublished = {\url{https://github.com/pageman/Capability-Cartography-Layer-4}},
  note      = {Current repository state is a narrow evidence-backed benchmark package for regime-theoretic forecasting heuristics and checkpointed mechanistic proxy analysis.}
}
```

## License

This repository is released under the MIT License. See [LICENSE](./LICENSE).
