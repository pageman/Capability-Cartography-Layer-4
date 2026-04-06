# Capability Cartography Layer 4: Predictive & Mechanistic

**Capability Cartography Layer 4 (CCL4)** is the synthesized evolution of the Capability Cartography hierarchy. While Layer 3 provided causal explanations for past failures, Layer 4 moves in two complementary directions:

1.  **Predictive (The "When")**: A regime-theoretic framework for forecasting AI capability trajectories *before* training begins, based on task descriptors and asymptotic analysis.
2.  **Mechanistic (The "How")**: A circuit-level interpretability engine for mapping capabilities to internal model implementations (ACDC, SAE) and verifying them through causal abstraction.

## The Layer Evolution

| Layer | Question | Verb | Key Addition |
|-------|----------|------|-------------|
| **1** | "What happened?" | **Measures** | Schemas, sweeps, surfaces, validation, falsifiable laws |
| **2** | "What kind of failure is this?" | **Classifies** | Failure atlas, visualization, agent briefs |
| **3** | "Why did it fail?" | **Explains** | Causal registry, middle-regime analysis, transfer diagnostics |
| **4** | **"How does it work & When will it scale?"** | **Forecasts & Maps** | **Regime-theoretic forecasting, circuit discovery, grokking analysis, Quantum-Fourier bridge** |

## Core Innovations

### 1. Regime-Theoretic Forecasting
Using the $(m,r,d,s^*)$ parameters from Schur et al. (2026), CCL4 classifies the asymptotic regime of a task and predicts its scaling trajectory:
-   **Power-law**: Gradual accumulation in classical regimes ($r \gg 1$).
-   **Emergent**: Phase transitions in sparse, high-dimensional regimes ($d > m, s^* \ll d$) with high relational depth.
-   **Step-function**: Sudden onset at resource thresholds or huge compressibility gaps ($>20\times$).

### 2. Mechanistic Circuit Mapping
Instead of treating the model as a black box, CCL4 identifies the minimal subgraphs (circuits) that implement specific capabilities:
-   **Induction Heads**: Mechanistic source of in-context learning.
-   **Fourier Circuits**: Trigonometric representations discovered in modular arithmetic tasks.

### 3. The Quantum Connection Hypothesis
CCL4 explores the novel research question of whether transformers trained on modular arithmetic independently discover the same Fourier-based computational strategies underlying quantum algorithms like **Shor's Algorithm** and the **Quantum Fourier Transform (QFT)**.

## Project Structure

-   `regime_forecaster.py`: Logic for pre-training trajectory prediction.
-   `circuit_discovery.py`: Methods for mapping capabilities to internal circuits.
-   `orchestration.py`: Unified pipeline for forecasting and verification.
-   `schemas.py`: Data structures for records, forecasts, and circuit definitions.
-   `demo.py`: Verification script across Scaling Laws, Transformers, and Modular Exponentiation.

## Quick Start

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
python3 capability_cartography_layer4/demo.py
```

## References
-   Schur, F. et al. (2026). *Many Experiments, Few Repetitions, Unpaired Data, and Sparse Effects*. arXiv:2601.15254
-   Pajo, P. (2026). *Predicting AI Capability Trajectories Before Training: A Regime-Theoretic Framework*.
-   Pajo, P. (2026). *Mechanistic Causal Analysis for LLM Capability Cartography*.
