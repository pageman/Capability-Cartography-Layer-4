# Capability Cartography Layer 4: The Research Report
## Predictive Forecasting & Mechanistic Causal Analysis

### 1. Executive Summary
Capability Cartography Layer 4 (CCL4) is currently best read as a narrow benchmark consolidation layer. While prior layers focused on post-hoc measurement (L1), symptom classification (L2), and statistical causal explanation (L3), the current repo adds a bounded forecasting package plus checkpointed toy and small-transformer discovery cases.

---

### 2. The Narrative Arc: From Measurement to Observable Narrow Evidence

**The "Why" of CCL4**: The central mystery of modern AI is its unpredictability. In March 2026, Terence Tao noted that we lack "reliable principles" to predict model behavior. The current repo addresses a narrow slice of that problem: how to register small benchmark forecasts, distinguish some false unpredictability from real unpredictability, and attach checkpointed mechanism evidence to limited benchmark families with explicit observability artifacts.

**The P04 Turning Point**: Earlier repo drafts described a much broader historical benchmark. The current repo no longer treats that as the evidentiary center. Instead, it centers frozen tiny cases, explicit falsifiers, narrow artifact-backed mechanism checks, and a deliverable manifest that freezes what was actually generated.

**The Dependency Turn**: Earlier active paths still implicitly relied on PyTorch being installed even when the repo claimed graceful fallback behavior. The current repo has now moved the benchmarked demo and suite paths to NumPy-only implementations, making the end-to-end evidence package less fragile and better aligned with its portability claims.

**The Convergence Turn**: Earlier benchmark files carried their own local mechanism summaries. The current repo now routes structured bundles from both the tiny-model suite and the small-transformer bundle through the same `circuit_discovery.py` engine, so feature ranking, Fourier scoring, and targeted-vs-random ablation use one shared contract.

**The Metadata Audit Turn**: A repo-wide stub sweep now distinguishes active-path implementation gaps from acceptable narrow placeholders. The most important closed gap is that `circuit_discovery.py` can now infer feature bundles directly from raw NumPy matrices when paired with a compatible linear/readout model, rather than dropping immediately to canned labels. The main remaining stubs are now isolated to three bounded areas: full graph recovery on realistic checkpoints, the optional quantum analogy layer, and heuristic fallbacks when neither a structured bundle nor an analyzable model is present.

**The Control Clarification Turn**: Earlier shared-rubric runs left the non-periodic tiny control graded as weak because it was effectively being judged against periodic-family criteria. The current repo now treats that family as a true smooth control: it uses a cleaner monotonic target, a representative held-out split, held-out score calibration metrics, and a control-specific grading rule that rewards low periodicity and the absence of special targeted-ablation effects.

**The Falsifier Clarification Turn**: The sparse relational family is no longer allowed to masquerade as a failed periodic discovery. The current repo now treats it as an explicit falsifier: forecast miss remains, the discovery engine reports no informative circuit under the current contract, and targeted ablation remains non-diagnostic. This is a better scientific outcome than a noisy pseudo-mechanism.

**The Quantum Bridge**: In the current repository, the quantum framing should be read as an optional analogy layer over Fourier-structured modular circuits. It is not yet evidence that the repo has established exact classical-quantum equivalence or that the quantum language adds predictive power.

> #### **Story Box 1: The P04 Correction (Overcoming Metadata Bias)**
> Historically, P04 was viewed as a purely theoretical paper on MDL (Minimum Description Length). CCL4's initial failure to classify it as a "Power-law" revealed a bias in our forecasting logic: we were weighting Publication Date and Title over the presence of MNIST experiments. By adding an "Empirical Override," we taught the framework that even a "Theory" paper can be a "Power-law" prototype if it contains a dataset.

> #### **Story Box 2: From One Mechanism Toy To A Small Evidence Bundle**
> The repo now includes both checkpointed toy modular evidence and a three-family small-transformer bundle. The periodic family plays the positive case, the smooth family plays the control case, and the sparse relational family now plays an explicit falsifier role inside the same artifact package.

> #### **Story Box 3: From Pure Heuristic Labeling To Rudimentary Real Analysis**
> Earlier versions of `circuit_discovery.py` mapped capability names to canned circuit labels. The current repo still does not perform full ACDC/SAE graph recovery, but it now does one real thing: when given a structured linear feature bundle, it ranks components by alignment, computes a real Fourier score, and compares targeted ablation against random control ablation.

> #### **Story Box 4: One Discovery Engine, Two Benchmark Paths**
> The repo now applies the same discovery engine to two different evidence paths. `real_tiny_case.py` supplies explicit feature bundles from its NumPy models, while `small_transformer_case.py` extracts final context-vector bundles from the checkpointed attention model. That does not make the analysis broad, but it does remove a previous source of methodological drift.

> #### **Story Box 5: The Stub Audit Compression**
> The repo no longer treats every weak area as equally urgent. A metadata sweep compressed the active stub surface down to a short list: (1) full ACDC/SAE-style graph recovery on realistic checkpoints, (2) optional quantum analogy logic, and (3) heuristic fallbacks for inputs that still lack analyzable structure. Everything else in the active benchmark path is now either implemented, tested, or explicitly marked as narrow.

> #### **Story Box 6: The Control Strengthening Pass**
> The non-periodic tiny family is no longer just “not periodic.” It is now a positive smooth control. Its held-out split spans the score range, its outputs stay highly calibrated, its monotonic agreement remains high, and targeted ablation is no more destructive than matched random ablation. That makes the contrast with the periodic case cleaner and the benchmark more scientifically useful.

> #### **Story Box 7: The Falsifier Cleanup**
> The sparse relational family used to fail in a muddled way: weak evidence, but still a misleading circuit label. The repo now treats that family more honestly. It is a falsifier case with a forecast miss, no informative circuit under the current discovery contract, and no meaningful causal leverage from ablation. That is a stronger benchmark artifact because it says exactly where the current theory stops.

---

### 3. The Methodological Arc: The End-to-End Pipeline With Observability

The CCL4 pipeline is a dual-engine system:

1.  **Forecasting Engine (Pre-training)**:
    -   **Input**: Task descriptors ($m, r, d, s^*$) and domain metadata.
    -   **Process**: Asymptotic regime analysis using Schur et al. (2026) parameters.
    -   **Output**: Trajectory Forecast (Power-law, Emergent, Step-function, Hybrid).

2.  **Mapping Engine (Post-training)**:
    -   **Input**: Model weights, structured feature bundles, and checkpoint history.
    -   **Process**: checkpointed proxy discovery on tiny NumPy models, shared rudimentary feature-level circuit analysis for structured linear bundles, plus a narrow implemented small-transformer attention-route extraction across positive/control/falsifier synthetic families.
    -   **Output**: benchmark-specific circuit summaries, causal verification via targeted-vs-random ablation, real Fourier scores for structured bundles, and explicit artifact-level `claim_coverage` / `failure_modes`.

3.  **Observability Layer**:
    -   **Input**: Generated artifacts and repo test runs.
    -   **Process**: root verification ledger plus deliverable manifest generation.
    -   **Output**: `verification.yaml`, `deliverable_manifest.json`, and machine-readable scope boundaries for each evidence package.

> #### **Method Box 1: (m, r, d, s*) Regime Boundary Optimization**
> Based on a Meta-Sweep of 30 foundational papers, we optimized the boundaries for the "Middle Regime." We found that the transition from Classical to Middle regimes occurs at $r=15.0$ (samples per environment), not $20.0$. This refined boundary significantly improved the accuracy of our "Scale Transfer Invariance" flags.

> #### **Method Box 2: Intrinsic Dimension as a Pre-training Proxy**
> To predict "Step-functions" (like AlexNet), we implemented the **ID-Ratio** ($IntrinsicDimension / ParamCount$). A high ID-Ratio (>0.1) combined with high relational depth (Depth > 3) is the primary precursor to sudden, non-linear capability onset.

> #### **Method Box 3: Rudimentary Circuit Analysis Contract**
> The current `circuit_discovery.py` no longer only emits canned labels. On structured linear bundles it now: (1) ranks features by alignment with the target score, (2) computes a Fourier signature from actual values, and (3) measures targeted-vs-random ablation drop through a minimal model adapter. This is still narrow analysis, but it is no longer purely descriptive.

> #### **Method Box 4: Shared Feature-Bundle Integration**
> The tiny-model suite now feeds explicit feature matrices into `circuit_discovery.py`, and the small-transformer bundle now feeds final context vectors from the trained attention model into the same discovery engine. This creates an end-to-end benchmark contract: generate bundle, discover components, compare targeted and random ablation, then record the result in the artifact layer.

> #### **Method Box 5: Minimum-Viable Stub Replacements**
> The repo now records a concrete replacement path for each remaining stub family. For raw matrix inputs, `circuit_discovery.py` infers a feature bundle from model outputs and analyzes it through the same scoring and ablation contract. For checkpointed attention traces, the next minimum-viable upgrade is to replace stable-position overlap with edge-level route extraction and matched control ablations. For the optional quantum layer, the minimum honest replacement is to keep it as analogy-only unless it improves benchmark prediction or mechanism selection measurably.

> #### **Method Box 6: Control-Specific Evidence Criteria**
> The shared rubric now distinguishes between positive mechanism cases and smooth control cases. Periodic families still need masking support, precursor rise, and targeted-ablation separation. The non-periodic control instead needs power-law RMSE, strong held-out score correlation, high monotonic agreement, low periodicity, and no special targeted-ablation advantage over random controls. This avoids grading the control as a failed periodic case.

---

### 4. Technical Consolidation: The Five Improvements

To improve narrow-benchmark credibility, the following end-to-end enhancements were implemented:

1.  **Empirical Override**: Scans task metadata for evidence of datasets, overriding "Theoretical" classifications.
2.  **Intrinsic Proxy**: Uses known dataset complexity to predict non-linear scaling.
3.  **L3 Pathology Feedback**: Connects Layer 3 "Unpaired Bias" flags to Layer 4 "Scale-Invariance" warnings.
4.  **Refined Boundaries**: Precision-tuned $r$ and $m$ thresholds for regime classification.
5.  **Hybrid Taxonomy**: Introduced support for MoE (Mixture of Experts) and Mamba (SSM) architectures.

---

### 5. Stub Audit Snapshot

The current stub inventory is narrower than earlier repo drafts:

1. **Closed in active path**:
   - raw NumPy demo inputs can now be analyzed through inferred feature bundles when paired with compatible linear/readout models;
   - structured tiny-model and small-transformer bundles already flow through the same discovery engine;
   - targeted-vs-random ablation is no longer purely hardcoded for structured linear bundles.

2. **Still narrow but acceptable**:
   - heuristic fallback labels for demo cases that do not supply analyzable structure;
   - position-level attention-route summaries in the small-transformer bundle;
   - operational heuristic regime forecasting over `(m, r, d, s*)`.

3. **Still planned**:
   - full ACDC/SAE-style graph recovery on realistic checkpoints;
   - large-model circuit discovery beyond synthetic families;
   - quantum-style motifs that add measurable predictive or mechanistic value.

---

### 6. Conclusion: Predicting the "Unpredictable"
CCL4 now supports a narrower conclusion. Some surprising jumps can be reframed as observation effects, and some controlled benchmark families can be forecast and tracked with checkpointed mechanism evidence. The repo now records those claims with explicit coverage, failure modes, deliverable provenance, and an active benchmark path that no longer depends on PyTorch. The stronger goal of general capability prediction across tasks remains open.
