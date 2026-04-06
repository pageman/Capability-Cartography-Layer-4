# Parameter Definitions

This document defines how CCL4 currently interprets the regime parameters `(m, r, d, s*)`.

These definitions are operational for the repository as it exists now. They are sufficient for the toy benchmark and small metadata-driven examples, but they are not yet a full theory of extraction for arbitrary training runs.

## Definitions

### `m`

`m` is the number of distinct environments, conditions, or task partitions that matter for the capability under study.

- Unit: count of environments.
- Typical interpretation:
  - modular task: number of residue classes or modular states
  - benchmark corpus: number of distinct task families or datasets
  - multi-domain training setup: number of environments with meaningfully different support

### `r`

`r` is the effective number of samples per environment.

- Unit: samples per environment.
- Current working formula:
  - `r = effective_samples / m`
- `effective_samples` may be smaller than raw rows when observations are deduplicated, heavily thresholded, or collapsed into environment-level summaries.

### `d`

`d` is the effective representational dimensionality of the task.

- Unit: feature dimensions.
- Current working interpretation:
  - number of active input dimensions, engineered features, or representational degrees of freedom that the learner must coordinate
  - for metadata-only forecasting, this is a coarse proxy rather than a measured latent dimension

### `s*`

`s*` is the effective sparsity of the mechanism or task-relevant representation.

- Unit: count of materially active factors.
- Current working interpretation:
  - number of dominant basis components, features, or constraints expected to carry the task
  - in modular periodic tasks, this can be approximated by the number of dominant Fourier-like basis terms

## Minimal Case Study Example

For the frozen modular masking case in [`case_study.py`](/Users/hifi/Capability-Cartography-Layer-4/capability_cartography_layer4/case_study.py):

- `m = 17`
  - one environment per residue class modulo 17
- `r = 1.0`
  - approximately one effective observation per environment in the registered toy setup
- `d = 64`
  - a coarse representational-difficulty proxy used by the forecasting engine
- `s* = 4`
  - four dominant basis terms in the toy periodic feature construction

## Uncertainty Notes

These quantities are currently estimated with different confidence levels:

- `m`: usually low uncertainty
- `r`: moderate uncertainty because "effective samples" depends on deduplication and task framing
- `d`: high uncertainty outside explicitly constructed toy settings
- `s*`: high uncertainty unless a sparse mechanism has already been measured

## Why This Is Still Partial

The current repository uses these parameters as a practical forecasting interface, not a fully justified universal extraction standard. To move from partial to strong evidence, the next required step is:

1. add extractor code that computes these quantities from frozen metadata schemas,
2. attach uncertainty intervals to every estimate,
3. compare extracted values against held-out benchmark outcomes,
4. document failure cases where the mapping from task to `(m, r, d, s*)` breaks down.
