# Transfer Scope

This document narrows what the repository claims about transferring SplitUP- and Schur-inspired ideas into capability forecasting.

## Current Position

CCL4 currently uses `(m, r, d, s*)` as **operational structural descriptors** for a frozen narrow benchmark. The repository does **not** claim that the underlying SplitUP or Schur results already provide a general theorem for AI capability emergence across tasks.

## What Is Defensible Now

1. SplitUP-style regimes motivate separating problems by:
   - number of environments or partitions (`m`)
   - effective samples per environment (`r`)
   - effective representational difficulty (`d`)
   - effective sparsity of the relevant mechanism (`s*`)

2. These quantities are useful enough to support:
   - toy benchmark forecasting
   - held-out small-suite comparisons
   - confidence adjustment when combined with checkpointed mechanistic evidence

3. The repository now demonstrates:
   - operational extraction of these quantities for frozen tiny cases
   - baseline comparisons showing they outperform trivial alternatives on the small benchmark
   - feedback loops where early mechanistic signals either support or weaken the initial forecast

## What Is Not Yet Defensible

1. A strong theorem that SplitUP-style asymptotics govern capability onset in transformers or LLMs.
2. A general proof that `(m, r, d, s*)` transfer cleanly from unpaired IV estimation to cross-task capability prediction.
3. A claim that the current benchmark establishes general cross-task prediction before training.

## Why The Transfer Is Still Useful

The transfer is still productive because it offers:

- a compact structural vocabulary for benchmarking
- regime-sensitive hypotheses that can be falsified
- a disciplined way to separate likely observational surprises from real remaining theory misses

## Path To Stronger Justification

To move from operational heuristic to stronger support, the repo would need:

1. a larger held-out task-family benchmark
2. extractor code with uncertainty estimates tied to real data/model metadata
3. evidence that these descriptors remain predictive across families, not just within a narrow toy suite
4. explicit negative cases where the descriptors fail, so the transfer boundary is empirically visible
