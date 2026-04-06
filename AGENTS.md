# GitNexus Agent Context

This repository supports an optional GitNexus observability layer.

In this repo, GitNexus should be treated as:
- a developer-facing architecture and dependency index,
- an aid for code navigation, impact analysis, and agent context,
- separate from the benchmark and evidence artifacts under `artifacts/`.

Recommended use:
- refresh the local index with `npx gitnexus analyze`
- optionally generate repo-local skills with `npx gitnexus analyze --skills`
- use GitNexus context and impact tools for code exploration, not as scientific evidence

Do not treat GitNexus output as:
- benchmark evidence,
- a substitute for `verification.yaml`,
- a replacement for the repo's explicit tiny-model and small-transformer artifacts

See [`observability/gitnexus/README.md`](/Users/hifi/Capability-Cartography-Layer-4/observability/gitnexus/README.md).
