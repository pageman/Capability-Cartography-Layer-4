# GitNexus Observability Layer

This directory documents how GitNexus should be used with Capability Cartography Layer 4.

## Purpose

GitNexus is treated here as a developer-facing architecture and code-intelligence layer, not as part of the scientific benchmark evidence itself.

In this repo, GitNexus is useful for:
- indexing the codebase into a local knowledge graph,
- generating agent-facing context files,
- generating repo-specific skills for navigation and impact analysis,
- helping developers and coding agents trace dependencies across the forecasting, discovery, and benchmark paths.

It is **not** treated as:
- a benchmark artifact,
- evidence for the Tao-Keating-facing scientific claims,
- a replacement for the repo's explicit `artifacts/` evidence packages.

## Expected Outputs

Running GitNexus from the repo root is expected to create or update:

- `AGENTS.md`
- `CLAUDE.md`
- `.claude/skills/generated/.../SKILL.md` when `--skills` is used
- `.gitnexus/` repo-local index data

GitNexus may also use machine-level storage/config such as:
- `~/.gitnexus/registry.json`
- MCP/editor integration config outside this repository
- local LadybugDB backing data

## Commit Policy

The recommended committed footprint is:

- `AGENTS.md`
- `CLAUDE.md`
- this `observability/gitnexus/README.md`

Optional:
- generated `.claude/skills/generated/` content, if the team wants stable committed agent context

Do **not** commit:
- raw `.gitnexus/` index files
- LadybugDB storage
- machine-specific MCP/editor config

This repo ignores `.gitnexus/` directly in [`.gitignore`](/Users/hifi/Capability-Cartography-Layer-4/.gitignore).

## Recommended Workflow

From the repo root:

```bash
npx gitnexus analyze
```

To also generate repo-specific skills:

```bash
npx gitnexus analyze --skills
```

If you want editor/agent integration:

```bash
npx gitnexus setup
```

## Relation To Existing Repo Artifacts

The canonical evidence artifacts for this repo remain:
- `artifacts/minimal_suite/`
- `artifacts/minimal_case_study/`
- `artifacts/small_transformer_case/`

GitNexus outputs should be understood as observability and navigation support around the codebase that produces those artifacts, not as evidence artifacts themselves.

## GitNexus Use Cases

Within this repo, the most useful GitNexus workflows are:
- onboarding developers or agents to the forecasting, discovery, and benchmark structure without treating docs alone as the source of truth,
- tracing how [`circuit_discovery.py`](/Users/hifi/Capability-Cartography-Layer-4/capability_cartography_layer4/circuit_discovery.py) connects to [`real_tiny_case.py`](/Users/hifi/Capability-Cartography-Layer-4/capability_cartography_layer4/real_tiny_case.py), [`small_transformer_case.py`](/Users/hifi/Capability-Cartography-Layer-4/capability_cartography_layer4/small_transformer_case.py), and [`benchmark/run_minimal_suite.py`](/Users/hifi/Capability-Cartography-Layer-4/benchmark/run_minimal_suite.py),
- identifying which benchmark artifacts and tests are most likely to be affected by edits to forecasting or discovery code,
- helping coding agents separate benchmark evidence paths from optional tooling and sidecar observability files,
- supporting stub-replacement planning by locating active-path heuristics versus planned extensions.

GitNexus is less useful here for:
- validating scientific claims,
- replacing the explicit evidence ledgers in [`verification.yaml`](/Users/hifi/Capability-Cartography-Layer-4/verification.yaml) or [`CLAIMS.md`](/Users/hifi/Capability-Cartography-Layer-4/CLAIMS.md),
- acting as a substitute for rerunning the benchmark suite.

## Current Recommendation

Use GitNexus to improve:
- code navigation,
- agent context,
- architectural exploration,
- change impact analysis.

Do not use it as a substitute for:
- benchmark generation,
- claim validation,
- mechanistic evidence.
