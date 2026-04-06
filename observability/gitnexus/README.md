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
