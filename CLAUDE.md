# GitNexus Claude Context

GitNexus integration in this repository is optional and scoped to developer observability.

Use it for:
- architecture exploration,
- dependency and impact analysis,
- code-navigation support for agents,
- local context refresh after substantial code changes

Keep these boundaries:
- GitNexus indexes and generated skills are not benchmark artifacts
- `.gitnexus/` and generated `.claude/` output are intentionally not committed
- scientific claims still depend on the explicit repo artifacts and tests

Refresh commands:

```bash
npx gitnexus analyze
npx gitnexus analyze --skills
```

For integration guidance, see [`observability/gitnexus/README.md`](/Users/hifi/Capability-Cartography-Layer-4/observability/gitnexus/README.md).
