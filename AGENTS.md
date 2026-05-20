# AGENTS.md

This file provides repository-specific guidance for agents working in this fork.

## Semantic Search Direction

When working on semantic search in this fork, optimize for this goal:

- Bring back the original upstream semantic-search experience and result quality as much as practical.
- Reduce the pain points that led upstream to remove semantic search, especially local model/runtime burden, native install fragility, and large-codebase stability issues.
- Do not reintroduce complexity casually. Prefer simpler designs unless extra complexity clearly removes a real pain point or preserves important behavior.
- Preserve local storage and local retrieval where practical; use remote embedding providers only where they simplify runtime burden.
- When choosing between fidelity to upstream behavior and extra infrastructure or metadata, prefer the simpler option unless correctness or user-visible behavior would clearly regress.
