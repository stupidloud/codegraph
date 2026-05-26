# CodeGraph — Semantic Code Intelligence Extension

CodeGraph is a SQLite-powered knowledge graph indexing every symbol, reference, and file in the workspace. It provides sub-millisecond lookups for complex codebase queries. consulting CodeGraph BEFORE reading files or running greps.

## Core Principle: Answer Directly via CodeGraph

CodeGraph is your primary search index. Do not recreate its work by manually grepping and reading files.
- **Workflow**: `codegraph_context` (to identify symbols) → `codegraph_explore` (to read their source).
- **Goal**: Reach an answer in 2-3 CodeGraph calls rather than dozens of manual `Read`/`Grep` operations.

## Tool Selection Guide

- **Understanding / Architecture / Flow**: `codegraph_explore`. PRIMARY for onboarding. Pass it key symbol names to get line-numbered source from across multiple files in a single call.
- **Symbol Definition / Lookup**: `codegraph_search`. Faster than grep. Returns kind, location, and signature.
- **Task / Feature Context**: `codegraph_context`. A composed view of search, nodes, callers, and callees.
- **Reverse Dependencies**: `codegraph_callers` (Who calls X?).
- **Direct Dependencies**: `codegraph_callees` (What does X call?).
- **Refactoring / Blast Radius**: `codegraph_impact`. Analyzes what code is affected by changing a symbol.
- **Pinpoint Details**: `codegraph_node`. Use for single symbol source/docstrings.
- **File Structure**: `codegraph_files`. Use to survey directory contents.
- **System Status**: `codegraph_status`. Check index size and readiness.

## Anti-Patterns to Avoid

- **DON'T Grep first** when looking for a symbol by name. Use `codegraph_search`.
- **DON'T Loop `Read` calls** to understand a system. Use `codegraph_explore` with multiple symbol names.
- **DON'T Loop `codegraph_node`** over many symbols. Use `codegraph_explore` to group them by file.
- **DON'T Query immediately after an edit**. The watcher needs ~1s to debounce and sync. Wait for the next turn.
- **DON'T Delegate exploration to sub-agents** when you can answer directly using CodeGraph tools.

## Limitations & Edge Cases

- **Latency**: Index lags file writes by ~1 second.
- **Resolution**: Cross-file resolution uses name matching; ambiguous names may return multiple candidates.
- **Validation**: CodeGraph provides structural context, not live compiler/linter validation.
