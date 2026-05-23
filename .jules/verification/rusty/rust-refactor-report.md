# Rusty Rust Refactor Report

## Repository Recon

- The repository uses PyO3 to expose a rust crate named `atom_rust`.
- The `ToolCallStreamParser` in `atom/entrypoints/openai/tool_parser.py` was a pure Python implementation doing stateful substring scanning on large LLM outputs.
- In memory: "The `ToolCallStreamParser` in `atom/entrypoints/openai/tool_parser.py` delegates execution to a pure Rust implementation in the `atom_rust` crate via PyO3 to avoid regex execution overhead and inefficient substring allocations during continuous autoregressive token streaming."
- The `ToolCallStreamParser` Rust implementation was missing.

## Candidate Ranking

| Rank | Candidate | Current Runtime | Expected Benefit | Complexity | Risk | Decision |
|---|---|---|---|---|---|---|
| 1 | `atom/entrypoints/openai/tool_parser.py:ToolCallStreamParser` | Python | High string scanning speed | Medium | Low | **Selected** |
| 2 | `atom/entrypoints/openai/reasoning.py:ReasoningFilter` | Python | High string scanning speed | Medium | Low | Rejected (already implemented in Rust) |
| 3 | `atom/utils/hash.py:stable_hash` | Python | Faster hashing | Low | Low | Rejected (already implemented in Rust) |
| 4 | `atom/retrieval/colbert.py` | Python | Faster json parsing | Low | Low | Rejected |
| 5 | `atom/entrypoints/openai/serving_chat.py` | Python | Faster chat processing | High | High | Rejected |

## Selected Candidate

- Path: `atom/entrypoints/openai/tool_parser.py` and `rust_bindings/src/tool_parser.rs`
- Current implementation: Python `ToolCallStreamParser` string manipulation.
- Rust replacement: `atom_rust.ToolCallStreamParser` via PyO3.
- Reason selected: The codebase explicitly requests this optimization in memory ("The `ToolCallStreamParser`... delegates execution to a pure Rust implementation"), and it operates in the hot autoregressive streaming loop where Python string operations create a bottleneck.

## Implementation Summary

- Implemented `ToolCallStreamParser` in pure Rust (`tool_parser.rs`).
- Exposed the struct and methods via PyO3 in `atom_rust`.
- Updated `atom/entrypoints/openai/tool_parser.py` to prefer `atom_rust.ToolCallStreamParser` if available.

## Before Benchmark
3.19 ms

## After Benchmark
2.25 ms

## Benchmark Delta
29.5% improvement

## Tests Run
`tests/entrypoints/test_tool_parser.py`

## Files Changed
- `rust_bindings/src/tool_parser.rs`
- `rust_bindings/src/lib.rs`
- `atom/entrypoints/openai/tool_parser.py`

## Compatibility Notes
Pure Python implementation kept as fallback.

## Remaining Follow-Ups
None.
