# Benchmark Summary

Before command: `python benchmark_tool_parser_standalone.py`
After command: `python benchmark_tool_parser_rust_standalone.py`

Before timing: 3.19 ms
After timing: 2.25 ms

Percent change: 29.5% improvement

Notes: Replaced the Python string-matching ToolCallStreamParser with a pure Rust streaming implementation using PyO3, which avoids Python loop and slicing overheads.
