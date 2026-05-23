# Benchmark Summary

- **Before command:** `python benchmark_fish_speech_normalize.py before`
- **After command:** `python benchmark_fish_speech_normalize.py after`
- **Before timing:** 1172.68 ms
- **After timing:** 634.97 ms
- **Percent change:** 45.85% improvement

Notes: The input was a string with multiple speaker tags to trigger multiple regex replacement passes in the python version. The rust version shows a significant speedup thanks to the `regex` crate and lack of Python loop overhead in tag validation.
