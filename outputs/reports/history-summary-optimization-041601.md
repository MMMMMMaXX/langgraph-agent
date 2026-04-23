# Eval Comparison Report

- Baseline: `outputs/eval_runs/20260416-174313-baseline.json`
- Candidate: `outputs/eval_runs/20260416-175610-concise.json`

## Overview

| Metric         |     Baseline |    Candidate |
| -------------- | -----------: | -----------: |
| Pass rate      | 100.0% (1/1) | 100.0% (1/1) |
| Avg request ms |      7339.40 |      6647.58 |

## Case Comparison

| Case                            | Category | Baseline Assertion | Candidate Assertion | Baseline ms | Candidate ms | Delta ms | Baseline Len | Candidate Len |
| ------------------------------- | -------- | ------------------ | ------------------- | ----------: | -----------: | -------: | -----------: | ------------: |
| history_summary_after_two_turns | memory   | pass               | pass                |     7339.40 |      6647.58 |  -691.82 |           38 |            38 |
