# Eval Comparison Report

- Baseline: `outputs/eval_runs/20260416-173434-baseline.json`
- Candidate: `outputs/eval_runs/20260416-174313-baseline.json`

## Overview

| Metric         |     Baseline |    Candidate |
| -------------- | -----------: | -----------: |
| Pass rate      | 100.0% (1/1) | 100.0% (1/1) |
| Avg request ms |      8495.08 |      7339.40 |

## Case Comparison

| Case                            | Category | Baseline Assertion | Candidate Assertion | Baseline ms | Candidate ms | Delta ms | Baseline Len | Candidate Len |
| ------------------------------- | -------- | ------------------ | ------------------- | ----------: | -----------: | -------: | -----------: | ------------: |
| history_summary_after_two_turns | memory   | pass               | pass                |     8495.08 |      7339.40 | -1155.68 |           38 |            38 |

## Changed Answers

### history_summary_after_two_turns

- Baseline: 刚刚的问题包括：

1. 用户询问上海的气候情况
2. 用户询问北京的气候情况

- Candidate: 刚刚的问题包括：

1. 用户询问北京气候怎么样
2. 用户询问上海气候怎么样
