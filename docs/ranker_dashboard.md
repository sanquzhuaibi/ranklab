# Ranker 线上看板

内部 wiki 摘录（2026-08 看板任务 `ranker.dashboard.prod`）。

## 覆盖与延迟

- 覆盖：有曝光且候选数 ≥ 2 的 request。
- 延迟：排序 RT p99 单独看，不进效果看板。
- 抽样：全量 request。

## 主效果指标：GAUC

任务名 `ranker.gauc.serving`。

1. 标签是点击 0/1。
2. 每个 request 内部算一次 ROC-AUC。
3. 只有单一标签的 request 跳过。
4. 再对 request 取平均，得到当天的 GAUC。

## 辅助指标

- NDCG@1 / NDCG@3
- 曝光覆盖率、空结果率

## 和离线实验

看板口径变更要单独发评审。上线结论需要能在这个看板上复现。
