# RankLab

内容流里的 **query 内物品排序** 实验仓库。当前主线是 2026-09-09 的 v2 pipeline，已经能端到端跑通。

这不是生产代码。数据是一份脱敏后的小样本 dump：train 用 bizdate 前约 20 天，test 用 `20260903` 单日；每个 query 下若干候选，`engage_score` 来自互动日志。

字段合同在 `data/feature_schema.json`。

## 环境

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 跑通现有 v2

```bash
python -m src.pipeline
```

会训练模型，并在 `outputs/run_metrics.json` 写出指标。仓库里预置了一份当时的结果：`outputs/baseline_metrics.json`。v2 变更说明见 `outputs/NOTES.md`。

## 你的任务

在现有 pipeline 上 **提升排序效果**。可以用 AI，不限制模型或库。

请提交：

1. 代码改动（直接改本仓库）
2. 一小段说明：你改了什么、怎么验证的
3. 新的指标文件（例如 `outputs/run_metrics.json`）

面试官会看你怎么动手，不看你能不能背公式。

## 相关文档

- `outputs/NOTES.md` — v2 变更说明
- `data/feature_schema.json` — 字段合同
- `docs/ranker_dashboard.md` — 线上看板

