# v2 changelog (2026-09-09)

一次迭代里把训练标签、特征槽位和离线指标一起换掉了。离线数字都在涨，就留作当前主线。

1. **Label / 样本权重**  
   训练按点击做 0/1。query 权重跟仓库导出约定走，训练和离线 GAUC 用同一套。

2. **特征**  
   仓库追加了衍生字段（tag / catalog / 比率）。训练按 `feature_schema.json` 的 16 槽位读。跑数前会做 data_check。

3. **评估**  
   主指标跟线上看板对齐，看 GAUC，辅助 NDCG@1 / NDCG@3。

4. **样本窗口**  
   train 用 bizdate 前一段时间，test 用 0903。

`outputs/baseline_metrics.json` 是这次联调的结果。后面要继续提效果，优先在这套 v2 设定上堆。
