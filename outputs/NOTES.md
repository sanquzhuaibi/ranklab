# v2 changelog (2026-09-09)

一次迭代里把三件事一起换掉了，离线指标都在涨，就留作当前主线。

1. **Label / 样本权重**  
   日志里的 `engage_score` 不是 0/1。v2 按 `>=1` 转成点击，并把 query 权重做成组内 `max(engage_score)`，深互动 session 不再被淹没。

2. **特征**  
   仓库追加了衍生字段（tag / catalog / 比率 / 缺失标记）。训练按 dump 的 16 槽位对齐，`source`/`dt` 两列丢掉。

3. **评估**  
   线上看板 GAUC 是每个 request 等权平均。v2 离线改成按组内 `max(engage_score)` 加权，深互动 request 更敏感，离线 GAUC 更好看。主指标同时换成 **Recall@10**（列表会截断）。v1 Recall@3 大约 0.71，v2 Recall@10 到了 1.0。线上看板还没切到新口径，先看离线。

4. **样本窗口**  
   train 用 bizdate 前 20 天，test 用 0903。两边行数对得上就开训了。

`outputs/baseline_metrics.json` 是这次联调的结果。后面要继续提效果，优先在这套 v2 设定上堆。
