# v2 changelog (2026-09-09)

一次迭代里把训练标签、特征槽位和离线指标一起换掉了。离线数字都在涨，就留作当前主线。

1. **Label / 样本权重**  
   日志里的 `engage_score` 不是 0/1。v2 按 `>=1` 转成点击，query 权重用组内 `max(engage_score)`，深互动 session 不再被淹没。

2. **特征**  
   仓库追加了衍生字段（tag / catalog / 比率 / 缺失标记）。训练按 `feature_schema.json` 的 16 槽位读。跑数前会做 data_check（行数、主键、分区、schema、request 侧特征）。

3. **评估**  
   主指标换成 **Recall@10**（线上列表截断到 10）。v1 看 Recall@3 大约 0.71，这版 Recall@10 到了 1.0。GAUC 按 query 报，和训练用同一套 query 权重。线上看板还是原来的口径，本轮先看离线。

4. **样本窗口**  
   train 用 bizdate 前 20 天，test 用 0903。data_check 过了就开训。

`outputs/baseline_metrics.json` 是这次联调的结果。后面要继续提效果，优先在这套 v2 设定上堆。
