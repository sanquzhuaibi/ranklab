# Serving GAUC

线上排序看板用的 GAUC，口径如下。离线脚本不要自己发明加权方式。

1. 标签是 **点击 0/1**，不是互动深度。
2. 每个 request 内部算一次 ROC-AUC。
3. 只有单一标签的 request 跳过。
4. 最终数字是 **request 等权平均**。不要按点击数加权，也不要按互动分加权。

监控任务名：`ranker.gauc.serving`。
