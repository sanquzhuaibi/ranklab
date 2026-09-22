-- Test dump for bizdate = 20260903
-- Impression window: bizdate.

SELECT
    i.query_id,
    i.item_id,
    i.engage_score,
    i.dt,
    f.quality,
    f.freshness,
    f.title_len,
    f.author_score,
    i.topic_match,
    f.hist_ctr,
    f.dwell_sec,
    f.share_rate,
    f.tag_n,
    f.tag_unique_n,
    f.catalog_type_n,
    f.bundle_seq_n,
    f.like_per_exp,
    f.share_per_exp,
    i.city_match,
    i.json_missing
FROM impression i
LEFT JOIN item_feat f
    ON i.item_id = f.item_id
   AND i.dt = f.dt
WHERE i.dt = '${bizdate}'
;
