-- 4-list_high_scores.sql
SELECT score, name
FROM second_table
WHERE score >= 10
ORDER BY score DESC;
