from collections import Counter
from pathlib import Path

from ayadb.queryshape import CLASSES, classify, is_complex
from ayadb.runinputs import evaluation_tasks


def test_query_shape_examples():
    assert classify("SELECT name FROM author WHERE id = 49") == "1-table lookup"
    assert classify("SELECT tagId FROM EventRecord ORDER BY count DESC LIMIT 1") == "1-table lookup"
    assert classify("SELECT COUNT(*) FROM t WHERE x = 1") == "1-table aggregate"
    assert classify("SELECT a, MAX(b) FROM t GROUP BY a") == "1-table aggregate"
    assert classify("SELECT a, ROW_NUMBER() OVER (ORDER BY b) FROM t") == "1-table aggregate"
    assert classify("SELECT u.email FROM users u JOIN orde o ON u.id = o.UserId") == "two tables"
    assert classify("SELECT e.name FROM emp e JOIN emp m ON e.boss = m.id") == "two tables"
    assert classify("SELECT a.x FROM a JOIN b ON a.id = b.id JOIN c ON b.id = c.id") == "3+ tables"
    assert classify("SELECT x FROM t WHERE id IN (SELECT id FROM u)") == "multi-step"
    assert classify("WITH s AS (SELECT a, SUM(b) AS n FROM t GROUP BY a) SELECT a FROM s") == "multi-step"
    assert classify("SELECT a FROM t UNION SELECT a FROM u") == "multi-step"
    assert is_complex("SELECT x FROM t WHERE id IN (SELECT id FROM u)")
    assert not is_complex("SELECT u.email FROM users u JOIN orde o ON u.id = o.UserId")


def test_frozen_dev_shape_counts():
    # Amendment 8 freezes these counts on the 95 effective dev tasks before the A1500 result exists.
    dev = evaluation_tasks(Path("data/sqale/data-build-v3/dev.tasks.jsonl"), purpose="dev")
    counts = Counter(classify(t["reference_sql"]) for t in dev)
    assert [counts[c] for c in CLASSES] == [37, 5, 23, 10, 20]
    assert sum(is_complex(t["reference_sql"]) for t in dev) == 30
