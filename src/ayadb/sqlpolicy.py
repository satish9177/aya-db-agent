"""Record tie-safe result ordering and exclude ambiguous top-level LIMIT boundaries."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import sqlglot
from sqlglot import exp

from .scoringlimits import SCORING_MAX_OPS, SCORING_SECONDS
from .sqlscore import MAX_ROWS, execute, rows_equal, top_level_ordered


def ordering_policy(db: Path, sql: str) -> dict:
    path = Path(db).resolve()
    stat = path.stat()
    # Database copies are frozen; stat in the key prevents reuse after a local edit.
    return dict(_policy(str(path), stat.st_size, stat.st_mtime_ns, sql))


@lru_cache(maxsize=16000)
def _policy(path: str, size: int, mtime: int, sql: str) -> dict:
    def policy(ordered, reason, eligible=True, **details):
        return {"ordered": ordered, "reason": reason, "eligible": eligible, **details}

    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except sqlglot.errors.SqlglotError:
        return policy(False, "unparsed_reference", False)
    has_limit = tree.args.get("limit") is not None or tree.args.get("offset") is not None
    if not top_level_ordered(sql):
        # LIMIT without order is only unambiguous when it includes the full result.
        if has_limit:
            unlimited = tree.copy()
            unlimited.set("limit", None)
            unlimited.set("offset", None)
            full = execute(Path(path), unlimited.sql(dialect="sqlite"), seconds=SCORING_SECONDS, max_ops=SCORING_MAX_OPS)
            selected = execute(Path(path), sql, seconds=SCORING_SECONDS, max_ops=SCORING_MAX_OPS)
            if full.error or selected.error or not rows_equal(full.rows, selected.rows):
                return policy(False, "limit_without_deterministic_order", False)
        return policy(False, "no_top_level_order")
    if not isinstance(tree, exp.Select):
        return policy(False, "compound_order_unresolved", not has_limit)
    gold = execute(Path(path), sql, seconds=SCORING_SECONDS, max_ops=SCORING_MAX_OPS)
    if gold.error:
        return policy(False, "invalid_reference: " + gold.error, False)
    try:
        limit, offset = None, 0
        if tree.args.get("limit"):
            value = tree.args["limit"].expression
            if not isinstance(value, exp.Literal) or value.is_string:
                raise ValueError("nonliteral_limit")
            limit = int(value.this)
            if limit < 0:
                limit = None
        if tree.args.get("offset"):
            value = tree.args["offset"].expression
            if not isinstance(value, exp.Literal) or value.is_string:
                raise ValueError("nonliteral_offset")
            offset = max(0, int(value.this))
        aliases = {e.alias.lower(): e.this for e in tree.expressions if isinstance(e, exp.Alias)}
        order = tree.args["order"].copy()
        for term in order.expressions:
            key = term.this
            if isinstance(key, exp.Literal) and not key.is_string:
                position = int(key.this) - 1
                if position < 0 or position >= len(tree.expressions):
                    raise ValueError("order_ordinal")
                expr = tree.expressions[position]
                key = expr.this if isinstance(expr, exp.Alias) else expr
            elif isinstance(key, exp.Column) and not key.table and key.name.lower() in aliases:
                key = aliases[key.name.lower()]
            if key.find(exp.Window) or isinstance(key, exp.Star):
                raise ValueError("window_order_or_star")
            term.set("this", key.copy())
        probe = tree.copy()
        probe.select(exp.alias_(exp.Window(this=exp.DenseRank(), order=order), "__v3_order_rank"), append=True, copy=False)
        probe.set("offset", None)
        if limit is not None:
            if offset + limit + 1 > MAX_ROWS:
                raise ValueError("limit_probe_over_row_cap")
            probe.set("limit", exp.Limit(expression=exp.Literal.number(offset + limit + 1)))
        else:
            probe.set("limit", None)
        result = execute(Path(path), probe.sql(dialect="sqlite"), seconds=SCORING_SECONDS, max_ops=SCORING_MAX_OPS)
        if result.error:
            raise ValueError("rank_probe: " + result.error)
        selected = result.rows[offset:offset + limit if limit is not None else None]
        if not rows_equal([r[:-1] for r in selected], gold.rows, ordered=True):
            raise ValueError("rank_probe_changed_reference")
        if selected and offset and result.rows[offset - 1][-1] == selected[0][-1]:
            return policy(False, "offset_cuts_tied_group", False)
        end = offset + len(selected)
        if selected and limit is not None and end < len(result.rows) and selected[-1][-1] == result.rows[end][-1]:
            return policy(False, "limit_cuts_tied_group", False)
        ranks = [r[-1] for r in selected]
        if len(set(ranks)) < len(ranks):
            return policy(False, "tied_order_keys_marked_unordered", tie_rows=len(ranks) - len(set(ranks)))
        return policy(True, "unique_order_keys")
    except (ValueError, TypeError, KeyError, IndexError, sqlglot.errors.SqlglotError) as error:
        return policy(False, "order_probe_unresolved: " + str(error), not has_limit)
