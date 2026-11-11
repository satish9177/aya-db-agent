"""Convert legacy scalar-answer shop references alongside their unchanged original records."""
import sqlglot
from sqlglot import exp


def scalar_reference(sql: str) -> str:
    tree = sqlglot.parse_one(sql, read="sqlite")
    if not isinstance(tree, exp.Select):
        raise ValueError("shop reference must be a SELECT")
    aliases = {e.alias.lower(): e.this for e in tree.expressions if isinstance(e, exp.Alias)}
    order = tree.args.get("order")
    if order:
        for term in order.expressions:
            key = term.this
            if isinstance(key, exp.Column) and not key.table and key.name.lower() in aliases:
                term.set("this", aliases[key.name.lower()].copy())
    tree.set("expressions", [tree.expressions[0]])
    # Legacy generators reject ties at the winner. The v3 ordering probe checks
    # that the projected query's LIMIT really has an unambiguous boundary.
    if order:
        tree.set("limit", exp.Limit(expression=exp.Literal.number(1)))
    return tree.sql(dialect="sqlite")
