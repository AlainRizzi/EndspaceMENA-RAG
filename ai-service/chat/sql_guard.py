import sqlglot
from sqlglot import exp

# The only tables the generated SQL may ever name - exactly what ai_readonly
# has SELECT on (schema.sql). Unqualified names are assumed to mean ai.*,
# since ai_readonly's search_path is ai, public and every view lives in ai.
ALLOWED_VIEWS = {
    "v_project", "v_task", "v_task_assignee", "v_task_activity", "v_scope",
    "v_invoice", "v_invoice_item", "v_expense", "v_quote", "v_budget", "v_rate_card",
    "v_announcement", "v_announcement_comment", "v_contact", "v_company_contact",
    "v_department", "v_position", "v_skill",
    "v_staff", "v_user_skill", "v_leave_request", "v_leave_policy", "v_staff_leave_balance",
    "v_feedback", "v_feedback_submission", "v_staff_note", "v_goal", "v_objective",
}

MAX_ROW_LIMIT = 500


class UnsafeQueryError(ValueError):
    pass


def validate_select(sql: str) -> str:
    """Rejects anything that isn't a single, plain SELECT touching only
    ALLOWED_VIEWS. Returns the query with a LIMIT enforced (caps it down if
    the generated query already has one higher than MAX_ROW_LIMIT, adds one
    if missing). Raises UnsafeQueryError with the reason otherwise.

    This is the query inspector from the implementation guide (Part B.5): the
    generated SQL is untrusted output from an LLM, which may have been
    steered by a prompt-injected document, so every check here assumes the
    worst rather than trusting the model followed instructions.
    """
    statements = sqlglot.parse(sql, read="postgres")
    if len(statements) != 1:
        raise UnsafeQueryError(f"expected exactly one statement, got {len(statements)}")

    stmt = statements[0]
    if stmt is None or not isinstance(stmt, exp.Select):
        raise UnsafeQueryError("only a single SELECT statement is allowed")

    # CTEs writing via DML, or a SELECT nested under WITH ... INSERT-like
    # constructs, would already fail the isinstance(Select) check above since
    # sqlglot parses those as their own statement types - but explicitly walk
    # for any DML/DDL node anywhere in the tree (e.g. hidden in a subquery)
    # as defense in depth rather than relying on the top-level type alone.
    forbidden_types = (exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create,
                       exp.Alter, exp.Grant, exp.Command, exp.Merge)
    for node in stmt.walk():
        if isinstance(node, forbidden_types):
            raise UnsafeQueryError(f"forbidden statement type: {type(node).__name__}")

    for table in stmt.find_all(exp.Table):
        table_name = table.name
        schema_name = table.db or ""  # sqlglot returns '' (not None) for an unqualified table
        if schema_name not in ("", "ai"):
            raise UnsafeQueryError(f"schema not allowed: {schema_name}")
        if table_name not in ALLOWED_VIEWS:
            raise UnsafeQueryError(f"table/view not allowed: {table_name}")

    # No function calls that could reach outside the row-filtering the views
    # already do - e.g. dblink, pg_read_file, current_setting() tampering.
    # Aggregate/scalar built-ins (count, sum, lower, coalesce, ...) are fine;
    # only flag ones capable of side effects or reading arbitrary session state.
    disallowed_functions = {"dblink", "pg_read_file", "pg_read_binary_file",
                            "lo_import", "lo_export", "set_config", "pg_sleep"}
    for func in stmt.find_all(exp.Anonymous, exp.Func):
        func_name = (getattr(func, "this", None) or "")
        if isinstance(func_name, str) and func_name.lower() in disallowed_functions:
            raise UnsafeQueryError(f"function not allowed: {func_name}")

    existing_limit = stmt.args.get("limit")
    if existing_limit is not None:
        try:
            limit_value = int(existing_limit.expression.this)
        except (AttributeError, ValueError, TypeError):
            limit_value = None
        if limit_value is None or limit_value > MAX_ROW_LIMIT:
            stmt.set("limit", exp.Limit(expression=exp.Literal.number(MAX_ROW_LIMIT)))
    else:
        stmt.set("limit", exp.Limit(expression=exp.Literal.number(MAX_ROW_LIMIT)))

    return stmt.sql(dialect="postgres")
