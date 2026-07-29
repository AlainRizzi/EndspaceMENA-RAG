import sqlglot
from sqlglot import exp

# The only tables the generated SQL may ever name - exactly what ai_readonly
# has SELECT on (schema.sql). Unqualified names are assumed to mean ai.*,
# since ai_readonly's search_path is ai, public and every view lives in ai.
ALLOWED_VIEWS = {
    "v_project", "v_project_member", "v_task", "v_task_assignee", "v_task_activity", "v_scope",
    "v_invoice", "v_invoice_item", "v_expense", "v_quote", "v_budget", "v_budget_data", "v_rate_card",
    "v_time_entry", "v_project_member_rate", "v_project_budget",
    "v_supplier", "v_scope_service", "v_retainer_period", "v_resourcing", "v_customer",
    "v_announcement", "v_announcement_comment", "v_contact", "v_company_contact",
    "v_department", "v_position", "v_skill",
    "v_staff_directory", "v_staff", "v_user_skill", "v_leave_request", "v_leave_policy", "v_staff_leave_balance",
    "v_feedback", "v_feedback_submission", "v_staff_note", "v_goal", "v_objective",
}

# Every mixed-case column across the views above. Postgres lowercases
# unquoted identifiers, so an LLM-generated query that writes projectSlug
# instead of "projectSlug" silently breaks (UndefinedColumnError, or worse,
# silently matches nothing). Rather than relying on prompt wording alone
# (unreliable - confirmed empirically, Gemini quotes inconsistently across
# a single query), every Column identifier matching one of these names
# case-insensitively gets force-quoted with the correct casing below.
_MIXED_CASE_COLUMNS = {
    "organisationSlug", "customId", "dueDate", "startDate", "managerId", "companyId",
    "createdAt", "updatedAt", "taskId", "projectSlug", "ownerId", "assigneeId",
    "estimatedTime", "taskName", "scopeSlug", "invoiceId", "unitPrice", "purchaserId",
    "purchaseDate", "financialYearId", "rateCardGroupId", "positionId", "hourlyRate",
    "dailyRate", "authorUserId", "startsAt", "endsAt", "isPinned", "publishedAt",
    "contentText", "departmentId", "userId", "skillId", "requestorId", "leaveStartDate",
    "leaveEndDate", "leavePolicyId", "requestedDuration", "durationUnit",
    "entitlementUnit", "recurringPeriod", "isPaid", "allowFullDay", "allowHalfDay",
    "staffUserId", "openingBalance", "createdById", "completionDate", "feedbackId",
    "submitterId", "firstAnswer", "secondAnswer", "goalId", "fullName", "jobTitle",
    "employmentStatus", "hireDate", "amountPaid", "paidAt", "paymentStatus", "issueDate",
    # v_leave_policy eligibility/accrual columns (added when applicableAfter
    # eligibility gating was built - previously missing here, meaning these
    # would have silently failed to auto-quote in any generated query).
    "applicableAfter", "applicableAfterUnit", "allowCarryForward", "accrualRate",
    "maxAccrual", "maxCarryForward",
    # v_scope money columns (Tier 1 GraySync Formulas integration)
    "subTotal", "estDeal", "estRevenue", "estCostOfSale", "forecastRevenue",
    "closeProbability", "wonAt",
    # v_time_entry / v_project_member_rate / v_project_budget (Tier 1)
    "memberId", "recordType", "dayCreated", "staffId",
    # v_supplier / v_scope_service / v_retainer_period / v_resourcing /
    # v_customer (Tier 2 GraySync Formulas integration)
    "supplierId", "paymentStatus", "mainTradingName", "sectionId", "serviceId",
    "totalCost", "totalAmount", "budgetedHours", "budgetedAmount", "usedHours",
    "incomeToDate", "periodName", "periodIndex", "futureResourcing", "markupType",
    "totalPaid",
}
_MIXED_CASE_COLUMNS_LOWER = {c.lower(): c for c in _MIXED_CASE_COLUMNS}

MAX_ROW_LIMIT = 500


class UnsafeQueryError(ValueError):
    pass


def validate_select(sql: str) -> tuple[str, set[str]]:
    """Rejects anything that isn't a single, plain SELECT touching only
    ALLOWED_VIEWS. Returns (normalized_sql, referenced_views) - the LIMIT is
    enforced (capped down if the generated query already has one higher than
    MAX_ROW_LIMIT, added if missing), and referenced_views is every view name
    the query touches, for the caller to cross-check against the caller's
    real abilities (see permissions.py) before running it. Raises
    UnsafeQueryError with the reason if the query is unsafe to run at all.

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

    referenced_views: set[str] = set()
    for table in stmt.find_all(exp.Table):
        table_name = table.name
        schema_name = table.db or ""  # sqlglot returns '' (not None) for an unqualified table
        if schema_name not in ("", "ai"):
            raise UnsafeQueryError(f"schema not allowed: {schema_name}")
        if table_name not in ALLOWED_VIEWS:
            raise UnsafeQueryError(f"table/view not allowed: {table_name}")
        referenced_views.add(table_name)

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

    # Force correct quoting/casing on every identifier that matches a known
    # mixed-case column, regardless of how the LLM wrote it (unquoted,
    # wrong-case, or correctly quoted already) - see _MIXED_CASE_COLUMNS.
    for identifier in stmt.find_all(exp.Identifier):
        correct_name = _MIXED_CASE_COLUMNS_LOWER.get(identifier.this.lower())
        if correct_name is not None:
            identifier.set("this", correct_name)
            identifier.set("quoted", True)

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

    return stmt.sql(dialect="postgres"), referenced_views
