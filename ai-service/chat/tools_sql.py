from chat.sql_guard import UnsafeQueryError, validate_select
from db import get_ai_readonly_pool
from llm_client import llm_client

_SCHEMA_DESCRIPTION = """\
Available read-only views (all already scoped to the caller's organisation and
to what they're allowed to see - never add your own organisationSlug/userId
filters, they're applied automatically):

v_project(id, slug, name, organisationSlug, customId, description, status, priority, type, dueDate, startDate, managerId, companyId, createdAt, updatedAt)
v_task(id, name, description, projectSlug, status, ownerId, startDate, dueDate, estimated, logged, taskType, flagged, isDeleted, createdAt, updatedAt)
v_task_assignee(id, taskId, projectSlug, assigneeId, estimatedTime, createdAt)
v_task_activity(id, taskId, taskName, projectSlug, createdAt)
v_scope(id, name, slug, customId, projectSlug, organisationSlug, status, type, dueDate, companyId, createdAt)
v_invoice(id, customId, organisationSlug, companyId, projectSlug, scopeSlug, type, issueDate, dueDate, amountPaid, paidAt, paymentStatus, balance)
v_invoice_item(id, invoiceId, description, quantity, unitPrice, discount, amount)
v_expense(id, customId, organisationSlug, projectSlug, purchaserId, purchaseDate, dueDate, cost, billed, profit, action)
v_quote(id, quote_number, job_title, organisationSlug, project_id, issued_on, subtotal, gst, total, status)
v_budget(id, name, organisationSlug, financialYearId, createdAt)
v_rate_card(id, rateCardGroupId, positionId, hourlyRate, dailyRate)
v_announcement(id, organisationSlug, authorUserId, type, status, title, contentText, startsAt, endsAt, isPinned, publishedAt, createdAt)
v_announcement_comment(id, announcementId, authorUserId, contentText, status, createdAt)
v_contact(id, slug, name, type, city, state, country, website, email, createdAt)
v_company_contact(id, customId, email, companyType, status, name)
v_department(id, name, organisationSlug)
v_position(id, name, departmentId, organisationSlug)
v_skill(id, name, organisationSlug)
v_staff(userId, fullName, organisationSlug, jobTitle, employmentStatus, hireDate, departmentId, positionId)
v_user_skill(id, userId, skillId, organisationSlug)
v_leave_request(id, requestorId, managerId, leaveStartDate, leaveEndDate, status, leavePolicyId, requestedDuration, durationUnit, createdAt, organisationSlug)
v_leave_policy(id, name, entitlement, entitlementUnit, recurringPeriod, isPaid, allowFullDay, allowHalfDay, organisationSlug)
v_staff_leave_balance(id, staffUserId, leavePolicyId, openingBalance, organisationSlug)
v_feedback(id, userId, createdById, completionDate, message, createdAt, organisationSlug)
v_feedback_submission(id, feedbackId, submitterId, firstAnswer, secondAnswer, status, createdAt, organisationSlug)
v_staff_note(id, userId, note, createdAt, organisationSlug)
v_goal(id, title, startDate, endDate, status, progress, userId, organisationSlug)
v_objective(id, detail, done, goalId, organisationSlug)

Enum columns only ever contain these exact UPPERCASE values - never guess or
invent a value (e.g. never write 'paid' or 'Paid'; the real value is 'PAID'):
- v_project.status: ACTIVE, INACTIVE
- v_project.priority: PREMIUM, BASIC
- v_project.type: BILLABLE, UNBILLABLE
- v_invoice.paymentStatus: DRAFT, AWAITING, OVERDUE, PAID, UNPAID, CANCELLED, VOIDED, REPEATING, PARTIALLY_PAID
- v_invoice.type / v_expense.action uses ServiceType or ExpenseAction respectively - see below
- v_expense.action: UNPAID, PAID, DEPOSIT, DRAFT, WAITING_FOR_APPROVAL
- v_scope.status: ACTIVE, LOST, DIFFER, WON
- v_scope.type / v_task.taskType: ONE_OFF, RECURRING, EXPENSE, MEDIA
- v_announcement.status: DRAFT, PUBLISHED, HIDDEN, ARCHIVED
- v_announcement.type: POST, EVENT
- v_company_contact.companyType: BUSINESS_CORPORATION, INTERNAL_ORGANISATION, SOLE_PROPRIETORSHIP, NON_PROFIT_CORPORATION
- v_company_contact.status: COMPLETED, ACTIVE, INACTIVE
- v_contact.type: COMPANY, SUPPLIER, MEDIA_VENDOR
- v_staff.employmentStatus: FULL_TIME, PART_TIME, CONTRACTOR, OTHER, TERMINATED
- v_leave_request.status: APPROVED, CANCELLED, REJECTED, WAITING
- v_leave_request.durationUnit / v_leave_policy.entitlementUnit: HOURS, DAYS, WEEKS, MONTHS, YEARS
- v_feedback_submission.status: PENDING, COMPLETED
- v_goal.status: IN_PROGRESS, COMPLETED, CLOSED, OVERDUE
Note: v_task.status is free-text (a status name like "In Progress"), not one
of these fixed enums - use ILIKE or check plausible values, don't assume
an exact fixed set for it.

Notes:
- v_staff intentionally has no salary/compensation columns - never claim or
  compute one; if asked, say that data isn't available through this tool.
- Values that aren't stored directly (totals, remainders, balances, counts,
  averages, etc.) should be computed in SQL by joining/aggregating the
  relevant views - e.g. a leave balance combines v_leave_policy.entitlement,
  v_staff_leave_balance.openingBalance, and a SUM over matching
  v_leave_request rows; an invoice's outstanding amount combines v_invoice
  and v_invoice_item. Work out which views apply to each specific question
  rather than assuming a fixed formula.
"""


class TextToSqlError(Exception):
    pass


async def generate_sql(question: str) -> str:
    """LLM drafts a SELECT against the curated views. The draft is untrusted
    output - it's validated by sql_guard.validate_select before ever being
    considered for execution (see run_text_to_sql).
    """
    from chat.schemas import GeneratedSql

    prompt = f"""You write a single read-only PostgreSQL SELECT query to answer a
question, using ONLY the views listed below. Never invent columns or tables.
Never add WHERE clauses for organisationSlug/userId - visibility is already
enforced by the views themselves.

Every mixed-case column name below (e.g. projectSlug, organisationSlug,
createdAt) MUST be double-quoted exactly as shown (e.g. "projectSlug") in the
query - Postgres silently lowercases unquoted identifiers, which would break
the reference. Lowercase columns like id, name, status need no quoting.

{_SCHEMA_DESCRIPTION}

Question: {question}

Return only the SQL query."""

    result, _usage = await llm_client.call_structured(prompt, GeneratedSql)
    return result.sql


async def run_text_to_sql(question: str, org_slug: str, user_id: int | None) -> list[dict]:
    """Full pipeline: LLM drafts SQL -> query inspector validates/caps it ->
    executes as ai_readonly with the session GUCs the views' RLS/ownership
    predicates read (see schema.sql: ai.session_user_id(), visible_project_slugs()).
    """
    raw_sql = await generate_sql(question)

    try:
        safe_sql = validate_select(raw_sql)
    except UnsafeQueryError as e:
        raise TextToSqlError(f"generated query rejected: {e}") from e

    pool = await get_ai_readonly_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            # Session-local GUCs (not connection-wide) - set_config's third
            # arg (is_local=true) scopes them to this transaction only, so a
            # pooled connection can never leak one request's org/user into
            # the next request that reuses it.
            await conn.execute("SELECT set_config('app.org_slug', $1, true)", org_slug)
            await conn.execute(
                "SELECT set_config('app.user_id', $1, true)", str(user_id) if user_id is not None else ""
            )
            rows = await conn.fetch(safe_sql)

    return [dict(r) for r in rows]
