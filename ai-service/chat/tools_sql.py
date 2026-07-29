from chat.permissions import find_missing_entities
from chat.sql_guard import UnsafeQueryError, validate_select
from db import get_ai_readonly_pool
from llm_client import llm_client

_SCHEMA_DESCRIPTION = """\
Available read-only views (all already filtered to what the caller is
permitted to see, based on their role's abilities and real ownership/
membership relations - never add your own organisationSlug/userId filters
for visibility, they're applied automatically). organisationSlug appears on
some views as a plain output column only - it is NOT a scoping boundary,
since a caller can legitimately have visible rows (e.g. project membership)
across many organisations at once. Never filter on organisationSlug unless
the question explicitly asks about a specific organisation by name.

Every view with a currency column has it because currency is set per
organisation, not globally - two rows in the same result can legitimately be
in different currencies. Whenever you SELECT any money amount from a view
that has a currency column, SELECT currency too and state it alongside every
number in the final answer (e.g. "AUD 4,320.00", not just "4,320.00") -
never state a bare number for money without its unit. If a question
aggregates money across rows that could span more than one organisation
(e.g. summed/compared across projects or orgs), and their currencies differ,
say so explicitly rather than presenting one combined total as if it were a
single currency.

Any column ending in "Id" (managerId, companyId, requestorId, purchaserId,
authorUserId, etc.) is a bare foreign-key reference, never a meaningful
answer on its own - a userId-shaped one resolves to a name via
v_staff_directory (join on userId), a companyId resolves via
v_company_contact (join on id). If the question needs to know WHO or WHAT a
foreign-key column points to (a manager, purchaser, author, company, ...),
JOIN to resolve the real name - never select the bare id alone and never
state the number itself as if it were an answer.

The SAME rule applies to any column ending in "Slug" (projectSlug, scopeSlug,
organisationSlug's exception noted separately above). A slug is an internal
identifier, not a display name, even though it's often readable text - never
select a bare *Slug column as the way to identify an entity in the answer
unless the user explicitly asked for the slug/id itself. If the question is
about which project/scope something belongs to, JOIN projectSlug to
v_project.slug for v_project.name, or scopeSlug to v_scope.slug for
v_scope.name, and use the real name in the answer - do not carry a raw
*Slug column through to the final result just because it happened to be on
the row you queried:

v_project(id, slug, name, organisationSlug, customId, description, status, priority, type, dueDate, startDate, managerId, companyId, createdAt, updatedAt)
  - contains every project the caller is PERMITTED to view, which for a
  manager/admin role can include projects they are not personally a member
  of. "my projects"/"projects I'm working on"/"projects I'm on" means actual
  team membership, NOT everything a broad role happens to permit viewing -
  for that meaning, use v_project_member (below), not v_project, even though
  v_project would also return a (larger, wrong-for-this-question) result.
  managerId/companyId are bare foreign-key ids, never meaningful on their
  own - if the question needs the manager's or company's NAME (e.g. "who
  manages this project", "give me this project's information"), JOIN
  managerId to v_staff_directory.userId for the manager's fullName, and
  companyId to v_company_contact.id for the company's name. Never select
  managerId/companyId alone and never state the bare number as if it were
  an answer.
v_project_member(projectSlug, userId) - actual project team membership. Use
  this, not v_project, for any "my"/"I'm on"/"working on" project question.
  userId here may belong to a different organisation than the caller's;
  that's real cross-org collaboration data, not a leak.
v_rag_source(id, sourceType, projectSlug, status, ingestedAt, name, fileType)
  - THIS is the inventory of what's been ingested into the knowledge base -
  use it for "what documents does project X have", "which documents are
  ingested/searchable", "list PDFs for project X" style questions. There is
  NO v_document/v_project_document/v_file view - do not invent one; this is
  the only view for this. name is the real ingested item's name/title
  (already resolved per sourceType - a project document's real filename, a
  scope document's filename, "Activity on task: X" for a task's activity
  log, an announcement's title, ...) - never confuse it with the enclosing
  project's own name. Filenames themselves do NOT reliably include an
  extension (confirmed live - a real invoice PDF was named just
  "INV-BCG1005", no ".pdf") - never filter file type via name ILIKE
  '%.pdf', use the fileType column instead. fileType is only populated for
  PROJECT_DOCUMENT/SCOPE_DOCUMENT/MEDIA_PLAN_DOCUMENT (NULL for the other
  sourceTypes, which have no underlying file) and its stored FORMAT is
  inconsistent across those three - sometimes a short code ('pdf', 'img'),
  sometimes a full MIME type ('application/pdf') - always use
  fileType ILIKE '%pdf%' (or the relevant type), never exact equality.
  sourceType is one of: PROJECT_DOCUMENT, SCOPE_DOCUMENT,
  MEDIA_PLAN_DOCUMENT, ACTIVITY_LOG, ANNOUNCEMENT, ANNOUNCEMENT_COMMENT,
  OBJECTIVE. This lists WHAT EXISTS, not document CONTENT - to search inside
  a document's actual text, use the search_knowledge_base tool instead, not
  this view.
v_task(id, name, description, projectSlug, status, ownerId, startDate, dueDate, estimated, logged, taskType, flagged, isDeleted, createdAt, updatedAt)
  - ownerId is who the task actually belongs to in this data - "who has/owns
  this task", "my tasks", "tasks assigned to X" should filter on v_task.ownerId
  first. v_task_assignee (below) is a separate, often-empty table - do not
  assume a task has an assignee there just because it has an owner here.
v_task_assignee(id, taskId, projectSlug, assigneeId, estimatedTime, createdAt)
v_task_activity(id, taskId, taskName, projectSlug, createdAt)
v_scope(id, name, slug, customId, projectSlug, organisationSlug, status, type, dueDate, companyId, createdAt, total, subTotal, estDeal, estRevenue, estCostOfSale, forecastRevenue, closeProbability, wonAt, currency)
  - total is the scope's approved/contracted amount ("Contracted Revenue" for
  the project it belongs to). estDeal/estRevenue/estCostOfSale/
  closeProbability are pre-close forecast figures for a not-yet-won scope;
  forecastRevenue = estDeal * closeProbability. wonAt is set once the scope
  is actually won (status = 'WON').
v_invoice(id, customId, organisationSlug, companyId, projectSlug, scopeSlug, type, issueDate, dueDate, amountPaid, paidAt, paymentStatus, balance, currency, billToFinancialDetailId)
v_invoice_item(id, invoiceId, description, quantity, unitPrice, discount, amount, currency)
v_expense(id, customId, organisationSlug, projectSlug, purchaserId, purchaseDate, dueDate, cost, billed, profit, action, currency, supplierId, markup, markupType, totalPaid, balance, status)
  - purchaserId is the internal staff member who made the purchase - NOT the
  supplier. supplierId identifies the actual supplier/vendor (join
  v_supplier below for supplier-level detail). "Cost of Goods" = cost;
  "Billed Amount" = billed; profit = billed - cost.
v_supplier(id, customId, paymentStatus, mainTradingName, total_expenses, total_cost_of_goods, total_billed, total_profit, total_outstanding)
  - one row per supplier, already rolled up across every expense linked to
  it that the caller can see. Use this instead of manually aggregating
  v_expense by supplierId.
v_scope_service(id, sectionId, scopeSlug, projectSlug, serviceName, quantity, totalCost, totalAmount, labour_cost_actual, currency)
  - per-service budget line within a scope (a scope's individual billable
  services, e.g. "Social Media Marketing", "Website Redesign"). totalAmount
  is the service's contracted price. KNOWN APPROXIMATION:
  labour_cost_actual sums ALL of the parent scope's logged time, not just
  time logged specifically against this one service - there is no
  per-service time-tracking link in this data. If a question needs an exact
  per-service labour cost, say this figure is scope-wide, not
  service-specific.
v_retainer_period(id, projectSlug, scopeSlug, periodName, periodIndex, startDate, endDate, budgetedHours, budgetedAmount, usedHours, incomeToDate, currency)
  - one row per retainer billing period (e.g. "May 2026") for a project on
  a retainer arrangement. budgetedHours/budgetedAmount can be NULL for a
  period that hasn't had a budget configured yet (confirmed live) - that is
  a real gap in that period's setup, not a zero; say so rather than
  treating NULL as 0 for these two columns specifically. usedHours/
  incomeToDate are always real numbers (default to 0, not NULL).
v_resourcing(id, scopeSlug, projectSlug, memberId, futureResourcing, month, year, allocated_hours, status)
  - planned staffing allocation per person per month for a scope. This is
  PLANNED allocation only, not actual logged time - for actual hours
  worked, use v_time_entry instead. There is no link between this and
  v_task/task estimates in this data - never compare or combine
  allocated_hours with v_task.estimated as if they were the same kind of
  number.
v_customer(id, companyId, name, email, abn, total_invoices, total_paid, total_outstanding_balance)
  - the customer/company an invoice is billed to (their financial/billing
  contact record - name, email, abn). To find the customer for a specific
  project/invoice: v_invoice.billToFinancialDetailId = v_customer.id (join
  on that column). Use this for "who is the customer for project X" or
  "what's the ABN on the latest invoice for project X" style questions -
  find the relevant invoice(s) in v_invoice first, then join to v_customer
  on billToFinancialDetailId.
v_quote(id, quote_number, job_title, organisationSlug, project_id, issued_on, subtotal, gst, total, status, currency)
v_rate_card(id, rateCardGroupId, positionId, hourlyRate, dailyRate)
v_announcement(id, organisationSlug, authorUserId, type, status, title, contentText, startsAt, endsAt, isPinned, publishedAt, createdAt)
v_announcement_comment(id, announcementId, authorUserId, contentText, status, createdAt)
v_contact(id, slug, name, type, city, state, country, website, email, createdAt)
v_company_contact(id, customId, email, companyType, status, name)
v_department(id, name, organisationSlug)
v_position(id, name, departmentId, organisationSlug)
v_skill(id, name, organisationSlug)
v_staff_directory(userId, fullName, organisationSlug, jobTitle, departmentId, positionId)
v_staff(userId, fullName, organisationSlug, jobTitle, employmentStatus, hireDate, departmentId, positionId)
v_user_skill(id, userId, skillId, organisationSlug)
v_leave_request(id, requestorId, managerId, leaveStartDate, leaveEndDate, status, leavePolicyId, requestedDuration, durationUnit, createdAt, organisationSlug)
v_leave_policy(id, name, entitlement, entitlementUnit, recurringPeriod, isPaid, allowFullDay, allowHalfDay, applicableAfter, applicableAfterUnit, allowCarryForward, accrualRate, maxAccrual, maxCarryForward, organisationSlug)
  - spans every organisation the caller has ANY relationship to (their own
  employer plus every org they have cross-org access/membership to), NOT
  just their own employer - a caller can have policies from several
  organisations show up here at once. For "how many leave days does person X
  have" (X's OWN leave, not a generic policy lookup), you MUST restrict
  v_leave_policy to X's own employing organisation first - join through
  v_staff (staffUserId/userId = X's id) or v_leave_request/
  v_staff_leave_balance (whichever already has X's rows) to get that
  organisationSlug, then match v_leave_policy.organisationSlug to it. Never
  join X to every organisationSlug v_leave_policy happens to return.
  A policy DOES NOT APPLY YET to a staff member whose tenure is under
  applicableAfter (in applicableAfterUnit, e.g. 12 MONTHS) - if not yet
  applicable, their remaining days for that policy is 0, not entitlement
  (confirmed live: a staff member hired under a year ago genuinely has 0
  Annual Leave available where that policy requires 12 months). To check
  this, ALWAYS use the function ai.leave_policy_is_eligible(target_user_id,
  policy_id) - e.g. WHERE ai.leave_policy_is_eligible(s."userId", lp.id) -
  never write the AGE()/interval comparison inline yourself. This is
  mandatory, not optional: a query that omits this eligibility check will
  silently show an ineligible policy's full entitlement as if the person
  already qualified for it.
  KNOWN GAP: allowCarryForward/accrualRate/maxAccrual/
  maxCarryForward are NOT factored into the remaining-leave formula below -
  a long-tenured staff member's real entitlement can exceed the flat
  `entitlement` value through accrual/carry-forward that this data cannot
  currently reproduce exactly (confirmed live: a real example showed a
  higher true balance than the formula below produces). If a computed
  number seems load-bearing for the user's decision, mention it's based on
  the base policy terms and may not reflect accrued/carried-forward leave.
v_staff_leave_balance(id, staffUserId, leavePolicyId, openingBalance, organisationSlug)
v_project_budget(projectSlug, budget_total_estimated, labour_cost_actual, expense_cost_actual, contracted_revenue_total, currency)
  - THIS is per-project budget data - use it for "what's this project's
  budget", "which project has the highest budget", budget used/remaining,
  profit/loss style questions. One row per project.
  budget_total_estimated = the project's allocated/estimated budget (sum of
  each task's estimated time x that task owner's project-specific rate -
  see v_project_member_rate below). KNOWN DATA GAP: if a project has tasks
  but budget_total_estimated is 0, that means no project-specific rates are
  recorded for its task owners in this environment - say so explicitly
  (e.g. "no rate data recorded for this project's team, so an estimated
  budget can't be computed") rather than stating 0 as the real budget.
  labour_cost_actual = actual logged labour cost so far (from v_time_entry).
  expense_cost_actual = actual expense cost so far (from v_expense).
  contracted_revenue_total = sum of the project's scopes' contracted amount
  (v_scope.total).
  Derive, don't expect stored: budget_remaining = budget_total_estimated -
  labour_cost_actual - expense_cost_actual; current_profit =
  contracted_revenue_total - labour_cost_actual - expense_cost_actual.
  For "over/under budget" or "budget overrun" questions SPECIFICALLY:
  budget_remaining being negative is only a real overrun when
  budget_total_estimated > 0 for that project - a project with
  budget_total_estimated = 0 will ALWAYS show as "negative remaining" purely
  because there's no recorded budget to compare against (see the data gap
  above), not because it was actually exceeded. Add
  WHERE budget_total_estimated > 0 ONLY for this specific question shape, so
  the comparison is only made among projects that actually have a real
  budget figure to be over or under.
  For "which project has the highest/lowest budget" or any other ranking/
  lookup of budget_total_estimated itself: do NOT filter out
  budget_total_estimated = 0 rows - if every project in scope has 0 (the
  data gap above, environment-wide), the query must still return those rows
  (e.g. ORDER BY budget_total_estimated DESC LIMIT 1, with no WHERE > 0) so
  the answer can honestly say no project has a recorded budget yet, rather
  than returning zero rows with no explanation of why.
v_time_entry(id, taskId, memberId, scopeSlug, invoiceId, recordType, duration, cost, total, dayCreated, createdAt)
  - logged work time per task. memberId is the staff member who logged it
  (join to v_staff_directory for their name). duration is in seconds. No
  billable/non-billable column exists - never claim a time entry is
  "billable" or filter on it, that distinction isn't tracked here. If a
  SUM(duration) for someone returns NULL (no rows at all, not zero rows
  summed to zero), that means 0 hours logged - state it as 0, do not say you
  don't have enough information, the same "no rows means zero, not no data"
  rule as v_leave_request/v_staff_leave_balance.
v_project_member_rate(projectSlug, staffId, hourlyRate)
  - a staff member's rate specifically on this project (NOT the org-wide
  v_rate_card, which is keyed by position, not by person - project-level
  budget/cost math must use this table when both could apply, matching how
  v_project_budget itself is computed).
v_budget(id, name, organisationSlug, financialYearId, createdAt, currency) - an
  org-wide financial budget for a year, with NO link to any project. Never
  use this for a "which project" or per-project budget question - use
  v_project_budget instead.
v_budget_data(id, accountBudgetId, budgetId, budgetName, organisationSlug, month, year, value, currency)
  - monthly dollar value per chart-of-accounts line within an org-wide
  budget (an account-level breakdown, not project-level). Use for "what's our budget
  for account X" or "total budgeted for year Y" style questions.
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
- To resolve a userId into a name (e.g. listing who's on a project/team),
  use v_staff_directory, not v_staff - v_staff is scoped to the caller's own
  record (or anyone's if a manager/admin), so joining it to something like
  v_project_member silently drops teammates the caller isn't otherwise
  allowed to see as individuals, even though the roster itself is visible.
- A count of who's on a project/team includes the caller if they're a member.
  A NAMED list of "my teammates"/"who else is on this" excludes the caller
  themselves if they're a member (they already know they're on it) - but if
  the caller is not a member of the project at all, a named list includes
  everyone, since there's no "themselves" to exclude.
- Values that aren't stored directly (totals, remainders, balances, counts,
  averages, etc.) should be computed in SQL by joining/aggregating the
  relevant views. Work out which views apply to each specific question rather
  than assuming a fixed formula.
- When a question asks "which project/company/policy/..." (i.e. wants that
  entity identified, not just referenced), SELECT its real name column via a
  join (e.g. v_task.projectSlug -> JOIN v_project ON slug = projectSlug,
  SELECT v_project.name), not just the slug/id foreign key column alone - a
  slug is not a name and must never be presented as one. If joining for the
  name isn't possible for some reason, select the slug/id explicitly so the
  answer can say it only has the identifier, rather than silently formatting
  a slug (e.g. "yeni-gate") to look like a name (e.g. "Yeni Gate").
- v_leave_policy always has a row per policy; v_staff_leave_balance and
  v_leave_request may have no rows at all for a given staff member (not
  everyone has an opening balance or has made a request) - that means "zero",
  not "no data", so anchor queries involving these on whichever view is
  guaranteed to have the rows you need, and join outward from there.
- A staff member's remaining leave for a policy is
  v_leave_policy.entitlement + v_staff_leave_balance.openingBalance -
  (sum of their APPROVED v_leave_request.requestedDuration for that policy).
  entitlement is required in this sum - it is not optional or a fallback.
  Different policies (Annual Leave, Sick Leave, Unpaid Leave, ...) are
  SEPARATE, non-fungible pools - compute and report this per policy name,
  never summed into one blended "days left" total across policies, since
  a policy someone hasn't touched still has its own full entitlement
  remaining and mixing pools together produces a number that isn't real.
- Always include an entity's slug/id column alongside its name when selecting
  it (e.g. v_project.slug with v_project.name) - the answer may be used to
  identify that same entity again in a later question.
- v_staff.fullName holds a full name, not just a first or last name - a name
  filter must use a wildcard partial match against it, not exact equality.
"""


class TextToSqlError(Exception):
    pass


class QueryNotPermittedError(TextToSqlError):
    """Raised specifically when the query inspector rejected the generated
    SQL for touching something outside the allowed views/columns (see
    sql_guard.UnsafeQueryError) - distinct from every other failure (a bad
    join, a timeout, a retry that still didn't work), which means "something
    went wrong," not "you're not allowed to see that."
    """


async def generate_sql(question: str, user_id: int | None, retry_error: str | None = None) -> str:
    """LLM drafts a SELECT against the curated views. The draft is untrusted
    output - it's validated by sql_guard.validate_select before ever being
    considered for execution (see run_text_to_sql).
    """
    from chat.schemas import GeneratedSql

    caller_line = (
        f"The caller's own userId is {user_id}."
        if user_id is not None
        else "The caller's userId is unknown."
    )

    retry_line = (
        f"\nYour previous attempt failed with this database error - fix it:\n{retry_error}\n"
        if retry_error else ""
    )

    prompt = f"""You write a single read-only PostgreSQL SELECT query to answer a
question, using ONLY the views listed below. Never invent columns or tables.

Never add WHERE clauses for organisationSlug - visibility is already enforced
by the views themselves via the caller's role abilities and real ownership/
membership relations, and organisationSlug is not a scoping boundary (a
caller can legitimately have visible rows spanning many organisations).

{caller_line} Two different things both use "userId" but must not be
confused: (a) VISIBILITY - which rows the caller is even allowed to see is
already enforced automatically by the views (a regular staff member only
ever sees their own leave/feedback/notes rows; a manager/admin sees
everyone's) - never add a userId filter just to enforce this. (b) SCOPE - if
the question is specifically about "me"/"my"/"I" (e.g. "how many leave days
do I have left", "my leave requests"), you MUST add an explicit
WHERE ... = {user_id} filter on the relevant user-identifying column
(requestorId, userId, staffUserId, ownerId, assigneeId, etc. depending on the
view) - otherwise a manager/admin's broader visibility means the query
aggregates or lists everyone's data instead of just theirs, which answers a
different question than what was asked. If the question is about people in
general (no "me"/"my"), do not add this filter.

Every mixed-case column name below (e.g. projectSlug, organisationSlug,
createdAt) MUST be double-quoted exactly as shown (e.g. "projectSlug") in the
query - Postgres silently lowercases unquoted identifiers, which would break
the reference. Lowercase columns like id, name, status need no quoting.

{_SCHEMA_DESCRIPTION}

Question: {question}
{retry_line}
Return only the SQL query."""

    result, _usage = await llm_client.call_structured(prompt, GeneratedSql)
    return result.sql


async def _validate_and_run(sql: str, user_id: int | None) -> list[dict]:
    safe_sql, referenced_views = validate_select(sql)  # UnsafeQueryError propagates uncaught - not retryable

    # Proactive check, before the query ever runs: if the caller's role holds
    # NONE of a referenced view's real abilities, the query is guaranteed to
    # come back empty specifically because of that - not because the data
    # happens to be empty. Catching this here (rather than inferring it from
    # an empty result afterward) is the only way to tell those two cases
    # apart, since a genuinely-empty allowed view and a permission-filtered
    # view are otherwise indistinguishable by the time rows come back.
    denied_views = await find_missing_entities(referenced_views, user_id)
    if denied_views:
        raise QueryNotPermittedError(
            f"caller's role has no read ability for: {', '.join(sorted(denied_views))}"
        )

    pool = await get_ai_readonly_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            # Session-local GUC (not connection-wide) - set_config's third
            # arg (is_local=true) scopes it to this transaction only, so a
            # pooled connection can never leak one request's user into the
            # next request that reuses it. No app.org_slug here - no view
            # reads it (see schema.sql: Ability is the only access gate).
            await conn.execute(
                "SELECT set_config('app.user_id', $1, true)", str(user_id) if user_id is not None else ""
            )
            rows = await conn.fetch(safe_sql)

    return [dict(r) for r in rows]


async def run_text_to_sql(question: str, user_id: int | None) -> list[dict]:
    """Full pipeline: LLM drafts SQL -> query inspector validates/caps it ->
    executes as ai_readonly with the session GUC the views' RLS/ownership
    predicates read (see schema.sql: ai.session_user_id(), visible_project_slugs()).

    One retry on a database error (e.g. a wrong column/alias) with the error
    fed back to the LLM - occasional SQL mistakes are expected from generated
    SQL and are usually fixable given the exact error, so this is cheaper and
    more reliable than prompt engineering for every possible mistake.
    """
    raw_sql = await generate_sql(question, user_id)
    try:
        return await _validate_and_run(raw_sql, user_id)
    except UnsafeQueryError as e:
        # Not retried - the model reaching for a disallowed table/column
        # isn't a syntax mistake a retry would fix, it's the fence working.
        raise QueryNotPermittedError(str(e)) from e
    except QueryNotPermittedError:
        # Not retried either - raised directly by _validate_and_run's ability
        # pre-check. A retry can't fix "the caller's role doesn't have this
        # ability" by rewriting the query differently.
        raise
    except Exception as e:
        retry_sql = await generate_sql(question, user_id, retry_error=str(e))
        try:
            return await _validate_and_run(retry_sql, user_id)
        except UnsafeQueryError as retry_e:
            raise QueryNotPermittedError(str(retry_e)) from retry_e
        except QueryNotPermittedError:
            raise
        except Exception as retry_e:
            raise TextToSqlError(f"query failed after retry: {retry_e}") from retry_e
