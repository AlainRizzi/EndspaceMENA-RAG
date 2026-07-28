-- Run this against the graysync database (or add these as Prisma migrations,
-- since Prisma should stay the source of truth for schema on the NestJS side).
--
-- All AI-owned tables live in a dedicated "ai" schema, kept separate from
-- GraySync's own tables in "public". db.py sets search_path=ai,public per
-- connection, so unqualified table names below still resolve without every
-- query needing an "ai." prefix.
--
-- One-time move from the old layout: these tables previously lived in
-- "public". If you have an existing DB from before this change, drop them
-- there first (data is not migrated - rerun `python ingest.py` afterwards
-- to repopulate RagSource/RagChunk under the new "ai" schema):
--   DROP TABLE IF EXISTS public."AiInvocationLog";
--   DROP TABLE IF EXISTS public."ProjectSummary";
--   DROP TABLE IF EXISTS public."RagChunk";
--   DROP TABLE IF EXISTS public."RagSource";
--   DROP TYPE IF EXISTS public."RagIngestStatus";
--   DROP TYPE IF EXISTS public."RagSourceType";

-- This file is safe to rerun end-to-end: every CREATE below is guarded
-- (IF NOT EXISTS / OR REPLACE / DROP ... IF EXISTS first for objects, like
-- policies, that support neither).

CREATE SCHEMA IF NOT EXISTS ai;

CREATE EXTENSION IF NOT EXISTS vector;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type t JOIN pg_namespace n ON n.oid = t.typnamespace
                   WHERE t.typname = 'RagSourceType' AND n.nspname = 'ai') THEN
        CREATE TYPE ai."RagSourceType" AS ENUM (
            'PROJECT_DOCUMENT',
            'TASK_DOCUMENT',
            'STAFF_DOCUMENT',
            'SCOPE_DOCUMENT',
            'MEDIA_PLAN_DOCUMENT',
            'COMPANY_DOCUMENT',
            'EXPENSE_IMPORT_DOCUMENT',
            'TASK_INBOUND_EMAIL',
            'SCOPE_DESCRIPTION',
            'FEEDBACK',
            'FEEDBACK_SUBMISSION',
            'QUESTION_ANSWER',
            'ACTIVITY_LOG',
            'STAFF_NOTE',
            'ANNOUNCEMENT',
            'ANNOUNCEMENT_COMMENT',
            'CONTACT_NOTES',
            'OBJECTIVE'
        );
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_type t JOIN pg_namespace n ON n.oid = t.typnamespace
                   WHERE t.typname = 'RagIngestStatus' AND n.nspname = 'ai') THEN
        CREATE TYPE ai."RagIngestStatus" AS ENUM ('PENDING', 'PROCESSING', 'COMPLETED', 'FAILED');
    END IF;
END
$$;

-- NOTE on organisationSlug:
-- organisationSlug (Organisation.slug) is the tenant boundary, always required.
-- subdomainName was dropped from this scoping - it identifies which hosting
-- instance a row lives on, not which company/organisation owns it, and
-- Organisation.slug is unique across the whole database today.

CREATE TABLE IF NOT EXISTS ai."RagSource" (
    id BIGSERIAL PRIMARY KEY,
    "organisationSlug" text NOT NULL,
    "sourceType" ai."RagSourceType" NOT NULL,
    "sourceId" text NOT NULL,
    "projectSlug" text,
    "taskId" integer,
    "scopeSlug" text,
    "userId" integer,
    "companyId" integer,
    checksum text,
    status ai."RagIngestStatus" NOT NULL DEFAULT 'PENDING',
    "errorMessage" text,
    "ingestedAt" timestamp,
    "createdAt" timestamp NOT NULL DEFAULT now(),
    "updatedAt" timestamp NOT NULL DEFAULT now(),
    UNIQUE ("sourceType", "sourceId")
);
CREATE INDEX IF NOT EXISTS "RagSource_organisationSlug_idx2" ON ai."RagSource" ("organisationSlug");
CREATE INDEX IF NOT EXISTS "RagSource_projectSlug_idx2" ON ai."RagSource" ("projectSlug");
CREATE INDEX IF NOT EXISTS "RagSource_taskId_idx2" ON ai."RagSource" ("taskId");

CREATE TABLE IF NOT EXISTS ai."RagChunk" (
    id BIGSERIAL PRIMARY KEY,
    "sourceId" bigint NOT NULL REFERENCES ai."RagSource"(id) ON DELETE CASCADE,
    "organisationSlug" text NOT NULL,   -- denormalized from RagSource so retrieval can filter without a join
    "chunkIndex" integer NOT NULL,
    content text NOT NULL,
    "tokenCount" integer,
    embedding vector(1024),  -- dimension must match your embedding model (Titan Embed Text v2 = 1024)
    metadata jsonb,
    "createdAt" timestamp NOT NULL DEFAULT now(),
    UNIQUE ("sourceId", "chunkIndex")
);
CREATE INDEX IF NOT EXISTS rag_chunk_embedding_idx ON ai."RagChunk" USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS "RagChunk_organisationSlug_idx2" ON ai."RagChunk" ("organisationSlug");

CREATE TABLE IF NOT EXISTS ai."ProjectSummary" (
    id SERIAL PRIMARY KEY,
    "projectId" integer NOT NULL REFERENCES public."Project"(id) ON DELETE CASCADE,
    "summaryText" text NOT NULL,
    "sourceHash" text NOT NULL,
    model text NOT NULL,
    "generatedAt" timestamp NOT NULL DEFAULT now(),
    UNIQUE ("projectId")
);

CREATE TABLE IF NOT EXISTS ai."AiInvocationLog" (
    id BIGSERIAL PRIMARY KEY,
    "organisationSlug" text NOT NULL,
    capability text NOT NULL,
    "contextHash" text,              -- hash of the resolved context, for EXACT cache matches
    "cacheScope" jsonb,               -- exact-match filters a semantic hit must still satisfy (e.g. skill list)
    embedding vector(1024),          -- embedding of the semantic cache text (e.g. title+description), for near-duplicate matches
    "inputPayload" jsonb NOT NULL,
    "outputPayload" jsonb,
    model text,
    "promptTokens" integer,
    "completionTokens" integer,
    "latencyMs" integer,
    status text NOT NULL,            -- SUCCESS / FAILED / EXACT_CACHE_HIT / SEMANTIC_CACHE_HIT
    "errorMessage" text,
    "createdAt" timestamp NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS "AiInvocationLog_organisationSlug_capability_idx2" ON ai."AiInvocationLog" ("organisationSlug", capability);
-- Fast lookup for EXACT cache matches: newest SUCCESS row for a given capability + tenant + exact context hash.
CREATE INDEX IF NOT EXISTS "AiInvocationLog_exact_cache_idx2" ON ai."AiInvocationLog" (capability, "organisationSlug", "contextHash", "createdAt" DESC)
    WHERE status = 'SUCCESS';
-- Fast lookup for SEMANTIC cache matches: vector similarity search scoped to successful calls only.
CREATE INDEX IF NOT EXISTS ai_invocation_log_embedding_idx ON ai."AiInvocationLog"
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 50);


-- ============================================================================
-- Chatbot: conversation state, audit log
-- ============================================================================
-- Auth is not wired up yet (see README) - organisationSlug/userId arrive as
-- plain request fields today, not a verified identity. RLS is still applied
-- to every ai-owned chat table and to the curated views below, keyed on
-- session GUCs (app.org_slug, app.user_id) that the FastAPI layer sets per
-- request. Only the "how do we trust these values" step changes once real
-- auth lands - the enforcement points and predicates stay the same.
--
-- Deliberately NOT applying RLS to GraySync's own "public" tables - out of
-- scope here. The exploratory text-to-SQL tool only ever reads through the
-- curated "ai".v_* views (safe columns only, already row-filtered), executed
-- by a dedicated read-only role - that's the fence for that path.

CREATE TABLE IF NOT EXISTS ai.conversations (
    id BIGSERIAL PRIMARY KEY,
    "organisationSlug" text NOT NULL,
    "userId" integer NOT NULL,
    title text,
    "createdAt" timestamp NOT NULL DEFAULT now(),
    "updatedAt" timestamp NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS "conversations_organisationSlug_userId_idx2" ON ai.conversations ("organisationSlug", "userId");

CREATE TABLE IF NOT EXISTS ai.messages (
    id BIGSERIAL PRIMARY KEY,
    "conversationId" bigint NOT NULL REFERENCES ai.conversations(id) ON DELETE CASCADE,
    "organisationSlug" text NOT NULL,  -- denormalized from conversations so RLS can filter without a join
    role text NOT NULL,                -- user / assistant / tool
    content text NOT NULL,
    "toolCalls" jsonb,                 -- tool invocations proposed/made while producing this message, if any
    "createdAt" timestamp NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS "messages_conversationId_createdAt_idx2" ON ai.messages ("conversationId", "createdAt");
CREATE INDEX IF NOT EXISTS "messages_organisationSlug_idx2" ON ai.messages ("organisationSlug");

-- One row per tool call the agent makes (read tool, text-to-SQL, etc.) -
-- security control + debugging aid, per the implementation guide (Part A, Step 12).
CREATE TABLE IF NOT EXISTS ai.audit_log (
    id BIGSERIAL PRIMARY KEY,
    "organisationSlug" text NOT NULL,
    "userId" integer NOT NULL,
    "conversationId" bigint REFERENCES ai.conversations(id) ON DELETE SET NULL,
    tool text NOT NULL,
    arguments jsonb NOT NULL,
    "authorizationDecision" text NOT NULL,  -- ALLOWED / DENIED - always ALLOWED today since auth isn't enforced yet
    result jsonb,
    "errorMessage" text,
    "createdAt" timestamp NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS "audit_log_organisationSlug_createdAt_idx2" ON ai.audit_log ("organisationSlug", "createdAt" DESC);
CREATE INDEX IF NOT EXISTS "audit_log_conversationId_idx2" ON ai.audit_log ("conversationId");

ALTER TABLE ai.conversations ENABLE ROW LEVEL SECURITY;
ALTER TABLE ai.messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE ai.audit_log ENABLE ROW LEVEL SECURITY;

-- Deny-by-default: no policy matches until app.org_slug is set on the session,
-- so a connection that forgets to set it sees zero rows rather than everything.
-- DROP + CREATE (not CREATE OR REPLACE - policies don't support it) keeps this rerunnable.
DROP POLICY IF EXISTS conversations_tenant_isolation ON ai.conversations;
CREATE POLICY conversations_tenant_isolation ON ai.conversations
    USING ("organisationSlug" = current_setting('app.org_slug', true));
DROP POLICY IF EXISTS messages_tenant_isolation ON ai.messages;
CREATE POLICY messages_tenant_isolation ON ai.messages
    USING ("organisationSlug" = current_setting('app.org_slug', true));
DROP POLICY IF EXISTS audit_log_tenant_isolation ON ai.audit_log;
CREATE POLICY audit_log_tenant_isolation ON ai.audit_log
    USING ("organisationSlug" = current_setting('app.org_slug', true));


-- ============================================================================
-- Chatbot: curated read-only views for the exploratory text-to-SQL tool
-- ============================================================================
-- Two scoping dimensions, mirroring the GraySync permission model (Ability.restrictTo):
--   1. Tenant: every view is implicitly limited to app.org_slug's organisation.
--   2. Ownership ("OWN" vs "ALL"): a regular user only sees rows tied to
--      projects they're on (or task-assigned to); isProjectManager/isSuperAdmin
--      users see everything in their organisation. Record-owned entities
--      (Feedback, StaffNote, Goal) use "is this about me" instead of project
--      membership. Org-wide entities (Announcement, Contact, Department,
--      Skill) have no per-user ownership dimension at all.
--
-- app.user_id is NOT a verified identity yet (see note above) - it is exactly
-- as trustworthy as the userId a caller puts in the request body today. This
-- still builds the real filtering logic (not a stub) so swapping in a verified
-- identity later is a one-line change in the FastAPI layer, not a schema change.

-- Safe parse of app.user_id: unset/empty/non-numeric all resolve to NULL
-- (matching the deny-by-default behavior of an unset GUC) instead of raising
-- "invalid input syntax for type integer" and failing the whole query. Since
-- app.user_id is an unverified, caller-supplied value (see note above), a
-- malformed one must degrade to "sees nothing," not a 500.
CREATE OR REPLACE FUNCTION ai.session_user_id() RETURNS integer AS $$
    SELECT CASE
        WHEN current_setting('app.user_id', true) ~ '^\d+$'
            THEN current_setting('app.user_id', true)::integer
        ELSE NULL
    END;
$$ LANGUAGE sql STABLE;

-- ---- Real permission model: Role -> Ability(entity, action[], restrictTo) ----
-- Replaces the earlier isProjectManager/isSuperAdmin approximation. Confirmed
-- against this tenant's live /v1/roles data (all 5 STAFF tiers - Employee,
-- Finance, Manager, High Management, Organisation Owner): every ability a
-- STAFF role holds is restrictTo=ALL; OWN only appears on auto-generated
-- CLIENT-holder roles (external client portal), and ORGANISATION is unused
-- entirely. So for STAFF, "can this role see this entity at all" (does a
-- READ Ability row exist for it, deny-by-default) is what actually varies
-- between tiers - not row-level ownership. Employee Level, for example, has
-- no INVOICE_* ability at all, so it can't reach invoice data regardless of
-- whose invoice it is; Organisation Owner has READ on everything.
--
-- CLIENT-holder roles (external client portal users) DO use restrictTo=OWN,
-- scoped to their own Contact record - kept as a separate, narrower check
-- (ai.session_client_restrict_to) rather than folded into the entity gate,
-- since it's a genuinely different mechanism (which Contact, not which org).

-- Which of the given candidate entities the caller's role actually holds a
-- READ ability for - used by the text-to-SQL tool (chat/permissions.py) to
-- proactively distinguish "this view returned zero rows because the role
-- has no ability for it" (a real access denial) from "zero rows because the
-- data happens to be empty" (not a denial), BEFORE running the generated
-- query - the row-level result alone can't tell those two apart after the
-- fact, since a genuinely-empty allowed view and a permission-filtered view
-- both just return no rows.
-- NOTE: none of these resolve or filter by organisationSlug. A person's
-- permissions come from their Role (User.id -> User.role_id -> Ability),
-- full stop - organisationSlug is tenant/ownership data on the rows
-- themselves (Project.organisationSlug, Invoice.organisationSlug, ...), not
-- an additional access gate layered on top of Ability. Confirmed necessary:
-- a single userId can hold real _members/ownerId/requestorId relationships
-- across many organisations (e.g. UserOrganisationAccess), and treating
-- organisationSlug as a hard WHERE filter on every view silently hid all
-- but one of them - the Ability model is meant to be the ONLY constraint.
CREATE OR REPLACE FUNCTION ai.session_held_read_entities(candidate_entities text[]) RETURNS TABLE(entity text) AS $$
    SELECT DISTINCT a.entity::text
    FROM public."User" u
    JOIN public."Ability" a ON a."roleId" = u.role_id
    WHERE u.id = ai.session_user_id()
      AND a.entity::text = ANY(candidate_entities)
      AND 'READ' = ANY(a.action::text[]);
$$ LANGUAGE sql STABLE;

-- Does the caller's role have a READ ability for the given entity? NULL/no
-- role, or no matching Ability row, means false (deny-by-default) - matches
-- how a real Employee Level user simply cannot query INVOICE data today.
CREATE OR REPLACE FUNCTION ai.session_has_read_ability(target_entity text) RETURNS boolean AS $$
    SELECT COALESCE(
        (
            SELECT true
            FROM public."User" u
            JOIN public."Ability" a ON a."roleId" = u.role_id
            WHERE u.id = ai.session_user_id()
              AND a.entity::text = target_entity
              AND 'READ' = ANY(a.action::text[])
            LIMIT 1
        ),
        false
    );
$$ LANGUAGE sql STABLE;

-- restrictTo for the caller's READ ability on the given entity - NULL if
-- they have no such ability (caller should check session_has_read_ability
-- first; this is for the one case, CLIENT roles, where restrictTo itself
-- still matters instead of being uniformly ALL).
CREATE OR REPLACE FUNCTION ai.session_read_restrict_to(target_entity text) RETURNS text AS $$
    SELECT a."restrictTo"::text
    FROM public."User" u
    JOIN public."Ability" a ON a."roleId" = u.role_id
    WHERE u.id = ai.session_user_id()
      AND a.entity::text = target_entity
      AND 'READ' = ANY(a.action::text[])
    ORDER BY CASE a."restrictTo"::text WHEN 'ALL' THEN 1 WHEN 'ORGANISATION' THEN 2 ELSE 3 END
    LIMIT 1;
$$ LANGUAGE sql STABLE;

-- General ability check: does the caller's role have a row for this entity
-- whose action array contains target_action? Needed beyond the READ-only
-- gate above for cases like LEAVES, where the real catalog's "own requests
-- only" vs "view/modify others'" distinction is UPDATE presence, not a
-- separate entity or restrictTo (confirmed: every non-Employee tier has
-- LEAVES:{CREATE,READ,UPDATE}, Employee Level has only {CREATE,READ}).
CREATE OR REPLACE FUNCTION ai.session_has_ability(target_entity text, target_action text) RETURNS boolean AS $$
    SELECT COALESCE(
        (
            SELECT true
            FROM public."User" u
            JOIN public."Ability" a ON a."roleId" = u.role_id
            WHERE u.id = ai.session_user_id()
              AND a.entity::text = target_entity
              AND target_action = ANY(a.action::text[])
            LIMIT 1
        ),
        false
    );
$$ LANGUAGE sql STABLE;

-- Whether the caller can see OTHER people's leave requests (not just their
-- own) - the real catalog's LEAVES:UPDATE ability ("View and modify leaves
-- of the other users"). A caller without it can still see their own via the
-- plain requestorId = self check in v_leave_request/v_staff_leave_balance.
CREATE OR REPLACE FUNCTION ai.session_can_see_others_leave() RETURNS boolean AS $$
    SELECT ai.session_has_ability('LEAVES', 'UPDATE');
$$ LANGUAGE sql STABLE;

-- Whether app.user_id (as currently set on this session) should see everything
-- in app.org_slug's organisation for project-scoped data. Kept for the
-- project-membership fallback below (PROJECT_VIEW_OTHERS is the "see other
-- users' projects" ability - without it, visibility still narrows to actual
-- project involvement even for someone who otherwise has PROJECT:READ).
CREATE OR REPLACE FUNCTION ai.session_user_sees_all_projects() RETURNS boolean AS $$
    SELECT ai.session_has_read_ability('PROJECT_VIEW_OTHERS');
$$ LANGUAGE sql STABLE;

-- Project slugs visible to the current session: every project in the org if
-- the caller's role can view other users' projects (PROJECT_VIEW_OTHERS),
-- otherwise only projects the session user is a team member of (_members),
-- owns a task in, or is assigned a task in. Callers of this function should
-- separately check session_has_read_ability('PROJECT') first - this only
-- resolves WHICH projects, not WHETHER project data is reachable at all.
CREATE OR REPLACE FUNCTION ai.visible_project_slugs() RETURNS SETOF text AS $$
    SELECT p.slug
    FROM public."Project" p
    WHERE ai.session_user_sees_all_projects()
       OR EXISTS (
           SELECT 1 FROM public."_members" m
           WHERE m."A" = p.id
             AND m."B" = ai.session_user_id()
       )
       OR EXISTS (
           SELECT 1 FROM public."Task" t
           WHERE t."projectSlug" = p.slug
             AND t."ownerId" = ai.session_user_id()
       )
       OR EXISTS (
           SELECT 1 FROM public."Task" t
           JOIN public."TaskAssignee" ta ON ta."taskId" = t.id
           WHERE t."projectSlug" = p.slug
             AND ta."assigneeId" = ai.session_user_id()
       );
$$ LANGUAGE sql STABLE;

-- Whether the given userId's own record is visible to the current session:
-- themselves, or a role that can view other users' people records
-- (PEOPLE_INTERNAL covers staff directory/leave/feedback-of-others per the
-- real catalog - see defaults.json "People" tab).
CREATE OR REPLACE FUNCTION ai.session_can_see_user(target_user_id integer) RETURNS boolean AS $$
    SELECT target_user_id = ai.session_user_id()
        OR ai.session_has_read_ability('PEOPLE_INTERNAL');
$$ LANGUAGE sql STABLE;

-- ---- Project/task-scoped entities (visible per ai.visible_project_slugs) ----

-- Deny-by-default entity gate: a role with none of these three project-read
-- abilities (PROJECT, PROJECT_VIEW_OTHERS, PROJECT_MODIFY_MEMBER - the real
-- catalog's only "can see project data" abilities) can't reach project data
-- at all, regardless of membership. Every real STAFF role today has at least
-- one, so this changes nothing in practice - it's the correct fence for a
-- hypothetical role that has none.
CREATE OR REPLACE FUNCTION ai.session_has_any_project_read() RETURNS boolean AS $$
    SELECT ai.session_has_read_ability('PROJECT')
        OR ai.session_has_read_ability('PROJECT_VIEW_OTHERS')
        OR ai.session_has_read_ability('PROJECT_MODIFY_MEMBER');
$$ LANGUAGE sql STABLE;

CREATE OR REPLACE VIEW ai.v_project AS
SELECT p.id, p.slug, p.name, p."organisationSlug", p."customId", p.description,
       p.status, p.priority, p.type, p."dueDate", p."startDate",
       p."managerId", p."companyId", p."createdAt", p."updatedAt"
FROM public."Project" p
WHERE ai.session_has_any_project_read()
  AND p.slug IN (SELECT * FROM ai.visible_project_slugs());

CREATE OR REPLACE VIEW ai.v_task AS
SELECT t.id, t.name, t.description, t."projectSlug", s.name AS status,
       t."ownerId", t."startDate", t."dueDate", t.estimated, t.logged,
       t."taskType", t.flagged, t."isDeleted", t."createdAt", t."updatedAt"
FROM public."Task" t
LEFT JOIN public."Status" s ON s.id = t."statusId"
WHERE ai.session_has_any_project_read()
  AND t."projectSlug" IN (SELECT * FROM ai.visible_project_slugs());

CREATE OR REPLACE VIEW ai.v_task_assignee AS
SELECT ta.id, ta."taskId", t."projectSlug", ta."assigneeId", ta."estimatedTime", ta."createdAt"
FROM public."TaskAssignee" ta
JOIN public."Task" t ON t.id = ta."taskId"
WHERE ai.session_has_any_project_read()
  AND t."projectSlug" IN (SELECT * FROM ai.visible_project_slugs());

CREATE OR REPLACE VIEW ai.v_project_member AS
-- Project team membership (_members: Project<->Staff) - without this, "which
-- projects am I on" is unanswerable by SQL: v_project's own visibility rule
-- already reflects membership internally, but doesn't expose the membership
-- fact itself as data, so a query has nothing to filter by beyond "everything
-- I can see" (which, for a manager/admin, is indistinguishable from "all
-- projects"). This surfaces the actual relationship.
SELECT p.slug AS "projectSlug", m."B" AS "userId"
FROM public."_members" m
JOIN public."Project" p ON p.id = m."A"
WHERE ai.session_has_any_project_read()
  AND p.slug IN (SELECT * FROM ai.visible_project_slugs());

CREATE OR REPLACE VIEW ai.v_task_activity AS
-- PROJECT_LOG is the real catalog entity for "View project logs" - task
-- activity is that log.
SELECT ta.id, ta."taskId", t.name AS "taskName", t."projectSlug", ta."createdAt"
FROM public."TaskActivity" ta
JOIN public."Task" t ON t.id = ta."taskId"
WHERE ai.session_has_read_ability('PROJECT_LOG')
  AND t."projectSlug" IN (SELECT * FROM ai.visible_project_slugs());
-- content is Slate/Plate rich-text jsonb (see rich_text.py) - deliberately not
-- exposed raw here; the RAG-ingested plain text (RagChunk, sourceType =
-- 'ACTIVITY_LOG') is the searchable text form of this same data.

CREATE OR REPLACE VIEW ai.v_scope AS
-- SCOPE (full edit), SCOPE_VIEW_ALL, SCOPE_VIEW_LINKED, SCOPE_VIEW_MEMBER are
-- the real catalog's scope-read abilities (defaults.json "Scopes" tab) -
-- any one grants read access here; which projects' scopes are visible still
-- follows the same project-read fallback as everything else project-scoped.
SELECT sc.id, sc.name, sc.slug, sc."customId", sc."projectSlug", sc."organisationSlug",
       sc.status, sc.type, sc."dueDate", sc."companyId", sc."createdAt"
FROM public."Scope" sc
WHERE (
    ai.session_has_read_ability('SCOPE')
    OR ai.session_has_read_ability('SCOPE_VIEW_ALL')
    OR ai.session_has_read_ability('SCOPE_VIEW_LINKED')
    OR ai.session_has_read_ability('SCOPE_VIEW_MEMBER')
  )
  AND (sc."projectSlug" IS NULL OR sc."projectSlug" IN (SELECT * FROM ai.visible_project_slugs()));

-- ---- Finance entities (project-scoped; amounts only, no bank/payment details) ----

CREATE OR REPLACE VIEW ai.v_invoice AS
-- INVOICE_VIEW_ALL vs INVOICE_VIEW_PROJECT_LINKED are the real catalog's two
-- separate invoice-read abilities (defaults.json "Finances" tab) - a role
-- can have either, both, or neither. ALL means every invoice in the org
-- regardless of project involvement; PROJECT_LINKED still narrows to
-- visible_project_slugs() same as everything else project-scoped.
SELECT i.id, i."customId", i."organisationSlug", i."companyId", i."projectSlug", i."scopeSlug",
       i.type, i."issueDate", i."dueDate", i."amountPaid", i."paidAt", i."paymentStatus", i.balance
FROM public."Invoice" i
WHERE ai.session_has_read_ability('INVOICE_VIEW_ALL')
   OR (
       ai.session_has_read_ability('INVOICE_VIEW_PROJECT_LINKED')
       AND i."projectSlug" IS NOT NULL
       AND i."projectSlug" IN (SELECT * FROM ai.visible_project_slugs())
   );

CREATE OR REPLACE VIEW ai.v_invoice_item AS
SELECT ii.id, ii."invoiceId", ii.description, ii.quantity, ii."unitPrice", ii.discount, ii.amount
FROM public."InvoiceItem" ii
JOIN public."Invoice" i ON i.id = ii."invoiceId"
WHERE ai.session_has_read_ability('INVOICE_VIEW_ALL')
   OR (
       ai.session_has_read_ability('INVOICE_VIEW_PROJECT_LINKED')
       AND i."projectSlug" IS NOT NULL
       AND i."projectSlug" IN (SELECT * FROM ai.visible_project_slugs())
   );

CREATE OR REPLACE VIEW ai.v_expense AS
-- Same two-ability shape as invoices: EXPENSE_READ_ALL_LIST (org-wide) vs
-- EXPENSE_VIEW_PROJECT_LINKED (project-scoped).
SELECT e.id, e."customId", e."organisationSlug", e."projectSlug", e."purchaserId",
       e."purchaseDate", e."dueDate", e.cost, e.billed, e.profit, e.action
FROM public."Expense" e
WHERE ai.session_has_read_ability('EXPENSE_READ_ALL_LIST')
   OR (
       ai.session_has_read_ability('EXPENSE_VIEW_PROJECT_LINKED')
       AND e."projectSlug" IS NOT NULL
       AND e."projectSlug" IN (SELECT * FROM ai.visible_project_slugs())
   );

CREATE OR REPLACE VIEW ai.v_quote AS
-- Quote has no dedicated Entity in the real permission catalog (confirmed
-- against the full defaults.json) - gated on SCOPE-family read abilities as
-- the closest real analog (a quote is a pre-scope sales document), which is
-- an approximation, not a literal mapping like the views above.
SELECT q.id, q.quote_number, q.job_title, q."organisationSlug", q.project_id,
       q.issued_on, q.subtotal, q.gst, q.total, q.status
FROM public."Quote" q
WHERE (
    ai.session_has_read_ability('SCOPE')
    OR ai.session_has_read_ability('SCOPE_VIEW_ALL')
    OR ai.session_has_read_ability('SCOPE_VIEW_LINKED')
  )
  AND (
    q.project_id IS NULL
    OR q.project_id IN (
        SELECT p.id FROM public."Project" p WHERE p.slug IN (SELECT * FROM ai.visible_project_slugs())
    )
  );

CREATE OR REPLACE VIEW ai.v_budget AS
SELECT b.id, b.name, b."organisationSlug", b."financialYearId", b."createdAt"
FROM public."Budget" b
WHERE ai.session_has_read_ability('PROJECT_BUDGET');

CREATE OR REPLACE VIEW ai.v_rate_card AS
-- RateCard has no dedicated Entity in the real catalog either - gated on
-- PROJECT_BUDGET (rate cards feed budget/cost calculations) as the closest
-- real analog, same caveat as v_quote above.
SELECT rc.id, rc."rateCardGroupId", rc."positionId", rc."hourlyRate", rc."dailyRate"
FROM public."RateCard" rc
WHERE ai.session_has_read_ability('PROJECT_BUDGET');

-- ---- Org-wide entities (no per-project ownership dimension) ----

CREATE OR REPLACE VIEW ai.v_announcement AS
SELECT a.id, a."organisationSlug", a."authorUserId", a.type, a.status, a.title,
       a."contentText", a."startsAt", a."endsAt", a."isPinned", a."publishedAt", a."createdAt"
FROM public."Announcement" a
WHERE ai.session_has_read_ability('ANNOUNCEMENT');

CREATE OR REPLACE VIEW ai.v_announcement_comment AS
SELECT ac.id, ac."announcementId", ac."authorUserId", ac."contentText", ac.status, ac."createdAt"
FROM public."AnnouncementComment" ac
JOIN public."Announcement" a ON a.id = ac."announcementId"
WHERE ai.session_has_read_ability('ANNOUNCEMENT');

CREATE OR REPLACE VIEW ai.v_contact AS
-- COMPANY is the real catalog entity for the "Contacts" tab (bundles
-- COMPANY+SUPPLIER - see defaults.json "Add contacts"/"View list of contacts").
SELECT c.id, c.slug, c.name, c.type, c.city, c.state, c.country, c.website, c.email, c."createdAt"
FROM public."Contact" c
JOIN public."ContactOnOrganisation" coo ON coo."contactId" = c.id
WHERE ai.session_has_read_ability('COMPANY');
-- notes/address intentionally excluded - CONTACT_NOTES already covered via
-- RAG ingestion (schema.sql RagSourceType), not needed raw here too.

CREATE OR REPLACE VIEW ai.v_company_contact AS
SELECT cc.id, cc."customId", cc.email, cc."companyType", cc.status, c.name
FROM public."CompanyContact" cc
JOIN public."Contact" c ON c.id = cc.id
JOIN public."ContactOnOrganisation" coo ON coo."contactId" = c.id
WHERE ai.session_has_read_ability('COMPANY');

CREATE OR REPLACE VIEW ai.v_department AS
SELECT d.id, d.name, d."organisationSlug"
FROM public."Department" d
WHERE ai.session_has_read_ability('ORG_DEPARTMENTS');

CREATE OR REPLACE VIEW ai.v_position AS
SELECT pos.id, pos.name, pos."departmentId", dep."organisationSlug"
FROM public."Position" pos
JOIN public."Department" dep ON dep.id = pos."departmentId"
WHERE ai.session_has_read_ability('ORG_DEPARTMENTS');

CREATE OR REPLACE VIEW ai.v_skill AS
SELECT s.id, s.name, s."organisationSlug"
FROM public."Skill" s
WHERE ai.session_has_read_ability('ORG_SKILLS');

-- ---- Record-owned entities ("own" = about me, or a manager/admin viewing anyone) ----

-- Directory-level identity: no personal-record ownership gate (see v_staff
-- below for that). A name/jobTitle lookup is needed to resolve WHO is on a
-- team/project (e.g. "who are my teammates"), which is not a
-- personal-privacy question the way leave/salary is.
--
-- Visible to the caller if they share a project with the person (via
-- ai.v_project_member, which is itself unrestricted by org - a project's
-- team can include people from other orgs, and that's real, intended
-- cross-org collaboration data, not a leak) OR the caller's role can see
-- anyone (PEOPLE_INTERNAL). Without this, a project's own member list
-- (v_project_member) would show userIds this view then couldn't resolve
-- into a name/job title - visible as a project member, but silently
-- anonymous, for no reason tied to an actual permission boundary.
CREATE OR REPLACE VIEW ai.v_staff_directory AS
SELECT DISTINCT st."userId", u."fullName", u."organisationSlug", st."jobTitle", st."departmentId", st."positionId"
FROM public."Staff" st
JOIN public."User" u ON u.id = st."userId"
WHERE ai.session_has_read_ability('PEOPLE_INTERNAL')
   OR EXISTS (
       SELECT 1 FROM ai.v_project_member pm WHERE pm."userId" = st."userId"
   );

CREATE OR REPLACE VIEW ai.v_staff AS
-- salary/annualGrossSalary/taxNumber/pension* deliberately excluded - see
-- schema note on Staff: compensation fields stay off the exploratory path
-- regardless of who's asking (PDF guidance: keep payroll on the trusted
-- GraySync-API path, not free-form SQL).
-- hireDate/employmentStatus are more personal than the directory-level
-- fields above (v_staff_directory) - stay gated to own record or manager/admin.
SELECT st."userId", u."fullName", u."organisationSlug", st."jobTitle", st."employmentStatus",
       st."hireDate", st."departmentId", st."positionId"
FROM public."Staff" st
JOIN public."User" u ON u.id = st."userId"
WHERE ai.session_can_see_user(st."userId");

CREATE OR REPLACE VIEW ai.v_user_skill AS
SELECT us.id, us."userId", us."skillId", us."organisationSlug"
FROM public."UserSkill" us
WHERE ai.session_can_see_user(us."userId");

CREATE OR REPLACE VIEW ai.v_leave_request AS
-- LEAVES:READ is required just to reach leave data at all (deny-by-default -
-- a role with no LEAVES ability sees nothing, even their own). Given that,
-- a regular row is visible if it's the caller's own request, OR the caller's
-- role has LEAVES:UPDATE (ai.session_can_see_others_leave - the real "view
-- other people's leave" ability, confirmed distinct from PEOPLE_INTERNAL).
-- This is the PDF's literal "own leave vs other people's leave" example
-- (Part B.3/B.5), now backed by the actual Ability data instead of an
-- isProjectManager/isSuperAdmin approximation.
SELECT lr.id, lr."requestorId", lr."managerId", lr."leaveStartDate", lr."leaveEndDate",
       lr.status, lr."leavePolicyId", lr."requestedDuration", lr."durationUnit", lr."createdAt",
       u."organisationSlug"
FROM public."LeaveRequest" lr
JOIN public."User" u ON u.id = lr."requestorId"
WHERE ai.session_has_read_ability('LEAVES')
  AND (lr."requestorId" = ai.session_user_id() OR ai.session_can_see_others_leave());

CREATE OR REPLACE VIEW ai.v_leave_policy AS
-- Policy terms (entitlement, accrual rules) aren't per-user data - org-wide,
-- no ownership dimension, needed to compute "how many days do I have left".
-- Still gated on LEAVES:READ - no point exposing policy terms to a role that
-- can't see any leave data at all.
SELECT lp.id, lp.name, lp.entitlement, lp."entitlementUnit", lp."recurringPeriod",
       lp."isPaid", lp."allowFullDay", lp."allowHalfDay", lg."organisationSlug"
FROM public."LeavePolicy" lp
JOIN public."LeaveGroup" lg ON lg.id = lp."groupId"
WHERE ai.session_has_read_ability('LEAVES');

CREATE OR REPLACE VIEW ai.v_staff_leave_balance AS
SELECT slob.id, slob."staffUserId", slob."leavePolicyId", slob."openingBalance", u."organisationSlug"
FROM public."StaffLeaveOpeningBalance" slob
JOIN public."User" u ON u.id = slob."staffUserId"
WHERE ai.session_has_read_ability('LEAVES')
  AND (slob."staffUserId" = ai.session_user_id() OR ai.session_can_see_others_leave());
-- Remaining balance = openingBalance (this view) + policy entitlement
-- (v_leave_policy) - sum(requestedDuration) of this staff member's APPROVED
-- LeaveRequest rows for that policy (v_leave_request) since the policy's
-- accrual period started. The agent composes this from the three views
-- rather than a single pre-computed "balance" column, since "remaining"
-- depends on the policy's recurringPeriod window.

CREATE OR REPLACE VIEW ai.v_feedback AS
-- FEEDBACK is a real, distinct catalog entity (confirmed: every non-Employee
-- tier has it with full CRUD, Employee Level lacks it entirely).
SELECT f.id, f."userId", f."createdById", f."completionDate", f.message, f."createdAt", u."organisationSlug"
FROM public."Feedback" f
JOIN public."User" u ON u.id = f."userId"
WHERE ai.session_has_read_ability('FEEDBACK')
  AND (f."userId" = ai.session_user_id() OR ai.session_has_read_ability('PEOPLE_INTERNAL'));

CREATE OR REPLACE VIEW ai.v_feedback_submission AS
SELECT fs.id, fs."feedbackId", fs."submitterId", fs."firstAnswer", fs."secondAnswer",
       fs.status, fs."createdAt", u."organisationSlug"
FROM public."FeedbackSubmission" fs
JOIN public."User" u ON u.id = fs."submitterId"
WHERE ai.session_has_read_ability('FEEDBACK')
  AND (fs."submitterId" = ai.session_user_id() OR ai.session_has_read_ability('PEOPLE_INTERNAL'));

CREATE OR REPLACE VIEW ai.v_staff_note AS
-- StaffNote has no dedicated catalog entity - PEOPLE_RELATED ("Modify other
-- users if related") is the closest real analog, an approximation like
-- v_quote/v_rate_card above, not a literal mapping.
SELECT sn.id, sn."userId", sn.note, sn."createdAt", u."organisationSlug"
FROM public."StaffNote" sn
JOIN public."User" u ON u.id = sn."userId"
WHERE sn."userId" = ai.session_user_id() OR ai.session_has_read_ability('PEOPLE_RELATED');

CREATE OR REPLACE VIEW ai.v_goal AS
-- Goal/Objective have no catalog entity at all (not in defaults.json under
-- any tab) - GraySync's real permission system doesn't cover this feature.
-- Falls back to ai.session_can_see_user (own record, or PEOPLE_INTERNAL for
-- anyone) as the least-arbitrary available check, same shape as the
-- record-owned views above.
SELECT g.id, g.title, g."startDate", g."endDate", g.status, g.progress, g."userId", u."organisationSlug"
FROM public."Goal" g
JOIN public."User" u ON u.id = g."userId"
WHERE ai.session_can_see_user(g."userId");

CREATE OR REPLACE VIEW ai.v_objective AS
SELECT o.id, o.detail, o.done, o."goalId", u."organisationSlug"
FROM public."Objective" o
JOIN public."Goal" g ON g.id = o."goalId"
JOIN public."User" u ON u.id = g."userId"
WHERE ai.session_can_see_user(g."userId");

-- ---- Least-privilege role + RLS on the ai-owned chat tables ----

-- Read-only role for the exploratory text-to-SQL tool: SELECT on the curated
-- views above only, never the raw GraySync tables. Short statement timeout so
-- a runaway/malicious generated query is cut off automatically (PDF Step 4).
-- psql -v ai_readonly_password='...' -f schema.sql - psql's :'var' client-side
-- substitution does NOT happen inside a dollar-quoted ($$...$$) body, so it
-- can't be referenced directly inside a DO block. \gexec runs whatever the
-- preceding query returns as a second statement, letting the substitution
-- happen in an ordinary SELECT instead.
SELECT format('CREATE ROLE ai_readonly LOGIN PASSWORD %L', :'ai_readonly_password')
WHERE NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'ai_readonly')
\gexec
ALTER ROLE ai_readonly SET statement_timeout = '5s';
GRANT USAGE ON SCHEMA ai TO ai_readonly;
GRANT SELECT ON
    ai.v_project, ai.v_project_member, ai.v_task, ai.v_task_assignee, ai.v_task_activity, ai.v_scope,
    ai.v_invoice, ai.v_invoice_item, ai.v_expense, ai.v_quote, ai.v_budget, ai.v_rate_card,
    ai.v_announcement, ai.v_announcement_comment, ai.v_contact, ai.v_company_contact,
    ai.v_department, ai.v_position, ai.v_skill,
    ai.v_staff_directory, ai.v_staff, ai.v_user_skill, ai.v_leave_request, ai.v_leave_policy, ai.v_staff_leave_balance,
    ai.v_feedback, ai.v_feedback_submission, ai.v_staff_note, ai.v_goal, ai.v_objective
    TO ai_readonly;
-- ai_readonly can execute the SECURITY-definer-free helper functions above
-- (SQL functions inherit caller privileges by default) since it already has
-- SELECT on the public tables they query indirectly through the views.
-- USAGE on the schema itself is required too - GRANT SELECT alone doesn't
-- make an object reachable if the containing schema is closed off.
GRANT USAGE ON SCHEMA public TO ai_readonly;
GRANT SELECT ON public."Project", public."Task", public."TaskAssignee", public."_members",
    public."User", public."Staff", public."LeaveGroup", public."Ability" TO ai_readonly;
