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

-- Which of the given candidate entity FAMILIES (exact name, or a name that
-- is the prefix of an underscore-separated sub-entity - same matching rule
-- as session_has_read_ability_family) the caller's role actually holds a
-- READ ability under - used by the text-to-SQL tool (chat/permissions.py)
-- to proactively distinguish "this view returned zero rows because the role
-- has no ability for it" (a real access denial) from "zero rows because the
-- data happens to be empty" (not a denial), BEFORE running the generated
-- query - the row-level result alone can't tell those two apart after the
-- fact, since a genuinely-empty allowed view and a permission-filtered view
-- both just return no rows. Matching by family here (not exact name) is
-- what keeps this pre-check in sync with the views' own family-matching
-- gates (v_expense, v_invoice, ...) without a hardcoded exact-name list on
-- either side - only the small set of family prefixes a view cares about
-- (chat/permissions.py VIEW_ENTITIES) is still a code-side decision; the
-- set of abilities that satisfy a given family is looked up live here.
-- NOTE: none of these resolve or filter by organisationSlug. A person's
-- permissions come from their Role (User.id -> User.role_id -> Ability),
-- full stop - organisationSlug is tenant/ownership data on the rows
-- themselves (Project.organisationSlug, Invoice.organisationSlug, ...), not
-- an additional access gate layered on top of Ability. Confirmed necessary:
-- a single userId can hold real _members/ownerId/requestorId relationships
-- across many organisations (e.g. UserOrganisationAccess), and treating
-- organisationSlug as a hard WHERE filter on every view silently hid all
-- but one of them - the Ability model is meant to be the ONLY constraint.
CREATE OR REPLACE FUNCTION ai.session_held_read_entity_families(candidate_prefixes text[]) RETURNS TABLE(family text) AS $$
    SELECT DISTINCT p
    FROM unnest(candidate_prefixes) AS p
    WHERE EXISTS (
        SELECT 1
        FROM public."User" u
        JOIN public."Ability" a ON a."roleId" = u.role_id
        WHERE u.id = ai.session_user_id()
          AND (a.entity::text = p OR a.entity::text LIKE p || '\_%' ESCAPE '\')
          AND 'READ' = ANY(a.action::text[])
    );
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

-- Same as session_has_read_ability, but matches ANY entity in a given family
-- (entity = prefix, or entity starting with prefix || '_') rather than one
-- exact name. Needed because the same organisation's Ability rows are not
-- all on one schema version: confirmed live in this DB that two roles
-- (id=1, id=37, both named "Owner") only hold the coarse legacy entity for a
-- family (e.g. plain EXPENSE, INVOICE) while every other role also or
-- instead holds the newer granular sub-abilities (EXPENSE_READ_ALL_LIST,
-- EXPENSE_APPROVE, INVOICE_VIEW_ALL, ...) - checking one exact name
-- silently denies whichever generation of role doesn't use it. Any
-- sub-ability in the family that carries READ counts (confirmed: every
-- real sub-ability's action array that includes any of
-- CREATE/UPDATE/DELETE/APPROVE also includes READ - there's no "can modify
-- but can't see" ability in this catalog), so this doesn't require a
-- separate list of "the read ones" within a family.
CREATE OR REPLACE FUNCTION ai.session_has_read_ability_family(entity_prefix text) RETURNS boolean AS $$
    SELECT COALESCE(
        (
            SELECT true
            FROM public."User" u
            JOIN public."Ability" a ON a."roleId" = u.role_id
            WHERE u.id = ai.session_user_id()
              AND (a.entity::text = entity_prefix OR a.entity::text LIKE entity_prefix || '\_%' ESCAPE '\')
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

-- Whether target_user_id's tenure (Staff.hireDate) has cleared the given
-- policy's applicableAfter/applicableAfterUnit requirement. Moved into a
-- function (not left as a WHERE-clause expression generated SQL had to
-- reproduce correctly every time) because that was structurally
-- non-deterministic - confirmed live: the same "how many leave days does
-- Amir have" question, asked twice, correctly showed 0 Annual Leave one run
-- and wrongly showed 15 the next, because the LLM sometimes omitted the
-- AGE(...) >= (...)::interval eligibility check from the generated query
-- and sometimes didn't. A single function call is either included in a
-- query or it isn't visibly missing - there's no equivalent way to
-- "partially" get a function call wrong the way a multi-line inline
-- expression can be silently dropped or malformed.
-- NULL applicableAfter means no tenure requirement at all - always eligible.
-- NULL hireDate (should not happen for a real Staff row, but degrade safely
-- rather than raise) - treated as not yet eligible for anything but a
-- policy with no requirement, matching the same deny-by-default posture
-- used everywhere else in this schema for missing/unverifiable data.
CREATE OR REPLACE FUNCTION ai.leave_policy_is_eligible(target_user_id integer, policy_id integer) RETURNS boolean AS $$
    SELECT CASE
        WHEN lp."applicableAfter" IS NULL THEN true
        WHEN st."hireDate" IS NULL THEN false
        ELSE AGE(CURRENT_DATE, st."hireDate"::date) >=
             (lp."applicableAfter" || ' ' || lp."applicableAfterUnit")::interval
    END
    FROM public."LeavePolicy" lp
    LEFT JOIN public."Staff" st ON st."userId" = target_user_id
    WHERE lp.id = policy_id;
$$ LANGUAGE sql STABLE;

-- Whether the caller can see OTHER people's time entries (not just their
-- own) - real catalog entities TIMESHEET_VIEW_ALL / TIMESHEET_MODIFY_OTHERS.
-- A caller without either can still see their own via the plain memberId =
-- self check in v_time_entry (matches ai.session_can_see_others_leave's
-- shape exactly - TIME_ENTRIES/TIMESHEET_VIEW_OWN is the "see my own"
-- baseline read ability, checked separately in v_time_entry's WHERE clause).
CREATE OR REPLACE FUNCTION ai.session_can_see_others_time_entries() RETURNS boolean AS $$
    SELECT ai.session_has_read_ability('TIMESHEET_VIEW_ALL')
        OR ai.session_has_read_ability('TIMESHEET_MODIFY_OTHERS');
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

-- Organisations the caller has a real relationship to: their own home org
-- (User.organisationSlug) plus every org UserOrganisationAccess explicitly
-- grants them (confirmed live: e.g. userId=6 has UserOrganisationAccess rows
-- for 8 different organisations, matching their real cross-org project
-- involvement). Needed for views like v_leave_policy/v_budget that are
-- "org-wide" but span MULTIPLE organisations in the table as a whole (unlike
-- v_announcement/v_department, where every row already belongs to one
-- self-evident org) - without this, "org-wide" degenerates to "every org in
-- the database," which is what let a High Management Level user at
-- endspace-mena see all 34 leave policies for every tenant, not just their
-- own. This is a data-relationship filter, not an Ability check - it runs
-- alongside the entity gate, not instead of it.
CREATE OR REPLACE FUNCTION ai.session_visible_orgs() RETURNS SETOF text AS $$
    SELECT u."organisationSlug"
    FROM public."User" u
    WHERE u.id = ai.session_user_id()
    UNION
    SELECT o.slug
    FROM public."UserOrganisationAccess" uoa
    JOIN public."Organisation" o ON o.id = uoa."organisationId"
    WHERE uoa."userId" = ai.session_user_id();
$$ LANGUAGE sql STABLE;

-- Currency symbol for an organisation (Organisation.slug ->
-- OrganisationFinance.currencyId -> Currency.symbol). Currency is org-level,
-- not per-row - confirmed live: all 20 organisations have a currency
-- configured, so this is reliable, unlike ProjectMemberRate's sparse data.
-- Every money-bearing view below joins this in so a dollar figure is never
-- presented without its unit - a bare "56,979.19" is ambiguous the moment
-- more than one organisation's data could be involved (e.g. a cross-org
-- comparison), and every organisation's currency isn't necessarily the same.
CREATE OR REPLACE FUNCTION ai.org_currency_symbol(org_slug text) RETURNS text AS $$
    SELECT c.symbol
    FROM public."OrganisationFinance" ofin
    JOIN public."Currency" c ON c.id = ofin."currencyId"
    WHERE ofin."organisationSlug" = org_slug;
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

CREATE OR REPLACE VIEW ai.v_rag_source AS
-- Inventory of what's actually been ingested into the knowledge base
-- (ai.RagSource/ai.RagChunk), for "what documents are in RAG" /
-- "which documents does project X have" style questions -
-- search_knowledge_base only searches document CONTENT semantically, it
-- cannot enumerate what exists, which is what this view is for.
--
-- Each RagSourceType is gated by the SAME visibility rule as its real
-- underlying entity's own existing view (v_task_activity, v_announcement,
-- v_objective, ...) - not a separate ability, since RagSource is metadata
-- describing an ingested copy of data the caller either can or can't
-- already see through the normal views. Only the 7 RagSourceType values
-- with real, populated ingestion as of this build are covered
-- (PROJECT_DOCUMENT, SCOPE_DOCUMENT, MEDIA_PLAN_DOCUMENT, ACTIVITY_LOG,
-- ANNOUNCEMENT, ANNOUNCEMENT_COMMENT, OBJECTIVE) - the other 11
-- RagSourceType enum values have zero rows in this environment and are
-- omitted rather than guessed at.
--
-- name is resolved per-type from the real backing table (ProjectDocument.
-- fileName, ScopeDocument.fileName, MediaPlanDocument.fileName, the parent
-- Task's name for an activity-log entry - TaskActivity.content is Slate/
-- Plate rich-text jsonb, not a usable name, see v_task_activity's note
-- above - Announcement.title, a truncated AnnouncementComment.contentText,
-- Objective.detail) - never the enclosing project/org's own name, which is
-- a different thing and was previously a source of confusion (a document
-- named "360 Marketing" does not mean the PROJECT named 360 Marketing is
-- itself a document).
-- fileType is added for the 3 document-backed branches (PROJECT_DOCUMENT/
-- SCOPE_DOCUMENT/MEDIA_PLAN_DOCUMENT) - NULL for the other 4, which have no
-- underlying file at all. Confirmed live the stored format is NOT
-- consistent across tables - e.g. ProjectDocument.fileType can be a short
-- code ('img', 'pdf'), while ScopeDocument/MediaPlanDocument.fileType can
-- be a full MIME type ('application/pdf') - use ILIKE '%pdf%', not exact
-- equality, when filtering by file type.
SELECT rs.id, rs."sourceType"::text AS "sourceType", rs."projectSlug", rs.status, rs."ingestedAt",
       pd."fileName" AS name, pd."fileType" AS "fileType"
FROM ai."RagSource" rs
JOIN public."ProjectDocument" pd ON pd.id::text = rs."sourceId"
WHERE rs."sourceType" = 'PROJECT_DOCUMENT'
  AND ai.session_has_any_project_read()
  AND rs."projectSlug" IN (SELECT * FROM ai.visible_project_slugs())

UNION ALL

SELECT rs.id, rs."sourceType"::text, sc."projectSlug", rs.status, rs."ingestedAt",
       sd."fileName" AS name, sd."fileType" AS "fileType"
FROM ai."RagSource" rs
JOIN public."ScopeDocument" sd ON sd.id::text = rs."sourceId"
JOIN public."Scope" sc ON sc.slug = sd."scopeSlug"
WHERE rs."sourceType" = 'SCOPE_DOCUMENT'
  AND (
    ai.session_has_read_ability('SCOPE')
    OR ai.session_has_read_ability('SCOPE_VIEW_ALL')
    OR ai.session_has_read_ability('SCOPE_VIEW_LINKED')
    OR ai.session_has_read_ability('SCOPE_VIEW_MEMBER')
  )
  AND (sc."projectSlug" IS NULL OR sc."projectSlug" IN (SELECT * FROM ai.visible_project_slugs()))

UNION ALL

SELECT rs.id, rs."sourceType"::text, rs."projectSlug", rs.status, rs."ingestedAt",
       mpd."fileName" AS name, mpd."fileType" AS "fileType"
FROM ai."RagSource" rs
JOIN public."MediaPlanDocument" mpd ON mpd.id::text = rs."sourceId"
WHERE rs."sourceType" = 'MEDIA_PLAN_DOCUMENT'
  AND ai.session_has_any_project_read()
  AND (rs."projectSlug" IS NULL OR rs."projectSlug" IN (SELECT * FROM ai.visible_project_slugs()))

UNION ALL

SELECT rs.id, rs."sourceType"::text, t."projectSlug", rs.status, rs."ingestedAt",
       ('Activity on task: ' || t.name) AS name, NULL::text AS "fileType"
FROM ai."RagSource" rs
JOIN public."TaskActivity" ta ON ta.id::text = rs."sourceId"
JOIN public."Task" t ON t.id = ta."taskId"
WHERE rs."sourceType" = 'ACTIVITY_LOG'
  AND ai.session_has_read_ability('PROJECT_LOG')
  AND t."projectSlug" IN (SELECT * FROM ai.visible_project_slugs())

UNION ALL

SELECT rs.id, rs."sourceType"::text, rs."projectSlug", rs.status, rs."ingestedAt",
       a.title AS name, NULL::text AS "fileType"
FROM ai."RagSource" rs
JOIN public."Announcement" a ON a.id::text = rs."sourceId"
WHERE rs."sourceType" = 'ANNOUNCEMENT'
  AND ai.session_has_read_ability('ANNOUNCEMENT')
  AND a."organisationSlug" IN (SELECT * FROM ai.session_visible_orgs())

UNION ALL

SELECT rs.id, rs."sourceType"::text, rs."projectSlug", rs.status, rs."ingestedAt",
       ('Comment on: ' || a.title) AS name, NULL::text AS "fileType"
FROM ai."RagSource" rs
JOIN public."AnnouncementComment" ac ON ac.id::text = rs."sourceId"
JOIN public."Announcement" a ON a.id = ac."announcementId"
WHERE rs."sourceType" = 'ANNOUNCEMENT_COMMENT'
  AND ai.session_has_read_ability('ANNOUNCEMENT')
  AND a."organisationSlug" IN (SELECT * FROM ai.session_visible_orgs())

UNION ALL

SELECT rs.id, rs."sourceType"::text, rs."projectSlug", rs.status, rs."ingestedAt",
       o.detail AS name, NULL::text AS "fileType"
FROM ai."RagSource" rs
JOIN public."Objective" o ON o.id::text = rs."sourceId"
JOIN public."Goal" g ON g.id = o."goalId"
WHERE rs."sourceType" = 'OBJECTIVE'
  AND ai.session_can_see_user(g."userId");

CREATE OR REPLACE VIEW ai.v_scope AS
-- SCOPE (full edit), SCOPE_VIEW_ALL, SCOPE_VIEW_LINKED, SCOPE_VIEW_MEMBER are
-- the real catalog's scope-read abilities (defaults.json "Scopes" tab) -
-- any one grants read access here; which projects' scopes are visible still
-- follows the same project-read fallback as everything else project-scoped.
-- Money columns (total, subTotal, estDeal, estRevenue, estCostOfSale,
-- forecastRevenue, closeProbability, wonAt) added per the GraySync Formulas
-- reference: Scope.total is the real "Contracted Revenue" for a project
-- (page 7: "Approved service amount from scope"), and estDeal/estRevenue/
-- estCostOfSale/closeProbability/forecastRevenue answer the scope-level
-- forecast fields (page 22) directly - all plain stored columns already on
-- Scope, no new joins or gating needed.
SELECT sc.id, sc.name, sc.slug, sc."customId", sc."projectSlug", sc."organisationSlug",
       sc.status, sc.type, sc."dueDate", sc."companyId", sc."createdAt",
       sc.total, sc."subTotal", sc."estDeal", sc."estRevenue", sc."estCostOfSale",
       sc."forecastRevenue", sc."closeProbability", sc."wonAt",
       ai.org_currency_symbol(sc."organisationSlug") AS currency
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
-- visible_project_slugs() same as everything else project-scoped. Plain
-- INVOICE (legacy, pre-split entity - confirmed still live on role ids 1
-- and 37, "Owner") behaves like VIEW_ALL: the old schema had no
-- project-linked-only concept, so its holders always saw every invoice.
SELECT i.id, i."customId", i."organisationSlug", i."companyId", i."projectSlug", i."scopeSlug",
       i.type, i."issueDate", i."dueDate", i."amountPaid", i."paidAt", i."paymentStatus", i.balance,
       ai.org_currency_symbol(i."organisationSlug") AS currency, i."billToFinancialDetailId"
FROM public."Invoice" i
WHERE ai.session_has_read_ability('INVOICE_VIEW_ALL')
   OR ai.session_has_read_ability('INVOICE')
   OR (
       ai.session_has_read_ability('INVOICE_VIEW_PROJECT_LINKED')
       AND i."projectSlug" IS NOT NULL
       AND i."projectSlug" IN (SELECT * FROM ai.visible_project_slugs())
   );

CREATE OR REPLACE VIEW ai.v_invoice_item AS
SELECT ii.id, ii."invoiceId", ii.description, ii.quantity, ii."unitPrice", ii.discount, ii.amount,
       ai.org_currency_symbol(i."organisationSlug") AS currency
FROM public."InvoiceItem" ii
JOIN public."Invoice" i ON i.id = ii."invoiceId"
WHERE ai.session_has_read_ability('INVOICE_VIEW_ALL')
   OR ai.session_has_read_ability('INVOICE')
   OR (
       ai.session_has_read_ability('INVOICE_VIEW_PROJECT_LINKED')
       AND i."projectSlug" IS NOT NULL
       AND i."projectSlug" IN (SELECT * FROM ai.visible_project_slugs())
   );

CREATE OR REPLACE VIEW ai.v_expense AS
-- Same two-ability shape as invoices: EXPENSE_READ_ALL_LIST (org-wide) vs
-- EXPENSE_VIEW_PROJECT_LINKED (project-scoped). Also accepts any other
-- EXPENSE_* sub-ability with READ (session_has_read_ability_family) - e.g.
-- EXPENSE_APPROVE, EXPENSE_MODIFY_OTHERS - since a role that can act on
-- expenses can necessarily see them, and plain legacy EXPENSE (role ids 1,
-- 37 - "Owner") behaves like org-wide READ_ALL_LIST for the same reason as
-- v_invoice's legacy INVOICE fallback above.
-- Cost-of-Goods columns (supplierId, markup, markupType, totalPaid, balance,
-- status) added per Tier 2 of the GraySync Formulas reference - all plain
-- stored columns already on Expense, no new joins. supplierId (NOT
-- purchaserId, which is the internal User who made the purchase, a
-- different concept) is the real link to ai.v_supplier below.
SELECT e.id, e."customId", e."organisationSlug", e."projectSlug", e."purchaserId",
       e."purchaseDate", e."dueDate", e.cost, e.billed, e.profit, e.action,
       ai.org_currency_symbol(e."organisationSlug") AS currency,
       e."supplierId", e.markup, e."markupType", e."totalPaid", e.balance, e.status
FROM public."Expense" e
WHERE ai.session_has_read_ability('EXPENSE_READ_ALL_LIST')
   OR ai.session_has_read_ability_family('EXPENSE')
   OR (
       ai.session_has_read_ability('EXPENSE_VIEW_PROJECT_LINKED')
       AND e."projectSlug" IS NOT NULL
       AND e."projectSlug" IN (SELECT * FROM ai.visible_project_slugs())
   );

-- ---- Tier 2 (GraySync Formulas reference): supplier, scope-service,
-- retainer, resourcing, and customer rollups ----

CREATE OR REPLACE VIEW ai.v_supplier AS
-- One row per supplier, visibility derived transitively through the
-- expenses linked to it (a supplier has no direct project/org column of
-- its own, and no dedicated SUPPLIER catalog entity exists - confirmed
-- against defaults.json during the original permission-model build) - same
-- ability gate as v_expense, since a supplier is only reachable through
-- expenses the caller can already see.
SELECT sc.id, sc."customId", sc."paymentStatus", sc."mainTradingName",
       COUNT(e.id) AS total_expenses,
       COALESCE(SUM(e.cost), 0) AS total_cost_of_goods,
       COALESCE(SUM(e.billed), 0) AS total_billed,
       COALESCE(SUM(e.profit), 0) AS total_profit,
       COALESCE(SUM(e.balance), 0) AS total_outstanding
FROM public."SupplierContact" sc
JOIN public."Expense" e ON e."supplierId" = sc.id
WHERE (
    ai.session_has_read_ability('EXPENSE_READ_ALL_LIST')
    OR ai.session_has_read_ability_family('EXPENSE')
    OR (
        ai.session_has_read_ability('EXPENSE_VIEW_PROJECT_LINKED')
        AND e."projectSlug" IS NOT NULL
        AND e."projectSlug" IN (SELECT * FROM ai.visible_project_slugs())
    )
  )
GROUP BY sc.id, sc."customId", sc."paymentStatus", sc."mainTradingName";

CREATE OR REPLACE VIEW ai.v_scope_service AS
-- Per-service budget breakdown (the SCO-10010001-style rows on a project's
-- Budgets page). labour_cost_actual sums ALL of a scope's TimeEntry rows,
-- NOT per-service - TimeEntry links to taskId/scopeSlug, not to a specific
-- ScopeService row, so a true per-service labour split isn't reachable from
-- the current schema. This is an approximation, documented in
-- chat/tools_sql.py's schema description - never present it as exact.
SELECT
    ss.id, ss."sectionId", sec."scopeSlug", sc."projectSlug",
    svc.name AS "serviceName", ss.quantity, ss."totalCost", ss."totalAmount",
    COALESCE((
        SELECT SUM(te.cost) FROM public."TimeEntry" te WHERE te."scopeSlug" = sec."scopeSlug"
    ), 0) AS labour_cost_actual,
    ai.org_currency_symbol(sc."organisationSlug") AS currency
FROM public."ScopeService" ss
JOIN public."ScopeSection" sec ON sec.id = ss."sectionId"
JOIN public."Scope" sc ON sc.slug = sec."scopeSlug"
LEFT JOIN public."Service" svc ON svc.id = ss."serviceId"
WHERE (
    ai.session_has_read_ability('SCOPE')
    OR ai.session_has_read_ability('SCOPE_VIEW_ALL')
    OR ai.session_has_read_ability('SCOPE_VIEW_LINKED')
    OR ai.session_has_read_ability('SCOPE_VIEW_MEMBER')
  )
  AND (sc."projectSlug" IS NULL OR sc."projectSlug" IN (SELECT * FROM ai.visible_project_slugs()));

CREATE OR REPLACE VIEW ai.v_retainer_period AS
-- Per-billing-period retainer tracking. budgetedHours/budgetedAmount can be
-- NULL for a period (confirmed live on a sample row) - that's a real,
-- possibly-not-yet-configured period, not a system error; treat NULL the
-- same "not a real zero" way as v_project_budget's data-gap columns, not as
-- 0.
SELECT rbp.id, rbp."projectSlug", rbp."scopeSlug", rbp."periodName", rbp."periodIndex",
       rbp."startDate", rbp."endDate", rbp."budgetedHours", rbp."budgetedAmount",
       rbp."usedHours", rbp."incomeToDate",
       ai.org_currency_symbol(rbp."organisationSlug") AS currency
FROM public."RetainerBillingPeriod" rbp
WHERE ai.session_has_any_project_read()
  AND rbp."projectSlug" IN (SELECT * FROM ai.visible_project_slugs());

CREATE OR REPLACE VIEW ai.v_resourcing AS
-- Planned allocation per scope/member/month - the well-defined half of
-- "Time Allocation" (GraySync Formulas page 9). Resourcing/ResourcingMonth
-- have NO foreign key to Task at all (confirmed against the schema) - the
-- source reference's "MAX allocation between resourcing and task details"
-- cannot be expressed as a direct SQL join here; this view only exposes the
-- resourcing-side allocation, not a task comparison.
SELECT r.id, r."scopeSlug", sc."projectSlug", r."memberId", r."futureResourcing",
       rm.month, rm.year, rm.value AS allocated_hours, rm.status
FROM public."Resourcing" r
JOIN public."Scope" sc ON sc.slug = r."scopeSlug"
LEFT JOIN public."ResourcingMonth" rm ON rm."resourcingId" = r.id
WHERE ai.session_has_any_project_read()
  AND sc."projectSlug" IN (SELECT * FROM ai.visible_project_slugs());

CREATE OR REPLACE VIEW ai.v_customer AS
-- Customer/company financial-detail rollup - covers both invoice-customer
-- aggregates and project-customer detail, since Project.financialDetailId
-- and Invoice.billToFinancialDetailId both resolve to the same
-- CompanyFinancialDetail table (confirmed during Tier 1's verification
-- pass). Same ability gate as v_invoice, since a customer is only reachable
-- through invoices billed to them - no separate CUSTOMER catalog entity.
-- deletedAt IS NULL excludes soft-deleted customer records.
SELECT cfd.id, cfd."companyId", cfd.name, cfd.email, cfd.abn,
       COUNT(DISTINCT i.id) AS total_invoices,
       COALESCE(SUM(i."amountPaid"), 0) AS total_paid,
       COALESCE(SUM(i.balance), 0) AS total_outstanding_balance
FROM public."CompanyFinancialDetail" cfd
JOIN public."Invoice" i ON i."billToFinancialDetailId" = cfd.id
WHERE cfd."deletedAt" IS NULL
  AND (
    ai.session_has_read_ability('INVOICE_VIEW_ALL')
    OR ai.session_has_read_ability('INVOICE')
    OR (
        ai.session_has_read_ability('INVOICE_VIEW_PROJECT_LINKED')
        AND i."projectSlug" IS NOT NULL
        AND i."projectSlug" IN (SELECT * FROM ai.visible_project_slugs())
    )
  )
GROUP BY cfd.id, cfd."companyId", cfd.name, cfd.email, cfd.abn;

CREATE OR REPLACE VIEW ai.v_quote AS
-- Quote has no dedicated Entity in the real permission catalog (confirmed
-- against the full defaults.json) - gated on SCOPE-family read abilities as
-- the closest real analog (a quote is a pre-scope sales document), which is
-- an approximation, not a literal mapping like the views above.
SELECT q.id, q.quote_number, q.job_title, q."organisationSlug", q.project_id,
       q.issued_on, q.subtotal, q.gst, q.total, q.status,
       ai.org_currency_symbol(q."organisationSlug") AS currency
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

-- ---- Project-level budget/labour cost (GraySync Formulas reference - see
-- the "Allocated Budget" bug this was built to fix: GraySync's own UI shows
-- a per-project budget figure that is NOT stored anywhere, it's computed
-- live as SUM(Task.estimated x that task's owner's ProjectMemberRate.hourlyRate).
-- Distinct from v_budget/v_budget_data below, which are org-wide financial
-- budgets with no project link at all - do not confuse the two. ----

CREATE OR REPLACE VIEW ai.v_time_entry AS
-- Labour hours/cost logged against a task. TIME_ENTRIES/TIMESHEET_VIEW_OWN
-- are the real catalog's own-time-tracking abilities (confirmed live:
-- Employee Level holds TIME_ENTRIES:READ + TIMESHEET_VIEW_OWN, distinct
-- from and NOT requiring PROJECT_BUDGET at all - that gate was wrong,
-- borrowed from v_project_budget's build without checking TimeEntry's own
-- real catalog entities). A caller with either sees their own entries
-- (memberId = self); TIMESHEET_VIEW_ALL/TIMESHEET_MODIFY_OTHERS
-- additionally allows seeing everyone's, same own-vs-others shape as
-- v_leave_request. Still project-scoped the same way as v_task (via the
-- task's own projectSlug, or via scopeSlug for the rare case - none
-- observed live, but taskId is nullable at the schema level - of a time
-- entry linked to a scope with no task). No "billable" column exists
-- anywhere on TimeEntry (confirmed) - never invent one.
SELECT te.id, te."taskId", te."memberId", te."scopeSlug", te."invoiceId",
       te."recordType", te.duration, te.cost, te.total, te."dayCreated", te."createdAt"
FROM public."TimeEntry" te
LEFT JOIN public."Task" t ON t.id = te."taskId"
WHERE (
    ai.session_has_read_ability('TIME_ENTRIES')
    OR ai.session_has_read_ability('TIMESHEET_VIEW_OWN')
    OR ai.session_has_read_ability('TIMESHEET')
  )
  AND (
    te."memberId" = ai.session_user_id()
    OR ai.session_can_see_others_time_entries()
  )
  AND (
    (t."projectSlug" IS NOT NULL AND t."projectSlug" IN (SELECT * FROM ai.visible_project_slugs()))
    OR (t."projectSlug" IS NULL AND te."scopeSlug" IN (
          SELECT slug FROM public."Scope" WHERE "projectSlug" IN (SELECT * FROM ai.visible_project_slugs())
    ))
  );

CREATE OR REPLACE VIEW ai.v_project_member_rate AS
-- Per-project rate override. Per the GraySync Formulas reference (verbatim,
-- their own red-text note): "Rate card is coming from the project people
-- table not from the organisation rate card settings" - i.e. for
-- project-level budget/labour-cost math, this table is the source of
-- truth, NOT ai.v_rate_card (which is the org-wide fallback, keyed by
-- position, not by person-on-this-project).
SELECT pmr.id, p.slug AS "projectSlug", pmr."staffId", pmr."hourlyRate"
FROM public."ProjectMemberRate" pmr
JOIN public."Project" p ON p.id = pmr."projectId"
WHERE ai.session_has_read_ability('PROJECT_BUDGET')
  AND p.slug IN (SELECT * FROM ai.visible_project_slugs());

-- Project-level financial rollup: the real "Allocated Budget" fix.
-- budget_total_estimated = SUM(Task.estimated x that task owner's
-- ProjectMemberRate.hourlyRate), per the GraySync Formulas reference
-- ("Budget Total (Allocated budget) = Estimated resource cost = Sum of
-- Estimated time for each task x Rate Card"). CONFIRMED LIVE: ProjectMemberRate
-- has only 3 rows in this database at time of writing - a project with
-- tasks but no matching ProjectMemberRate row will show
-- budget_total_estimated = 0, which is a data-completeness gap, not a real
-- zero budget (see the schema note in chat/tools_sql.py, which tells the
-- agent to say so rather than state 0 as fact).
--
-- UNRESOLVED, FLAG FOR THE GRAYSYNC TEAM: labour_cost_actual sums
-- TimeEntry.cost (matching the reference's literal field name, "Cost =
-- Tracking Time * Cost Rate"), but this is NOT fully confirmed against
-- real duration-based rate math - checked live: for one staff member,
-- TimeEntry.total / (duration in hours) was a flat, consistent rate across
-- every entry (e.g. always $180/hr), while TimeEntry.cost / duration
-- varied wildly entry-to-entry (65, 360, 515, 540 per hour) for the SAME
-- person - meaning .total behaves like a clean rate x duration figure and
-- .cost does not, for at least this sample. Left as .cost per the
-- reference's literal wording, but this should be confirmed against
-- GraySync's actual backend calculation before being treated as authoritative.
--
-- current_profit and budget_remaining are deliberately NOT stored columns
-- here - derived by the agent from the four columns below
-- (contracted_revenue_total - labour_cost_actual - expense_cost_actual,
-- and budget_total_estimated - labour_cost_actual - expense_cost_actual
-- respectively), same pattern as v_leave_policy's remaining-balance formula.
CREATE OR REPLACE VIEW ai.v_project_budget AS
SELECT
    p.slug AS "projectSlug",
    COALESCE(SUM(t.estimated * pmr."hourlyRate") FILTER (WHERE t.id IS NOT NULL), 0) AS budget_total_estimated,
    COALESCE((SELECT SUM(te.cost) FROM public."TimeEntry" te JOIN public."Task" t2 ON t2.id = te."taskId" WHERE t2."projectSlug" = p.slug), 0) AS labour_cost_actual,
    COALESCE((SELECT SUM(e.cost) FROM public."Expense" e WHERE e."projectSlug" = p.slug), 0) AS expense_cost_actual,
    COALESCE((SELECT SUM(sc.total) FROM public."Scope" sc WHERE sc."projectSlug" = p.slug), 0) AS contracted_revenue_total,
    ai.org_currency_symbol(p."organisationSlug") AS currency
FROM public."Project" p
LEFT JOIN public."Task" t ON t."projectSlug" = p.slug
LEFT JOIN public."ProjectMemberRate" pmr ON pmr."projectId" = p.id AND pmr."staffId" = t."ownerId"
WHERE ai.session_has_any_project_read()
  AND p.slug IN (SELECT * FROM ai.visible_project_slugs())
GROUP BY p.slug, p."organisationSlug";

CREATE OR REPLACE VIEW ai.v_budget AS
-- Budget spans every organisation in the database in one table (confirmed:
-- 41 rows across 20 orgs) with no natural per-row scoping otherwise - same
-- leak shape as v_leave_policy had, fixed the same way.
SELECT b.id, b.name, b."organisationSlug", b."financialYearId", b."createdAt",
       ai.org_currency_symbol(b."organisationSlug") AS currency
FROM public."Budget" b
WHERE ai.session_has_read_ability('PROJECT_BUDGET')
  AND b."organisationSlug" IN (SELECT * FROM ai.session_visible_orgs());

CREATE OR REPLACE VIEW ai.v_budget_data AS
-- Budget -> AccountBudget (one row per chart-of-accounts line per budget) ->
-- BudgetData (one row per month/year with a dollar value) - this is an
-- org-wide financial budget broken down by account category and month, NOT
-- a per-project budget. There is no projectId/projectSlug anywhere in this
-- chain (confirmed against the real schema) - "which project has the
-- highest budget" cannot be answered from this data; project-level
-- financials live in v_invoice/v_expense/v_scope instead (see schema note
-- in tools_sql.py).
SELECT bd.id, bd."accountBudgetId", ab."budgetId", b.name AS "budgetName",
       ab."organisationSlug", bd.month, bd.year, bd.value,
       ai.org_currency_symbol(ab."organisationSlug") AS currency
FROM public."BudgetData" bd
JOIN public."AccountBudget" ab ON ab.id = bd."accountBudgetId"
JOIN public."Budget" b ON b.id = ab."budgetId"
WHERE ai.session_has_read_ability('PROJECT_BUDGET')
  AND ab."organisationSlug" IN (SELECT * FROM ai.session_visible_orgs());

CREATE OR REPLACE VIEW ai.v_rate_card AS
-- RateCard has no dedicated Entity in the real catalog either - gated on
-- PROJECT_BUDGET (rate cards feed budget/cost calculations) as the closest
-- real analog, same caveat as v_quote above.
SELECT rc.id, rc."rateCardGroupId", rc."positionId", rc."hourlyRate", rc."dailyRate"
FROM public."RateCard" rc
WHERE ai.session_has_read_ability('PROJECT_BUDGET');

-- ---- Org-wide entities (no per-project ownership dimension) ----

-- The views below (announcement/contact/department/skill) each span every
-- organisation in the database in one table with no natural per-row
-- scoping otherwise - same leak shape as v_leave_policy had (confirmed:
-- Announcement 2 distinct orgs, Department 20, Skill 2), fixed the same way
-- via ai.session_visible_orgs().

CREATE OR REPLACE VIEW ai.v_announcement AS
SELECT a.id, a."organisationSlug", a."authorUserId", a.type, a.status, a.title,
       a."contentText", a."startsAt", a."endsAt", a."isPinned", a."publishedAt", a."createdAt"
FROM public."Announcement" a
WHERE ai.session_has_read_ability('ANNOUNCEMENT')
  AND a."organisationSlug" IN (SELECT * FROM ai.session_visible_orgs());

CREATE OR REPLACE VIEW ai.v_announcement_comment AS
SELECT ac.id, ac."announcementId", ac."authorUserId", ac."contentText", ac.status, ac."createdAt"
FROM public."AnnouncementComment" ac
JOIN public."Announcement" a ON a.id = ac."announcementId"
WHERE ai.session_has_read_ability('ANNOUNCEMENT')
  AND a."organisationSlug" IN (SELECT * FROM ai.session_visible_orgs());

CREATE OR REPLACE VIEW ai.v_contact AS
-- COMPANY is the real catalog entity for the "Contacts" tab (bundles
-- COMPANY+SUPPLIER - see defaults.json "Add contacts"/"View list of contacts").
SELECT c.id, c.slug, c.name, c.type, c.city, c.state, c.country, c.website, c.email, c."createdAt"
FROM public."Contact" c
JOIN public."ContactOnOrganisation" coo ON coo."contactId" = c.id
WHERE ai.session_has_read_ability('COMPANY')
  AND coo."organisationSlug" IN (SELECT * FROM ai.session_visible_orgs());
-- notes/address intentionally excluded - CONTACT_NOTES already covered via
-- RAG ingestion (schema.sql RagSourceType), not needed raw here too.

CREATE OR REPLACE VIEW ai.v_company_contact AS
SELECT cc.id, cc."customId", cc.email, cc."companyType", cc.status, c.name
FROM public."CompanyContact" cc
JOIN public."Contact" c ON c.id = cc.id
JOIN public."ContactOnOrganisation" coo ON coo."contactId" = c.id
WHERE ai.session_has_read_ability('COMPANY')
  AND coo."organisationSlug" IN (SELECT * FROM ai.session_visible_orgs());

CREATE OR REPLACE VIEW ai.v_department AS
SELECT d.id, d.name, d."organisationSlug"
FROM public."Department" d
WHERE ai.session_has_read_ability('ORG_DEPARTMENTS')
  AND d."organisationSlug" IN (SELECT * FROM ai.session_visible_orgs());

CREATE OR REPLACE VIEW ai.v_position AS
SELECT pos.id, pos.name, pos."departmentId", dep."organisationSlug"
FROM public."Position" pos
JOIN public."Department" dep ON dep.id = pos."departmentId"
WHERE ai.session_has_read_ability('ORG_DEPARTMENTS')
  AND dep."organisationSlug" IN (SELECT * FROM ai.session_visible_orgs());

CREATE OR REPLACE VIEW ai.v_skill AS
SELECT s.id, s.name, s."organisationSlug"
FROM public."Skill" s
WHERE ai.session_has_read_ability('ORG_SKILLS')
  AND s."organisationSlug" IN (SELECT * FROM ai.session_visible_orgs());

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
-- Policy terms (entitlement, accrual rules) aren't per-user data - no
-- per-user ownership dimension within an org, needed to compute "how many
-- days do I have left". Still gated on LEAVES:READ - no point exposing
-- policy terms to a role that can't see any leave data at all. UNLIKE other
-- org-wide views (v_announcement, v_department, ...), LeavePolicy spans
-- every organisation in the whole database in one table with no natural
-- per-row scoping otherwise - confirmed this was returning all ~34 policies
-- system-wide to any caller with LEAVES:READ before this org filter was
-- added, corrupting "how many days do I have left" with other tenants'
-- differently-named, differently-valued policies.
-- applicableAfter/applicableAfterUnit added (confirmed live: Amir Moadad,
-- hired 2025-11-01, genuinely has 0 Annual Leave available today per
-- GraySync's own UI - not a bug - because that policy's applicableAfter is
-- 12 MONTHS and he isn't there yet). allowCarryForward/accrualRate/
-- maxAccrual/maxCarryForward also added but NOT yet factored into the
-- documented remaining-leave formula (see tools_sql.py schema notes) -
-- confirmed live a second policy (Rony Chiha's Sick Leave) shows a real
-- entitlement in GraySync's UI (30) that plain LeavePolicy.entitlement (10)
-- does not explain; this is very likely accrual/carry-forward math this
-- schema does not yet reproduce, left as a known, documented gap rather
-- than a guessed-at formula.
SELECT lp.id, lp.name, lp.entitlement, lp."entitlementUnit", lp."recurringPeriod",
       lp."isPaid", lp."allowFullDay", lp."allowHalfDay", lg."organisationSlug",
       lp."applicableAfter", lp."applicableAfterUnit", lp."allowCarryForward",
       lp."accrualRate", lp."maxAccrual", lp."maxCarryForward"
FROM public."LeavePolicy" lp
JOIN public."LeaveGroup" lg ON lg.id = lp."groupId"
WHERE ai.session_has_read_ability('LEAVES')
  AND lg."organisationSlug" IN (SELECT * FROM ai.session_visible_orgs());

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
    ai.v_invoice, ai.v_invoice_item, ai.v_expense, ai.v_quote, ai.v_budget, ai.v_budget_data, ai.v_rate_card,
    ai.v_time_entry, ai.v_project_member_rate, ai.v_project_budget,
    ai.v_supplier, ai.v_scope_service, ai.v_retainer_period, ai.v_resourcing, ai.v_customer,
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
    public."User", public."Staff", public."LeaveGroup", public."Ability",
    public."UserOrganisationAccess", public."Organisation", public."BudgetData", public."AccountBudget",
    public."TimeEntry", public."ProjectMemberRate", public."Scope", public."Expense",
    public."OrganisationFinance", public."Currency",
    public."SupplierContact", public."ScopeService", public."ScopeSection", public."Service",
    public."RetainerBillingPeriod", public."Resourcing", public."ResourcingMonth", public."CompanyFinancialDetail",
    public."Invoice", public."LeavePolicy"
    TO ai_readonly;
