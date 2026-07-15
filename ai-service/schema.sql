-- Run this against the graysync database (or add these as Prisma migrations,
-- since Prisma should stay the source of truth for schema on the NestJS side).

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TYPE "RagSourceType" AS ENUM (
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

CREATE TYPE "RagIngestStatus" AS ENUM ('PENDING', 'PROCESSING', 'COMPLETED', 'FAILED');

-- NOTE on subdomainName / organisationSlug:
-- subdomainName is the real tenant boundary and is always required. organisationSlug
-- is nullable and, when present, is the finer-grained scope within that tenant. These
-- two are NEVER compared against each other as if they share one namespace (no more
-- "scopeKey" collapsing them into a single string) - two different tenants could
-- coincidentally reuse the same slug/subdomain string, and matching on that alone
-- would leak one tenant's data into another's results. Every query filters
-- subdomainName exactly first, then organisationSlug as a second, independent check.

CREATE TABLE "RagSource" (
    id BIGSERIAL PRIMARY KEY,
    "subdomainName" text NOT NULL,
    "organisationSlug" text,            -- nullable: NULL = general/subdomain-wide, set = specific to that organisation
    "sourceType" "RagSourceType" NOT NULL,
    "sourceId" text NOT NULL,
    "projectSlug" text,
    "taskId" integer,
    "scopeSlug" text,
    "userId" integer,
    "companyId" integer,
    checksum text,
    status "RagIngestStatus" NOT NULL DEFAULT 'PENDING',
    "errorMessage" text,
    "ingestedAt" timestamp,
    "createdAt" timestamp NOT NULL DEFAULT now(),
    "updatedAt" timestamp NOT NULL DEFAULT now(),
    UNIQUE ("sourceType", "sourceId")
);
CREATE INDEX ON "RagSource" ("subdomainName", "organisationSlug");
CREATE INDEX ON "RagSource" ("projectSlug");
CREATE INDEX ON "RagSource" ("taskId");

CREATE TABLE "RagChunk" (
    id BIGSERIAL PRIMARY KEY,
    "sourceId" bigint NOT NULL REFERENCES "RagSource"(id) ON DELETE CASCADE,
    "subdomainName" text NOT NULL,      -- denormalized from RagSource so retrieval can filter without a join
    "organisationSlug" text,            -- denormalized too, same nullable rule
    "chunkIndex" integer NOT NULL,
    content text NOT NULL,
    "tokenCount" integer,
    embedding vector(1024),  -- dimension must match your embedding model (voyage-3 = 1024)
    metadata jsonb,
    "createdAt" timestamp NOT NULL DEFAULT now(),
    UNIQUE ("sourceId", "chunkIndex")
);
CREATE INDEX rag_chunk_embedding_idx ON "RagChunk" USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX ON "RagChunk" ("subdomainName", "organisationSlug");

CREATE TABLE "ProjectSummary" (
    id SERIAL PRIMARY KEY,
    "projectId" integer NOT NULL REFERENCES "Project"(id) ON DELETE CASCADE,
    "summaryText" text NOT NULL,
    "sourceHash" text NOT NULL,
    model text NOT NULL,
    "generatedAt" timestamp NOT NULL DEFAULT now(),
    UNIQUE ("projectId")
);

CREATE TABLE "AiInvocationLog" (
    id BIGSERIAL PRIMARY KEY,
    "subdomainName" text NOT NULL,
    "organisationSlug" text,            -- nullable: same rule as RagSource/RagChunk
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
CREATE INDEX ON "AiInvocationLog" ("subdomainName", "organisationSlug", capability);
-- Fast lookup for EXACT cache matches: newest SUCCESS row for a given capability + tenant + exact context hash.
CREATE INDEX ON "AiInvocationLog" (capability, "subdomainName", "contextHash", "createdAt" DESC)
    WHERE status = 'SUCCESS';
-- Fast lookup for SEMANTIC cache matches: vector similarity search scoped to successful calls only.
CREATE INDEX ai_invocation_log_embedding_idx ON "AiInvocationLog"
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 50);
