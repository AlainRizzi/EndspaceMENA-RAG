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

-- NOTE on organisationSlug:
-- organisationSlug (Organisation.slug) is the tenant boundary, always required.
-- subdomainName was dropped from this scoping - it identifies which hosting
-- instance a row lives on, not which company/organisation owns it, and
-- Organisation.slug is unique across the whole database today.

CREATE TABLE "RagSource" (
    id BIGSERIAL PRIMARY KEY,
    "organisationSlug" text NOT NULL,
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
CREATE INDEX ON "RagSource" ("organisationSlug");
CREATE INDEX ON "RagSource" ("projectSlug");
CREATE INDEX ON "RagSource" ("taskId");

CREATE TABLE "RagChunk" (
    id BIGSERIAL PRIMARY KEY,
    "sourceId" bigint NOT NULL REFERENCES "RagSource"(id) ON DELETE CASCADE,
    "organisationSlug" text NOT NULL,   -- denormalized from RagSource so retrieval can filter without a join
    "chunkIndex" integer NOT NULL,
    content text NOT NULL,
    "tokenCount" integer,
    embedding vector(1024),  -- dimension must match your embedding model (Titan Embed Text v2 = 1024)
    metadata jsonb,
    "createdAt" timestamp NOT NULL DEFAULT now(),
    UNIQUE ("sourceId", "chunkIndex")
);
CREATE INDEX rag_chunk_embedding_idx ON "RagChunk" USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX ON "RagChunk" ("organisationSlug");

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
CREATE INDEX ON "AiInvocationLog" ("organisationSlug", capability);
-- Fast lookup for EXACT cache matches: newest SUCCESS row for a given capability + tenant + exact context hash.
CREATE INDEX ON "AiInvocationLog" (capability, "organisationSlug", "contextHash", "createdAt" DESC)
    WHERE status = 'SUCCESS';
-- Fast lookup for SEMANTIC cache matches: vector similarity search scoped to successful calls only.
CREATE INDEX ai_invocation_log_embedding_idx ON "AiInvocationLog"
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 50);
