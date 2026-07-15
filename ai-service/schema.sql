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

CREATE TABLE "RagSource" (
    id BIGSERIAL PRIMARY KEY,
    "subdomainName" text NOT NULL,
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
CREATE INDEX ON "RagSource" ("subdomainName");
CREATE INDEX ON "RagSource" ("projectSlug");
CREATE INDEX ON "RagSource" ("taskId");

CREATE TABLE "RagChunk" (
    id BIGSERIAL PRIMARY KEY,
    "sourceId" bigint NOT NULL REFERENCES "RagSource"(id) ON DELETE CASCADE,
    "subdomainName" text NOT NULL,
    "chunkIndex" integer NOT NULL,
    content text NOT NULL,
    "tokenCount" integer,
    embedding vector(1024),  -- dimension must match your embedding model (voyage-3 = 1024)
    metadata jsonb,
    "createdAt" timestamp NOT NULL DEFAULT now(),
    UNIQUE ("sourceId", "chunkIndex")
);
CREATE INDEX rag_chunk_embedding_idx ON "RagChunk" USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX ON "RagChunk" ("subdomainName");

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
    capability text NOT NULL,
    "inputPayload" jsonb NOT NULL,
    "outputPayload" jsonb,
    model text,
    "promptTokens" integer,
    "completionTokens" integer,
    "latencyMs" integer,
    status text NOT NULL,
    "errorMessage" text,
    "createdAt" timestamp NOT NULL DEFAULT now()
);
CREATE INDEX ON "AiInvocationLog" ("subdomainName", capability);
