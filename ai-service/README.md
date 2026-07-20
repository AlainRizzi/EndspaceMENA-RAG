# GraySync AI Service

FastAPI service exposing AI capabilities (`suggest_skills`, `summarize_project`) backed by
Gemini, AWS Bedrock (embeddings + rerank), and a RAG pipeline over `graysync` Postgres data.

## Prerequisites

- Postgres 18 with the `pgvector` extension installed, `schema.sql` applied to `graysync`.
- Redis reachable at `REDIS_URL` (e.g. `docker run -d --name redis -p 6379:6379 redis:7-alpine`).
- `.env` filled in (see `.env.example`) — `DATABASE_URL`, `GEMINI_API_KEY`, AWS keys with
  `bedrock:InvokeModel` (and ideally `bedrock:Rerank`) plus `s3:GetObject` on the target bucket.

## Setup

```powershell
pip install -r requirements.txt
```

## Running

Two processes must run at the same time, **in separate terminals**:

**Terminal 1 — API server:**
```powershell
uvicorn main:app --reload
```
Serves `http://127.0.0.1:8000`. Auto-reloads on file changes.

**Terminal 2 — background job worker** (required for async capabilities like `summarize_project`):
```powershell
arq worker.WorkerSettings
```
Does **not** auto-reload — restart it manually (`Ctrl+C`, then re-run) after editing any
capability, schema, or service file it imports.

## Testing

Run these from a third terminal once both processes above are up. PowerShell's built-in `curl`
alias doesn't support `-H`/`-d` the way real curl does — use `curl.exe` explicitly.

### `suggest_skills` (sync, no queue needed)

**1.**
```powershell
curl.exe -X POST http://127.0.0.1:8000/ai/suggest_skills -H "Content-Type: application/json" -d "{\"organisationSlug\": \"endspace-mena\", \"title\": \"Launch summer campaign\", \"description\": \"Plan and execute a social media campaign for summer promotions\"}"
```

**2.**
```powershell
curl.exe -X POST http://127.0.0.1:8000/ai/suggest_skills -H "Content-Type: application/json" -d "{\"organisationSlug\": \"adcreators-mena\", \"title\": \"Rebuild client onboarding flow\", \"description\": \"Design and implement a smoother onboarding experience for new clients\"}"
```

**3.**
```powershell
curl.exe -X POST http://127.0.0.1:8000/ai/suggest_skills -H "Content-Type: application/json" -d "{\"organisationSlug\": \"endspace-mena\", \"title\": \"Fix production API bug\", \"description\": \"Investigate and resolve a reported bug in the backend API affecting checkout\"}"
```

### `summarize_project` (async — submits a job, then poll for the result)

**1.**
```powershell
curl.exe -X POST http://127.0.0.1:8000/ai/summarize_project -H "Content-Type: application/json" -d "{\"organisationSlug\": \"adcreators-australia\", \"projectSlug\": \"360-marketing\"}"
```

**2.**
```powershell
curl.exe -X POST http://127.0.0.1:8000/ai/summarize_project -H "Content-Type: application/json" -d "{\"organisationSlug\": \"born-creators-australia\", \"projectSlug\": \"internal-bcg-projects\"}"
```

**3.**
```powershell
curl.exe -X POST http://127.0.0.1:8000/ai/summarize_project -H "Content-Type: application/json" -d "{\"organisationSlug\": \"adcreators-mena\", \"projectSlug\": \"marketing_15\"}"
```

Each returns `{"status": "PROCESSING", "jobId": "..."}`. Poll with the returned `jobId`:
```powershell
curl.exe http://127.0.0.1:8000/ai/jobs/<jobId>
```
Repeat until `"status": "COMPLETED"`.

## Ingesting RAG data

Populates `RagSource`/`RagChunk` from `graysync` Postgres + S3 documents. Safe to re-run —
skips unchanged content via checksum, and prunes rows whose source was deleted.

```powershell
python ingest.py                    # all source types
python ingest.py ANNOUNCEMENT       # just one type
```

Only documents stored in `S3_BUCKET_NAME` (preprod) are ingested; documents in other buckets
are skipped and logged.

## Notes

- `organisationSlug` (`Organisation.slug`) is the tenant boundary throughout — required on
  every capability input, unique across the whole database today.
- Rerank failures (e.g. missing `bedrock:Rerank` permission) degrade gracefully to
  pgvector-similarity ordering rather than failing the whole search.
