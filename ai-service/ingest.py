import asyncio
import hashlib
import logging
from dataclasses import dataclass
from typing import Any

from chunking import chunk_text
from db import get_pool
from retrieval_service import retrieval_service
from s3_client import fetch_and_extract

logger = logging.getLogger("ingest")


@dataclass
class SourceRow:
    """One ingestable unit resolved from a Postgres table row, before chunking.
    organisationSlug (Organisation.slug) is the tenant boundary everywhere here -
    every spec below resolves it via a join, since most source tables only carry
    subdomainName (which identifies a hosting instance, not an organisation).
    """
    source_type: str
    source_id: str
    organisation_slug: str
    project_slug: str | None
    task_id: int | None
    scope_slug: str | None
    user_id: int | None
    company_id: int | None
    content: str | None  # None => nothing to embed (e.g. failed S3 fetch, image file)


# Tables that store an uploaded file (fileUrl) rather than text directly.
# content is fetched from S3 and text-extracted at ingest time.
_DOCUMENT_SOURCES: list[dict[str, Any]] = [
    {
        "source_type": "PROJECT_DOCUMENT",
        "query": (
            'SELECT pd.id, p."organisationSlug", pd."projectSlug", pd."fileUrl", pd."fileType" '
            'FROM "ProjectDocument" pd JOIN "Project" p ON p.slug = pd."projectSlug" '
            'WHERE p."organisationSlug" IS NOT NULL'
        ),
        "row_to_source": lambda r: SourceRow(
            "PROJECT_DOCUMENT", str(r["id"]), r["organisationSlug"],
            r["projectSlug"], None, None, None, None, content=None,
        ),
    },
    {
        "source_type": "TASK_DOCUMENT",
        "query": (
            'SELECT td.id, p."organisationSlug", t."projectSlug", td."taskId", td."fileUrl", td."fileType" '
            'FROM "TaskDocument" td '
            'JOIN "Task" t ON t.id = td."taskId" '
            'JOIN "Project" p ON p.slug = t."projectSlug" '
            'WHERE p."organisationSlug" IS NOT NULL'
        ),
        "row_to_source": lambda r: SourceRow(
            "TASK_DOCUMENT", str(r["id"]), r["organisationSlug"],
            r["projectSlug"], r["taskId"], None, None, None, content=None,
        ),
    },
    {
        "source_type": "STAFF_DOCUMENT",
        "query": (
            'SELECT sd.id, u."organisationSlug", sd."userId", sd."fileUrl", sd."fileType" '
            'FROM "StaffDocument" sd JOIN "User" u ON u.id = sd."userId" '
            'WHERE u."organisationSlug" IS NOT NULL'
        ),
        "row_to_source": lambda r: SourceRow(
            "STAFF_DOCUMENT", str(r["id"]), r["organisationSlug"],
            None, None, None, r["userId"], None, content=None,
        ),
    },
    {
        "source_type": "SCOPE_DOCUMENT",
        "query": (
            'SELECT sd.id, s."organisationSlug", sd."scopeSlug", sd."fileUrl", sd."fileType" '
            'FROM "ScopeDocument" sd JOIN "Scope" s ON s.slug = sd."scopeSlug" '
            'WHERE s."organisationSlug" IS NOT NULL'
        ),
        "row_to_source": lambda r: SourceRow(
            "SCOPE_DOCUMENT", str(r["id"]), r["organisationSlug"],
            None, None, r["scopeSlug"], None, None, content=None,
        ),
    },
    {
        "source_type": "MEDIA_PLAN_DOCUMENT",
        "query": (
            'SELECT mpd.id, mp."organisationSlug", mpd."fileUrl", mpd."fileType" '
            'FROM "MediaPlanDocument" mpd JOIN "MediaPlan" mp ON mp.slug = mpd."mediaPlanSlug" '
            'WHERE mp."organisationSlug" IS NOT NULL'
        ),
        "row_to_source": lambda r: SourceRow(
            "MEDIA_PLAN_DOCUMENT", str(r["id"]), r["organisationSlug"],
            None, None, None, None, None, content=None,
        ),
    },
    {
        # companyId -> Contact, which is many-to-many with Organisation via
        # ContactOnOrganisation - one row here can resolve to several orgs, so this
        # yields one SourceRow per (document, organisation) pair. source_id is
        # suffixed with the org slug to keep it unique per RagSource row.
        "source_type": "COMPANY_DOCUMENT",
        "query": (
            'SELECT cd.id, coo."organisationSlug", cd."companyId", cd."fileUrl", cd."fileType" '
            'FROM "CompanyDocument" cd '
            'JOIN "ContactOnOrganisation" coo ON coo."contactId" = cd."companyId"'
        ),
        "row_to_source": lambda r: SourceRow(
            "COMPANY_DOCUMENT", f"{r['id']}:{r['organisationSlug']}", r["organisationSlug"],
            None, None, None, None, r["companyId"], content=None,
        ),
    },
]

# Tables whose relevant text already lives in a column - no S3 fetch needed.
_TEXT_SOURCES: list[dict[str, Any]] = [
    {
        "source_type": "FEEDBACK",
        "query": (
            'SELECT f.id, u."organisationSlug", f."userId", f.message '
            'FROM "Feedback" f JOIN "User" u ON u.id = f."userId" '
            'WHERE u."organisationSlug" IS NOT NULL'
        ),
        "row_to_source": lambda r: SourceRow(
            "FEEDBACK", str(r["id"]), r["organisationSlug"],
            None, None, None, r["userId"], None, content=r["message"],
        ),
    },
    {
        "source_type": "FEEDBACK_SUBMISSION",
        "query": (
            'SELECT fs.id, u."organisationSlug", fs."submitterId", fs."firstAnswer", fs."secondAnswer" '
            'FROM "FeedbackSubmission" fs JOIN "User" u ON u.id = fs."submitterId" '
            'WHERE u."organisationSlug" IS NOT NULL'
        ),
        "row_to_source": lambda r: SourceRow(
            "FEEDBACK_SUBMISSION", str(r["id"]), r["organisationSlug"],
            None, None, None, r["submitterId"], None,
            content="\n".join(filter(None, [r["firstAnswer"], r["secondAnswer"]])) or None,
        ),
    },
    {
        "source_type": "STAFF_NOTE",
        "query": (
            'SELECT sn.id, u."organisationSlug", sn."userId", sn.note '
            'FROM "StaffNote" sn JOIN "User" u ON u.id = sn."userId" '
            'WHERE u."organisationSlug" IS NOT NULL'
        ),
        "row_to_source": lambda r: SourceRow(
            "STAFF_NOTE", str(r["id"]), r["organisationSlug"],
            None, None, None, r["userId"], None, content=r["note"],
        ),
    },
    {
        "source_type": "ANNOUNCEMENT",
        "select": 'id, "organisationSlug", title, "contentText"',
        "table": "Announcement",
        "where": '"organisationSlug" IS NOT NULL',
        "row_to_source": lambda r: SourceRow(
            "ANNOUNCEMENT", str(r["id"]), r["organisationSlug"],
            None, None, None, None, None,
            content="\n".join(filter(None, [r["title"], r["contentText"]])) or None,
        ),
    },
    {
        "source_type": "ANNOUNCEMENT_COMMENT",
        "query": (
            'SELECT ac.id, a."organisationSlug", ac."contentText" '
            'FROM "AnnouncementComment" ac JOIN "Announcement" a ON a.id = ac."announcementId" '
            'WHERE a."organisationSlug" IS NOT NULL'
        ),
        "row_to_source": lambda r: SourceRow(
            "ANNOUNCEMENT_COMMENT", str(r["id"]), r["organisationSlug"],
            None, None, None, None, None, content=r["contentText"],
        ),
    },
    {
        # Contact<->Organisation is many-to-many - one SourceRow per (contact, org) pair,
        # same reasoning as COMPANY_DOCUMENT above.
        "source_type": "CONTACT_NOTES",
        "query": (
            'SELECT c.id, coo."organisationSlug", c.notes '
            'FROM "Contact" c JOIN "ContactOnOrganisation" coo ON coo."contactId" = c.id '
            'WHERE c.notes IS NOT NULL'
        ),
        "row_to_source": lambda r: SourceRow(
            "CONTACT_NOTES", f"{r['id']}:{r['organisationSlug']}", r["organisationSlug"],
            None, None, None, None, None, content=r["notes"],
        ),
    },
    {
        "source_type": "OBJECTIVE",
        "query": (
            'SELECT o.id, u."organisationSlug", o.detail, g."userId" '
            'FROM "Objective" o '
            'JOIN "Goal" g ON g.id = o."goalId" '
            'JOIN "User" u ON u.id = g."userId" '
            'WHERE u."organisationSlug" IS NOT NULL'
        ),
        "row_to_source": lambda r: SourceRow(
            "OBJECTIVE", str(r["id"]), r["organisationSlug"],
            None, None, None, r["userId"], None, content=r["detail"],
        ),
    },
]


async def _fetch_rows(spec: dict) -> list[dict]:
    if "query" in spec:
        query = spec["query"]
    else:
        query = f'SELECT {spec["select"]} FROM "{spec["table"]}"'
        if spec.get("where"):
            query += f" WHERE {spec['where']}"

    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(query)
        return [dict(r) for r in rows]


async def _resolve_document_source(spec: dict, row: dict) -> SourceRow:
    source = spec["row_to_source"](row)
    try:
        source.content = fetch_and_extract(row["fileUrl"], row["fileType"])
    except Exception as e:
        logger.warning("s3 fetch failed for %s %s: %s", spec["source_type"], row["id"], e)
        source.content = None
    return source


async def _iter_document_sources(spec: dict):
    for row in await _fetch_rows(spec):
        yield await _resolve_document_source(spec, row)


async def _iter_text_sources(spec: dict):
    for row in await _fetch_rows(spec):
        yield spec["row_to_source"](row)


async def _upsert_source(conn, source: SourceRow, checksum: str, status: str, error: str | None) -> tuple[int, bool]:
    """Returns (source_id, changed). changed=False means the checksum matched the
    existing row and nothing was written - caller should leave chunks untouched.
    """
    row = await conn.fetchrow(
        """
        INSERT INTO "RagSource"
            ("organisationSlug", "sourceType", "sourceId", "projectSlug",
             "taskId", "scopeSlug", "userId", "companyId", checksum, status, "errorMessage",
             "ingestedAt", "updatedAt")
        VALUES ($1, $2::"RagSourceType", $3, $4, $5, $6, $7, $8, $9, $10::"RagIngestStatus", $11, now(), now())
        ON CONFLICT ("sourceType", "sourceId") DO UPDATE SET
            "organisationSlug" = EXCLUDED."organisationSlug",
            "projectSlug" = EXCLUDED."projectSlug",
            "taskId" = EXCLUDED."taskId",
            "scopeSlug" = EXCLUDED."scopeSlug",
            "userId" = EXCLUDED."userId",
            "companyId" = EXCLUDED."companyId",
            checksum = EXCLUDED.checksum,
            status = EXCLUDED.status,
            "errorMessage" = EXCLUDED."errorMessage",
            "ingestedAt" = now(),
            "updatedAt" = now()
        WHERE "RagSource".checksum IS DISTINCT FROM EXCLUDED.checksum
        RETURNING id
        """,
        source.organisation_slug, source.source_type, source.source_id,
        source.project_slug, source.task_id, source.scope_slug, source.user_id, source.company_id,
        checksum, status, error,
    )
    if row is not None:
        return row["id"], True

    # Checksum unchanged (or conflicting row untouched by the WHERE clause) - fetch its
    # id without touching chunks.
    existing = await conn.fetchrow(
        'SELECT id FROM "RagSource" WHERE "sourceType" = $1::"RagSourceType" AND "sourceId" = $2',
        source.source_type, source.source_id,
    )
    return existing["id"], False


async def _replace_chunks(conn, source_id: int, organisation_slug: str, chunks: list[str]):
    await conn.execute('DELETE FROM "RagChunk" WHERE "sourceId" = $1', source_id)
    for index, content in enumerate(chunks):
        embedding = await retrieval_service.embed(content)
        await conn.execute(
            """
            INSERT INTO "RagChunk"
                ("sourceId", "organisationSlug", "chunkIndex", content, "tokenCount", embedding)
            VALUES ($1, $2, $3, $4, $5, $6::vector)
            """,
            source_id, organisation_slug, index, content, len(content), str(embedding),
        )


async def ingest_source(source: SourceRow) -> None:
    checksum = hashlib.sha256((source.content or "").encode("utf-8")).hexdigest()
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            if source.content is None:
                await _upsert_source(conn, source, checksum, "FAILED", "no extractable content")
                return

            source_id, changed = await _upsert_source(conn, source, checksum, "PROCESSING", None)
            if not changed:
                return  # content identical to last ingest - chunks are already current

            chunks = chunk_text(source.content)
            await _replace_chunks(conn, source_id, source.organisation_slug, chunks)
            await conn.execute(
                'UPDATE "RagSource" SET status = $1::"RagIngestStatus" WHERE id = $2',
                "COMPLETED", source_id,
            )


async def ingest_all(source_types: list[str] | None = None, concurrency: int = 4) -> None:
    specs = [s for s in (*_DOCUMENT_SOURCES, *_TEXT_SOURCES) if not source_types or s["source_type"] in source_types]
    semaphore = asyncio.Semaphore(concurrency)

    async def _run(source: SourceRow):
        async with semaphore:
            try:
                await ingest_source(source)
            except Exception:
                logger.exception("ingest failed for %s %s", source.source_type, source.source_id)

    tasks = []
    for spec in specs:
        is_document = spec in _DOCUMENT_SOURCES
        iterator = _iter_document_sources(spec) if is_document else _iter_text_sources(spec)
        async for source in iterator:
            tasks.append(asyncio.create_task(_run(source)))
    if tasks:
        await asyncio.gather(*tasks)


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    types = sys.argv[1:] or None
    asyncio.run(ingest_all(source_types=types))
