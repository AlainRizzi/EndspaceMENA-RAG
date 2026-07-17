import io
from urllib.parse import unquote, urlparse

import boto3
from docx import Document as DocxDocument
from pypdf import PdfReader

from config import settings

_TEXT_EXTENSIONS = {"txt", "csv", "json", "md"}

_s3 = boto3.client(
    "s3",
    aws_access_key_id=settings.aws_access_key_id,
    aws_secret_access_key=settings.aws_access_key_secret,
    region_name=settings.aws_region,
)


def parse_s3_url(url: str) -> tuple[str, str]:
    """Documents store the full https://<bucket>.s3.<region>.amazonaws.com/<key> URL,
    not a bare key, so the bucket must be parsed out per-document rather than assumed
    from S3_BUCKET_NAME (older rows may point at a different bucket/environment).
    """
    parsed = urlparse(url)
    bucket = parsed.netloc.split(".s3.")[0]
    key = unquote(parsed.path.lstrip("/"))
    return bucket, key


def extract_text(file_bytes: bytes, file_type: str) -> str | None:
    """Best-effort text extraction. Returns None for types we don't know how to
    read (images, unrecognized binary, scanned/text-less PDFs) rather than raising,
    so ingestion can skip those sources without failing the whole batch.
    """
    ext = file_type.lower().lstrip(".").split("/")[-1]

    if ext == "pdf":
        reader = PdfReader(io.BytesIO(file_bytes))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages).strip() or None

    if ext == "docx":
        doc = DocxDocument(io.BytesIO(file_bytes))
        return "\n".join(p.text for p in doc.paragraphs).strip() or None

    if ext in _TEXT_EXTENSIONS:
        return file_bytes.decode("utf-8", errors="ignore").strip() or None

    return None


def fetch_and_extract(url: str, file_type: str) -> str | None:
    bucket, key = parse_s3_url(url)
    obj = _s3.get_object(Bucket=bucket, Key=key)
    return extract_text(obj["Body"].read(), file_type)
