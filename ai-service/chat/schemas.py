from pydantic import BaseModel


# --- planning ---

class PlannedStep(BaseModel):
    tool: str
    # JSON-encoded object, e.g. '{"project_slug": "yeni-gate"}' - Gemini's
    # Developer API structured-output mode rejects an open-ended dict/
    # additionalProperties field, so args travels as a string and gets
    # json.loads'd by the caller (see chat/graph.py) instead.
    args_json: str
    rationale: str


class Plan(BaseModel):
    steps: list[PlannedStep]


# --- text-to-sql tool ---

class TextToSqlArgs(BaseModel):
    question: str


class GeneratedSql(BaseModel):
    sql: str


# --- chat api ---

class ChatIn(BaseModel):
    organisationSlug: str
    userId: int
    conversationId: int | None = None
    message: str


class ChatMessageOut(BaseModel):
    id: int
    role: str
    content: str
    createdAt: str
