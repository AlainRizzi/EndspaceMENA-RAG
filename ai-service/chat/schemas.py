from pydantic import BaseModel


# --- planning ---

class PlannedStep(BaseModel):
    tool: str
    args: dict
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
