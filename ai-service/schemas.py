from pydantic import BaseModel


# --- suggest_skills ---

class SuggestSkillsInput(BaseModel):
    subdomainName: str
    organisationSlug: str | None = None
    title: str
    description: str
    departmentId: int


class SubtaskSuggestion(BaseModel):
    title: str
    description: str
    requiredSkill: str | None = None


class SkillSuggestionResult(BaseModel):
    subtasks: list[SubtaskSuggestion]


# --- summarize_project ---

class SummarizeProjectInput(BaseModel):
    subdomainName: str
    organisationSlug: str | None = None
    projectId: int
    projectSlug: str


class ProjectSummaryResult(BaseModel):
    summary: str
    keyRisks: list[str] = []
    nextSteps: list[str] = []
