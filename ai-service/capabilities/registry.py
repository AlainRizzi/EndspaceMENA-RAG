from capabilities.base import AiCapability
from capabilities.suggest_skills import SuggestSkillsCapability
from capabilities.summarize_project import SummarizeProjectCapability

# Adding a new AI feature = write one AiCapability subclass + add it here.
# Nothing else in the service needs to change.
CAPABILITIES: dict[str, AiCapability] = {
    "suggest_skills": SuggestSkillsCapability(),
    "summarize_project": SummarizeProjectCapability(),
}
