from typing import Any

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)


class SourceItem(BaseModel):
    filename: str
    chunk_id: int


class AssistantAnswer(BaseModel):
    answer: str
    sources: list[SourceItem]
    used_tool: str | None = None
    grounded: bool


class IngestResponse(BaseModel):
    filename: str
    chunks_indexed: int


class AgentStep(BaseModel):
    iteration: int
    action: str
    reason: str = ""
    ok: bool = True
    detail: dict[str, Any] = Field(default_factory=dict)


class AgentAnswer(BaseModel):
    answer: str
    sources: list[SourceItem] = Field(default_factory=list)
    grounded: bool
    status: str
    iterations: int
    total_tokens: int
    trajectory: list[AgentStep] = Field(default_factory=list)
    clarification_question: str | None = None
