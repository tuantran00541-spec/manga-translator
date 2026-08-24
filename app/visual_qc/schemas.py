from pydantic import BaseModel, field_validator


class VisualQCChapterRequest(BaseModel):
    chapter_id: str
    concurrency: int = 2

    @field_validator("concurrency")
    @classmethod
    def _bounded_concurrency(cls, value: int) -> int:
        if value < 1 or value > 4:
            raise ValueError("concurrency must be between 1 and 4")
        return value
