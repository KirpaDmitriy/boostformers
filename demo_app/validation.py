from pydantic import BaseModel


class UploadHandlerPostBody(BaseModel):
    description: str | None = None
    targets: list[str | None]
