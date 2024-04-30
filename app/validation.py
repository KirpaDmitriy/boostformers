from pydantic import BaseModel


class UploadHandlerPostBody(BaseModel):
    description: str
    # targets: list[str | None]
