import os
import uuid

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .utils import (get_dataframe_path, get_semantic_table, save_data,
                    train_semantic_table)
from .validation import UploadHandlerPostBody

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/ping")
async def ping():
    return "pong"


@app.post("/upload")
async def upload_data(file: UploadFile = File(...)):
    try:
        dataframe_id = str(uuid.uuid4())
        save_data(dataframe_id, file.file.read())
        get_semantic_table(dataframe_id)
        return JSONResponse(
            content={"dataframe_id": dataframe_id}, status_code=status.HTTP_201_CREATED
        )
    except UnicodeEncodeError:
        return HTTPException(
            status_code=422,
            detail="Dataset has either incorrect format or broken encoding",
        )
    except Exception as failure:
        print(f"/upload failed with error: {failure}")
        return HTTPException(
            status_code=422,
            detail="Dataset is broken",
        )
    finally:
        file.file.close()


@app.post("/train_dataset")
async def upload_data(dataframe_id: str, body_metadata: UploadHandlerPostBody):
    if not os.path.exists(get_dataframe_path(dataframe_id)):
        return HTTPException(
            status_code=404,
            detail="Dataset was not upload",
        )
    semantic_dataframe = get_semantic_table(dataframe_id)
    semantic_dataframe.set_description(body_metadata.description)
    train_semantic_table(dataframe_id, semantic_dataframe, body_metadata.targets)
    return JSONResponse(
        content={"dataframe_id": dataframe_id}, status_code=status.HTTP_201_CREATED
    )


@app.get("/search")
async def upload_data(query: str, dataframe_id: str | None = None):
    if not os.path.exists(get_dataframe_path(dataframe_id)):
        return HTTPException(
            status_code=404,
            detail="Dataset was not upload",
        )
    if dataframe_id:
        semantic_dataframe = get_semantic_table(dataframe_id)
        found = semantic_dataframe.search_lines(query)
        return found.to_json(orient="records")

    return {}
