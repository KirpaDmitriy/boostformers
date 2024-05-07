import os
import uuid

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import demo_app.utils.dataframe as df_utils

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
async def upload_data(description: str | None = None, file: UploadFile = File(...)):
    try:
        dataframe_id = str(uuid.uuid4())
        df_utils.save_data(dataframe_id, file.file.read())
        semantic_dataframe = df_utils.get_semantic_table(dataframe_id)
        if description:
            semantic_dataframe.set_description(description)
            df_utils.dataFramesStorage.add_semantic_dataframe(semantic_dataframe)
        return JSONResponse(
            content={"dataframe_id": dataframe_id}, status_code=status.HTTP_201_CREATED
        )
    except UnicodeEncodeError:
        return HTTPException(
            status_code=422,
            detail="Dataframe has either incorrect format or broken encoding",
        )
    except Exception as failure:
        print(f"/upload failed with error: {failure}")
        return HTTPException(
            status_code=422,
            detail="Dataframe is broken",
        )
    finally:
        file.file.close()


@app.post("/train_dataframe")
async def train_dataframe(dataframe_id: str, body_metadata: UploadHandlerPostBody):
    if not os.path.exists(df_utils.get_dataframe_path(dataframe_id)):
        return HTTPException(
            status_code=404,
            detail="Dataframe was not upload",
        )
    semantic_dataframe = df_utils.get_semantic_table(dataframe_id)
    if body_metadata.description:
        semantic_dataframe.set_description(body_metadata.description)
        df_utils.dataFramesStorage.add_semantic_dataframe(semantic_dataframe)
    df_utils.dataFramesStorage.add_semantic_dataframe(dataframe_id, semantic_dataframe)
    df_utils.train_semantic_table(
        dataframe_id, semantic_dataframe, body_metadata.targets
    )
    return JSONResponse(
        content={"dataframe_id": dataframe_id}, status_code=status.HTTP_200_OK
    )


@app.get("/search")
async def search(query: str, dataframe_id: str | None = None, n_results: int = 5):
    if not os.path.exists(df_utils.get_dataframe_path(dataframe_id)):
        return HTTPException(
            status_code=404,
            detail="Dataframe was not upload",
        )
    if dataframe_id:
        semantic_dataframe = df_utils.get_semantic_table(dataframe_id)
        found = semantic_dataframe.search_lines(query, n_results=n_results)
        return found.to_json(orient="records")

    return [
        (
            found_dataframe_id,
            df_utils.get_semantic_table(found_dataframe_id).description,
        )
        for found_dataframe_id in df_utils.dataFramesStorage.search_dataframes(
            query, n_results=n_results
        )
    ]
