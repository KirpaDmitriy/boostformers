import os
import uuid

import utils
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

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
async def upload_data(description: str, file: UploadFile = File(...)):
    try:
        dataset_id = str(uuid.uuid4())
        utils.save_data(dataset_id, file.file.read())
        utils.get_semantic_table(dataset_id)
        print(description)
        return JSONResponse(
            content={"dataset_id": dataset_id}, status_code=status.HTTP_201_CREATED
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


@app.post("/train_df")
async def upload_data(dataframe_id: str, targets: list[str | None]):
    if not os.path.exists(utils.get_dataframe_path(dataframe_id)):
        return HTTPException(
            status_code=404,
            detail="Dataset was not upload",
        )
    semantic_dataframe = utils.get_semantic_table(dataframe_id)
    utils.train_semantic_table(semantic_dataframe, targets)
    return {"dataset_id": dataframe_id}


@app.get("/search")
async def upload_data(dataframe_id: str, query: str):
    if not os.path.exists(utils.get_dataframe_path(dataframe_id)):
        return HTTPException(
            status_code=404,
            detail="Dataset was not upload",
        )
    semantic_dataframe = utils.get_semantic_table(dataframe_id)
    found = semantic_dataframe.search_lines(query)
    return found.to_json(orient="records")
