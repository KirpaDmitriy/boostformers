from fastapi import FastAPI, UploadFile, HTTPException, File, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uuid
from pathlib import Path
import pandas as pd
import os
from functools import lru_cache
from demo.kirpandas.semantic_df import SemanticDataFrame

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = "./data"
Path(DATA_DIR).mkdir(parents=True, exist_ok=True)


def load_dataframe(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        if len(df.columns) == 1 and ";" in df.columns[0]:
          df = pd.read_csv(path, sep=";")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="cp1251")
        if len(df.columns) == 1 and ";" in df.columns[0]:
          df = pd.read_csv(path, sep=";", encoding="cp1251")

    if df is None:
        raise UnicodeDecodeError

    return df


def get_dataframe_path(dataset_id: str) -> str:
    return f"{DATA_DIR}/{dataset_id}.csv"


def save_data(dataset_id: str, data: bytes) -> None:
    with open(get_dataframe_path(dataset_id), "wb") as f:
        f.write(data)


@lru_cache(maxsize=500)
def get_semantic_table(dataset_id: str):
    dataframe = load_dataframe(get_dataframe_path(dataset_id))
    return SemanticDataFrame(
        dataframe,
        columns=dataframe.columns,
    )


@app.get("/ping")
async def ping():
    return "pong"


@app.post("/upload")
async def upload_data(description: str, file: UploadFile = File(...)):
    try:
        dataset_id = str(uuid.uuid4())
        save_data(dataset_id, file.file.read())
        get_semantic_table(dataset_id)
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
    if not os.path.exists(get_dataframe_path(dataframe_id)):
        return HTTPException(
            status_code=404,
            detail="Dataset was not upload",
        )
    semantic_dataframe = get_semantic_table(dataframe_id)
    semantic_dataframe.set_train_texts(pd.Series(targets))
    semantic_dataframe.fit_embedder()
    _ = semantic_dataframe.embeddings_index
    return "Ok"


@app.get("/search")
async def upload_data(dataframe_id: str, query: str):
    if not os.path.exists(get_dataframe_path(dataframe_id)):
        return HTTPException(
            status_code=404,
            detail="Dataset was not upload",
        )
    semantic_dataframe = get_semantic_table(dataframe_id)
    found = semantic_dataframe.search_lines(query)
    return found.to_json(orient="records")
