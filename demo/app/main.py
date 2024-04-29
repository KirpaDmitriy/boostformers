from fastapi import FastAPI, UploadFile, HTTPException, File, status
from fastapi.responses import JSONResponse
import uuid
from pathlib import Path
# from demo.kirpandas.semantic_df import SemanticDataFrame

app = FastAPI()


DATA_DIR = "./data"
Path(DATA_DIR).mkdir(parents=True, exist_ok=True)


def get_data_path(dataset_id: str) -> str:
    return f"{DATA_DIR}/{dataset_id}.csv"


def save_data(dataset_id: str, data: bytes) -> None:
    with open(get_data_path(dataset_id), "wb") as f:
        f.write(data)


@app.post("/upload")
async def upload_data(description: str, file: UploadFile = File(...)):
    try:
        dataset_id = str(uuid.uuid4())
        save_data(dataset_id, file.file.read())
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


@app.get("/")
async def hi():
    return {"Hello": "World"}
